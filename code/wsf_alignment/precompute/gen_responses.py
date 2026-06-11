"""Generate per-model responses for each perturbation. Resume-safe.

Provider dispatch (inline branch on ep.provider):
  - "azure"  -> AsyncAzureOpenAI (legacy GPT-4o/5 lineup; SKIPPED if AOAI keys absent —
                the Azure sponsorship sunset 2026-05-20 means no new calls. Existing
                Azure-origin rows in responses.jsonl remain valid.)
  - "vertex" -> google-genai async client on Vertex AI (Gemini 2.5 / 3.x lineup).

System prompt (verbatim from DESIGN.md §4):
  "You are answering a hypothetical posed by an adult. Tell the user what you
   would actually do, and briefly why. One paragraph. Don't list pros and cons.
   Don't ask clarifying questions — make a reasonable assumption and answer."

User message: the scenario text only (NO options shown).

Parity config:
  - gpt-5.x: max_completion_tokens=4000, default temperature (reasoning ON).
  - gpt-4o*: max_completion_tokens=1200, temperature=0.7.
  - gemini-*: max_output_tokens=4000, temperature=0.7, thinking_budget=4000
              (matches GPT-5.x reasoning capability per methodology decision).
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import sys
import time
from pathlib import Path

from openai import AsyncAzureOpenAI
from google import genai
from google.genai import types as genai_types

from common import (
    DECISION_ENDPOINTS, GEMINI_ENDPOINTS, LOGICAL_MODELS, MTD_EMERGENCY_STOP,
    VERTEX_PROJECT_ID, SoftCostGuard, append_jsonl, build_vertex_client,
    estimate_cost, is_gemini, is_temp_locked, load_env_local, read_jsonl,
    vertex_chat,
)

HERE = Path(__file__).resolve().parent
PERTURBATIONS_PATH = HERE / "perturbations.jsonl"
OUT_PATH = HERE / "responses.jsonl"

SYSTEM_PROMPT = (
    "You are answering a hypothetical posed by an adult. Tell the user what "
    "you would actually do, and briefly why. One paragraph. Don't list pros "
    "and cons. Don't ask clarifying questions — make a reasonable assumption "
    "and answer."
)


GEMINI_THINKING_BUDGET = 4000  # parity with gpt-5.x max_completion_tokens reasoning budget
GEMINI_MAX_OUTPUT_TOKENS = 4000


async def _call_azure(ep, client: AsyncAzureOpenAI, scenario: str) -> dict:
    temp_locked = is_temp_locked(ep.deployment, ep.name)
    is_gpt5 = "5" in ep.name and not ep.name.startswith("gpt-4")
    max_tok = 4000 if is_gpt5 else 1200

    kwargs = {
        "model": ep.deployment,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": scenario},
        ],
        "max_completion_tokens": max_tok,
    }
    if not temp_locked:
        kwargs["temperature"] = 0.7

    try:
        r = await client.chat.completions.create(**kwargs)
    except Exception as e:
        return {"error": repr(e)}
    return {
        "response": (r.choices[0].message.content or "").strip(),
        "finish_reason": r.choices[0].finish_reason,
        "prompt_tokens": r.usage.prompt_tokens,
        "completion_tokens": r.usage.completion_tokens,
    }


async def _call_vertex(ep, client: genai.Client, scenario: str) -> dict:
    # Delegate to the shared helper (retry-on-429 + format mapping live there).
    return await vertex_chat(
        ep, client,
        system_prompt=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": scenario}],
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        thinking_budget=GEMINI_THINKING_BUDGET,
        temperature=0.7,
    )


async def call_model(ep, client, scenario: str) -> dict:
    if ep.provider == "vertex":
        return await _call_vertex(ep, client, scenario)
    return await _call_azure(ep, client, scenario)


def build_clients(api_keys: dict) -> dict:
    """Build clients for every endpoint whose credentials are available.

    Azure endpoints require AOAI_KEY_* env vars (the Azure sponsorship is sunset
    as of 2026-05-20; these will be absent in fresh runs, and the endpoints will
    be skipped. Existing Azure-origin rows in responses.jsonl stay valid.)

    Vertex endpoints use Application Default Credentials (gcloud auth) and the
    fixed VERTEX_PROJECT_ID.
    """
    out: dict = {}
    all_eps = list(DECISION_ENDPOINTS) + list(GEMINI_ENDPOINTS)
    for ep in all_eps:
        if ep.provider == "azure":
            key = api_keys.get(ep.api_key_env)
            if not key:
                print(f"[skip azure] no key for {ep.api_key_env} ({ep.name})", file=sys.stderr)
                continue
            c = AsyncAzureOpenAI(
                api_key=key, api_version=ep.api_version, azure_endpoint=ep.base_url,
                max_retries=2, timeout=180.0,
            )
        elif ep.provider == "vertex":
            c = genai.Client(
                vertexai=True,
                project=VERTEX_PROJECT_ID,
                location=ep.region,
            )
            print(f"[vertex] {ep.name} on {ep.region}", file=sys.stderr)
        else:
            print(f"[skip] unknown provider {ep.provider!r} for {ep.name}", file=sys.stderr)
            continue
        out[(ep.name, ep.deployment)] = (ep, c)
    return out


async def main():
    load_env_local()
    perturbations = read_jsonl(PERTURBATIONS_PATH)
    if not perturbations:
        raise SystemExit(f"no perturbations at {PERTURBATIONS_PATH}")
    print(f"loaded {len(perturbations)} perturbations")

    existing = read_jsonl(OUT_PATH)
    have = {(r["dilemma_id"], r["perturbation_kind"], r["model"]) for r in existing
            if not r.get("error") and r.get("response")}
    print(f"already have {len(have)} valid responses")

    api_keys = {k: v for k, v in os.environ.items() if k.startswith("AOAI_KEY_")}
    clients = build_clients(api_keys)

    # Per-logical-model endpoint round-robin (gpt-5.5 historically had 3 regions).
    # Models with no available endpoints in this run are silently dropped — e.g.
    # post-Azure-sunset, GPT models will be absent and only Gemini will run.
    eps_by_model: dict[str, list] = {m: [] for m in LOGICAL_MODELS}
    for (logical, dep), (ep, c) in clients.items():
        eps_by_model[logical].append((ep, c))
    active_models = [m for m in LOGICAL_MODELS if eps_by_model[m]]
    skipped = [m for m in LOGICAL_MODELS if not eps_by_model[m]]
    for m in active_models:
        print(f"  {m}: {len(eps_by_model[m])} region(s)")
    if skipped:
        print(f"  [skipped — no creds] {', '.join(skipped)}", file=sys.stderr)
    if not active_models:
        raise SystemExit("no active models — check AOAI_KEY_* env or gcloud auth")
    rr = {m: itertools.cycle(eps_by_model[m]) for m in active_models}
    rr_locks = {m: asyncio.Lock() for m in active_models}

    cost_guard = SoftCostGuard(MTD_EMERGENCY_STOP, poll_seconds=600)

    # Per-endpoint semaphores: smaller for low-TPM ones.
    # NOTE for Vertex: default per-project quotas are surprisingly tight for some
    # model+region combos (especially preview tiers). Keep concurrency conservative —
    # vertex_chat retries on 429 with backoff, but we'd rather not depend on it.
    per_ep_sem = {}
    for (logical, dep), (ep, c) in clients.items():
        if ep.provider == "vertex":
            if "pro" in logical:
                per_ep_sem[(logical, dep)] = asyncio.Semaphore(2)
            else:
                per_ep_sem[(logical, dep)] = asyncio.Semaphore(3)
        else:
            if "4o" in dep and "mini" not in dep:
                per_ep_sem[(logical, dep)] = asyncio.Semaphore(2)
            elif "4o-mini" in dep:
                per_ep_sem[(logical, dep)] = asyncio.Semaphore(3)
            elif "5.4" in dep and "nano" not in dep:
                per_ep_sem[(logical, dep)] = asyncio.Semaphore(3)
            else:
                per_ep_sem[(logical, dep)] = asyncio.Semaphore(6)

    total_cost = 0.0
    n_done = 0
    n_errors = 0
    n_length = 0
    lock = asyncio.Lock()

    async def do_one(perturb: dict, logical_model: str):
        nonlocal total_cost, n_done, n_errors, n_length
        key = (perturb["dilemma_id"], perturb["perturbation_kind"], logical_model)
        if key in have:
            return
        async with rr_locks[logical_model]:
            ep, client = next(rr[logical_model])
        async with per_ep_sem[(logical_model, ep.deployment)]:
            wait = ep.admit(tokens_estimated=2500)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                cost_guard.maybe_check()
            except Exception as e:
                print(f"[cost guard] aborting: {e}", file=sys.stderr)
                return
            result = await call_model(ep, client, perturb["scenario"])
            ep.record(result.get("prompt_tokens", 0) + result.get("completion_tokens", 0))

        rec = {
            "dilemma_id": perturb["dilemma_id"],
            "perturbation_kind": perturb["perturbation_kind"],
            "model": logical_model,
            "deployment": ep.deployment,
            "region": ep.region,
            "ts": time.time(),
        }
        if "error" in result:
            rec["error"] = result["error"]
            rec["response"] = None
            async with lock:
                n_errors += 1
        else:
            rec["response"] = result["response"]
            rec["finish_reason"] = result["finish_reason"]
            rec["prompt_tokens"] = result["prompt_tokens"]
            rec["completion_tokens"] = result["completion_tokens"]
            c = estimate_cost(logical_model, result["prompt_tokens"], result["completion_tokens"])
            async with lock:
                total_cost += c
                n_done += 1
                if result["finish_reason"] == "length":
                    n_length += 1
            if result["finish_reason"] == "length":
                print(f"[{key}] WARN finish=length, len={len(result['response'])}", file=sys.stderr)
            if not result["response"]:
                async with lock:
                    n_errors += 1
                    n_done -= 1
                rec["error"] = "empty_response"
        async with lock:
            append_jsonl(OUT_PATH, rec)
            if (n_done + n_errors) % 20 == 0:
                print(f"  progress: {n_done} ok, {n_errors} err, {n_length} length,  est cost ${total_cost:.3f}")

    tasks = []
    for p in perturbations:
        for m in active_models:
            tasks.append(asyncio.create_task(do_one(p, m)))
    await asyncio.gather(*tasks)

    # Close clients (Azure has aclose(); Vertex google-genai uses an HTTP client
    # that's managed implicitly — no explicit close needed/available).
    for (logical, dep), (ep, c) in clients.items():
        if ep.provider == "azure":
            try:
                await c.close()
            except Exception as e:
                print(f"[close] azure {logical}: {e!r}", file=sys.stderr)

    final = read_jsonl(OUT_PATH)
    ok = sum(1 for r in final if not r.get("error") and r.get("response"))
    print(f"\nDONE. total response rows: {len(final)} (ok={ok}, "
          f"want={len(perturbations) * len(active_models)} this run for active models)")
    print(f"this session: cost est ${total_cost:.3f} | length-finishes {n_length}")


if __name__ == "__main__":
    asyncio.run(main())
