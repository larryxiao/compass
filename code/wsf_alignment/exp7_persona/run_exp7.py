"""Exp 7 runner -- persona modulation probe.

For each of 75 persona-prompts (15 dilemmas x 5 personas) and each of 5 models,
make ONE generation call. System prompt = <persona_text> + "\n\n" + BASE_ELICIT.
User message = the dilemma scenario (byte-identical to Exp 2's V1 sourcing).

Models (per task spec; identical to Exp 2/3):
  - gpt-5.5 via southcentralus ONLY
  - gpt-5.4, gpt-5.4-nano (your-aoai-resource-2 / eastus2)
  - gpt-4o, gpt-4o-mini (your-aoai-resource-1 / eastus)

Total: 75 prompts x 5 models = 375 generation calls.

Idempotent on (prompt_id, model). Output: responses.jsonl (append).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncAzureOpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "precompute"))
from common import (  # noqa: E402
    Endpoint, GEMINI_ENDPOINTS, MTD_EMERGENCY_STOP, SoftCostGuard,
    append_jsonl, build_vertex_client, estimate_cost, is_gemini,
    is_temp_locked, load_env_local, read_jsonl, vertex_chat,
)

PROMPTS_PATH = HERE / "prompts" / "persona_prompts.jsonl"
OUT_PATH = HERE / "responses.jsonl"


# Azure GPT (sunset 2026-05-20) + Gemini from common.GEMINI_ENDPOINTS.
ENDPOINTS: list[Endpoint] = [
    Endpoint("gpt-5.5",      "https://southcentralus.api.cognitive.microsoft.com/",
             "2024-10-21", "aoai-southcentralus",
             "AOAI_KEY_AOAI_2_SOUTHCENTRALUS", 10000, "southcentralus"),
    Endpoint("gpt-5.4",      "https://your-aoai-resource-2.cognitiveservices.azure.com/",
             "2024-10-21", "gpt-5.4",
             "AOAI_KEY_AOAI_RESOURCE_2", 5000, "eastus2"),
    Endpoint("gpt-5.4-nano", "https://your-aoai-resource-2.cognitiveservices.azure.com/",
             "2024-10-21", "gpt-5.4-nano",
             "AOAI_KEY_AOAI_RESOURCE_2", 75000, "eastus2"),
    Endpoint("gpt-4o",       "https://your-aoai-resource-1.cognitiveservices.azure.com/",
             "2024-10-21", "gpt-4o",
             "AOAI_KEY_AOAI_RESOURCE_1", 450, "eastus"),
    Endpoint("gpt-4o-mini",  "https://your-aoai-resource-1.cognitiveservices.azure.com/",
             "2024-10-21", "gpt-4o-mini",
             "AOAI_KEY_AOAI_RESOURCE_1", 1021, "eastus"),
] + list(GEMINI_ENDPOINTS)


def is_gpt5(name: str) -> bool:
    return name.lower().startswith("gpt-5")


async def call_one(ep: Endpoint, client,
                   messages: list[dict]) -> dict:
    max_tok = 4000 if (is_gpt5(ep.name) or is_gemini(ep.name)) else 1200

    if ep.provider == "vertex":
        system_msg = next((m["content"] for m in messages if m.get("role") == "system"), None)
        return await vertex_chat(
            ep, client,
            system_prompt=system_msg,
            messages=messages,
            max_output_tokens=max_tok,
            thinking_budget=max_tok,
            temperature=1.0,
        )

    temp_locked = is_temp_locked(ep.deployment, ep.name)
    kwargs = {
        "model": ep.deployment,
        "messages": messages,
        "max_completion_tokens": max_tok,
    }
    if not temp_locked:
        kwargs["temperature"] = 1.0
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


def build_clients(api_keys: dict) -> dict[str, tuple[Endpoint, object]]:
    out = {}
    for ep in ENDPOINTS:
        if ep.provider == "vertex":
            out[ep.name] = (ep, build_vertex_client(ep))
            continue
        key = api_keys.get(ep.api_key_env)
        if not key:
            print(f"[skip azure] no key for {ep.api_key_env} ({ep.name})", file=sys.stderr)
            continue
        c = AsyncAzureOpenAI(
            api_key=key, api_version=ep.api_version, azure_endpoint=ep.base_url,
            max_retries=4, timeout=180.0,
        )
        out[ep.name] = (ep, c)
    return out


async def main() -> None:
    load_env_local()
    prompts = read_jsonl(PROMPTS_PATH)
    if not prompts:
        raise SystemExit(f"no prompts at {PROMPTS_PATH}")
    print(f"loaded {len(prompts)} persona-prompts (15 dilemmas x 5 personas)")

    existing = read_jsonl(OUT_PATH)
    have = {(r["prompt_id"], r["model"]) for r in existing
            if not r.get("error") and r.get("response")}
    print(f"already have {len(have)} valid responses")

    api_keys = {k: v for k, v in os.environ.items() if k.startswith("AOAI_KEY_")}
    clients = build_clients(api_keys)
    if not clients:
        raise SystemExit("no clients constructed -- env not loaded?")
    print(f"clients: {list(clients)}")

    cost_guard = SoftCostGuard(MTD_EMERGENCY_STOP, poll_seconds=600)

    SEM_PER_MODEL = {
        "gpt-5.5": 4, "gpt-5.4": 3, "gpt-5.4-nano": 6,
        "gpt-4o": 2, "gpt-4o-mini": 3,
        "gemini-3.1-pro-preview": 2, "gemini-3.5-flash": 3,
        "gemini-3.1-flash-lite": 3, "gemini-2.5-pro": 2,
        "gemini-2.5-flash": 3, "gemini-2.5-flash-lite": 3,
    }
    sems = {m: asyncio.Semaphore(SEM_PER_MODEL.get(m, 3)) for m in clients}
    lock = asyncio.Lock()
    total_cost = 0.0
    n_ok = n_err = 0

    async def do_one(prompt: dict, model: str) -> None:
        nonlocal n_ok, n_err, total_cost
        key = (prompt["prompt_id"], model)
        if key in have:
            return
        ep, client = clients[model]
        async with sems[model]:
            wait = ep.admit(tokens_estimated=2500)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                cost_guard.maybe_check()
            except Exception as e:
                print(f"[cost guard] aborting: {e}", file=sys.stderr)
                return
            msgs = [
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": prompt["scenario"]},
            ]
            res = await call_one(ep, client, msgs)
            ep.record(res.get("prompt_tokens", 0) + res.get("completion_tokens", 0))

        rec = {
            "prompt_id": prompt["prompt_id"],
            "dilemma_id": prompt["dilemma_id"],
            "persona_id": prompt["persona_id"],
            "persona_name": prompt["persona_name"],
            "category": prompt["category"],
            "axes_in_play": prompt["axes_in_play"],
            "model": model,
            "deployment": ep.deployment,
            "region": ep.region,
            "system_prompt": prompt["system_prompt"],
            "scenario": prompt["scenario"],
            "response": res.get("response"),
            "finish_reason": res.get("finish_reason"),
            "prompt_tokens": res.get("prompt_tokens"),
            "completion_tokens": res.get("completion_tokens"),
            "error": res.get("error"),
            "ts": time.time(),
        }
        c = estimate_cost(model, res.get("prompt_tokens", 0),
                          res.get("completion_tokens", 0))
        async with lock:
            append_jsonl(OUT_PATH, rec)
            total_cost += c
            if res.get("error") or not res.get("response"):
                n_err += 1
            else:
                n_ok += 1
            done = n_ok + n_err
            if done % 25 == 0:
                print(f"  progress: {n_ok} ok, {n_err} err, est ${total_cost:.3f}")

    tasks = []
    for p in prompts:
        for m in clients:
            tasks.append(asyncio.create_task(do_one(p, m)))
    await asyncio.gather(*tasks)

    for ep, c in clients.values():
        if ep.provider == "azure":
            try:
                await c.close()
            except Exception as e:
                print(f"[close] {ep.name}: {e!r}", file=sys.stderr)

    final = read_jsonl(OUT_PATH)
    ok = sum(1 for r in final if r.get("response") and not r.get("error"))
    print(f"\nDONE. {len(final)} rows ({ok} ok). this session: est ${total_cost:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
