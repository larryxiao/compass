"""Exp 3 runner -- 2-turn leave-taking probe.

Per (opener × model): one real turn-1 API call, then for each of the 3
conditions, a turn-2 call where the leave-taking phrasing is the only
varying element (turn-1 user + real turn-1 assistant text are held constant).

Models (per task spec):
  - gpt-5.5 via southcentralus (the task pins this region)
  - gpt-5.4
  - gpt-5.4-nano
  - gpt-4o
  - gpt-4o-mini

Budget structure:
  20 openers × 5 models × (1 turn-1 + 3 turn-2) = 400 calls.

Output: responses.jsonl, appended row-per-call. Idempotent on
(seed_id, model, turn).
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

# Reuse the env loader + cost guard from the precompute helper module.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "precompute"))
from common import (  # noqa: E402
    Endpoint, GEMINI_ENDPOINTS, MTD_EMERGENCY_STOP, SoftCostGuard,
    append_jsonl, build_vertex_client, estimate_cost, is_gemini,
    is_temp_locked, load_env_local, read_jsonl, vertex_chat,
)

PROMPTS_PATH = HERE / "prompts" / "conversations.jsonl"
OUT_PATH = HERE / "responses.jsonl"

SYSTEM_PROMPT = "You are a helpful assistant."


# Azure GPT endpoints (sunset 2026-05-20 — SKIPPED if AOAI_KEY_* env absent).
ENDPOINTS: list[Endpoint] = [
    Endpoint("gpt-5.5",     "https://southcentralus.api.cognitive.microsoft.com/",
             "2024-10-21", "aoai-southcentralus",
             "AOAI_KEY_AOAI_2_SOUTHCENTRALUS", 10000, "southcentralus"),
    Endpoint("gpt-5.4",     "https://your-aoai-resource-2.cognitiveservices.azure.com/",
             "2024-10-21", "gpt-5.4",
             "AOAI_KEY_AOAI_RESOURCE_2", 5000, "eastus2"),
    Endpoint("gpt-5.4-nano", "https://your-aoai-resource-2.cognitiveservices.azure.com/",
             "2024-10-21", "gpt-5.4-nano",
             "AOAI_KEY_AOAI_RESOURCE_2", 75000, "eastus2"),
    Endpoint("gpt-4o",      "https://your-aoai-resource-1.cognitiveservices.azure.com/",
             "2024-10-21", "gpt-4o",
             "AOAI_KEY_AOAI_RESOURCE_1", 450, "eastus"),
    Endpoint("gpt-4o-mini", "https://your-aoai-resource-1.cognitiveservices.azure.com/",
             "2024-10-21", "gpt-4o-mini",
             "AOAI_KEY_AOAI_RESOURCE_1", 1021, "eastus"),
] + list(GEMINI_ENDPOINTS)


def is_gpt5(name: str) -> bool:
    n = name.lower()
    return n.startswith("gpt-5")


async def call_one(ep: Endpoint, client,
                   messages: list[dict]) -> dict:
    max_tok = 4000 if (is_gpt5(ep.name) or is_gemini(ep.name)) else 1200

    if ep.provider == "vertex":
        system_msg = next((m["content"] for m in messages if m.get("role") == "system"), SYSTEM_PROMPT)
        return await vertex_chat(
            ep, client,
            system_prompt=system_msg,
            messages=messages,
            max_output_tokens=max_tok,
            thinking_budget=max_tok,
            temperature=0.7,
        )

    temp_locked = is_temp_locked(ep.deployment, ep.name)
    kwargs = {
        "model": ep.deployment,
        "messages": messages,
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


async def main():
    load_env_local()
    seeds = read_jsonl(PROMPTS_PATH)
    if not seeds:
        raise SystemExit(f"no seeds at {PROMPTS_PATH}")
    print(f"loaded {len(seeds)} seeds")

    existing = read_jsonl(OUT_PATH)
    have = {(r["seed_id"], r["model"], r["turn"]) for r in existing
            if not r.get("error") and r.get("response")}
    have_turn1 = {(r["opener_id"], r["model"]): r["response"]
                  for r in existing if r.get("turn") == 1 and r.get("response")
                  and not r.get("error")}
    print(f"already have {len(have)} valid call records, "
          f"{len(have_turn1)} unique turn-1 responses cached")

    api_keys = {k: v for k, v in os.environ.items() if k.startswith("AOAI_KEY_")}
    clients = build_clients(api_keys)
    if not clients:
        raise SystemExit("no clients constructed -- env not loaded?")
    print(f"clients: {list(clients)}")

    cost_guard = SoftCostGuard(MTD_EMERGENCY_STOP, poll_seconds=600)

    # Per-model semaphore. Match WS-B pattern: 4 in-flight is plenty for our
    # TPM caps and avoids hammering the small gpt-4o quota (450 tpm).
    SEM_PER_MODEL = {
        "gpt-5.5": 4, "gpt-5.4": 3, "gpt-5.4-nano": 6,
        "gpt-4o": 2, "gpt-4o-mini": 3,
        "gemini-3.1-pro-preview": 2, "gemini-3.5-flash": 3,
        "gemini-3.1-flash-lite": 3, "gemini-2.5-pro": 2,
        "gemini-2.5-flash": 3, "gemini-2.5-flash-lite": 3,
    }
    sems = {m: asyncio.Semaphore(SEM_PER_MODEL.get(m, 3)) for m in clients}
    lock = asyncio.Lock()
    turn1_lock = asyncio.Lock()
    turn1_cache: dict[tuple[str, str], str] = dict(have_turn1)
    inflight_turn1: dict[tuple[str, str], asyncio.Event] = {}
    n_ok = n_err = 0
    total_cost = 0.0

    # Group seeds by (opener_id, model) and ensure turn-1 happens before turn-2.
    seeds_by_opener: dict[str, list[dict]] = {}
    for s in seeds:
        seeds_by_opener.setdefault(s["opener_id"], []).append(s)

    async def ensure_turn1(opener_id: str, seed: dict, model: str) -> str | None:
        """Get-or-create the turn-1 assistant response for (opener, model)."""
        nonlocal n_ok, n_err, total_cost
        key = (opener_id, model)
        async with turn1_lock:
            if key in turn1_cache:
                return turn1_cache[key]
            ev = inflight_turn1.get(key)
            if ev is None:
                ev = asyncio.Event()
                inflight_turn1[key] = ev
                first = True
            else:
                first = False
        if not first:
            await ev.wait()
            return turn1_cache.get(key)

        ep, client = clients[model]
        async with sems[model]:
            wait = ep.admit(tokens_estimated=2000)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                cost_guard.maybe_check()
            except Exception as e:
                print(f"[cost guard] turn1 abort: {e}", file=sys.stderr)
                async with turn1_lock:
                    ev.set()
                return None
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": seed["turn1_user"]},
            ]
            res = await call_one(ep, client, msgs)
            ep.record(res.get("prompt_tokens", 0) + res.get("completion_tokens", 0))

        rec = {
            "seed_id": None,
            "opener_id": opener_id,
            "context": seed["context"],
            "condition": None,
            "model": model,
            "deployment": ep.deployment,
            "region": ep.region,
            "turn": 1,
            "turn1_user": seed["turn1_user"],
            "turn2_user": None,
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
        async with turn1_lock:
            if res.get("response") and not res.get("error"):
                turn1_cache[key] = res["response"]
            ev.set()
        return turn1_cache.get(key)

    async def do_turn2(seed: dict, model: str, turn1_assistant: str):
        nonlocal n_ok, n_err, total_cost
        key3 = (seed["seed_id"], model, 2)
        if key3 in have:
            return
        ep, client = clients[model]
        async with sems[model]:
            wait = ep.admit(tokens_estimated=2500)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                cost_guard.maybe_check()
            except Exception as e:
                print(f"[cost guard] turn2 abort: {e}", file=sys.stderr)
                return
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": seed["turn1_user"]},
                {"role": "assistant", "content": turn1_assistant},
                {"role": "user", "content": seed["turn2_user"]},
            ]
            res = await call_one(ep, client, msgs)
            ep.record(res.get("prompt_tokens", 0) + res.get("completion_tokens", 0))

        rec = {
            "seed_id": seed["seed_id"],
            "opener_id": seed["opener_id"],
            "context": seed["context"],
            "condition": seed["condition"],
            "model": model,
            "deployment": ep.deployment,
            "region": ep.region,
            "turn": 2,
            "turn1_user": seed["turn1_user"],
            "turn2_user": seed["turn2_user"],
            "turn1_assistant": turn1_assistant,
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
            if done % 20 == 0:
                print(f"  progress: {n_ok} ok, {n_err} err, est ${total_cost:.3f}")

    async def do_opener_model(opener_seeds: list[dict], model: str):
        # All 3 condition-seeds for one opener share the same turn-1 prompt.
        first = opener_seeds[0]
        turn1 = await ensure_turn1(first["opener_id"], first, model)
        if not turn1:
            print(f"[skip turn2] no turn1 for {first['opener_id']} @ {model}",
                  file=sys.stderr)
            return
        await asyncio.gather(*[do_turn2(s, model, turn1) for s in opener_seeds])

    tasks = []
    for opener_id, opener_seeds in seeds_by_opener.items():
        for model in clients:
            tasks.append(asyncio.create_task(do_opener_model(opener_seeds, model)))
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
