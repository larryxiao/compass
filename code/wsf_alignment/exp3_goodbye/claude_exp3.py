"""Claude probe for exp3 (2-turn leave-taking retention probe). Reproduces
run_exp3.py's exact 2-turn message build:

  Turn 1 (per opener x model, shared across the 3 conditions):
      system = SYSTEM_PROMPT, user = seed["turn1_user"]
  Turn 2 (per seed/condition x model):
      system = SYSTEM_PROMPT,
      user = seed["turn1_user"], assistant = <real turn-1 text>,
      user = seed["turn2_user"]

SYSTEM_PROMPT is copied VERBATIM from run_exp3.py. The 3 conditions
(terminal / continuing / neutral) differ ONLY in seed["turn2_user"], which is
already baked into prompts/conversations.jsonl; the runner does not add any
extra per-condition text, so neither do we.

Each (opener, model) gets one real turn-1 call; that assistant text is held
constant across the opener's 3 turn-2 calls (claude_cli.pack packs the
[user, assistant, user] transcript). Resume-safe on the runner's idempotency
key (seed_id, model, turn); turn-1 rows carry seed_id=None like the runner.
Appends Claude rows (elicitation="claude_code_cli") to responses.jsonl.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "precompute"))
from claude_cli import CLAUDE_EXP_MODELS, claude_chat, read_jsonl, run_batch  # noqa: E402

PROMPTS_PATH = HERE / "prompts" / "conversations.jsonl"
OUT_PATH = HERE / "responses.jsonl"

# Copied VERBATIM from run_exp3.py.
SYSTEM_PROMPT = "You are a helpful assistant."


def build_turn1_row(item):
    """item = (opener_id, seed, model). One real turn-1 call. Mirrors the
    runner's ensure_turn1 row: seed_id=None, condition=None, turn=1."""
    opener_id, seed, model = item
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": seed["turn1_user"]},
    ]
    res = claude_chat(model, msgs)
    rec = {
        "seed_id": None,
        "opener_id": opener_id,
        "context": seed["context"],
        "condition": None,
        "model": model,
        "deployment": model,
        "region": "claude_code_cli",
        "elicitation": "claude_code_cli",
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
    if res.get("cost_usd") is not None:
        rec["cost_usd"] = res["cost_usd"]
    if res.get("duration_ms") is not None:
        rec["duration_ms"] = res["duration_ms"]
    if "error" in res:
        rec["response"] = None
    return rec


def build_turn2_row(item):
    """item = (seed, model, turn1_assistant). One turn-2 call where the
    leave-taking phrasing (seed["turn2_user"]) is the only varying element."""
    seed, model, turn1_assistant = item
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": seed["turn1_user"]},
        {"role": "assistant", "content": turn1_assistant},
        {"role": "user", "content": seed["turn2_user"]},
    ]
    res = claude_chat(model, msgs)
    rec = {
        "seed_id": seed["seed_id"],
        "opener_id": seed["opener_id"],
        "context": seed["context"],
        "condition": seed["condition"],
        "model": model,
        "deployment": model,
        "region": "claude_code_cli",
        "elicitation": "claude_code_cli",
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
    if res.get("cost_usd") is not None:
        rec["cost_usd"] = res["cost_usd"]
    if res.get("duration_ms") is not None:
        rec["duration_ms"] = res["duration_ms"]
    if "error" in res:
        rec["response"] = None
    return rec


def run(models=None, limit=None):
    models = models or CLAUDE_EXP_MODELS
    seeds = read_jsonl(PROMPTS_PATH)

    # Group seeds by opener (the 3 condition-seeds for one opener share turn-1).
    seeds_by_opener: dict[str, list[dict]] = {}
    for s in seeds:
        seeds_by_opener.setdefault(s["opener_id"], []).append(s)

    existing = read_jsonl(OUT_PATH)
    # Runner idempotency key for valid records: (seed_id, model, turn).
    have = {(r["seed_id"], r["model"], r["turn"]) for r in existing
            if not r.get("error") and r.get("response")}
    # Cached turn-1 assistant text per (opener_id, model) -- Claude rows only,
    # so the side-series stays self-contained.
    turn1_cache = {(r["opener_id"], r["model"]): r["response"]
                   for r in existing
                   if r.get("turn") == 1 and r.get("response") and not r.get("error")
                   and r.get("elicitation") == "claude_code_cli"}

    total = {"ok": 0, "err": 0, "cost": 0.0}

    def merge(stats):
        total["ok"] += stats.get("ok", 0)
        total["err"] += stats.get("err", 0)
        total["cost"] += stats.get("cost", 0.0)

    # ===== Phase 1: turn-1 (one per opener x model, not yet cached) =====
    turn1_work = []
    for opener_id, opener_seeds in seeds_by_opener.items():
        first = opener_seeds[0]
        for m in models:
            if (opener_id, m) in turn1_cache:
                continue
            turn1_work.append((opener_id, first, m))
    if limit:
        turn1_work = turn1_work[:limit]
    print(f"exp3: {len(seeds_by_opener)} openers x {len(models)} models; "
          f"{len(turn1_work)} turn-1 to run", flush=True)
    if turn1_work:
        stats = run_batch(turn1_work, build_turn1_row, OUT_PATH, "exp3-t1")
        merge(stats)
        # Refresh the turn-1 cache from the rows just written.
        for r in read_jsonl(OUT_PATH):
            if (r.get("turn") == 1 and r.get("response") and not r.get("error")
                    and r.get("elicitation") == "claude_code_cli"):
                turn1_cache[(r["opener_id"], r["model"])] = r["response"]

    # ===== Phase 2: turn-2 (per seed/condition x model) =====
    turn2_work = []
    for s in seeds:
        for m in models:
            if (s["seed_id"], m, 2) in have:
                continue
            t1 = turn1_cache.get((s["opener_id"], m))
            if not t1:
                # No turn-1 for this (opener, model) -- skip like the runner does.
                continue
            turn2_work.append((s, m, t1))
    if limit:
        turn2_work = turn2_work[:limit]
    print(f"exp3: {len(turn2_work)} turn-2 to run", flush=True)
    if turn2_work:
        stats = run_batch(turn2_work, build_turn2_row, OUT_PATH, "exp3-t2")
        merge(stats)

    if not turn1_work and not turn2_work:
        print("exp3: nothing to do", flush=True)
    return total


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(limit=a.limit)
