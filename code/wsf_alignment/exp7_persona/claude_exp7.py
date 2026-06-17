"""Claude probe for exp7 (persona modulation). Reproduces run_exp7.py's exact
message build: system = prompt["system_prompt"] (persona + elicit), user =
prompt["scenario"]. One call per (prompt_id, model). Resume-safe; appends Claude
rows (elicitation="claude_code_cli") to responses.jsonl.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "precompute"))
from claude_cli import CLAUDE_EXP_MODELS, claude_chat, read_jsonl, run_batch  # noqa: E402

PROMPTS_PATH = HERE / "prompts" / "persona_prompts.jsonl"
OUT_PATH = HERE / "responses.jsonl"


def build_row(item):
    prompt, model = item
    msgs = [
        {"role": "system", "content": prompt["system_prompt"]},
        {"role": "user", "content": prompt["scenario"]},
    ]
    res = claude_chat(model, msgs)
    rec = {
        "prompt_id": prompt["prompt_id"], "dilemma_id": prompt["dilemma_id"],
        "persona_id": prompt["persona_id"], "persona_name": prompt["persona_name"],
        "category": prompt["category"], "axes_in_play": prompt["axes_in_play"],
        "model": model, "deployment": model, "region": "claude_code_cli",
        "elicitation": "claude_code_cli",
        "system_prompt": prompt["system_prompt"], "scenario": prompt["scenario"],
        "ts": time.time(),
    }
    rec.update(res)
    if "error" in res:
        rec["response"] = None
    return rec


def run(models=None, limit=None):
    models = models or CLAUDE_EXP_MODELS
    prompts = read_jsonl(PROMPTS_PATH)
    have = {(r["prompt_id"], r["model"]) for r in read_jsonl(OUT_PATH)
            if not r.get("error") and r.get("response")}
    work = [(p, m) for p in prompts for m in models if (p["prompt_id"], m) not in have]
    if limit:
        work = work[:limit]
    print(f"exp7: {len(prompts)} prompts x {len(models)} models; {len(work)} to run", flush=True)
    if not work:
        print("exp7: nothing to do", flush=True)
        return {"ok": 0, "err": 0, "cost": 0.0}
    return run_batch(work, build_row, OUT_PATH, "exp7")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(limit=a.limit)
