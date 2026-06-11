"""Generate 2 perturbations per dilemma (60 rows total) via gpt-4o.

- original: as written
- gender_swap: swap gender on at least one named character
- reversed_rapport: subtle rewording shifting how the situation is framed,
  so the user's "preferred" outcome is less / more obvious.

Idempotent: skips dilemmas already present in perturbations.jsonl.

Output schema (one row per perturbation):
{
  "dilemma_id": "D001",
  "perturbation_kind": "original"|"gender_swap"|"reversed_rapport",
  "scenario": "...",
  "axes_in_play": [...],
  "options": [...],
  "judge_rubric": "...",
  "title": "...",
  "category": "..."
}
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from openai import AsyncAzureOpenAI

from common import (
    DECISION_ENDPOINTS, JUDGE_ENDPOINTS, PERTURBATION_KINDS,
    append_jsonl, load_env_local, read_jsonl, is_temp_locked,
)

HERE = Path(__file__).resolve().parent
DILEMMAS_PATH = HERE.parent / "dilemmas" / "dilemmas.jsonl"
OUT_PATH = HERE / "perturbations.jsonl"


PERTURB_SYSTEM = (
    "You are a careful editor. Given a moral-dilemma scenario, you produce a "
    "modified version per the user's instructions. You preserve the structure, "
    "stakes, and decision logic of the scenario. You output ONLY the rewritten "
    "scenario text — no preamble, no commentary."
)


def gender_swap_prompt(scenario: str, characters_hint: str) -> str:
    return (
        "Rewrite the following scenario, swapping the gender of the named "
        "characters where natural. Use new names of the swapped gender; keep "
        "every concrete detail otherwise identical (numbers, times, places, "
        "relationships). Keep the same word count (within ~15 words). "
        "Maintain the second-person, present-tense voice.\n\n"
        f"Named characters in the scenario: {characters_hint}\n\n"
        "ORIGINAL:\n"
        f"{scenario}\n\n"
        "REWRITTEN (only the rewritten scenario text):"
    )


def reversed_rapport_prompt(scenario: str) -> str:
    return (
        "Rewrite the following scenario with a subtle reframing that shifts "
        "the implicit sympathy. Keep ALL facts the same — same names, same "
        "numbers, same outcomes — but adjust 2-4 small wording choices so the "
        "narrator's emotional pull leans the OPPOSITE direction from the "
        "original. (E.g. if the original makes the 'tell the truth' option "
        "feel obviously right, your rewrite should make the 'protect the "
        "relationship' option feel obviously right — without changing what "
        "actually happened.) Keep the same word count (within ~15 words). "
        "Maintain the second-person, present-tense voice.\n\n"
        "ORIGINAL:\n"
        f"{scenario}\n\n"
        "REWRITTEN (only the rewritten scenario text):"
    )


def extract_characters(dilemma: dict) -> str:
    """Best-effort: scrape names from scenario + a few common templates."""
    scenario = dilemma["scenario"]
    # Heuristic: capitalized words 3+ letters, not at start of sentence,
    # excluding common nouns. The model gets this only as a hint anyway.
    import re
    candidates = set(re.findall(r"\b[A-Z][a-z]{2,}\b", scenario))
    # Drop obvious non-names
    stop = {"The", "You", "Your", "This", "That", "Sunday", "Monday", "Tuesday",
            "Wednesday", "Thursday", "Friday", "Saturday", "January", "February",
            "March", "April", "May", "June", "July", "August", "September",
            "October", "November", "December", "Italian", "Twitter", "TikTok",
            "Slack", "Series", "Facebook", "Reddit", "Instagram", "Christmas",
            "Mom", "Dad", "Mother", "Father", "ICE"}
    names = sorted(candidates - stop)
    if not names:
        return "(infer from text)"
    return ", ".join(names[:6])


async def gen_perturbation(client: AsyncAzureOpenAI, deployment: str,
                           prompt: str) -> tuple[str | None, dict]:
    kwargs = {
        "model": deployment,
        "messages": [
            {"role": "system", "content": PERTURB_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": 700,
    }
    if not is_temp_locked(deployment):
        kwargs["temperature"] = 0.7
    try:
        r = await client.chat.completions.create(**kwargs)
    except Exception as e:
        return None, {"error": repr(e)}
    text = (r.choices[0].message.content or "").strip()
    return text, {
        "finish_reason": r.choices[0].finish_reason,
        "prompt_tokens": r.usage.prompt_tokens,
        "completion_tokens": r.usage.completion_tokens,
    }


async def main():
    load_env_local()
    dilemmas = read_jsonl(DILEMMAS_PATH)
    if not dilemmas:
        print(f"no dilemmas at {DILEMMAS_PATH}", file=sys.stderr)
        return
    print(f"loaded {len(dilemmas)} dilemmas")

    existing = read_jsonl(OUT_PATH)
    have = {(r["dilemma_id"], r["perturbation_kind"]) for r in existing}
    print(f"existing perturbations on disk: {len(have)}")

    # Use gpt-4o for perturbation generation (good at edits, fast).
    ep = JUDGE_ENDPOINTS["gpt-4o"]
    import os
    api_key = os.environ.get(ep.api_key_env)
    if not api_key:
        raise SystemExit(f"missing env {ep.api_key_env}")
    client = AsyncAzureOpenAI(
        api_key=api_key, api_version=ep.api_version, azure_endpoint=ep.base_url,
        max_retries=2, timeout=120.0,
    )

    sem = asyncio.Semaphore(2)  # gpt-4o has 450 tpm cap; be gentle

    async def do_one(d: dict, kind: str):
        if (d["id"], kind) in have:
            return
        async with sem:
            if kind == "original":
                text = d["scenario"]
                meta = {"finish_reason": "n/a", "prompt_tokens": 0, "completion_tokens": 0}
            elif kind == "gender_swap":
                prompt = gender_swap_prompt(d["scenario"], extract_characters(d))
                text, meta = await gen_perturbation(client, ep.deployment, prompt)
            elif kind == "reversed_rapport":
                prompt = reversed_rapport_prompt(d["scenario"])
                text, meta = await gen_perturbation(client, ep.deployment, prompt)
            else:
                return
            if not text:
                print(f"[{d['id']}/{kind}] failed: {meta}")
                return
            rec = {
                "dilemma_id": d["id"],
                "title": d["title"],
                "category": d["category"],
                "perturbation_kind": kind,
                "scenario": text,
                "axes_in_play": d["axes_in_play"],
                "options": d["options"],
                "judge_rubric": d["judge_rubric"],
                "perturb_meta": meta,
                "ts": time.time(),
            }
            append_jsonl(OUT_PATH, rec)
            print(f"[{d['id']}/{kind}] wrote ({len(text)} chars)")

    tasks = []
    for d in dilemmas:
        for kind in PERTURBATION_KINDS:
            tasks.append(asyncio.create_task(do_one(d, kind)))
    await asyncio.gather(*tasks)

    await client.close()
    final = read_jsonl(OUT_PATH)
    print(f"\nDONE. total perturbation rows on disk: {len(final)}  "
          f"({len(dilemmas)} dilemmas × {len(PERTURBATION_KINDS)} kinds = "
          f"{len(dilemmas)*len(PERTURBATION_KINDS)} expected)")


if __name__ == "__main__":
    asyncio.run(main())
