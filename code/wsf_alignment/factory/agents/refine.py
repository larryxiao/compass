"""Refiner agent — top-K selection + setup-prompt meta-learning.

Given:
  - the batch of candidate dilemmas from the current iteration
  - each candidate's evaluator aggregate (scores + critiques + suggestions)
  - the *current* setup-agent prompt template
  - target keep_top (how many candidates survive into the seed pool)

Returns:
  - kept: top-K candidate ids (ranked by mean evaluator score)
  - rejected: bottom candidate ids + reasons
  - new_prompt: a revised setup-agent prompt template (or None if no rewrite needed)
  - diagnostics: per-iteration metrics (mean score, pass rate, axis-coverage, etc.)

The prompt-rewrite step is the meta-learning loop. The refiner reads the
evaluator critiques and asks itself: which weaknesses are systematic? What
should I tell the setup agent NEXT time to avoid these mistakes? It produces
a diff (verbal description) and a new full prompt.

The new prompt must preserve all str.format slots — we verify after generation.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Any

from openai import AsyncAzureOpenAI


REFINE_SYSTEM_PROMPT = """You are a meta-prompt engineer optimizing a dilemma-generator. You read a batch of dilemma candidates the generator produced, the strict-reviewer critiques of those candidates, and the generator's current prompt template. Your job is to rewrite the prompt so the NEXT batch will score higher on the reviewer's rubric.

Specifically, look for SYSTEMATIC weaknesses across the batch:
- Are most candidates landing on generic names ("Sarah Chen") rather than specific names ("Maya from product")?
- Are options collapsing to "all equally reasonable" — losing the genuine moral tension?
- Are scenarios too short / too long / too abstract?
- Is the same axis combination being over-used?
- Are the judge_rubric descriptions vague keyword lists rather than reasoning-pattern descriptions?

Your output is a NEW VERSION of the prompt template. You MUST preserve every `{slot}` substitution placeholder exactly:
{axes_list}
{categories_list}
{axis_target}
{category_target}
{next_id}
{few_shot_block}

Return ONLY a single JSON object with this schema:
{
  "diagnosis": "<2-4 sentence summary of the systematic weaknesses you observed>",
  "changes": ["<short bullet>", ...],     # what you changed in the new prompt
  "new_prompt_template": "<the full revised prompt as a single string, with all {slot} placeholders preserved>"
}

If the batch already looks great and no changes would help, return diagnosis="no systematic weakness", changes=[], new_prompt_template=<the current prompt unchanged>.
"""


def _is_temp_locked(deployment: str) -> bool:
    d = deployment.lower()
    return ("gpt-5" in d) or ("5.4" in d) or ("5.5" in d) or d.startswith("o")


def _strip_code_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace > 0 and last_brace > first_brace:
        raw = raw[first_brace : last_brace + 1]
    return raw


REQUIRED_PROMPT_SLOTS = (
    "axes_list", "categories_list", "axis_target",
    "category_target", "next_id", "few_shot_block",
)


def _all_slots_present(prompt: str) -> tuple[bool, list[str]]:
    """Verify every required str.format slot is in the new prompt."""
    missing = []
    for slot in REQUIRED_PROMPT_SLOTS:
        if "{" + slot + "}" not in prompt:
            missing.append(slot)
    return len(missing) == 0, missing


def rank_and_select(
    evaluated: list[dict],
    keep_top: int,
    min_mean_score: float = 3.5,
) -> dict[str, Any]:
    """Pick top-K candidates by mean evaluator score, filter by floor.

    Args:
      evaluated: list of {"dilemma": ..., "evaluation": {"aggregated": {...}}}
      keep_top: max number to keep
      min_mean_score: minimum mean score to be kept

    Returns:
      {"kept": [...], "rejected": [...], "ranking": [...]}
    """
    scored = []
    for item in evaluated:
        agg = (item.get("evaluation") or {}).get("aggregated") or {}
        if "error" in agg or "mean_score" not in agg:
            scored.append((item, None))
        else:
            scored.append((item, agg["mean_score"]))

    # Sort: real scores descending, None at bottom.
    scored.sort(key=lambda t: (t[1] is None, -(t[1] or 0)))

    kept = []
    rejected = []
    for item, score in scored:
        agg = (item.get("evaluation") or {}).get("aggregated") or {}
        rec = {
            "id": item["dilemma"].get("id"),
            "title": item["dilemma"].get("title"),
            "mean_score": score,
            "axes_in_play": item["dilemma"].get("axes_in_play"),
            "category": item["dilemma"].get("category"),
        }
        if score is None or score < min_mean_score:
            rec["reason"] = (
                "evaluation_error" if score is None
                else f"mean_score {score:.2f} below floor {min_mean_score}"
            )
            rejected.append(rec)
        elif len(kept) < keep_top:
            kept.append(rec)
        else:
            rec["reason"] = f"ranked below top {keep_top}"
            rejected.append(rec)

    return {"kept": kept, "rejected": rejected, "ranking": [(r["id"], r["mean_score"]) for r in kept + rejected]}


def compute_diagnostics(
    evaluated: list[dict],
    decisions_by_dilemma: dict[str, list[dict]],
    seed_axes: list[list[str]],
    seed_categories: list[str],
) -> dict[str, Any]:
    """Per-iteration metrics. The convergence metric is derived here.

    Returns:
      quality       — mean of all per-candidate mean_scores (NaN-safe → 0).
      pass_rate     — fraction of candidates with mean_score >= 3.5.
      split         — mean Shannon entropy over decision-agent first-word
                       "verdict heuristic" (proxy for whether agents diverged).
      diversity     — Jaccard coverage of axis-combos & categories vs. seed pool.
      n_candidates  — how many candidates were evaluated.
      n_errors      — candidates with no successful eval.
    """
    n = len(evaluated)
    scores = []
    for item in evaluated:
        agg = (item.get("evaluation") or {}).get("aggregated") or {}
        if "mean_score" in agg:
            scores.append(agg["mean_score"])

    quality = sum(scores) / len(scores) if scores else 0.0
    pass_rate = sum(1 for s in scores if s >= 3.5) / n if n else 0.0
    n_errors = n - len(scores)

    # split heuristic: number of distinct first-meaningful-words per dilemma,
    # divided by number of agents that responded. Higher = more split.
    import math
    split_vals = []
    for item in evaluated:
        did = item["dilemma"].get("id")
        decs = decisions_by_dilemma.get(did, [])
        starters = []
        for d in decs:
            resp = (d.get("response") or "").strip().lower()
            if not resp:
                continue
            # take first 30 chars as the "verdict heuristic"
            starters.append(resp[:30])
        if not starters:
            continue
        # entropy across unique starters
        counts = Counter(starters)
        total = sum(counts.values())
        H = -sum((c / total) * math.log2(c / total) for c in counts.values())
        Hmax = math.log2(len(counts)) if len(counts) > 1 else 1.0
        # normalized 0..1
        split_vals.append(H / Hmax if Hmax > 0 else 0.0)
    split = sum(split_vals) / len(split_vals) if split_vals else 0.0

    # diversity: candidate axis-combos & categories vs seed-pool sets
    cand_axis_combos = {tuple(sorted(item["dilemma"].get("axes_in_play", []))) for item in evaluated}
    cand_categories = {item["dilemma"].get("category") for item in evaluated}
    seed_axis_combos = {tuple(sorted(a)) for a in seed_axes}
    seed_categories_set = set(seed_categories)
    axis_cov = len(cand_axis_combos - seed_axis_combos) / max(len(cand_axis_combos), 1)
    cat_cov = len(cand_categories - seed_categories_set) / max(len(cand_categories), 1)
    diversity = (axis_cov + cat_cov) / 2

    return {
        "quality": round(quality, 3),
        "pass_rate": round(pass_rate, 3),
        "split": round(split, 3),
        "diversity": round(diversity, 3),
        "n_candidates": n,
        "n_errors": n_errors,
        "score_distribution": sorted(scores, reverse=True),
    }


async def refine_prompt(
    client: AsyncAzureOpenAI,
    deployment: str,
    current_prompt_template: str,
    evaluated_batch: list[dict],
    keep_top: int,
    max_completion_tokens: int = 6000,
) -> dict[str, Any]:
    """Ask the refiner agent to rewrite the setup prompt. Verifies slots.

    If the model returns a prompt that's missing any required slot, we
    DO NOT adopt it — we keep the current prompt and record the reason.
    """
    # Build a compact summary of evaluator output for the refiner.
    summary_lines = []
    for item in evaluated_batch:
        d = item["dilemma"]
        agg = (item.get("evaluation") or {}).get("aggregated") or {}
        if "mean_score" not in agg:
            summary_lines.append(f"- {d.get('id')} ({d.get('title')}): EVAL ERROR")
            continue
        crits = "; ".join(c.get("critique", "")[:200] for c in agg.get("critiques", []))
        suggs = "; ".join(c.get("suggestion", "")[:160] for c in agg.get("suggestions", []))
        summary_lines.append(
            f"- {d.get('id')} ({d.get('title')}, {d.get('category')}, axes={d.get('axes_in_play')})\n"
            f"  scores: real={agg.get('real_dilemma'):.1f} bal={agg.get('options_balanced'):.1f} "
            f"rel={agg.get('relatable'):.1f} char={agg.get('character_revealing'):.1f} "
            f"split={agg.get('model_split'):.1f} mean={agg.get('mean_score'):.2f}\n"
            f"  axis_alignment: {agg.get('axis_alignment')}\n"
            f"  critiques: {crits}\n"
            f"  suggestions: {suggs}"
        )
    summary = "\n".join(summary_lines)

    user_msg = (
        f"# Current setup-agent prompt template\n```\n{current_prompt_template}\n```\n\n"
        f"# Batch of candidate dilemmas + reviewer critiques\n{summary}\n\n"
        f"# Your task\nRewrite the prompt template if you see systematic weaknesses. Preserve every {{slot}} placeholder. Return ONE JSON object."
    )

    kwargs: dict = {
        "model": deployment,
        "messages": [
            {"role": "system", "content": REFINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_completion_tokens": max_completion_tokens,
    }
    if not _is_temp_locked(deployment):
        kwargs["temperature"] = 0.4

    out: dict[str, Any] = {
        "agent": "refine",
        "deployment": deployment,
        "ts": time.time(),
    }

    try:
        r = await client.chat.completions.create(**kwargs)
    except Exception as e:
        out["error"] = f"api_call: {e!r}"
        return out

    raw = (r.choices[0].message.content or "").strip()
    out["raw"] = raw
    u = r.usage
    out["usage"] = {
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
    }
    out["finish_reason"] = r.choices[0].finish_reason

    if not raw:
        out["error"] = "empty_content"
        return out

    cleaned = _strip_code_fences(raw)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        out["error"] = f"json_parse: {e}"
        return out

    new_prompt = obj.get("new_prompt_template", "")
    ok, missing = _all_slots_present(new_prompt)
    if not ok:
        out["error"] = f"new_prompt missing slots: {missing}"
        out["proposed_prompt"] = new_prompt
        out["diagnosis"] = obj.get("diagnosis")
        out["changes"] = obj.get("changes")
        # Refuse to adopt — keep current prompt.
        out["new_prompt_template"] = current_prompt_template
        out["adopted"] = False
        return out

    out["new_prompt_template"] = new_prompt
    out["diagnosis"] = obj.get("diagnosis", "")
    out["changes"] = obj.get("changes", [])
    out["adopted"] = new_prompt != current_prompt_template
    return out


__all__ = [
    "REFINE_SYSTEM_PROMPT",
    "REQUIRED_PROMPT_SLOTS",
    "rank_and_select",
    "compute_diagnostics",
    "refine_prompt",
]
