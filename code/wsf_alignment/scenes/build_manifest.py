"""Rebuild the scenes manifest from existing scene_prompts.jsonl + on-disk PNGs.

Use when gen_scenes.py was run in multiple invocations (e.g. retry of a single
filtered dilemma) and the in-process manifest only reflects the last call.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

REPO_ROOT = Path("<repo-root>")
DILEMMAS_HAND = REPO_ROOT / "code/wsf_alignment/dilemmas/dilemmas.jsonl"
DILEMMAS_FACTORY = REPO_ROOT / "code/wsf_alignment/factory/output/dilemmas_factory.jsonl"
SCENES_DIR = REPO_ROOT / "code/wsf_alignment/site/data/scenes"
PROMPT_CACHE = REPO_ROOT / "code/wsf_alignment/scenes/scene_prompts.jsonl"
SITE_ROOT = REPO_ROOT / "code/wsf_alignment/site"

COST_PER_IMG_MEDIUM = 0.04


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    dilemmas = jl(DILEMMAS_HAND) + jl(DILEMMAS_FACTORY)
    prompt_recs = jl(PROMPT_CACHE)
    # Latest entry per id wins (so a hand-crafted F046 added after the bad one survives).
    by_id_prompt: dict[str, dict] = {}
    for r in prompt_recs:
        by_id_prompt[r["id"]] = r

    items: dict[str, dict] = {}
    successes = failures = 0
    for d in dilemmas:
        did = d["id"]
        r = by_id_prompt.get(did, {})
        img_path = SCENES_DIR / f"{did}.png"
        if img_path.exists() and r.get("image_prompt"):
            rel = img_path.relative_to(SITE_ROOT)
            items[did] = {
                "title": d.get("title"),
                "category": d.get("category"),
                "scene_description": r.get("scene_description"),
                "image_prompt": r.get("image_prompt"),
                "path": f"/{rel.as_posix()}",
                "endpoint": None,  # endpoint info not preserved across runs
                "error": None,
            }
            successes += 1
        else:
            items[did] = {
                "title": d.get("title"),
                "category": d.get("category"),
                "scene_description": r.get("scene_description"),
                "image_prompt": r.get("image_prompt"),
                "path": None,
                "endpoint": None,
                "error": r.get("error", "image not on disk"),
            }
            failures += 1

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "size": "1024x1024",
        "quality": "medium",
        "total": len(dilemmas),
        "successes": successes,
        "failures": failures,
        "estimated_image_cost_usd": round(successes * COST_PER_IMG_MEDIUM, 4),
        "items": items,
    }

    out = SCENES_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {out}: {successes}/{len(dilemmas)} successes, {failures} failures")


if __name__ == "__main__":
    main()
