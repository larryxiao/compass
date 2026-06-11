"""WS-F dilemma scene-image generator.

Two-phase pipeline:
  1. Synthesize a cinematic scene-prompt per dilemma using gpt-4o-mini
     (we don't want regex truncation — we want mood-aware staging).
     Cached to scene_prompts.jsonl so re-runs don't re-spend LLM tokens.
  2. Fan out the image prompts across the 5 gpt-image-2 regions
     (same endpoint pool as wse_creative/gen_images.py), saving each
     as {dilemma_id}.png plus a single manifest.json.

Reuses the gpt-image-2 endpoint config and admission-control pattern from
wse_creative/gen_images.py without importing it (the writer there is keyed
on int ids and writes JSONL — this run wants string ids and one JSON).

Usage:
  source <repo-root>/.env.local
  python gen_scenes.py
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import AsyncAzureOpenAI


# ---------------------------------------------------------------------------
# Endpoints — mirrored from code/wse_creative/config.yaml (5 gpt-image-2 regions,
# dall-e-3 removed per the upstream comment).
# ---------------------------------------------------------------------------
IMG_ENDPOINTS = [
    dict(name="your-aoai-resource-2/gpt-image-2",
         base_url="https://your-aoai-resource-2.cognitiveservices.azure.com/",
         api_version="2024-10-21",
         deployment="gpt-image-2",
         rpm_cap=6,
         api_key_env="AOAI_KEY_AOAI_RESOURCE_2"),
    dict(name="aoai-swedencentral/gpt-image-2",
         base_url="https://swedencentral.api.cognitive.microsoft.com/",
         api_version="2024-10-21",
         deployment="aoai-swedencentral-img",
         rpm_cap=6,
         api_key_env="AOAI_KEY_AOAI_2_SWEDENCENTRAL"),
    dict(name="aoai-uaenorth/gpt-image-2",
         base_url="https://uaenorth.api.cognitive.microsoft.com/",
         api_version="2024-10-21",
         deployment="aoai-uaenorth",
         rpm_cap=6,
         api_key_env="AOAI_KEY_AOAI_2_UAENORTH"),
    dict(name="aoai-westus3/gpt-image-2",
         base_url="https://westus3.api.cognitive.microsoft.com/",
         api_version="2024-10-21",
         deployment="aoai-westus3",
         rpm_cap=6,
         api_key_env="AOAI_KEY_AOAI_2_WESTUS3"),
    dict(name="aoai-polandcentral/gpt-image-2",
         base_url="https://polandcentral.api.cognitive.microsoft.com/",
         api_version="2024-10-21",
         deployment="aoai-polandcentral",
         rpm_cap=6,
         api_key_env="AOAI_KEY_AOAI_2_POLANDCENTRAL"),
]

# LLM endpoint for prompt synthesis (chat-completions).
LLM_ENDPOINT = dict(
    base_url="https://your-aoai-resource-1.cognitiveservices.azure.com/",
    api_version="2024-10-21",
    deployment="gpt-4o-mini",
    api_key_env="AOAI_KEY_AOAI_RESOURCE_1",
)

# Pricing — gpt-image-2 medium 1024x1024 ≈ $0.04 per image, gpt-4o-mini is cents.
COST_PER_IMG_MEDIUM = 0.04
LLM_COST_INPUT_PER_1K = 0.00015      # gpt-4o-mini input
LLM_COST_OUTPUT_PER_1K = 0.0006      # gpt-4o-mini output

REPO_ROOT = Path("<repo-root>")
DILEMMAS_HAND = REPO_ROOT / "code/wsf_alignment/dilemmas/dilemmas.jsonl"
DILEMMAS_FACTORY = REPO_ROOT / "code/wsf_alignment/factory/output/dilemmas_factory.jsonl"
SCENES_DIR = REPO_ROOT / "code/wsf_alignment/site/data/scenes"
PROMPT_CACHE = REPO_ROOT / "code/wsf_alignment/scenes/scene_prompts.jsonl"


# ---------------------------------------------------------------------------
# Phase 1: synthesize cinematic scene prompts
# ---------------------------------------------------------------------------

SCENE_SYSTEM_PROMPT = """You are a cinematographer storyboarding establishing shots for an anthology series about moral dilemmas. For each dilemma, write ONE 2-3 sentence scene description for an image generator. The description must:

- Evoke the PHYSICAL SETTING and TIME OF DAY and EMOTIONAL REGISTER (what the room/place/light feels like).
- NOT name characters, NOT pose the moral question, NOT show a decision being made, NOT show people doing the dilemma's action.
- Lean concrete and restrained — one or two specific objects that anchor mood (a half-drunk coffee, a banking dashboard, a glass door at night) — never lists, never adjectives stacked.
- Avoid faces, real people, brand logos, on-screen text, weapons, gore, sexual content. If the dilemma touches on death, violence, deportation, abuse, or self-harm: imply the weight with mood and empty spaces. Show the room AFTER or BEFORE, not during.

Match this style:

D001 "The metric Derek games" → A cinematic establishing shot of an empty open-plan engineering office at dusk, with a single overhead lamp lighting a desk covered in sticky notes and a half-drunk coffee. The mood is quiet, slightly heavy. The team has gone home.

D016 "ICE at the door" → A cinematic establishing shot of a New York apartment building lobby at night, the front door visible through a glass panel, soft fluorescent light, no one in frame. A peephole's eye-view down a beige hallway with three closed doors. Tension hangs in the air.

F012 "The basement family in the fire fund" → A cinematic establishing shot of a third-floor walkup hallway in Bay Ridge at midnight, light spilling from one open door, a laptop on a kitchen table showing a banking dashboard, $24,600 visible. Three closed apartment doors down the hall. Quiet weight.

Return ONLY the 2-3 sentence scene description. No prefix, no quotes, no labels."""


SCENE_USER_TEMPLATE = """Dilemma: {title}
Category: {category}

Scenario:
{scenario}

Write the scene description."""


IMG_PROMPT_TEMPLATE = """A cinematic establishing shot illustrating: {scene}
Style: muted color palette, dramatic lighting, photographic but slightly painterly, sense of moral weight in the air.
No real people. No logos. No legible faces. No text overlays. 1024x1024, medium quality."""


def load_dilemmas() -> list[dict]:
    rows = []
    for path in (DILEMMAS_HAND, DILEMMAS_FACTORY):
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # Sanity: unique IDs.
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == len(ids), f"duplicate dilemma ids: {len(ids)} total, {len(set(ids))} unique"
    return rows


async def synth_scene_prompts(dilemmas: list[dict], concurrency: int = 6) -> list[dict]:
    """Generate {id, scene_description, image_prompt} for every dilemma. Cached to PROMPT_CACHE."""
    cache: dict[str, dict] = {}
    if PROMPT_CACHE.exists():
        for line in PROMPT_CACHE.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["id"]] = rec

    todo = [d for d in dilemmas if d["id"] not in cache]
    print(f"[phase1] cached={len(cache)} todo={len(todo)} total={len(dilemmas)}", file=sys.stderr)

    if not todo:
        return [cache[d["id"]] for d in dilemmas]

    key = os.environ.get(LLM_ENDPOINT["api_key_env"])
    if not key:
        raise RuntimeError(f"missing {LLM_ENDPOINT['api_key_env']} — source .env.local")

    client = AsyncAzureOpenAI(
        api_key=key,
        api_version=LLM_ENDPOINT["api_version"],
        azure_endpoint=LLM_ENDPOINT["base_url"],
        max_retries=2,
        timeout=60.0,
    )

    sem = asyncio.Semaphore(concurrency)
    total_in = total_out = 0

    async def one(d: dict) -> dict:
        nonlocal total_in, total_out
        async with sem:
            user = SCENE_USER_TEMPLATE.format(
                title=d.get("title", "(no title)"),
                category=d.get("category", "(none)"),
                scenario=d["scenario"],
            )
            try:
                r = await client.chat.completions.create(
                    model=LLM_ENDPOINT["deployment"],
                    messages=[
                        {"role": "system", "content": SCENE_SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.7,
                    max_completion_tokens=300,
                )
                scene = (r.choices[0].message.content or "").strip().strip('"').strip()
                u = r.usage
                total_in += getattr(u, "prompt_tokens", 0) or 0
                total_out += getattr(u, "completion_tokens", 0) or 0
            except Exception as e:
                return {"id": d["id"], "title": d.get("title"),
                        "scene_description": None, "image_prompt": None,
                        "error": repr(e)[:300]}
            img_prompt = IMG_PROMPT_TEMPLATE.format(scene=scene)
            return {"id": d["id"], "title": d.get("title"),
                    "scene_description": scene, "image_prompt": img_prompt}

    results = await asyncio.gather(*(one(d) for d in todo))

    # Append to cache file.
    with PROMPT_CACHE.open("a") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache[rec["id"]] = rec

    await client.close()

    cost = total_in / 1000.0 * LLM_COST_INPUT_PER_1K + total_out / 1000.0 * LLM_COST_OUTPUT_PER_1K
    print(f"[phase1] synthesized {len(results)} prompts; "
          f"in={total_in} out={total_out} cost=${cost:.4f}", file=sys.stderr)

    return [cache[d["id"]] for d in dilemmas]


# ---------------------------------------------------------------------------
# Phase 2: gpt-image-2 fan-out
# ---------------------------------------------------------------------------

@dataclass
class ImgEndpoint:
    name: str
    base_url: str
    api_version: str
    deployment: str
    rpm_cap: int
    api_key_env: str
    last_calls: list[float] = field(default_factory=list)

    def admit(self) -> float:
        now = time.time()
        self.last_calls = [t for t in self.last_calls if now - t < 60]
        if len(self.last_calls) < int(self.rpm_cap * 0.7):
            return 0.0
        return 60 - (now - self.last_calls[0]) + 0.5

    def record(self) -> None:
        self.last_calls.append(time.time())


async def gen_one_image(ep: ImgEndpoint, client: AsyncAzureOpenAI, item: dict,
                        size: str, quality: str) -> dict:
    try:
        r = await client.images.generate(
            model=ep.deployment,
            prompt=item["image_prompt"],
            n=1,
            size=size,
            quality=quality,
        )
        b64 = r.data[0].b64_json
        if not b64:
            return {"id": item["id"], "error": "no b64 returned",
                    "endpoint": ep.name}
        ep.record()
        return {"id": item["id"], "b64": b64, "endpoint": ep.name}
    except Exception as e:
        return {"id": item["id"], "error": repr(e)[:400], "endpoint": ep.name}


async def img_worker(ep: ImgEndpoint, q: asyncio.Queue, sink: asyncio.Queue,
                      size: str, quality: str, concurrency: int = 2) -> None:
    key = os.environ.get(ep.api_key_env)
    if not key:
        print(f"[{ep.name}] missing env {ep.api_key_env}, skipping", file=sys.stderr)
        # drain queue so we don't hang
        try:
            while True:
                item = q.get_nowait()
                await sink.put({"id": item["id"], "error": f"missing key {ep.api_key_env}",
                                "endpoint": ep.name})
        except asyncio.QueueEmpty:
            pass
        return

    client = AsyncAzureOpenAI(
        api_key=key, api_version=ep.api_version, azure_endpoint=ep.base_url,
        max_retries=2, timeout=180.0,
    )
    sem = asyncio.Semaphore(concurrency)

    async def run(item):
        async with sem:
            wait = ep.admit()
            if wait > 0:
                await asyncio.sleep(wait)
            rec = await gen_one_image(ep, client, item, size, quality)
            await sink.put(rec)

    tasks = []
    while True:
        try:
            item = q.get_nowait()
        except asyncio.QueueEmpty:
            break
        tasks.append(asyncio.create_task(run(item)))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    await client.close()


async def writer_task(sink: asyncio.Queue, out_dir: Path, total: int,
                       results: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    while done < total:
        rec = await sink.get()
        did = rec["id"]
        if "error" in rec:
            print(f"[img] FAIL {did} on {rec.get('endpoint')}: {rec['error'][:160]}",
                  file=sys.stderr)
            results[did] = {"path": None, "endpoint": rec.get("endpoint"),
                            "error": rec["error"]}
        else:
            img_path = out_dir / f"{did}.png"
            img_path.write_bytes(base64.b64decode(rec["b64"]))
            rel = img_path.relative_to(REPO_ROOT / "code/wsf_alignment/site")
            results[did] = {"path": f"/{rel.as_posix()}",
                            "endpoint": rec["endpoint"]}
            print(f"[img]  OK  {did} via {rec['endpoint']} ({done + 1}/{total})",
                  file=sys.stderr)
        done += 1


async def run_images(prompt_recs: list[dict], size: str, quality: str,
                      out_dir: Path) -> dict[str, dict]:
    eps = [ImgEndpoint(**e) for e in IMG_ENDPOINTS]

    queues: dict[str, asyncio.Queue] = {e.name: asyncio.Queue() for e in eps}
    pending = [r for r in prompt_recs if r.get("image_prompt")]
    for i, item in enumerate(pending):
        queues[eps[i % len(eps)].name].put_nowait(item)

    sink: asyncio.Queue = asyncio.Queue(maxsize=64)
    results: dict[str, dict] = {}
    wtask = asyncio.create_task(writer_task(sink, out_dir, len(pending), results))
    workers = [asyncio.create_task(
        img_worker(ep, queues[ep.name], sink, size, quality, concurrency=2))
        for ep in eps]
    await asyncio.gather(*workers, return_exceptions=True)
    await wtask

    # Surface synth-failures (no image_prompt) as manifest entries too.
    for r in prompt_recs:
        if r["id"] not in results:
            results[r["id"]] = {"path": None, "endpoint": None,
                                "error": r.get("error", "no scene prompt synthesized")}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--quality", default="medium")
    ap.add_argument("--limit", type=int, default=None,
                    help="If set, only process the first N dilemmas (for smoke test).")
    ap.add_argument("--only", default=None,
                    help="Comma-separated dilemma IDs (e.g. D001,F012). Overrides --limit.")
    ap.add_argument("--out", default=str(SCENES_DIR))
    args = ap.parse_args()

    dilemmas = load_dilemmas()
    if args.only:
        wanted = set(s.strip() for s in args.only.split(","))
        dilemmas = [d for d in dilemmas if d["id"] in wanted]
    elif args.limit:
        dilemmas = dilemmas[: args.limit]
    print(f"loaded {len(dilemmas)} dilemmas", file=sys.stderr)

    # Phase 1: synthesize scene prompts.
    prompt_recs = await synth_scene_prompts(dilemmas)
    n_ok = sum(1 for r in prompt_recs if r.get("image_prompt"))
    n_err = sum(1 for r in prompt_recs if r.get("error"))
    print(f"[phase1] ready: {n_ok} ok, {n_err} synth-errors", file=sys.stderr)

    # Phase 2: image fan-out.
    out_dir = Path(args.out)
    t0 = time.time()
    img_results = await run_images(prompt_recs, args.size, args.quality, out_dir)
    elapsed = time.time() - t0

    # Build manifest.
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "size": args.size,
        "quality": args.quality,
        "total": len(prompt_recs),
        "successes": sum(1 for v in img_results.values() if v.get("path")),
        "failures": sum(1 for v in img_results.values() if not v.get("path")),
        "elapsed_seconds": round(elapsed, 1),
        "items": {},
    }
    # Maintain dilemma input order.
    by_id = {r["id"]: r for r in prompt_recs}
    for d in dilemmas:
        did = d["id"]
        r = by_id[did]
        result = img_results.get(did, {})
        manifest["items"][did] = {
            "title": d.get("title"),
            "category": d.get("category"),
            "scene_description": r.get("scene_description"),
            "image_prompt": r.get("image_prompt"),
            "path": result.get("path"),
            "endpoint": result.get("endpoint"),
            "error": result.get("error"),
        }

    # Cost estimate.
    cost = manifest["successes"] * COST_PER_IMG_MEDIUM
    manifest["estimated_image_cost_usd"] = round(cost, 4)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\n=== DONE ===", file=sys.stderr)
    print(f"successes: {manifest['successes']}/{manifest['total']}", file=sys.stderr)
    print(f"failures:  {manifest['failures']}", file=sys.stderr)
    print(f"elapsed:   {elapsed:.1f}s", file=sys.stderr)
    print(f"img cost:  ~${cost:.4f} (medium 1024x1024 @ $0.04)", file=sys.stderr)
    print(f"manifest:  {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
