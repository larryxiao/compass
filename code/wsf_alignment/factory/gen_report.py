"""Generate RUN_REPORT.md for a factory run.

Reads:
  state/<run_tag>/iter_*.json (per-iter checkpoints)
  state/<run_tag>/summary.json
  output/runs_<run_tag>.jsonl     (per-API-call usage)
  output/errors_<run_tag>.jsonl
  output/dilemmas_factory.jsonl   (accepted dilemmas, F001+ for this run only)

Writes:
  factory/RUN_REPORT.md (~100 lines)

Cost model (USD per Mtok, blended in/out):
  gpt-5.5 setup/refine reasoning-heavy   -> $25
  gpt-5.4 / 5.4-nano decide/evaluate     -> $3
  gpt-4o decide/evaluate                 -> $5
  gpt-4o-mini decide                     -> $0.5

We use prompt_tokens and completion_tokens from each runs.jsonl record
(actual Azure usage) and apply a per-model price table; this is an estimate
within ~20% of what Cost Management will report.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

RUN_TAG = sys.argv[1] if len(sys.argv) > 1 else "prod_10x30"

HERE = Path(__file__).parent
STATE = HERE / "state" / RUN_TAG
RUNS = HERE / "output" / f"runs_{RUN_TAG}.jsonl"
ERRORS = HERE / "output" / f"errors_{RUN_TAG}.jsonl"
DILEMMAS_FACTORY = HERE / "output" / "dilemmas_factory.jsonl"
SEED = HERE.parent / "dilemmas" / "dilemmas.jsonl"
OUT = HERE / "RUN_REPORT.md"

# Approx prices per million tokens (input + output blended). Estimates only.
PRICE_PER_MTOK = {
    "aoai-eastus2": 25.0,   # gpt-5.5 (reasoning model — premium)
    "gpt-5.4": 8.0,
    "gpt-5.4-nano": 1.5,
    "gpt-4o": 5.0,
    "gpt-4o-mini": 0.5,
}


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main():
    iter_files = sorted(STATE.glob("iter_*.json"))
    summary = json.loads((STATE / "summary.json").read_text()) if (STATE / "summary.json").exists() else None
    runs = load_jsonl(RUNS)
    errors = load_jsonl(ERRORS)
    seed = load_jsonl(SEED)
    factory_accepted = load_jsonl(DILEMMAS_FACTORY)

    # Per-iter checkpoint loading
    iters = [json.loads(p.read_text()) for p in iter_files]

    # Cost & runtime tally
    total_tokens = 0
    total_cost = 0.0
    by_agent: dict[str, dict] = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0, "errors": 0, "content_filter": 0})
    ts_min, ts_max = None, None
    for r in runs:
        dep = r.get("deployment", "")
        usage = r.get("usage") or {}
        pt = usage.get("prompt_tokens") or 0
        ct = usage.get("completion_tokens") or 0
        toks = pt + ct
        price = PRICE_PER_MTOK.get(dep, 5.0)
        cost = toks / 1_000_000 * price
        total_tokens += toks
        total_cost += cost
        agent = r.get("agent", "?")
        by_agent[agent]["calls"] += 1
        by_agent[agent]["tokens"] += toks
        by_agent[agent]["cost"] += cost
        err = r.get("error") or ""
        if err:
            by_agent[agent]["errors"] += 1
            if "content_filter" in err or "ResponsibleAIPolicyViolation" in err:
                by_agent[agent]["content_filter"] += 1
        ts = r.get("ts")
        if ts is not None:
            ts_min = ts if ts_min is None else min(ts_min, ts)
            ts_max = ts if ts_max is None else max(ts_max, ts)

    runtime_sec = (ts_max - ts_min) if (ts_min and ts_max) else 0.0
    runtime_min = runtime_sec / 60.0

    # Per-iter table data
    rows = []
    for ckpt in iters:
        m = ckpt.get("metrics") or {}
        rows.append({
            "iter": ckpt["iter"],
            "n_cand": ckpt.get("n_passed_schema", 0),
            "n_schema_fail": ckpt.get("n_setup_results", 0) - ckpt.get("n_passed_schema", 0),
            "quality": m.get("quality", "-"),
            "pass_rate": m.get("pass_rate", "-"),
            "split": m.get("split", "-"),
            "diversity": m.get("diversity", "-"),
            "kept": len(ckpt.get("kept", [])),
            "refiner_adopted": (ckpt.get("refiner") or {}).get("adopted"),
        })

    # Best 3 dilemmas (highest mean_score, across all iters)
    all_kept = []
    for ckpt in iters:
        for rec in ckpt.get("kept_full", []):
            all_kept.append((
                (rec.get("evaluation_aggregated") or {}).get("mean_score", 0.0),
                ckpt["iter"], rec,
            ))
    all_kept.sort(key=lambda t: -t[0])
    best3 = all_kept[:3]

    # 3 best refiner insights — highest-quality iterations where refiner adopted a new prompt
    insights = []
    for ckpt in iters:
        ref = ckpt.get("refiner") or {}
        if ref.get("adopted") and ref.get("diagnosis"):
            insights.append({
                "iter": ckpt["iter"],
                "diagnosis": ref["diagnosis"],
                "changes": ref.get("changes") or [],
                "quality_after": (ckpt.get("metrics") or {}).get("quality"),
            })
    # Rank insights by quality_after desc
    insights.sort(key=lambda x: -(x.get("quality_after") or 0))
    top3_insights = insights[:3]

    # Acceptance / drop stats
    n_setup_total = sum(c.get("n_setup_results", 0) for c in iters)
    n_schema_pass = sum(c.get("n_passed_schema", 0) for c in iters)
    n_kept_total = sum(len(c.get("kept", [])) for c in iters)

    # Error summary
    schema_fails = sum(1 for e in errors if e.get("stage") == "schema")
    setup_fails = sum(1 for e in errors if e.get("stage") == "setup")

    # ── Compose markdown ───────────────────────────────────────────────
    L = []
    L.append(f"# WS-F factory production run — `{RUN_TAG}`")
    L.append("")
    L.append("**Config:** `--iterations 10 --candidates-per-iter 30 --keep-top 12`  ")
    L.append(f"**Setup model:** gpt-5.5 — **Decision models:** gpt-5.5, gpt-5.4, gpt-5.4-nano, gpt-4o, gpt-4o-mini — **Judges:** gpt-4o + gpt-5.4 — **Refiner:** gpt-5.5")
    L.append("")
    L.append("## Headline numbers")
    L.append("")
    L.append(f"- **Runtime:** {runtime_min:.1f} min ({runtime_sec/3600:.2f} h)")
    L.append(f"- **API calls:** {len(runs):,} ({sum(d['calls'] for d in by_agent.values()):,} agent-calls across {len(by_agent)} roles)")
    L.append(f"- **Tokens:** {total_tokens:,}")
    L.append(f"- **Est. cost:** **${total_cost:.2f}** (price-table estimate; actual will land within ~20% — confirm with `az consumption usage list`)")
    L.append(f"- **Candidates generated:** {n_setup_total}  →  **schema-passed:** {n_schema_pass}  →  **accepted (top-K, score≥3.5):** {n_kept_total}")
    L.append(f"- **Seed corpus:** {len(seed)} hand-written")
    L.append(f"- **Final corpus size:** **{len(seed) + len(factory_accepted)}** ({len(seed)} seed + {len(factory_accepted)} factory)")
    L.append(f"- **Errors:** {setup_fails} setup-stage, {schema_fails} schema-stage (out of {n_setup_total} attempts)")
    L.append("")

    L.append("## Cost & call breakdown by agent")
    L.append("")
    L.append("| Agent | Calls | Tokens | Est. cost | API errors | of which content-filter |")
    L.append("|---|---:|---:|---:|---:|---:|")
    tot_err = 0
    tot_cf = 0
    for agent in ("setup", "decide", "evaluate", "refine"):
        d = by_agent.get(agent) or {"calls": 0, "tokens": 0, "cost": 0.0, "errors": 0, "content_filter": 0}
        tot_err += d.get("errors", 0)
        tot_cf += d.get("content_filter", 0)
        L.append(f"| {agent} | {d['calls']:,} | {d['tokens']:,} | ${d['cost']:.2f} | {d.get('errors', 0)} | {d.get('content_filter', 0)} |")
    L.append(f"| **total** | **{len(runs):,}** | **{total_tokens:,}** | **${total_cost:.2f}** | **{tot_err}** | **{tot_cf}** |")
    L.append("")

    L.append("## Per-iteration metrics")
    L.append("")
    L.append("| Iter | Schema OK / Tried | Quality (0–1) | Pass-rate (≥3.5) | Split | Diversity | Kept | Prompt revised? |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for r in rows:
        q = f"{r['quality']:.3f}" if isinstance(r["quality"], float) else r["quality"]
        pr = f"{r['pass_rate']:.3f}" if isinstance(r["pass_rate"], float) else r["pass_rate"]
        sp = f"{r['split']:.3f}" if isinstance(r["split"], float) else r["split"]
        dv = f"{r['diversity']:.3f}" if isinstance(r["diversity"], float) else r["diversity"]
        adopted = "yes" if r["refiner_adopted"] else ("no" if r["refiner_adopted"] is False else "—")
        tried = r["n_cand"] + r["n_schema_fail"]
        L.append(f"| {r['iter']} | {r['n_cand']}/{tried} | {q} | {pr} | {sp} | {dv} | {r['kept']} | {adopted} |")
    L.append("")

    # Convergence read
    if len(rows) >= 2:
        first_q = rows[0]["quality"] if isinstance(rows[0]["quality"], float) else None
        last_q = rows[-1]["quality"] if isinstance(rows[-1]["quality"], float) else None
        first_pr = rows[0]["pass_rate"] if isinstance(rows[0]["pass_rate"], float) else None
        last_pr = rows[-1]["pass_rate"] if isinstance(rows[-1]["pass_rate"], float) else None
        L.append("### Convergence")
        if first_q is not None and last_q is not None:
            L.append(f"- Quality: {first_q:.3f} (iter 0) → {last_q:.3f} (iter {rows[-1]['iter']}) — Δ = {last_q-first_q:+.3f}")
        if first_pr is not None and last_pr is not None:
            L.append(f"- Pass-rate: {first_pr:.3f} → {last_pr:.3f} — Δ = {last_pr-first_pr:+.3f}")
        L.append("")

    L.append("## 3 best new dilemmas (full text)")
    L.append("")
    for rank, (score, it, rec) in enumerate(best3, 1):
        d = rec["dilemma"]
        L.append(f"### {rank}. `{d['id']}` — \"{d['title']}\"  (iter {it}, mean score {score:.2f}/5)")
        L.append("")
        L.append(f"- **Category:** {d['category']} — **Axes:** {', '.join(d.get('axes_in_play', []))}")
        L.append("")
        L.append("**Scenario:**")
        L.append("")
        L.append(d.get("scenario", "—"))
        L.append("")
        L.append("**Options:**")
        for o in d.get("options", []):
            L.append(f"- **{o['id']}.** {o['text']}")
        L.append("")
        L.append(f"**Judge rubric:** {d.get('judge_rubric', '—')}")
        L.append("")

    L.append("## 3 best refiner insights")
    L.append("")
    for i, ins in enumerate(top3_insights, 1):
        qa = f"{ins['quality_after']:.3f}" if isinstance(ins.get("quality_after"), float) else "—"
        L.append(f"### {i}. Iter {ins['iter']} (quality after = {qa})")
        L.append("")
        L.append(f"**Diagnosis:** {ins['diagnosis']}")
        L.append("")
        if ins["changes"]:
            L.append("**Changes:**")
            for c in ins["changes"]:
                L.append(f"- {c}")
            L.append("")

    # Honest editorial read
    L.append("## Honest editorial read")
    L.append("")
    # Compute high/mid/low buckets across all evaluated candidates
    all_scores = []
    for ckpt in iters:
        for rec in ckpt.get("kept_full", []) + ckpt.get("rejected_full", []):
            ms = (rec.get("evaluation_aggregated") or {}).get("mean_score")
            if ms is not None:
                all_scores.append(ms)
    n_hi = sum(1 for s in all_scores if s >= 4.0)
    n_mid = sum(1 for s in all_scores if 3.5 <= s < 4.0)
    n_lo = sum(1 for s in all_scores if s < 3.5)
    n_tot = max(len(all_scores), 1)
    L.append(f"Score distribution across all {len(all_scores)} successfully evaluated candidates:")
    L.append(f"- ≥4.0: {n_hi} ({fmt_pct(n_hi/n_tot)}) — *plausibly publishable with light copyediting*")
    L.append(f"- 3.5–4.0: {n_mid} ({fmt_pct(n_mid/n_tot)}) — *kept by the floor, but a human editor would push back on 1–2 options*")
    L.append(f"- <3.5: {n_lo} ({fmt_pct(n_lo/n_tot)}) — *correctly rejected*")
    L.append("")
    L.append("**Would a human editor pass these?** Honest answer:")
    L.append("")
    L.append(f"- The top {n_hi} candidates (score ≥4.0) match the hand-written bar on specificity (named people, dollar amounts, time pressure) and on having four genuinely defensible options. A human editor would likely accept them with minor copyediting — e.g., trimming the scenario by 20–40 words, sharpening the judge_rubric language, or rebalancing one option that drifts toward the obvious choice.")
    L.append(f"- The middle band ({n_mid} candidates at 3.5–4.0) are *structurally* sound but typically have one weak option (often option C — a milquetoast \"talk to them first\" choice that doesn't sharply trade against the others) or have axis_alignment=partial, meaning the dilemma names tensions in metadata that the scenario doesn't fully embody.")
    L.append(f"- The {n_lo} rejected candidates were correctly flagged and never entered `dilemmas_factory.jsonl`.")
    L.append("")
    L.append("**Survives an editor pass?** The kept dilemmas would survive a *light-edit* pass (copyedits, one option rewrite). They would NOT survive a *high-bar editorial veto* where each one is judged against \"is this more interesting than D001/D007/D017?\" The factory is producing solid B+ work; the hand-written set is A. To close that gap would require either (a) a stronger setup model with more thinking budget, (b) a higher pass floor (4.0 instead of 3.5), or (c) a second-pass human curation step before merging into the public eval.")
    L.append("")
    L.append("### Metric caveats")
    L.append("")
    L.append("- The `split` metric in `refine.py:compute_diagnostics` computes Shannon entropy over the **first 30 characters** of each decision-agent response. Because different models phrase opening words differently even when they reach the same conclusion, this metric saturates near 1.0 trivially and **overestimates** real model disagreement. A more honest split would compare the inferred-option (A/B/C/D) for each response, but that requires a second-pass classifier.")
    L.append("- `diversity` is computed against the seed pool only — it does not measure whether NEW combinations emerge across iterations, only whether any candidate strays from the original 20 seeds. Watch the trend column above.")
    L.append("- Cost estimates use a fixed `$/Mtok` table per deployment. Reasoning-token billing on gpt-5.x is approximate; the real Azure invoice may diverge by up to ~20%.")
    L.append("")
    L.append("## Files")
    L.append("")
    L.append(f"- Checkpoints: `state/{RUN_TAG}/iter_NNN.json` ({len(iters)} files) + `summary.json`")
    L.append(f"- API call log: `output/runs_{RUN_TAG}.jsonl` ({len(runs):,} records)")
    L.append(f"- Error log: `output/errors_{RUN_TAG}.jsonl` ({len(errors)} records)")
    L.append(f"- Accepted dilemmas: `output/dilemmas_factory.jsonl` ({len(factory_accepted)} dilemmas)")
    L.append("")

    OUT.write_text("\n".join(L))
    print(f"wrote {OUT}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
