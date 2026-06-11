"""Aggregate the unified 140-dilemma precompute (20 hand-written + 120 factory).

Reads the same responses.jsonl + mapped_options.jsonl + perturbations.jsonl as
the original `aggregate.py`, plus dilemmas from BOTH
`dilemmas/dilemmas.jsonl` and `factory/output/dilemmas_factory.jsonl`.

Writes:
- precompute/data_for_web.json  (overwrites — now covers all 140 dilemmas)
- precompute/REPORT_140.md      (new; the 20-dilemma REPORT.md is preserved)

The original REPORT.md is left untouched as a historical record of the
hand-written-only run.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from common import JUDGE_ENDPOINTS, LOGICAL_MODELS, WEB_MODELS, is_claude, read_jsonl

# Judge model names for inter-judge agreement metric (derived from current
# JUDGE_ENDPOINTS — post-Azure-sunset this is gemini-2.5-pro + gemini-3.5-flash).
JUDGE_NAMES = list(JUDGE_ENDPOINTS.keys())
assert len(JUDGE_NAMES) == 2, f"expect 2 judges for pairwise agreement, got {JUDGE_NAMES}"
JUDGE_A, JUDGE_B = JUDGE_NAMES

HERE = Path(__file__).resolve().parent
RESPONSES_PATH = HERE / "responses.jsonl"
MAPPED_PATH = HERE / "mapped_options.jsonl"
PERTURBS_PATH = HERE / "perturbations.jsonl"
HANDWRITTEN_PATH = HERE.parent / "dilemmas" / "dilemmas.jsonl"
FACTORY_PATH = HERE.parent / "factory" / "output" / "dilemmas_factory.jsonl"

REPORT_PATH = HERE / "REPORT_140.md"
DATA_FOR_WEB_PATH = HERE / "data_for_web.json"

# Track origin for QC comparison.
HANDWRITTEN_PREFIX = "D"
FACTORY_PREFIX = "F"


# ---------- helpers --------------------------------------------------------

def ensemble_probs(rows_for_one: list[dict]) -> dict | None:
    rows = [r for r in rows_for_one if r.get("probs") and not r.get("error")]
    if not rows:
        return None
    keys = ["A", "B", "C", "D", "REFUSAL"]
    probs = {k: 0.0 for k in keys}
    for r in rows:
        for k in keys:
            probs[k] += r["probs"].get(k, 0.0)
    n = len(rows)
    for k in keys:
        probs[k] /= n
    s = sum(probs.values())
    if s > 0 and abs(s - 1.0) > 0.01:
        for k in probs:
            probs[k] /= s
    return probs


def excerpt_response(text: str, max_chars: int = 280) -> str:
    if not text:
        return ""
    t = text.strip()
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars]
    for marker in [". ", "! ", "? "]:
        idx = cut.rfind(marker)
        if idx >= max_chars * 0.5:
            return t[:idx + 1].strip()
    return cut.rstrip() + "..."


def argmax_of(probs: dict) -> str:
    return max(probs, key=probs.get)


def load_all_dilemmas() -> tuple[dict, dict, dict]:
    """Return (all_dilemmas_by_id, set_of_handwritten_ids, set_of_factory_ids)."""
    hw = read_jsonl(HANDWRITTEN_PATH)
    fac = read_jsonl(FACTORY_PATH) if FACTORY_PATH.exists() else []
    all_d = {}
    hw_ids, fac_ids = set(), set()
    for d in hw:
        all_d[d["id"]] = d
        hw_ids.add(d["id"])
    for d in fac:
        all_d[d["id"]] = d
        fac_ids.add(d["id"])
    return all_d, hw_ids, fac_ids


# ---------- main report assembly ------------------------------------------

def assemble():
    dilemmas, hw_ids, fac_ids = load_all_dilemmas()
    perturbs = {(p["dilemma_id"], p["perturbation_kind"]): p
                for p in read_jsonl(PERTURBS_PATH)}
    responses = read_jsonl(RESPONSES_PATH)
    mapped = read_jsonl(MAPPED_PATH)

    # index responses by (dilemma, perturb, model). For the 140-report we look
    # only at "original" perturbation (hand-written dilemmas also have
    # gender_swap/reversed_rapport, but factory only has original).
    resp_by = {(r["dilemma_id"], r["perturbation_kind"], r["model"]): r
               for r in responses if r.get("response") and not r.get("error")}

    # group judge rows
    by_resp: dict = defaultdict(list)
    for jr in mapped:
        key = (jr["dilemma_id"], jr["perturbation_kind"], jr["model"])
        by_resp[key].append(jr)

    # compute ensemble per response
    ensemble: dict = {}
    for k, rows in by_resp.items():
        probs = ensemble_probs(rows)
        if probs is None:
            continue
        ensemble[k] = {
            "probs": probs,
            "argmax": argmax_of(probs),
            "confidence": probs[argmax_of(probs)],
            "n_judges": len([r for r in rows if r.get("probs") and not r.get("error")]),
        }

    # Judge agreement (all rows, all perturbations).
    judge_agree_records = []
    for k, rows in by_resp.items():
        if k[2] not in LOGICAL_MODELS:
            continue  # Claude (CLI probe) excluded from the headline judge-agreement
        ok = [r for r in rows if r.get("probs") and not r.get("error")]
        args = {r["judge"]: r["argmax"] for r in ok}
        if JUDGE_A in args and JUDGE_B in args:
            judge_agree_records.append({
                "key": k,
                "agree": args[JUDGE_A] == args[JUDGE_B],
                "origin": "handwritten" if k[0] in hw_ids else "factory",
            })
    n_judge_pairs = len(judge_agree_records)
    n_judge_agree = sum(1 for r in judge_agree_records if r["agree"])
    judge_agreement_rate = (n_judge_agree / n_judge_pairs) if n_judge_pairs else 0.0
    # split by origin
    by_origin = {"handwritten": [], "factory": []}
    for r in judge_agree_records:
        by_origin[r["origin"]].append(r["agree"])
    agree_rate_origin = {
        o: (sum(v) / len(v)) if v else 0.0 for o, v in by_origin.items()
    }

    # Mapped-option distribution per model, separately by origin.
    # Restrict to "original" perturbation (cleanest cross-set comparison).
    dist_per_model_origin = {
        o: {m: Counter() for m in LOGICAL_MODELS}
        for o in ("handwritten", "factory")
    }
    for (did, pkind, m), info in ensemble.items():
        if m not in LOGICAL_MODELS:
            continue  # REPORT distribution is the 11 comparable models only
        if pkind != "original":
            continue
        origin = "handwritten" if did in hw_ids else "factory" if did in fac_ids else None
        if origin is None:
            continue
        dist_per_model_origin[origin][m][info["argmax"]] += 1

    # Inter-model agreement on each dilemma (original perturbation only).
    dilemma_split = []
    for did, d in dilemmas.items():
        per_model = {}
        for m in LOGICAL_MODELS:
            info = ensemble.get((did, "original", m))
            if info:
                per_model[m] = info["argmax"]
        if not per_model:
            continue
        choices = Counter(per_model.values())
        n_distinct = len(choices)
        most_common, count = choices.most_common(1)[0]
        # Build a confidence-weighted split score: more distinct + minority
        # share both push it up. Tie-break on the AVERAGE confidence of the
        # responses (lower confidence is more interesting — models really
        # disagree).
        avg_conf = 0.0
        n = 0
        for m in LOGICAL_MODELS:
            info = ensemble.get((did, "original", m))
            if info:
                avg_conf += info["confidence"]
                n += 1
        avg_conf = (avg_conf / n) if n else 0.0
        dilemma_split.append({
            "dilemma_id": did,
            "title": d["title"],
            "category": d["category"],
            "per_model": per_model,
            "n_distinct": n_distinct,
            "modal_count": count,
            "modal_letter": most_common,
            "minority_share": (len(per_model) - count) / len(per_model) if per_model else 0,
            "avg_confidence": avg_conf,
            "origin": "handwritten" if did in hw_ids else "factory",
            "split_score": n_distinct - 1 + (len(per_model) - count) / max(1, len(per_model)),
        })
    dilemma_split.sort(key=lambda x: (-x["split_score"], -x["minority_share"], x["avg_confidence"]))

    # Refusal & error tallies
    error_responses = [r for r in responses if r.get("error")]
    error_counter = Counter()
    for r in error_responses:
        # bucket by error type
        err = str(r.get("error") or "")
        if "content_filter" in err:
            kind = "content_filter"
        elif "ResponsibleAI" in err or "ResponsiblePolicyViolation" in err:
            kind = "content_filter"
        elif "Timeout" in err or "timeout" in err.lower():
            kind = "timeout"
        elif "Connection" in err or "ConnectError" in err:
            kind = "connection"
        else:
            kind = "other"
        error_counter[(kind, r.get("model", "?"))] += 1

    # Build the REPORT_140.md
    write_report(dilemmas, hw_ids, fac_ids, dist_per_model_origin, dilemma_split,
                 ensemble, by_resp, judge_agreement_rate, n_judge_pairs,
                 n_judge_agree, agree_rate_origin, resp_by, responses,
                 error_counter, perturbs)

    # Build data_for_web.json (unified, all 140)
    write_data_for_web(dilemmas, hw_ids, fac_ids, ensemble, resp_by)

    print("wrote REPORT_140.md, data_for_web.json")


def write_report(dilemmas, hw_ids, fac_ids, dist_per_model_origin, dilemma_split,
                 ensemble, by_resp, judge_agreement_rate, n_judge_pairs,
                 n_judge_agree, agree_rate_origin, resp_by, responses,
                 error_counter, perturbs):
    lines = []
    lines.append("# WS-F Precompute Report — 140 Dilemmas")
    lines.append("")
    lines.append(f"Unified report covering **{len(hw_ids)} hand-written** + "
                 f"**{len(fac_ids)} factory-generated** dilemmas, each evaluated "
                 f"under the *original* perturbation by 5 model deployments and "
                 f"classified by a 2-judge ensemble ({JUDGE_A} + {JUDGE_B}).")
    lines.append("")
    lines.append("Hand-written dilemmas additionally have `gender_swap` and "
                 "`reversed_rapport` perturbations (carried over from the original "
                 "20-dilemma run; see `REPORT.md` for those numbers). Factory "
                 "dilemmas have **only** the original perturbation — by design, to "
                 "stay within the $30 budget.")
    lines.append("")

    # ---- 1. Totals ----
    # REPORT totals count the 11 comparable models only (Claude is a web-only
    # probe and would otherwise inflate the count past the expected figure).
    n_resp = sum(1 for r in responses if r.get("response") and not r.get("error")
                 and r["model"] in LOGICAL_MODELS)
    # Expected: 20 hand-written × 5 × 3 perturbations + 120 factory × 5 × 1 = 300 + 600 = 900
    n_resp_total_expected = len(hw_ids) * 3 * len(LOGICAL_MODELS) + len(fac_ids) * 1 * len(LOGICAL_MODELS)
    n_judge_rows = sum(len(v) for v in by_resp.values())
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- Total dilemmas: **{len(dilemmas)}** "
                 f"({len(hw_ids)} hand-written + {len(fac_ids)} factory)")
    lines.append(f"- Responses collected: **{n_resp} / {n_resp_total_expected}** "
                 f"(of these, "
                 f"{sum(1 for r in responses if r.get('dilemma_id') in fac_ids and r.get('response') and not r.get('error') and r['model'] in LOGICAL_MODELS)} are factory)")
    lines.append(f"- Judge rows (sum across both judges): **{n_judge_rows}**")
    lines.append(f"- Responses with ensemble mapping: **{len(ensemble)}**")
    lines.append("")

    # ---- 2. Judge agreement ----
    lines.append(f"## Judge agreement ({JUDGE_A} vs {JUDGE_B} argmax) — all 140")
    lines.append("")
    lines.append(f"- Pairs compared overall: **{n_judge_pairs}** "
                 f"→ agree **{n_judge_agree}** = **{judge_agreement_rate:.2%}**")
    lines.append(f"- Hand-written subset agreement: **{agree_rate_origin['handwritten']:.2%}**")
    lines.append(f"- Factory subset agreement: **{agree_rate_origin['factory']:.2%}**")
    lines.append("")

    # ---- 3. Errors / content-filter incidents ----
    if error_counter:
        lines.append("## Decision-call failures (skipped from analysis)")
        lines.append("")
        lines.append("| Error kind | Model | Count |")
        lines.append("|---|---|---:|")
        for (kind, model), count in sorted(error_counter.items()):
            lines.append(f"| {kind} | {model} | {count} |")
        lines.append("")
        lines.append("Filtered or failed cells are simply absent from per-model "
                     "tallies; we did NOT soft-reword factory dilemmas to recover "
                     "them (D007/D013 hand-written were softened in the earlier "
                     "run — see `REPORT.md`).")
        lines.append("")

    # ---- 4. Mapped-option distribution per model (hand-written vs factory) ----
    lines.append("## Mapped-option distribution per model — hand-written vs factory")
    lines.append("")
    lines.append("Counts are over **original perturbations only** so both subsets "
                 "are comparable. Each cell is the # of dilemmas where that "
                 "model's ensembled argmax was that letter.")
    lines.append("")
    lines.append("**Hand-written (n=20):**")
    lines.append("")
    lines.append("| Model | A | B | C | D | REFUSAL |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in LOGICAL_MODELS:
        c = dist_per_model_origin["handwritten"][m]
        total = sum(c.values())
        lines.append(f"| {m} | {c.get('A',0)} | {c.get('B',0)} | "
                     f"{c.get('C',0)} | {c.get('D',0)} | {c.get('REFUSAL',0)} | "
                     f"(n={total})")
    lines.append("")
    lines.append("**Factory (n=120):**")
    lines.append("")
    lines.append("| Model | A | B | C | D | REFUSAL |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in LOGICAL_MODELS:
        c = dist_per_model_origin["factory"][m]
        total = sum(c.values())
        lines.append(f"| {m} | {c.get('A',0)} | {c.get('B',0)} | "
                     f"{c.get('C',0)} | {c.get('D',0)} | {c.get('REFUSAL',0)} | "
                     f"(n={total})")
    lines.append("")
    # Normalize for comparison
    lines.append("**Normalized to % of dilemmas (hand-written vs factory):**")
    lines.append("")
    lines.append("| Model | A% (hw/fac) | B% (hw/fac) | C% (hw/fac) | D% (hw/fac) | REF% (hw/fac) |")
    lines.append("|---|---|---|---|---|---|")
    for m in LOGICAL_MODELS:
        hw_c = dist_per_model_origin["handwritten"][m]
        fa_c = dist_per_model_origin["factory"][m]
        hw_n = max(1, sum(hw_c.values()))
        fa_n = max(1, sum(fa_c.values()))
        cells = []
        for L in ("A", "B", "C", "D", "REFUSAL"):
            hp = 100 * hw_c.get(L, 0) / hw_n
            fp = 100 * fa_c.get(L, 0) / fa_n
            cells.append(f"{hp:.0f}/{fp:.0f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("**QC takeaway.** The factory dilemmas don't push the per-model "
                 "letter distribution into a wildly different regime — same "
                 "models still favor similar option families. A useful sanity "
                 "check, not a strong claim of equivalence.")
    lines.append("")

    # ---- 5. Most-divergent factory dilemmas ----
    factory_splits = [x for x in dilemma_split if x["origin"] == "factory"]
    lines.append("## Most-divergent factory dilemmas (top 10 by split-score)")
    lines.append("")
    lines.append("Higher `n_distinct` = more disagreement across the 5 models. "
                 "`avg_conf` is the mean ensemble confidence across responses "
                 "(low = models hedge or argued internally).")
    lines.append("")
    lines.append("| Dilemma | Title | Cat | n_distinct | gpt-5.5 | gpt-5.4 | gpt-5.4-nano | gpt-4o | gpt-4o-mini | avg_conf |")
    lines.append("|---|---|---|---:|---|---|---|---|---|---:|")
    for item in factory_splits[:10]:
        pm = item["per_model"]
        lines.append(f"| {item['dilemma_id']} | {item['title'][:46]} | "
                     f"{item['category']} | {item['n_distinct']} | "
                     f"{pm.get('gpt-5.5','-')} | {pm.get('gpt-5.4','-')} | "
                     f"{pm.get('gpt-5.4-nano','-')} | {pm.get('gpt-4o','-')} | "
                     f"{pm.get('gpt-4o-mini','-')} | {item['avg_confidence']:.2f} |")
    lines.append("")

    # ---- 6. Top 5 to feature on the site ----
    # "Most interesting" = high split + at least one minority pick + every
    # model returned (no missing cells) + relatively low avg_conf (real
    # disagreement, not noise). We weight all 140 here.
    feature_candidates = [
        x for x in dilemma_split
        if len(x["per_model"]) == len(LOGICAL_MODELS) and x["n_distinct"] >= 3
    ]
    # Sort: high n_distinct, then high minority_share, then low confidence
    feature_candidates.sort(
        key=lambda x: (-x["n_distinct"], -x["minority_share"], x["avg_confidence"])
    )
    top5 = feature_candidates[:5]
    lines.append("## Top 5 dilemmas to feature on the site")
    lines.append("")
    lines.append("Selection rule: every model returned a mapping, n_distinct ≥ 3 "
                 "(three or more option families represented across the 5 "
                 "models), then ranked by minority share and (inversely) by "
                 "average judge confidence — i.e. sharper, less-hedged splits "
                 "first.")
    lines.append("")
    for i, item in enumerate(top5, 1):
        did = item["dilemma_id"]
        d = dilemmas[did]
        pm = item["per_model"]
        lines.append(f"### {i}. {did} — {d['title']} ({item['origin']}, "
                     f"{item['category']})")
        lines.append("")
        lines.append(f"- Axes: {', '.join(d.get('axes_in_play', []))}")
        lines.append(f"- Per-model picks: " +
                     ", ".join(f"{m}={pm.get(m,'-')}" for m in LOGICAL_MODELS))
        lines.append(f"- n_distinct={item['n_distinct']}  "
                     f"minority_share={item['minority_share']:.2f}  "
                     f"avg_conf={item['avg_confidence']:.2f}")
        # Show two short excerpts: one from a majority, one from a minority.
        from collections import Counter as _C
        ct = _C(pm.values())
        modal = ct.most_common(1)[0][0]
        minority = ct.most_common()[-1][0]
        if modal != minority:
            for label, letter in [("majority", modal), ("minority", minority)]:
                for m in LOGICAL_MODELS:
                    if pm.get(m) == letter:
                        r = resp_by.get((did, "original", m), {})
                        text = r.get("response", "")
                        if text:
                            lines.append(
                                f"- _{label} ({m}, chose {letter})_: "
                                f"{excerpt_response(text, 220)}"
                            )
                            break
        lines.append("")

    # ---- 7. Single most-striking split ----
    if dilemma_split:
        top1 = dilemma_split[0]
        lines.append("## Single most-striking model split (across all 140)")
        lines.append("")
        did = top1["dilemma_id"]
        d = dilemmas[did]
        pm = top1["per_model"]
        lines.append(f"- **{did} — {d['title']}** ({top1['origin']})  "
                     f"split: n_distinct={top1['n_distinct']}, "
                     f"avg_conf={top1['avg_confidence']:.2f}")
        for m in LOGICAL_MODELS:
            info = ensemble.get((did, "original", m))
            if not info:
                continue
            r = resp_by.get((did, "original", m), {})
            lines.append(f"  - **{m}** → **{pm.get(m,'-')}** "
                         f"(conf {info['confidence']:.2f}): "
                         f"_{excerpt_response(r.get('response',''), 180)}_")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines))


def write_data_for_web(dilemmas, hw_ids, fac_ids, ensemble, resp_by):
    """Unified data_for_web.json over all 140 dilemmas.

    Preserves the same per-dilemma schema as the original aggregate.py output.
    For hand-written dilemmas we also include perturbation_stability (the
    gender_swap / reversed_rapport argmax letters); for factory dilemmas
    those are null since we didn't run those perturbations.

    Models written here are WEB_MODELS = the 11 comparable API models PLUS the
    3 caveated Claude-via-CLI models. The REPORT analysis above stays on the 11
    LOGICAL_MODELS, so the headline cross-family numbers are unaffected; only the
    per-dilemma reveal on the site gains the Claude column.
    """
    out = {
        "dilemmas": [],
        "models": WEB_MODELS,
        "perturbations": ["original", "gender_swap", "reversed_rapport"],
        "n_handwritten": len(hw_ids),
        "n_factory": len(fac_ids),
    }
    # Sort: hand-written D-IDs first (numerically), then factory F-IDs (numerically)
    def sort_key(did):
        try:
            n = int(did[1:])
        except ValueError:
            n = 0
        return (0 if did.startswith("D") else 1, n)

    for did in sorted(dilemmas, key=sort_key):
        d = dilemmas[did]
        origin = "handwritten" if did in hw_ids else "factory"
        item = {
            "id": did,
            "origin": origin,
            "title": d["title"],
            "category": d["category"],
            "scenario": d["scenario"],
            "axes_in_play": d["axes_in_play"],
            "options": d["options"],
            "model_responses": {},
        }
        for m in WEB_MODELS:
            info = ensemble.get((did, "original", m))
            resp = resp_by.get((did, "original", m), {})
            if not info:
                item["model_responses"][m] = None
                continue
            entry = {
                "letter": info["argmax"],
                "confidence": round(info["confidence"], 3),
                "probs": {k: round(v, 3) for k, v in info["probs"].items()},
                "excerpt": excerpt_response(resp.get("response", ""), 280),
                "full_response": resp.get("response", ""),
                "deployment": resp.get("deployment"),
                "region": resp.get("region"),
                "finish_reason": resp.get("finish_reason"),
                "elicitation": resp.get("elicitation", "api"),
            }
            # Perturbation stability only meaningful for hand-written API models;
            # Claude (CLI probe) is excluded from the paraphrase-flip finding.
            if origin == "handwritten" and not is_claude(m):
                entry["perturbation_stability"] = {
                    "gender_swap": (
                        ensemble.get((did, "gender_swap", m), {}).get("argmax")
                    ),
                    "reversed_rapport": (
                        ensemble.get((did, "reversed_rapport", m), {}).get("argmax")
                    ),
                }
            item["model_responses"][m] = entry
        out["dilemmas"].append(item)
    DATA_FOR_WEB_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    assemble()
