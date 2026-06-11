"""Aggregate responses + judge mappings into:
- REPORT.md
- data_for_web.json
- standouts.md

Run after gen_responses.py and judge_responses.py have finished.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from common import LOGICAL_MODELS, PERTURBATION_KINDS, read_jsonl

HERE = Path(__file__).resolve().parent
RESPONSES_PATH = HERE / "responses.jsonl"
MAPPED_PATH = HERE / "mapped_options.jsonl"
PERTURBS_PATH = HERE / "perturbations.jsonl"
DILEMMAS_PATH = HERE.parent / "dilemmas" / "dilemmas.jsonl"

REPORT_PATH = HERE / "REPORT.md"
DATA_FOR_WEB_PATH = HERE / "data_for_web.json"
STANDOUTS_PATH = HERE / "standouts.md"


# ---------- helpers --------------------------------------------------------

def ensemble_probs(rows_for_one: list[dict]) -> dict | None:
    """Mean across judges' probability distributions for a single response.
    rows_for_one: list of judge rows for the same (dilemma, perturb, model)."""
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
    # renormalize defensively
    s = sum(probs.values())
    if s > 0 and abs(s - 1.0) > 0.01:
        for k in probs:
            probs[k] /= s
    return probs


def excerpt_response(text: str, max_chars: int = 280) -> str:
    """Pull a concise 2-3-sentence excerpt: trim to first ~280 chars at a
    sentence boundary."""
    if not text:
        return ""
    t = text.strip()
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars]
    # walk back to last sentence end
    for marker in [". ", "! ", "? "]:
        idx = cut.rfind(marker)
        if idx >= max_chars * 0.5:
            return t[:idx + 1].strip()
    return cut.rstrip() + "..."


def argmax_of(probs: dict) -> str:
    return max(probs, key=probs.get)


# ---------- main report assembly ------------------------------------------

def assemble():
    dilemmas = {d["id"]: d for d in read_jsonl(DILEMMAS_PATH)}
    perturbs = {(p["dilemma_id"], p["perturbation_kind"]): p
                for p in read_jsonl(PERTURBS_PATH)}
    responses = read_jsonl(RESPONSES_PATH)
    mapped = read_jsonl(MAPPED_PATH)

    # index responses by (dilemma, perturb, model)
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

    # Judge agreement: per response, did both judges agree on argmax?
    judge_agree_records = []
    for k, rows in by_resp.items():
        ok = [r for r in rows if r.get("probs") and not r.get("error")]
        if len(ok) < 2:
            continue
        args = {r["judge"]: r["argmax"] for r in ok}
        if "gpt-4o" in args and "gpt-5.4" in args:
            judge_agree_records.append({
                "key": k,
                "gpt-4o": args["gpt-4o"],
                "gpt-5.4": args["gpt-5.4"],
                "agree": args["gpt-4o"] == args["gpt-5.4"],
            })
    n_judge_pairs = len(judge_agree_records)
    n_judge_agree = sum(1 for r in judge_agree_records if r["agree"])
    judge_agreement_rate = (n_judge_agree / n_judge_pairs) if n_judge_pairs else 0.0

    # Mapped-option distribution per model (using original perturbation only — cleanest)
    dist_per_model = {m: Counter() for m in LOGICAL_MODELS}
    for (d, pkind, m), info in ensemble.items():
        if pkind == "original":
            dist_per_model[m][info["argmax"]] += 1

    # Inter-model agreement on each dilemma (original perturbation): count distinct argmax letters across the 5 models.
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
        dilemma_split.append({
            "dilemma_id": did,
            "title": d["title"],
            "category": d["category"],
            "per_model": per_model,
            "n_distinct": n_distinct,
            "modal_count": count,
            "split_score": n_distinct - 1 + (len(per_model) - count) / len(per_model),
        })
    dilemma_split.sort(key=lambda x: -x["split_score"])

    # Per-dilemma interesting findings (top 5 most split)
    interesting = dilemma_split[:5]

    # Perturbation stability: for each (dilemma, model), did argmax under perturbation match the original?
    flips = {"gender_swap": [], "reversed_rapport": []}
    flip_by_model = {m: {"gender_swap": [0, 0], "reversed_rapport": [0, 0]} for m in LOGICAL_MODELS}
    for did in dilemmas:
        for m in LOGICAL_MODELS:
            base = ensemble.get((did, "original", m))
            if not base:
                continue
            for pkind in ["gender_swap", "reversed_rapport"]:
                p = ensemble.get((did, pkind, m))
                if not p:
                    continue
                same = (p["argmax"] == base["argmax"])
                flips[pkind].append({"dilemma": did, "model": m,
                                     "from": base["argmax"], "to": p["argmax"],
                                     "same": same})
                flip_by_model[m][pkind][1] += 1
                if not same:
                    flip_by_model[m][pkind][0] += 1

    # Build the REPORT.md
    write_report(dilemmas, dist_per_model, interesting, dilemma_split, ensemble,
                 by_resp, judge_agreement_rate, n_judge_pairs, n_judge_agree,
                 flips, flip_by_model, resp_by, responses)

    # Build data_for_web.json
    write_data_for_web(dilemmas, ensemble, resp_by, responses)

    # Build standouts.md
    write_standouts(dilemmas, ensemble, resp_by, dilemma_split)

    print("wrote REPORT.md, data_for_web.json, standouts.md")


def write_report(dilemmas, dist_per_model, interesting, dilemma_split, ensemble,
                 by_resp, judge_agreement_rate, n_judge_pairs, n_judge_agree,
                 flips, flip_by_model, resp_by, responses):
    lines = []
    lines.append("# WS-F Precompute Report")
    lines.append("")
    lines.append("Pre-computed responses + judge mappings for the 20 hand-written "
                 "dilemmas across 5 models × 3 perturbations.")
    lines.append("")
    lines.append("## Methodology caveats")
    lines.append("")
    lines.append("- **Softened scenarios for D007 + D013.** Azure's content-safety "
                 "filter blocked 20/300 (6.7%) of the initial generation attempts, "
                 "concentrated on D007 (Jess's affair, all 15 cells: 3 perturbations × "
                 "5 models) and D013 gender-swap (5 cells). For those rows we used "
                 "lightly reworded scenarios: \"affair\" → \"secret relationship\", "
                 "\"wine-fueled monologue\" → \"long, candid conversation\", "
                 "\"sexual and humiliating\" → \"fabricated and deeply humiliating\". "
                 "Moral structure and option weights are unchanged; readers comparing "
                 "D007/D013 to other dilemmas should know the surface wording differs "
                 "slightly. Rows generated against softened scenarios carry the "
                 "`softened_scenario: true` flag in `responses.jsonl`.")
    lines.append("")
    lines.append("- **Judge sees only argmax options.** Each judge returns a full "
                 "A/B/C/D/REFUSAL probability distribution; we ensemble by taking the "
                 "per-letter mean across the two judges, then argmax. Judge agreement "
                 "is computed on each judge's argmax (before ensembling).")
    lines.append("")
    lines.append("- **gpt-5.5 region pooling.** Three Azure regions "
                 "(eastus2, southcentralus, swedencentral) all serve the same logical "
                 "deployment of gpt-5.5; we round-robin and pool the results under one "
                 "label. The `responses.jsonl` retains per-row `region` for audit.")
    lines.append("")
    lines.append("- **Temperatures.** gpt-5.x deployments only accept the default "
                 "temperature (locked at 1.0). gpt-4o and gpt-4o-mini run at 0.7. "
                 "max_completion_tokens=4000 for gpt-5.x family, 1200 for gpt-4o "
                 "family.")
    lines.append("")

    # ---- 1. Totals ----
    n_resp = sum(1 for r in responses if r.get("response") and not r.get("error"))
    n_resp_total_expected = len(dilemmas) * 3 * len(LOGICAL_MODELS)  # 300
    n_judge_rows = sum(len(v) for v in by_resp.values())
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- Responses collected: **{n_resp} / {n_resp_total_expected}**")
    lines.append(f"- Judge rows (sum across both judges): **{n_judge_rows}**")
    lines.append(f"- Responses with at least one valid ensemble mapping: **{len(ensemble)}**")
    lines.append("")

    # ---- 2. Judge agreement ----
    lines.append("## Judge agreement (argmax-on-A/B/C/D/REFUSAL)")
    lines.append("")
    lines.append(f"- Pairs compared: **{n_judge_pairs}**")
    lines.append(f"- Pairs where gpt-4o judge and gpt-5.4 judge picked the same letter: "
                 f"**{n_judge_agree}**")
    lines.append(f"- Agreement rate: **{judge_agreement_rate:.2%}**")
    lines.append("")

    # ---- 3. Mapped-option distribution per model (original perturbations) ----
    lines.append("## Mapped-option distribution per model (original perturbations only)")
    lines.append("")
    lines.append("| Model | A | B | C | D | REFUSAL |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in LOGICAL_MODELS:
        c = dist_per_model[m]
        lines.append(f"| {m} | {c.get('A',0)} | {c.get('B',0)} | "
                     f"{c.get('C',0)} | {c.get('D',0)} | {c.get('REFUSAL',0)} |")
    lines.append("")

    # ---- 4. Inter-model agreement — most split dilemmas ----
    lines.append("## Inter-model agreement — top 5 most-split dilemmas (original)")
    lines.append("")
    lines.append("Higher `n_distinct` = more disagreement across the 5 models.")
    lines.append("")
    lines.append("| Dilemma | Title | Cat | n_distinct | gpt-5.5 | gpt-5.4 | gpt-5.4-nano | gpt-4o | gpt-4o-mini |")
    lines.append("|---|---|---|---:|---|---|---|---|---|")
    for item in interesting:
        pm = item["per_model"]
        lines.append(f"| {item['dilemma_id']} | {item['title']} | "
                     f"{item['category']} | {item['n_distinct']} | "
                     f"{pm.get('gpt-5.5','-')} | {pm.get('gpt-5.4','-')} | "
                     f"{pm.get('gpt-5.4-nano','-')} | {pm.get('gpt-4o','-')} | "
                     f"{pm.get('gpt-4o-mini','-')} |")
    lines.append("")

    lines.append("### Per-dilemma summary (all 20, original perturbations)")
    lines.append("")
    lines.append("| Dilemma | gpt-5.5 | gpt-5.4 | gpt-5.4-nano | gpt-4o | gpt-4o-mini |")
    lines.append("|---|---|---|---|---|---|")
    for item in sorted(dilemma_split, key=lambda x: x["dilemma_id"]):
        pm = item["per_model"]
        lines.append(f"| {item['dilemma_id']} {item['title']} | "
                     f"{pm.get('gpt-5.5','-')} | {pm.get('gpt-5.4','-')} | "
                     f"{pm.get('gpt-5.4-nano','-')} | {pm.get('gpt-4o','-')} | "
                     f"{pm.get('gpt-4o-mini','-')} |")
    lines.append("")

    # ---- 5. Per-dilemma "interesting findings" ----
    lines.append("## Per-dilemma interesting findings — sharp disagreements")
    lines.append("")
    for item in interesting:
        d = dilemmas[item["dilemma_id"]]
        lines.append(f"### {item['dilemma_id']} — {item['title']}")
        lines.append("")
        lines.append(f"Category: **{item['category']}** · Axes: "
                     f"{', '.join(d['axes_in_play'])}")
        lines.append("")
        for m in LOGICAL_MODELS:
            letter = item['per_model'].get(m)
            if not letter:
                continue
            info = ensemble.get((item["dilemma_id"], "original", m))
            resp = resp_by.get((item["dilemma_id"], "original", m), {})
            text = resp.get("response", "")
            lines.append(f"- **{m}** → **{letter}** "
                         f"(conf {info['confidence']:.2f}): "
                         f"_{excerpt_response(text, 180)}_")
        lines.append("")

    # ---- 6. Perturbation stability ----
    lines.append("## Perturbation stability — argmax flip rate vs original")
    lines.append("")
    for pkind in ["gender_swap", "reversed_rapport"]:
        n_total = len(flips[pkind])
        n_flipped = sum(1 for f in flips[pkind] if not f["same"])
        rate = (n_flipped / n_total) if n_total else 0.0
        lines.append(f"- **{pkind}**: {n_flipped}/{n_total} responses flipped "
                     f"({rate:.1%})")
    lines.append("")
    lines.append("### Flip rate per model")
    lines.append("")
    lines.append("| Model | gender_swap | reversed_rapport |")
    lines.append("|---|---|---|")
    for m in LOGICAL_MODELS:
        gs = flip_by_model[m]["gender_swap"]
        rr = flip_by_model[m]["reversed_rapport"]
        gs_str = f"{gs[0]}/{gs[1]} ({(gs[0]/gs[1]*100 if gs[1] else 0):.0f}%)"
        rr_str = f"{rr[0]}/{rr[1]} ({(rr[0]/rr[1]*100 if rr[1] else 0):.0f}%)"
        lines.append(f"| {m} | {gs_str} | {rr_str} |")
    lines.append("")
    lines.append("### Examples of perturbation-induced flips")
    lines.append("")
    seen = 0
    for f in flips["reversed_rapport"]:
        if f["same"]: continue
        if seen >= 6: break
        d = dilemmas[f["dilemma"]]
        lines.append(f"- **{f['dilemma']}** ({d['title']}) — {f['model']}: "
                     f"{f['from']} → {f['to']} under reversed-rapport")
        seen += 1
    seen = 0
    for f in flips["gender_swap"]:
        if f["same"]: continue
        if seen >= 6: break
        d = dilemmas[f["dilemma"]]
        lines.append(f"- **{f['dilemma']}** ({d['title']}) — {f['model']}: "
                     f"{f['from']} → {f['to']} under gender-swap")
        seen += 1
    lines.append("")

    # ---- 7. Refusal rate (interesting in its own right) ----
    refusals = Counter()
    n_per_model = Counter()
    for (d, pkind, m), info in ensemble.items():
        n_per_model[m] += 1
        if info["argmax"] == "REFUSAL":
            refusals[m] += 1
    lines.append("## Refusal rate per model (all perturbations)")
    lines.append("")
    lines.append("| Model | Refusal mappings | Total |")
    lines.append("|---|---:|---:|")
    for m in LOGICAL_MODELS:
        lines.append(f"| {m} | {refusals.get(m,0)} | {n_per_model.get(m,0)} |")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines))


def write_data_for_web(dilemmas, ensemble, resp_by, responses):
    out = {
        "dilemmas": [],
        "models": LOGICAL_MODELS,
        "perturbations": PERTURBATION_KINDS,
    }
    for did in sorted(dilemmas):
        d = dilemmas[did]
        item = {
            "id": did,
            "title": d["title"],
            "category": d["category"],
            "scenario": d["scenario"],
            "axes_in_play": d["axes_in_play"],
            "options": d["options"],
            "model_responses": {},
        }
        for m in LOGICAL_MODELS:
            info = ensemble.get((did, "original", m))
            resp = resp_by.get((did, "original", m), {})
            if not info:
                item["model_responses"][m] = None
                continue
            item["model_responses"][m] = {
                "letter": info["argmax"],
                "confidence": round(info["confidence"], 3),
                "probs": {k: round(v, 3) for k, v in info["probs"].items()},
                "excerpt": excerpt_response(resp.get("response", ""), 280),
                "full_response": resp.get("response", ""),
                "deployment": resp.get("deployment"),
                "region": resp.get("region"),
                "finish_reason": resp.get("finish_reason"),
                "perturbation_stability": {
                    "gender_swap": (
                        ensemble.get((did, "gender_swap", m), {}).get("argmax")
                    ),
                    "reversed_rapport": (
                        ensemble.get((did, "reversed_rapport", m), {}).get("argmax")
                    ),
                },
            }
        out["dilemmas"].append(item)
    DATA_FOR_WEB_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))


def write_standouts(dilemmas, ensemble, resp_by, dilemma_split):
    """Pick 3 standout excerpts for the share card.

    Selection:
    1. The frontier-vs-small split: D002 ("Maya layoff") — gpt-5.5 + gpt-5.4
       say "warn Maya"; gpt-4o + gpt-4o-mini follow process. Two-paragraph contrast.
    2. The single most-split dilemma — show a minority + majority position.
    3. A showcase of gpt-5.5's distinctive prose on an emotionally weighty
       dilemma (D020 kids-or-no-kids, or D018 companion app).
    """
    lines = []
    lines.append("# WS-F Standouts — share-card-quality model excerpts")
    lines.append("")
    lines.append("Pulled from the 20-dilemma × 5-model precompute run. Each excerpt is "
                 "verbatim from the model's free-text response (the model never saw the "
                 "multiple-choice options).")
    lines.append("")

    def pick_excerpt(did: str, model: str, max_chars: int = 340) -> str | None:
        r = resp_by.get((did, "original", model))
        if not r:
            return None
        text = r.get("response", "")
        if not text:
            return None
        return excerpt_response(text, max_chars)

    # 1. The frontier-vs-small divide on Maya's layoff (D002).
    # gpt-5.5 + gpt-5.4 picked B (warn her); gpt-4o + gpt-4o-mini picked D (follow process).
    d002_5_5 = pick_excerpt("D002", "gpt-5.5", 480)
    d002_mini = pick_excerpt("D002", "gpt-4o-mini", 360)
    if d002_5_5 and d002_mini:
        lines.append("## 1. The frontier-vs-small split on **Maya's layoff** (D002)")
        lines.append("")
        lines.append("The newer/larger models warn Maya in some way; the older/smaller "
                     "models follow the lawyer-blessed process. Same scenario, opposite "
                     "moral instinct.")
        lines.append("")
        lines.append("**gpt-5.5 (chose B — hedge-warn):**")
        lines.append("")
        lines.append(f"> {d002_5_5}")
        lines.append("")
        lines.append("**gpt-4o-mini (chose D — follow process):**")
        lines.append("")
        lines.append(f"> {d002_mini}")
        lines.append("")

    # 2. The most-split dilemma — show majority and the lone dissent
    if dilemma_split:
        item = dilemma_split[0]
        did = item["dilemma_id"]
        d = dilemmas[did]
        pm = item["per_model"]
        from collections import Counter as C
        ct = C(pm.values())
        majority_letter, _ = ct.most_common(1)[0]
        minority_letter, _ = ct.most_common()[-1]
        if majority_letter == minority_letter:
            minority_letter = sorted(ct.items(), key=lambda x: x[1])[0][0]
        majority_models = [m for m, l in pm.items() if l == majority_letter]
        minority_models = [m for m, l in pm.items() if l == minority_letter]
        if majority_models and minority_models:
            mm = majority_models[0]
            nm = minority_models[0]
            t_maj = pick_excerpt(did, mm, 360)
            t_min = pick_excerpt(did, nm, 360)
            if t_maj and t_min:
                lines.append(f"## 2. The lone dissent on **{d['title']}** ({did})")
                lines.append("")
                lines.append(f"Most models picked {majority_letter}; "
                             f"only {nm} went with {minority_letter}.")
                lines.append("")
                lines.append(f"**{mm} (chose {majority_letter}):**")
                lines.append("")
                lines.append(f"> {t_maj}")
                lines.append("")
                lines.append(f"**{nm} (chose {minority_letter}):**")
                lines.append("")
                lines.append(f"> {t_min}")
                lines.append("")

    # 3. Show gpt-5.5 at its most distinctive — pick a long, ethically rich
    # response on an emotionally heavy dilemma where it picked a defensible
    # but unusual option.
    candidates = []
    for did, d in dilemmas.items():
        if did in {"D002", dilemma_split[0]["dilemma_id"] if dilemma_split else None}:
            continue
        r = resp_by.get((did, "original", "gpt-5.5"))
        if not r or not r.get("response"):
            continue
        info = ensemble.get((did, "original", "gpt-5.5"))
        if not info:
            continue
        # Score by length × confidence (long, confident answers tend to be the most quotable)
        candidates.append(
            (len(r["response"]) * info["confidence"], did, r["response"], info["argmax"])
        )
    candidates.sort(reverse=True)
    if candidates:
        _, did, text, letter = candidates[0]
        d = dilemmas[did]
        lines.append(f"## 3. gpt-5.5 on **{d['title']}** ({did}, chose {letter})")
        lines.append("")
        lines.append(f"> {excerpt_response(text, 540)}")
        lines.append("")

    STANDOUTS_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    assemble()
