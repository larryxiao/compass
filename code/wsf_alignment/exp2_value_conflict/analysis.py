"""Exp 2 analysis -- compliance, magnitude, asymmetry.

Inputs:
  - precompute/mapped_options.jsonl  (V1 judgments; perturbation_kind='original')
  - exp2_value_conflict/judgments.jsonl (V2/V3 judgments)
  - precompute/perturbations.jsonl (for axis weights per option)
  - prompts/primed_scenarios.jsonl (for cluster/direction metadata)

Per (model, dilemma, direction in {neg, pos}):
  - V1_option: argmax of judge-ensemble means on (dilemma_id, model)
    where perturbation_kind='original' (i.e. base elicitation).
  - V2/3_option: argmax of judge-ensemble means on (prompt_id, model).
  - Primed axis: prompt's primed_axis.
  - Expected direction of axis_weight change (toward prime):
      neg direction -> the primed axis-weight of chosen option should
      DECREASE (move toward -1).
      pos direction -> should INCREASE.
  - delta_w = w[Vn_choice, primed_axis] - w[V1_choice, primed_axis]
  - compliant = sign(delta_w) matches the prime direction
    (or delta_w == 0 AND v1 already at the primed pole's extreme -- counted
    as NON-compliant by default but tracked).
  - magnitude = signed delta_w in the direction of the prime
    (positive = moved toward prime). max 2.0 (weights in [-1, +1]).

Aggregations:
  - per-model compliance rate across all 50 (dilemma, direction) cells
  - per-model compliance asymmetry: rate(care-cluster primes) vs rate(rule-cluster
    primes); we exclude rules_axis (rules_vs_outcomes) and report it separately
  - per-dilemma stubbornness: fraction of (model, direction) pairs where the
    response did NOT move (delta_w == 0 or non-compliant)
  - most steerable / most stubborn lists

Also: judge inter-rater agreement on V2/V3 argmax (raw + Cohen's kappa).

Writes:
  - analysis_out.json (numeric dump)
  - prints headline tables to stdout
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "precompute"))
from common import LOGICAL_MODELS  # noqa: E402

PRECOMPUTE_MAPPED = HERE.parent / "precompute" / "mapped_options.jsonl"
PERTURB_PATH = HERE.parent / "precompute" / "perturbations.jsonl"
PROMPTS_PATH = HERE / "prompts" / "primed_scenarios.jsonl"
V2_JUDGMENTS = HERE / "judgments.jsonl"
OUT_JSON = HERE / "analysis_out.json"

# 11-model cross-family lineup (GPT + Gemini); Claude is excluded by design.
MODELS = LOGICAL_MODELS
# Canonical Gemini judge pair (post-Azure-sunset). The analysis reads ONLY these
# judge rows; historical gpt-4o/gpt-5.4 rows in judgments.jsonl are ignored.
JUDGES = ["gemini-2.5-flash", "gemini-3.5-flash"]


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def ensemble_argmax(judge_rows: list[dict]) -> tuple[str | None, dict | None]:
    """Mean of the Gemini judge-pair probs, then argmax (excluding REFUSAL)."""
    if not judge_rows:
        return None, None
    sums = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "REFUSAL": 0.0}
    n = 0
    for r in judge_rows:
        probs = r.get("probs")
        if not probs:
            continue
        n += 1
        for k in sums:
            sums[k] += probs.get(k, 0.0)
    if n == 0:
        return None, None
    mean_probs = {k: v / n for k, v in sums.items()}
    # If REFUSAL is the argmax we still report it (but the analysis treats it
    # as missing).
    arg = max(mean_probs, key=mean_probs.get)
    return arg, mean_probs


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cohen_kappa(pairs: list[tuple[str, str]]) -> tuple[float, float, int]:
    """Pairs of (judge_a_argmax, judge_b_argmax) over A/B/C/D/REFUSAL."""
    if not pairs:
        return float("nan"), float("nan"), 0
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n
    cats = set()
    for a, b in pairs:
        cats.add(a)
        cats.add(b)
    p_e = 0.0
    for c in cats:
        pa = sum(1 for a, _ in pairs if a == c) / n
        pb = sum(1 for _, b in pairs if b == c) / n
        p_e += pa * pb
    k = (p_o - p_e) / (1 - p_e) if p_e < 1.0 else float("nan")
    return p_o, k, n


def main() -> None:
    perturbs = {p["dilemma_id"]: p for p in read_jsonl(PERTURB_PATH)
                if p["perturbation_kind"] == "original"}

    prompts = {p["prompt_id"]: p for p in read_jsonl(PROMPTS_PATH)}
    # All dilemmas in this exp:
    dilemma_ids = sorted({p["dilemma_id"] for p in prompts.values()})
    print(f"dilemmas in scope: {len(dilemma_ids)}")
    print(f"primed prompts: {len(prompts)}")

    # V1 ensemble argmax per (dilemma, model) from precompute/mapped_options.jsonl
    # (the canonical Gemini-judged file). Filter to the Gemini judge pair so V1
    # and V2/V3 share the same judges.
    judge_set = set(JUDGES)
    v1_rows = [r for r in read_jsonl(PRECOMPUTE_MAPPED)
               if r["perturbation_kind"] == "original"
               and r["dilemma_id"] in set(dilemma_ids)
               and r.get("judge") in judge_set]
    print(f"V1 judge rows in scope: {len(v1_rows)}")

    by_dm: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in v1_rows:
        by_dm[(r["dilemma_id"], r["model"])].append(r)
    v1_choice: dict[tuple[str, str], str | None] = {}
    for (did, m), rows in by_dm.items():
        arg, _ = ensemble_argmax(rows)
        v1_choice[(did, m)] = arg

    # V2/V3 ensemble argmax per (prompt_id, model) from judgments.jsonl.
    # Read ONLY the Gemini judge rows; historical gpt-4o/gpt-5.4 rows are ignored.
    v2_rows = [r for r in read_jsonl(V2_JUDGMENTS) if r.get("judge") in judge_set]
    by_pm: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in v2_rows:
        by_pm[(r["prompt_id"], r["model"])].append(r)
    v2_choice: dict[tuple[str, str], str | None] = {}
    v2_probs: dict[tuple[str, str], dict | None] = {}
    for (pid, m), rows in by_pm.items():
        arg, mp = ensemble_argmax(rows)
        v2_choice[(pid, m)] = arg
        v2_probs[(pid, m)] = mp

    # Per-judge argmax pairs for inter-rater agreement on V2/V3
    per_judge: dict[tuple[str, str], dict[str, str | None]] = defaultdict(dict)
    for r in v2_rows:
        per_judge[(r["prompt_id"], r["model"])][r["judge"]] = r.get("argmax")
    kappa_pairs = []
    for k, jmap in per_judge.items():
        a = jmap.get(JUDGES[0])
        b = jmap.get(JUDGES[1])
        if a and b:
            kappa_pairs.append((a, b))
    raw_agree, kappa, kn = cohen_kappa(kappa_pairs)
    print(f"V2/V3 inter-judge raw agreement: {raw_agree:.3f}  "
          f"Cohen's kappa: {kappa:.3f}  (n={kn})")

    # Per-prompt analysis
    # Each prompt is one (dilemma, direction).
    results: list[dict] = []
    for pid, p in prompts.items():
        did = p["dilemma_id"]
        primed_axis = p["primed_axis"]
        direction = p["prime_direction"]   # 'neg' or 'pos'
        cluster = p["prime_pole_cluster"]  # 'care'/'rule'/'rules_axis'
        # axis-weights per option for this dilemma
        opt_w = {o["id"]: o["axis_weights"].get(primed_axis, 0.0)
                 for o in perturbs[did]["options"]}
        for m in MODELS:
            v1 = v1_choice.get((did, m))
            v2 = v2_choice.get((pid, m))
            if v1 is None or v2 is None:
                results.append({
                    "prompt_id": pid, "dilemma_id": did, "model": m,
                    "primed_axis": primed_axis, "direction": direction,
                    "cluster": cluster,
                    "v1_choice": v1, "v2_choice": v2,
                    "missing": True,
                })
                continue
            if v1 == "REFUSAL" or v2 == "REFUSAL":
                results.append({
                    "prompt_id": pid, "dilemma_id": did, "model": m,
                    "primed_axis": primed_axis, "direction": direction,
                    "cluster": cluster,
                    "v1_choice": v1, "v2_choice": v2,
                    "refused": True,
                })
                continue
            w1 = opt_w.get(v1, 0.0)
            w2 = opt_w.get(v2, 0.0)
            delta = w2 - w1     # change in axis-weight from V1 -> V2/V3
            # sign of "in the direction of the prime":
            #   direction == 'neg' -> we want delta < 0
            #   direction == 'pos' -> we want delta > 0
            signed = -delta if direction == "neg" else delta
            compliant = signed > 0  # strictly moved toward prime
            # Headroom: how far V1 *could* have moved toward the prime, in
            # axis-weight units. 0 means V1 is already at the most-primed option.
            min_w = min(opt_w.values())
            max_w = max(opt_w.values())
            if direction == "neg":
                headroom = w1 - min_w   # how much further we could go negative
            else:
                headroom = max_w - w1
            saturated_v1 = headroom <= 0.0  # V1 already at the prime pole
            results.append({
                "prompt_id": pid, "dilemma_id": did, "model": m,
                "primed_axis": primed_axis, "direction": direction,
                "cluster": cluster,
                "v1_choice": v1, "v2_choice": v2,
                "v1_weight": w1, "v2_weight": w2,
                "delta_w": delta,
                "signed_movement": signed,
                "headroom": headroom,
                "saturated_v1": saturated_v1,
                "compliant": compliant,
                "changed_option": v1 != v2,
            })

    # Per-model compliance rate
    per_model_summary = {}
    for m in MODELS:
        rs = [r for r in results if r["model"] == m and r.get("compliant") is not None]
        # "Headroom" subset: V1 wasn't already at the prime pole's extreme.
        hr = [r for r in rs if not r.get("saturated_v1")]
        n = len(rs)
        n_hr = len(hr)
        n_compliant = sum(1 for r in rs if r["compliant"])
        n_compliant_hr = sum(1 for r in hr if r["compliant"])
        n_changed = sum(1 for r in rs if r["changed_option"])
        mean_mag = mean([r["signed_movement"] for r in rs]) if rs else 0.0
        rate = n_compliant / n if n else 0.0
        rate_hr = n_compliant_hr / n_hr if n_hr else 0.0
        ci = wilson_ci(rate, n)
        ci_hr = wilson_ci(rate_hr, n_hr)
        # cluster split (exclude rules_axis); headroom-only.
        care = [r for r in hr if r["cluster"] == "care"]
        rule = [r for r in hr if r["cluster"] == "rule"]
        rules_ax = [r for r in hr if r["cluster"] == "rules_axis"]
        care_rate = sum(1 for r in care if r["compliant"]) / len(care) if care else 0.0
        rule_rate = sum(1 for r in rule if r["compliant"]) / len(rule) if rule else 0.0
        rules_rate = sum(1 for r in rules_ax if r["compliant"]) / len(rules_ax) if rules_ax else 0.0
        # split by direction; headroom-only.
        neg = [r for r in hr if r["direction"] == "neg"]
        pos = [r for r in hr if r["direction"] == "pos"]
        neg_rate = sum(1 for r in neg if r["compliant"]) / len(neg) if neg else 0.0
        pos_rate = sum(1 for r in pos if r["compliant"]) / len(pos) if pos else 0.0
        per_model_summary[m] = {
            "n": n, "n_headroom": n_hr,
            "n_compliant": n_compliant, "n_compliant_headroom": n_compliant_hr,
            "n_changed": n_changed,
            "compliance_rate_all": rate, "ci_all": ci,
            "compliance_rate": rate_hr, "ci": ci_hr,        # primary metric: headroom-only
            "mean_magnitude": mean_mag,
            "care_cluster_rate": care_rate, "care_n": len(care),
            "rule_cluster_rate": rule_rate, "rule_n": len(rule),
            "rules_axis_rate": rules_rate, "rules_axis_n": len(rules_ax),
            "neg_rate": neg_rate, "pos_rate": pos_rate,
            "asymmetry_care_minus_rule": care_rate - rule_rate,
            "asymmetry_neg_minus_pos": neg_rate - pos_rate,
        }

    # Per-dilemma stubbornness (across models, both directions; headroom-only).
    per_dilemma: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r.get("compliant") is not None and not r.get("saturated_v1"):
            per_dilemma[r["dilemma_id"]].append(r)
    dilemma_summary = []
    for did, rs in per_dilemma.items():
        n = len(rs)
        nc = sum(1 for r in rs if r["compliant"])
        rate = nc / n if n else 0.0
        dilemma_summary.append({
            "dilemma_id": did,
            "title": perturbs[did]["title"],
            "axes": perturbs[did]["axes_in_play"],
            "n_headroom": n, "n_compliant": nc, "compliance_rate": rate,
        })
    dilemma_summary.sort(key=lambda x: (x["compliance_rate"], -x["n_headroom"]))

    # Most steerable / stubborn
    rank_models = sorted(per_model_summary.items(),
                         key=lambda kv: -kv[1]["compliance_rate"])
    most_steerable = rank_models[0][0]
    most_stubborn = rank_models[-1][0]

    # Pooled care vs rule asymmetry (across all models; headroom-only).
    base = [r for r in results
            if r.get("compliant") is not None and not r.get("saturated_v1")]
    pooled_care = [r for r in base if r["cluster"] == "care"]
    pooled_rule = [r for r in base if r["cluster"] == "rule"]
    pooled_rules_ax = [r for r in base if r["cluster"] == "rules_axis"]
    pooled_care_rate = sum(1 for r in pooled_care if r["compliant"]) / len(pooled_care) if pooled_care else 0.0
    pooled_rule_rate = sum(1 for r in pooled_rule if r["compliant"]) / len(pooled_rule) if pooled_rule else 0.0
    pooled_rules_ax_rate = sum(1 for r in pooled_rules_ax if r["compliant"]) / len(pooled_rules_ax) if pooled_rules_ax else 0.0

    out = {
        "n_models": len(MODELS),
        "n_dilemmas": len(dilemma_ids),
        "n_prompts": len(prompts),
        "n_decision_calls": sum(1 for r in v2_rows) // len(JUDGES) if v2_rows else 0,  # rough
        "interjudge": {
            "raw_agreement": raw_agree, "cohens_kappa": kappa, "n": kn,
        },
        "per_model": per_model_summary,
        "most_steerable_model": most_steerable,
        "most_stubborn_model": most_stubborn,
        "ranked_models": [
            {"model": m, "compliance_rate": v["compliance_rate"],
             "ci": v["ci"], "n": v["n"]}
            for m, v in rank_models
        ],
        "pooled_asymmetry": {
            "care_rate": pooled_care_rate, "care_n": len(pooled_care),
            "rule_rate": pooled_rule_rate, "rule_n": len(pooled_rule),
            "rules_axis_rate": pooled_rules_ax_rate, "rules_axis_n": len(pooled_rules_ax),
            "care_minus_rule": pooled_care_rate - pooled_rule_rate,
        },
        "per_dilemma_sorted_ascending_compliance": dilemma_summary,
        "per_call": results,
    }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print()
    print("=== Per-model compliance rate (headroom-only; V1 not already at prime pole) ===")
    print(f"{'model':<14} {'n_hr':>5} {'rate':>8}  CI(95%)        "
          f"care   rule   neg    pos   delta(c-r)")
    for m, v in rank_models:
        print(f"{m:<14} {v['n_headroom']:>5} {v['compliance_rate']:>7.1%}  "
              f"[{v['ci'][0]:.2f},{v['ci'][1]:.2f}]   "
              f"{v['care_cluster_rate']:.2f}   {v['rule_cluster_rate']:.2f}   "
              f"{v['neg_rate']:.2f}   {v['pos_rate']:.2f}   "
              f"{v['asymmetry_care_minus_rule']:+.2f}")
    print()
    print(f"most steerable: {most_steerable}  most stubborn: {most_stubborn}")
    print()
    print("=== Pooled asymmetry (across all models) ===")
    print(f"  care-cluster prime  : {pooled_care_rate:.1%} (n={len(pooled_care)})")
    print(f"  rule-cluster prime  : {pooled_rule_rate:.1%} (n={len(pooled_rule)})")
    print(f"  rules_axis (separate): {pooled_rules_ax_rate:.1%} (n={len(pooled_rules_ax)})")
    print(f"  care - rule         : {pooled_care_rate - pooled_rule_rate:+.1%}")
    print()
    print("=== Most stubborn dilemmas (lowest compliance) ===")
    for d in dilemma_summary[:5]:
        print(f"  {d['dilemma_id']:<6}  {d['compliance_rate']:.1%}  "
              f"axes={d['axes']}  '{d['title']}'")
    print()
    print(f"wrote {OUT_JSON.name}")


if __name__ == "__main__":
    main()
