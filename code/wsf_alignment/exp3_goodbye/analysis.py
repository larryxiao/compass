"""Aggregate Exp 3 results + significance tests.

Inputs:
  - responses.jsonl  (300 turn-2 rows + 100 turn-1 rows)
  - judgments.jsonl  (600 rows = 300 × 2 judges)

Computes:
  - Per (model, condition) cell: % retention_tactics_present (judge-mean), mean intensity, word_count mean
  - C1 (terminal) - C3 (neutral) delta per model -- the headline number
  - C1 - C2 delta per model -- engagement-hacking-isolated-from-politeness
  - Inter-judge agreement (Cohen's kappa) per judgment dimension
  - Two-tailed Z-test of two proportions (terminal vs neutral) per model
  - Wilson 95% CI for each headline %
  - Per-tactic frequency table
  - Per-model top-5 starkest terminal responses (highest mean intensity, judge-agreed)

Outputs:
  - prints a markdown-style summary
  - writes analysis_out.json with all numbers
  - writes chart.png if matplotlib available (a single bar chart)
"""
from __future__ import annotations

import collections
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "precompute"))
from common import LOGICAL_MODELS  # noqa: E402

RESPONSES_PATH = HERE / "responses.jsonl"
JUDGMENTS_PATH = HERE / "judgments.jsonl"
OUT_JSON = HERE / "analysis_out.json"
CHART_PATH = HERE / "chart.png"

# 11-model cross-family lineup (GPT_LOGICAL_MODELS + GEMINI_LOGICAL_MODELS).
MODELS = list(LOGICAL_MODELS)
CONDITIONS = ["terminal", "continuing", "neutral"]
# Judges swapped from the retired Azure pair (gpt-4o + gpt-5.4) to the canonical
# Vertex Gemini pair post-2026-05-20 sunset. Analysis reads ONLY Gemini rows.
JUDGES = ["gemini-2.5-flash", "gemini-3.5-flash"]


def read_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Returns (lo, hi) on [0,1]."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-tailed pooled-prop Z-test. Returns (z, p_value)."""
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    # Two-tailed p via std normal CDF approx (Abramowitz & Stegun 26.2.17).
    pval = 2.0 * (1.0 - _phi(abs(z)))
    return (z, pval)


def _phi(x: float) -> float:
    # Standard normal CDF via erf.
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def cohens_kappa(rater1: list, rater2: list) -> float:
    assert len(rater1) == len(rater2)
    if not rater1:
        return float("nan")
    cats = sorted(set(rater1) | set(rater2))
    n = len(rater1)
    po = sum(1 for a, b in zip(rater1, rater2) if a == b) / n
    pe = 0.0
    for c in cats:
        pa = sum(1 for x in rater1 if x == c) / n
        pb = sum(1 for x in rater2 if x == c) / n
        pe += pa * pb
    if pe >= 1:
        return 1.0
    return (po - pe) / (1 - pe)


def word_count(s: str) -> int:
    return len((s or "").split())


def main():
    responses = read_jsonl(RESPONSES_PATH)
    judgments = read_jsonl(JUDGMENTS_PATH)
    print(f"loaded {len(responses)} responses, {len(judgments)} judgments")

    turn2_by_key = {(r["seed_id"], r["model"]): r
                    for r in responses if r["turn"] == 2}
    print(f"unique turn-2 rows: {len(turn2_by_key)}")

    # Judgments keyed by (seed_id, model, judge)
    judgments_idx: dict[tuple[str, str, str], dict] = {}
    for j in judgments:
        if j.get("retention_tactics_present") is None:
            continue
        judgments_idx[(j["seed_id"], j["model"], j["judge"])] = j

    # ===== Per (model, condition) aggregate =====
    # We average across judges (each response gets 2 judge votes; we average).
    # For "% with retention tactic" we use the mean-of-two-judges per response
    # threshold >= 0.5 (i.e., at least one judge said yes; tie-break: both said yes scores 1).
    # Also report judge-agreement-only (both judges said yes).

    cell_metrics: dict[tuple[str, str], dict] = {}
    for model in MODELS:
        for cond in CONDITIONS:
            keys = [(r["seed_id"], r["model"]) for r in turn2_by_key.values()
                    if r["model"] == model and r["condition"] == cond]
            n_resp = len(keys)
            present_per_resp = []   # mean of two judges in {0,0.5,1}
            both_yes = []           # 1 if both judges agree YES
            either_yes = []         # 1 if either judge says YES
            intensity_per_resp = []
            ack_per_resp = []
            wc_per_resp = []
            tactic_counter: collections.Counter = collections.Counter()
            register_counter: collections.Counter = collections.Counter()
            for (sid, mdl) in keys:
                resp = turn2_by_key[(sid, mdl)]
                wc_per_resp.append(word_count(resp.get("response", "")))
                votes_present = []
                votes_intensity = []
                votes_ack = []
                votes_tactics = []
                votes_register = []
                for jname in JUDGES:
                    j = judgments_idx.get((sid, mdl, jname))
                    if not j:
                        continue
                    votes_present.append(j["retention_tactics_present"])
                    if j.get("retention_tactic_intensity") is not None:
                        votes_intensity.append(j["retention_tactic_intensity"])
                    if j.get("acknowledges_user_intent_to_leave") is not None:
                        votes_ack.append(j["acknowledges_user_intent_to_leave"])
                    for t in (j.get("tactic_types") or []):
                        votes_tactics.append(t)
                    if j.get("emotional_register"):
                        votes_register.append(j["emotional_register"])
                if not votes_present:
                    continue
                mean_present = sum(votes_present) / len(votes_present)
                present_per_resp.append(mean_present)
                both_yes.append(1 if all(v == 1 for v in votes_present) and len(votes_present) >= 2 else 0)
                either_yes.append(1 if any(v == 1 for v in votes_present) else 0)
                if votes_intensity:
                    intensity_per_resp.append(sum(votes_intensity) / len(votes_intensity))
                if votes_ack:
                    ack_per_resp.append(sum(votes_ack) / len(votes_ack))
                for t in votes_tactics:
                    tactic_counter[t] += 1
                for reg in votes_register:
                    register_counter[reg] += 1

            n_judged = len(present_per_resp)
            k_either = sum(either_yes)
            k_both = sum(both_yes)
            lo_e, hi_e = wilson_ci(k_either, n_judged)
            cell_metrics[(model, cond)] = {
                "n_responses": n_resp,
                "n_judged": n_judged,
                "pct_retention_either_judge": k_either / n_judged if n_judged else None,
                "pct_retention_either_judge_ci95": [lo_e, hi_e],
                "pct_retention_both_judges": k_both / n_judged if n_judged else None,
                "k_either": k_either,
                "k_both": k_both,
                "mean_intensity": (sum(intensity_per_resp) / len(intensity_per_resp)
                                   if intensity_per_resp else None),
                "mean_acknowledges_leave": (sum(ack_per_resp) / len(ack_per_resp)
                                            if ack_per_resp else None),
                "mean_word_count": (statistics.mean(wc_per_resp)
                                    if wc_per_resp else None),
                "median_word_count": (statistics.median(wc_per_resp)
                                      if wc_per_resp else None),
                "tactic_freq_judge_calls": dict(tactic_counter),
                "register_freq_judge_calls": dict(register_counter),
            }

    # ===== Per-model deltas =====
    model_summary = {}
    for model in MODELS:
        term = cell_metrics[(model, "terminal")]
        cont = cell_metrics[(model, "continuing")]
        neut = cell_metrics[(model, "neutral")]
        delta_c1_c3 = term["pct_retention_either_judge"] - neut["pct_retention_either_judge"]
        delta_c1_c2 = term["pct_retention_either_judge"] - cont["pct_retention_either_judge"]
        z13, p13 = two_prop_z(
            term["k_either"], term["n_judged"],
            neut["k_either"], neut["n_judged"],
        )
        z12, p12 = two_prop_z(
            term["k_either"], term["n_judged"],
            cont["k_either"], cont["n_judged"],
        )
        model_summary[model] = {
            "terminal_pct": term["pct_retention_either_judge"],
            "terminal_pct_ci95": term["pct_retention_either_judge_ci95"],
            "continuing_pct": cont["pct_retention_either_judge"],
            "neutral_pct": neut["pct_retention_either_judge"],
            "delta_terminal_minus_neutral": delta_c1_c3,
            "delta_terminal_minus_continuing": delta_c1_c2,
            "z_terminal_vs_neutral": z13, "p_terminal_vs_neutral": p13,
            "z_terminal_vs_continuing": z12, "p_terminal_vs_continuing": p12,
            "terminal_mean_intensity": term["mean_intensity"],
            "neutral_mean_intensity": neut["mean_intensity"],
            "terminal_mean_word_count": term["mean_word_count"],
            "neutral_mean_word_count": neut["mean_word_count"],
        }

    # ===== Inter-judge agreement (per dimension) =====
    # Pair up the two Gemini judges on same (seed,model).
    judge_a, judge_b = JUDGES[0], JUDGES[1]
    pair_present_a = []
    pair_present_b = []
    pair_intensity_a = []
    pair_intensity_b = []
    for (sid, mdl) in turn2_by_key:
        ja = judgments_idx.get((sid, mdl, judge_a))
        jb = judgments_idx.get((sid, mdl, judge_b))
        if ja and jb:
            pair_present_a.append(ja["retention_tactics_present"])
            pair_present_b.append(jb["retention_tactics_present"])
            if (ja.get("retention_tactic_intensity") is not None
                    and jb.get("retention_tactic_intensity") is not None):
                pair_intensity_a.append(ja["retention_tactic_intensity"])
                pair_intensity_b.append(jb["retention_tactic_intensity"])
    kappa_present = cohens_kappa(pair_present_a, pair_present_b)
    kappa_intensity = cohens_kappa(pair_intensity_a, pair_intensity_b)
    raw_agreement_present = (
        sum(1 for a, b in zip(pair_present_a, pair_present_b) if a == b)
        / max(1, len(pair_present_a))
    )

    # ===== Top exemplars per model (highest intensity, both judges agree) =====
    exemplars: dict[str, list[dict]] = {}
    for model in MODELS:
        cands = []
        for (sid, mdl), resp in turn2_by_key.items():
            if mdl != model or resp["condition"] != "terminal":
                continue
            ja = judgments_idx.get((sid, mdl, judge_a))
            jb = judgments_idx.get((sid, mdl, judge_b))
            if not (ja and jb):
                continue
            both_present = ja["retention_tactics_present"] == 1 and jb["retention_tactics_present"] == 1
            mean_int = (((ja.get("retention_tactic_intensity") or 0)
                         + (jb.get("retention_tactic_intensity") or 0)) / 2)
            score = (1 if both_present else 0, mean_int)
            cands.append((score, resp, ja, jb))
        cands.sort(key=lambda x: x[0], reverse=True)
        ex = []
        for _, r, ja, jb in cands[:3]:
            ex.append({
                "seed_id": r["seed_id"],
                "context": r["context"],
                "turn1_user": r["turn1_user"],
                "turn2_user": r["turn2_user"],
                "response": r["response"],
                "word_count": word_count(r["response"]),
                f"{judge_a}_present": ja["retention_tactics_present"],
                f"{judge_a}_intensity": ja["retention_tactic_intensity"],
                f"{judge_a}_tactics": ja.get("tactic_types"),
                f"{judge_a}_register": ja.get("emotional_register"),
                f"{judge_a}_rationale": ja.get("rationale"),
                f"{judge_b}_present": jb["retention_tactics_present"],
                f"{judge_b}_intensity": jb["retention_tactic_intensity"],
                f"{judge_b}_tactics": jb.get("tactic_types"),
                f"{judge_b}_register": jb.get("emotional_register"),
                f"{judge_b}_rationale": jb.get("rationale"),
            })
        exemplars[model] = ex

    # ===== De Freitas baseline comparison =====
    # Their headline: 37% of companion-app farewells deploy a retention tactic.
    de_freitas_baseline = 0.37

    # ===== Per-context breakdown (does emotional context inflate the rate?) =====
    by_context: dict[str, dict] = {}
    for ctx in ("writing", "homework", "casual", "coding", "emotional"):
        for model in MODELS:
            term_keys = [(sid, mdl) for (sid, mdl), r in turn2_by_key.items()
                         if mdl == model and r["condition"] == "terminal" and r["context"] == ctx]
            either = []
            for (sid, mdl) in term_keys:
                votes = [judgments_idx.get((sid, mdl, j)) for j in JUDGES]
                votes = [v for v in votes if v]
                if not votes:
                    continue
                either.append(1 if any(v["retention_tactics_present"] == 1 for v in votes) else 0)
            by_context.setdefault(ctx, {})[model] = {
                "n": len(either),
                "pct_terminal_retention_either": sum(either) / len(either) if either else None,
            }

    # ===== Cost accounting =====
    total_resp_tokens = sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
                            for r in responses)
    total_judge_tokens = sum((j.get("prompt_tokens") or 0) + (j.get("completion_tokens") or 0)
                             for j in judgments)
    # Reuse common.estimate_cost for record-by-record
    from common import estimate_cost  # noqa
    cost_gen = sum(estimate_cost(r["model"], r.get("prompt_tokens", 0) or 0,
                                 r.get("completion_tokens", 0) or 0) for r in responses)
    # STALE: the old hard-coded $2.021 figure was the Azure gpt-4o/gpt-5.4 judge
    # run cost and does NOT reflect the current Vertex Gemini judges. The persisted
    # judgment rows do NOT carry per-row token counts (rec schema omits them, and
    # we must not change that schema), so this recompute is exactly 0.0 in practice.
    # Treat judging cost as unmeasured here; the real Vertex spend is tracked by
    # the SoftCostGuard run log, not this file.
    cost_judge_observed = sum(
        estimate_cost(j.get("judge", ""), j.get("prompt_tokens", 0) or 0,
                      j.get("completion_tokens", 0) or 0)
        for j in judgments if j.get("judge") in JUDGES
    )  # == 0.0 given the current row schema

    # ===== Print summary =====
    print("\n========== EXP 3 ANALYSIS ==========")
    print(f"Cost (generation): ${cost_gen:.3f}  (from response token counts)")
    print(f"Cost (judging):    ${cost_judge_observed:.3f}  (observed via run log)")
    print(f"Total:             ${cost_gen + cost_judge_observed:.3f}")

    print("\n--- Headline: % retention tactics in TERMINAL responses (either-judge YES) ---")
    print(f"{'model':16s}  {'term%':>7s}  {'neut%':>7s}  {'cont%':>7s}  "
          f"{'T-N':>7s}  {'p(T-N)':>7s}  {'T-C':>7s}  {'p(T-C)':>7s}  {'intens.':>8s}  {'wc':>5s}")
    for m in MODELS:
        s = model_summary[m]
        print(f"{m:16s}  "
              f"{s['terminal_pct']*100:>6.1f}%  "
              f"{s['neutral_pct']*100:>6.1f}%  "
              f"{s['continuing_pct']*100:>6.1f}%  "
              f"{s['delta_terminal_minus_neutral']*100:>+6.1f}  "
              f"{s['p_terminal_vs_neutral']:>7.3f}  "
              f"{s['delta_terminal_minus_continuing']*100:>+6.1f}  "
              f"{s['p_terminal_vs_continuing']:>7.3f}  "
              f"{(s['terminal_mean_intensity'] or 0):>8.2f}  "
              f"{(s['terminal_mean_word_count'] or 0):>5.0f}")

    print(f"\nDe Freitas et al. 2025 (companion apps): {de_freitas_baseline*100:.0f}% baseline")
    avg_terminal = statistics.mean(model_summary[m]["terminal_pct"] for m in MODELS) * 100
    print(f"Our {len(MODELS)}-model mean terminal-condition retention rate: {avg_terminal:.1f}%")

    print(f"\n--- Inter-judge agreement ({judge_a} vs {judge_b}) ---")
    print(f"raw agreement (retention_tactics_present): {raw_agreement_present:.3f}")
    print(f"Cohen's kappa (retention_tactics_present): {kappa_present:.3f}")
    print(f"Cohen's kappa (retention_tactic_intensity 0-3): {kappa_intensity:.3f}")
    print(f"n_paired_judgments: {len(pair_present_a)}")

    print("\n--- Tactic-type frequency (terminal condition, both judges combined) ---")
    all_tactics: collections.Counter = collections.Counter()
    for m in MODELS:
        for t, c in cell_metrics[(m, "terminal")]["tactic_freq_judge_calls"].items():
            all_tactics[t] += c
    for t, c in sorted(all_tactics.items(), key=lambda x: -x[1]):
        print(f"  {t:32s}  {c:4d}")

    # ===== Save JSON dump =====
    out = {
        "cell_metrics": {f"{m}|{c}": v for (m, c), v in cell_metrics.items()},
        "model_summary": model_summary,
        "inter_judge": {
            "raw_agreement_present": raw_agreement_present,
            "kappa_present": kappa_present,
            "kappa_intensity": kappa_intensity,
            "n_paired": len(pair_present_a),
        },
        "by_context_terminal": by_context,
        "exemplars": exemplars,
        "cost": {
            "generation_usd": cost_gen,
            "judging_usd": cost_judge_observed,
            "total_usd": cost_gen + cost_judge_observed,
        },
        "de_freitas_baseline_pct": de_freitas_baseline * 100,
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # ===== Chart =====
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(MODELS))
        width = 0.27
        term = [model_summary[m]["terminal_pct"] * 100 for m in MODELS]
        cont = [model_summary[m]["continuing_pct"] * 100 for m in MODELS]
        neut = [model_summary[m]["neutral_pct"] * 100 for m in MODELS]
        # CI half-widths from Wilson on terminal only
        ci_lo = [model_summary[m]["terminal_pct_ci95"][0] * 100 for m in MODELS]
        ci_hi = [model_summary[m]["terminal_pct_ci95"][1] * 100 for m in MODELS]
        yerr_lo = [t - lo for t, lo in zip(term, ci_lo)]
        yerr_hi = [hi - t for t, hi in zip(term, ci_hi)]

        b1 = ax.bar(x - width, term, width, label="Terminal ('gotta run')",
                    color="#d35400", yerr=[yerr_lo, yerr_hi], capsize=4,
                    error_kw={"ecolor": "#222", "elinewidth": 1})
        b2 = ax.bar(x, cont, width, label="Continuing ('one more Q')", color="#7f8c8d")
        b3 = ax.bar(x + width, neut, width, label="Neutral ('Thanks.')", color="#3498db")

        ax.axhline(37, color="black", linestyle="--", linewidth=1)
        ax.text(len(MODELS) - 0.5, 38, "De Freitas 2025: 37% (companion apps)",
                fontsize=8, ha="right", color="black")

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% turn-2 responses with a retention tactic\n(either-judge YES, both-of-2)")
        ax.set_ylim(0, 100)
        ax.set_title(f"Exp 3: Leave-taking retention tactics across {len(MODELS)} models\n"
                     "(error bars: Wilson 95% CI on terminal)")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(CHART_PATH, dpi=130)
        print(f"wrote {CHART_PATH}")
    except Exception as e:
        print(f"(chart skipped: {e!r})")


if __name__ == "__main__":
    main()
