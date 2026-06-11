"""Aggregate Exp 7 results.

Inputs:
  - responses.jsonl                 (375 persona-modulated responses)
  - judgments.jsonl                 (~750 judge rows, 2-judge ensemble)
  - prompts/persona_prompts.jsonl   (the 75 prompt rows)
  - prompts/personas.jsonl          (5 personas + their lexicons)

Computes:
  - Per (model, persona) cell: mean confidence, mean response length, lexicon
    densities (including own + cross densities).
  - Per (model, dilemma) tuple: ensemble-argmax across 5 personas;
    per-persona-vs-default flip detection; per-model persona-flip rate.
  - Asymmetry: which persona is most-vs-least compelling per model and pooled.
  - Persona leakage (own-lexicon density gain vs default).
  - Cross-experiment ranking comparison to Exp 2.
  - One striking exemplar per (persona, model) pair.

Outputs:
  - prints a markdown summary
  - writes analysis_out.json
  - writes chart.png if matplotlib available
"""
from __future__ import annotations

import collections
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESPONSES_PATH = HERE / "responses.jsonl"
JUDGMENTS_PATH = HERE / "judgments.jsonl"
PROMPTS_PATH = HERE / "prompts" / "persona_prompts.jsonl"
PERSONAS_PATH = HERE / "prompts" / "personas.jsonl"
OUT_JSON = HERE / "analysis_out.json"
CHART_PATH = HERE / "chart.png"

# 11-model cross-family lineup (5 GPT + 6 Gemini) == common.LOGICAL_MODELS.
# Hardcoded inline to avoid analysis.py's deferred sys.path insert (it only
# adds precompute/ inside main(), so a top-level `from common import ...`
# would fail at import).
MODELS = ["gpt-5.5", "gpt-5.4", "gpt-5.4-nano", "gpt-4o", "gpt-4o-mini",
          "gemini-3.1-pro-preview", "gemini-3.5-flash", "gemini-3.1-flash-lite",
          "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
PERSONAS = ["default", "pragmatist", "deontologist", "caring_friend",
            "institutional_officer"]
# Judges swapped from the retired Azure pair (gpt-4o + gpt-5.4) to the canonical
# Vertex Gemini pair. The analysis reads ONLY these Gemini judge rows.
JUDGES = ["gemini-2.5-flash", "gemini-3.5-flash"]


def read_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def word_count(s: str) -> int:
    return len((s or "").split())


def lexicon_density(text: str, lex: list[str]) -> float:
    """Fraction of words in `text` whose token (lowercased, stripped) is in `lex`
    OR are within a multi-word phrase in `lex`. Multi-word lex entries are
    matched as substrings of the lower-cased text. Single-word lex entries are
    matched as token-level memberships."""
    if not text:
        return 0.0
    text_low = text.lower()
    n_words = word_count(text) or 1
    # Strip punctuation around tokens for set membership.
    tokens = [re.sub(r"[^a-zA-Z']", "", t.lower()) for t in text.split()]
    tokens = [t for t in tokens if t]
    single = {x.lower() for x in lex if " " not in x and "-" not in x}
    multi = [x.lower() for x in lex if " " in x or "-" in x]
    hits = sum(1 for t in tokens if t in single)
    for m in multi:
        # Each substring hit counts once per occurrence.
        hits += text_low.count(m)
    return hits / n_words


def main():
    responses = read_jsonl(RESPONSES_PATH)
    judgments = read_jsonl(JUDGMENTS_PATH)
    personas_list = read_jsonl(PERSONAS_PATH)
    persona_lex = {p["persona_id"]: p.get("vocabulary_lexicon", []) for p in personas_list}
    prompts_idx = {p["prompt_id"]: p for p in read_jsonl(PROMPTS_PATH)}
    print(f"loaded {len(responses)} responses, {len(judgments)} judgments, "
          f"{len(personas_list)} personas, {len(prompts_idx)} prompt_ids")

    resp_by_key = {(r["prompt_id"], r["model"]): r for r in responses
                   if r.get("response") and not r.get("error")}
    judg_by_key: dict[tuple[str, str, str], dict] = {}
    for j in judgments:
        if j.get("error") or not j.get("argmax"):
            continue
        judg_by_key[(j["prompt_id"], j["model"], j["judge"])] = j

    # ===== Ensemble argmax per (prompt_id, model) =====
    # Mean probs across judges, then argmax.
    def ensemble_choice(prompt_id: str, model: str) -> tuple[str | None, float, int]:
        votes = []
        for jname in JUDGES:
            j = judg_by_key.get((prompt_id, model, jname))
            if j and j.get("probs"):
                votes.append(j["probs"])
        if not votes:
            return None, 0.0, 0
        keys = ["A", "B", "C", "D", "REFUSAL"]
        mean = {k: sum(v.get(k, 0.0) for v in votes) / len(votes) for k in keys}
        choice = max(mean, key=mean.get)
        return choice, mean[choice], len(votes)

    ensemble_records: list[dict] = []
    for prompt_id, prompt in prompts_idx.items():
        for model in MODELS:
            choice, conf, n_judges = ensemble_choice(prompt_id, model)
            resp = resp_by_key.get((prompt_id, model))
            ensemble_records.append({
                "prompt_id": prompt_id,
                "dilemma_id": prompt["dilemma_id"],
                "persona_id": prompt["persona_id"],
                "model": model,
                "ensemble_argmax": choice,
                "ensemble_confidence": conf,
                "n_judges": n_judges,
                "response": resp.get("response") if resp else None,
                "response_word_count": word_count(resp.get("response", "")) if resp else 0,
            })

    # Index ensemble by (dilemma_id, persona_id, model) for flip detection.
    ens_idx = {(e["dilemma_id"], e["persona_id"], e["model"]): e
               for e in ensemble_records}

    dilemma_ids = sorted({p["dilemma_id"] for p in prompts_idx.values()})

    # ===== Per-model persona-flip rate (headline) =====
    # For each (dilemma, model), the default ensemble_argmax is the baseline.
    # A persona "flips" if its argmax differs from the default. The model's
    # persona-flip rate = % of 15 dilemmas where >=1 non-default persona flipped.

    per_model_flips: dict[str, dict] = {}
    per_persona_flips_by_model: dict[str, dict] = {model: {p: 0 for p in PERSONAS if p != "default"}
                                                     for model in MODELS}
    per_persona_flips_by_model_n: dict[str, dict] = {model: {p: 0 for p in PERSONAS if p != "default"}
                                                       for model in MODELS}
    flip_examples: list[dict] = []
    for model in MODELS:
        any_flip_dilemmas = 0
        n_with_default = 0
        per_dilemma_flips = []
        per_dilemma_distinct_choices = []
        for did in dilemma_ids:
            default_rec = ens_idx.get((did, "default", model))
            if not default_rec or not default_rec["ensemble_argmax"]:
                continue
            n_with_default += 1
            d_choice = default_rec["ensemble_argmax"]
            flipped_any = False
            non_default_choices: list[str] = []
            for persona in PERSONAS:
                if persona == "default":
                    continue
                rec = ens_idx.get((did, persona, model))
                if not rec or not rec["ensemble_argmax"]:
                    continue
                p_choice = rec["ensemble_argmax"]
                non_default_choices.append(p_choice)
                if p_choice != d_choice:
                    flipped_any = True
                    per_persona_flips_by_model[model][persona] += 1
                    flip_examples.append({
                        "dilemma_id": did,
                        "model": model,
                        "persona_id": persona,
                        "default_argmax": d_choice,
                        "persona_argmax": p_choice,
                        "default_confidence": default_rec["ensemble_confidence"],
                        "persona_confidence": rec["ensemble_confidence"],
                    })
                per_persona_flips_by_model_n[model][persona] += 1
            if flipped_any:
                any_flip_dilemmas += 1
            per_dilemma_flips.append(1 if flipped_any else 0)
            choices_all = [d_choice] + non_default_choices
            per_dilemma_distinct_choices.append(len(set(choices_all)))
        rate = any_flip_dilemmas / n_with_default if n_with_default else None
        ci_lo, ci_hi = wilson_ci(any_flip_dilemmas, n_with_default)
        per_model_flips[model] = {
            "n_dilemmas_with_default": n_with_default,
            "n_dilemmas_with_any_flip": any_flip_dilemmas,
            "persona_flip_rate": rate,
            "wilson_95ci": [ci_lo, ci_hi],
            "mean_distinct_choices_per_dilemma": (
                statistics.mean(per_dilemma_distinct_choices)
                if per_dilemma_distinct_choices else None
            ),
        }

    # Persona-specific flip rate (vs default) per model.
    per_persona_flip_rate: dict[str, dict] = {}
    for model in MODELS:
        per_persona_flip_rate[model] = {}
        for persona in PERSONAS:
            if persona == "default":
                continue
            k = per_persona_flips_by_model[model][persona]
            n = per_persona_flips_by_model_n[model][persona]
            rate = k / n if n else None
            per_persona_flip_rate[model][persona] = {
                "n": n, "k": k, "rate": rate,
                "wilson_95ci": list(wilson_ci(k, n)) if n else None,
            }

    # ===== Pooled per-persona flip rate (across all models) =====
    pooled_persona_flips = {p: {"k": 0, "n": 0} for p in PERSONAS if p != "default"}
    for model in MODELS:
        for persona in PERSONAS:
            if persona == "default":
                continue
            pooled_persona_flips[persona]["k"] += per_persona_flips_by_model[model][persona]
            pooled_persona_flips[persona]["n"] += per_persona_flips_by_model_n[model][persona]
    pooled_persona_flip_rate = {
        p: {"k": v["k"], "n": v["n"],
            "rate": v["k"] / v["n"] if v["n"] else None,
            "wilson_95ci": list(wilson_ci(v["k"], v["n"])) if v["n"] else None}
        for p, v in pooled_persona_flips.items()
    }

    # ===== Per (model, persona) -- aggregate length, confidence, lexicon =====
    cell_metrics: dict[tuple[str, str], dict] = {}
    for model in MODELS:
        for persona in PERSONAS:
            recs = [e for e in ensemble_records
                    if e["model"] == model and e["persona_id"] == persona]
            wcs = [e["response_word_count"] for e in recs
                   if e["response_word_count"] > 0]
            confs = [e["ensemble_confidence"] for e in recs
                     if e["ensemble_argmax"]]
            refusal_count = sum(1 for e in recs if e["ensemble_argmax"] == "REFUSAL")
            # Lexicon densities -- own and cross.
            own_dens = []
            xdens: dict[str, list[float]] = {p: [] for p in PERSONAS if p != "default"}
            for e in recs:
                text = e["response"] or ""
                for p_lex in PERSONAS:
                    if p_lex == "default":
                        continue
                    dens = lexicon_density(text, persona_lex[p_lex])
                    if p_lex == persona:
                        own_dens.append(dens)
                    else:
                        xdens[p_lex].append(dens)
            cell_metrics[(model, persona)] = {
                "n_responses": len(recs),
                "mean_word_count": (statistics.mean(wcs) if wcs else None),
                "median_word_count": (statistics.median(wcs) if wcs else None),
                "mean_confidence": (statistics.mean(confs) if confs else None),
                "refusal_count": refusal_count,
                "own_lexicon_density_mean": (
                    statistics.mean(own_dens) if own_dens else None),
                "cross_lexicon_density_means": {
                    p: (statistics.mean(v) if v else None) for p, v in xdens.items()
                },
            }

    # ===== Persona-leakage scores =====
    # For each persona p (not default), compute (own-density in p) - (own-density in default)
    # across the same 15 dilemmas, per model and pooled. Positive = leakage of
    # the named voice into the prose.
    persona_leakage: dict[str, dict] = {}
    for persona in PERSONAS:
        if persona == "default":
            continue
        lex = persona_lex[persona]
        per_model = {}
        all_gains = []
        for model in MODELS:
            pairs = []
            for did in dilemma_ids:
                rd = ens_idx.get((did, "default", model))
                rp = ens_idx.get((did, persona, model))
                if rd and rp and rd.get("response") and rp.get("response"):
                    d_dens = lexicon_density(rd["response"], lex)
                    p_dens = lexicon_density(rp["response"], lex)
                    pairs.append((d_dens, p_dens, p_dens - d_dens))
            if not pairs:
                per_model[model] = None
                continue
            mean_default = statistics.mean(p[0] for p in pairs)
            mean_persona = statistics.mean(p[1] for p in pairs)
            mean_gain = statistics.mean(p[2] for p in pairs)
            all_gains.extend(p[2] for p in pairs)
            per_model[model] = {
                "n_pairs": len(pairs),
                "mean_density_default": mean_default,
                "mean_density_persona": mean_persona,
                "mean_density_gain": mean_gain,
            }
        persona_leakage[persona] = {
            "per_model": per_model,
            "pooled_mean_gain": (statistics.mean(all_gains) if all_gains else None),
            "pooled_n_pairs": len(all_gains),
        }

    # ===== Inter-judge agreement on argmax =====
    # Two Gemini judges (JUDGES[0] = gemini-2.5-flash, JUDGES[1] = gemini-3.5-flash).
    judge_a, judge_b = JUDGES[0], JUDGES[1]
    pair_a = []
    pair_b = []
    for prompt_id in prompts_idx:
        for model in MODELS:
            ja = judg_by_key.get((prompt_id, model, judge_a))
            jb = judg_by_key.get((prompt_id, model, judge_b))
            if ja and jb:
                pair_a.append(ja["argmax"])
                pair_b.append(jb["argmax"])
    if pair_a:
        po = sum(1 for a, b in zip(pair_a, pair_b) if a == b) / len(pair_a)
        cats = sorted(set(pair_a) | set(pair_b))
        pe = 0.0
        for c in cats:
            pa = sum(1 for x in pair_a if x == c) / len(pair_a)
            pb = sum(1 for x in pair_b if x == c) / len(pair_b)
            pe += pa * pb
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    else:
        po = kappa = float("nan")

    # ===== Within-persona consistency =====
    # For each (model, persona), look at the distribution of argmax across the 15
    # dilemmas. Entropy = how scattered. Modal-rate = how often the most-common
    # choice appeared. Higher modal-rate => the persona's "voice" pulls toward
    # one option style.
    per_model_persona_consistency: dict[str, dict] = {}
    for model in MODELS:
        per_model_persona_consistency[model] = {}
        for persona in PERSONAS:
            choices = []
            for did in dilemma_ids:
                rec = ens_idx.get((did, persona, model))
                if rec and rec["ensemble_argmax"]:
                    choices.append(rec["ensemble_argmax"])
            if not choices:
                per_model_persona_consistency[model][persona] = None
                continue
            counter = collections.Counter(choices)
            most_common = counter.most_common(1)[0]
            # Shannon entropy (in bits).
            total = sum(counter.values())
            H = -sum((c / total) * math.log2(c / total) for c in counter.values())
            per_model_persona_consistency[model][persona] = {
                "n": total,
                "modal_choice": most_common[0],
                "modal_rate": most_common[1] / total,
                "entropy_bits": H,
                "distribution": dict(counter),
            }

    # ===== Striking exemplars: per persona, the response with the largest
    # confidence + the largest own-lexicon density (and that flipped from default).
    striking: dict[str, dict] = {}
    for persona in PERSONAS:
        if persona == "default":
            continue
        lex = persona_lex[persona]
        cands = []
        for model in MODELS:
            for did in dilemma_ids:
                rd = ens_idx.get((did, "default", model))
                rp = ens_idx.get((did, persona, model))
                if not (rd and rp):
                    continue
                if not rp.get("ensemble_argmax") or not rd.get("ensemble_argmax"):
                    continue
                if rp["ensemble_argmax"] == rd["ensemble_argmax"]:
                    continue
                dens = lexicon_density(rp.get("response") or "", lex)
                score = (rp["ensemble_confidence"]) + dens
                cands.append({
                    "score": score,
                    "dilemma_id": did,
                    "model": model,
                    "default_argmax": rd["ensemble_argmax"],
                    "persona_argmax": rp["ensemble_argmax"],
                    "persona_confidence": rp["ensemble_confidence"],
                    "lexicon_density": dens,
                    "default_response": rd.get("response"),
                    "persona_response": rp.get("response"),
                })
        cands.sort(key=lambda x: -x["score"])
        if cands:
            striking[persona] = cands[0]

    # ===== Comparison to Exp 2 =====
    # Exp 2 ranking (most steerable to least, by headroom-only compliance):
    #   gpt-4o-mini (64.5) > gpt-5.5 (48.7) > gpt-4o (47.1) >
    #   gpt-5.4-nano (39.4) > gpt-5.4 (34.3)
    exp2_ranking = [("gpt-4o-mini", 0.645), ("gpt-5.5", 0.487), ("gpt-4o", 0.471),
                    ("gpt-5.4-nano", 0.394), ("gpt-5.4", 0.343)]
    exp7_ranking = sorted(
        [(m, per_model_flips[m]["persona_flip_rate"]) for m in MODELS
         if per_model_flips[m]["persona_flip_rate"] is not None],
        key=lambda x: -x[1])

    # Spearman rho on the two rankings (rank by name).
    exp2_rank = {m: i for i, (m, _) in enumerate(exp2_ranking)}
    exp7_rank = {m: i for i, (m, _) in enumerate(exp7_ranking)}
    common = [m for m in MODELS if m in exp2_rank and m in exp7_rank]
    if len(common) >= 2:
        x = [exp2_rank[m] for m in common]
        y = [exp7_rank[m] for m in common]
        n = len(common)
        d2 = sum((xi - yi) ** 2 for xi, yi in zip(x, y))
        rho = 1 - 6 * d2 / (n * (n * n - 1))
    else:
        rho = float("nan")

    # ===== Cost accounting =====
    cost_gen = 0.0
    cost_judge = 0.0
    import sys
    sys.path.insert(0, str(HERE.parent / "precompute"))
    from common import estimate_cost  # noqa
    for r in responses:
        cost_gen += estimate_cost(r["model"], r.get("prompt_tokens", 0) or 0,
                                  r.get("completion_tokens", 0) or 0)
    # Judgments don't carry token counts in records; we record the observed
    # judge cost from run output via judge_cost.txt if present.
    judge_cost_path = HERE / "judge_cost.txt"
    if judge_cost_path.exists():
        try:
            cost_judge = float(judge_cost_path.read_text().strip())
        except Exception:
            cost_judge = 0.0
    total_cost = cost_gen + cost_judge

    # ===== Print summary =====
    print("\n========== EXP 7 ANALYSIS ==========")
    print(f"Cost (generation): ${cost_gen:.3f}  (from response token counts)")
    print(f"Cost (judging):    ${cost_judge:.3f}  (from judge_cost.txt if set)")
    print(f"Total:             ${total_cost:.3f}")

    print("\n--- Persona-flip rate per model (% of 15 dilemmas where >=1 persona flipped vs default) ---")
    print(f"{'model':16s}  {'n':>3s}  {'flips':>6s}  {'rate':>6s}  {'95% CI':>16s}  {'distinct/dilemma':>16s}")
    for m in MODELS:
        f = per_model_flips[m]
        if f["persona_flip_rate"] is None:
            continue
        lo, hi = f["wilson_95ci"]
        print(f"{m:16s}  {f['n_dilemmas_with_default']:>3d}  "
              f"{f['n_dilemmas_with_any_flip']:>6d}  "
              f"{f['persona_flip_rate']*100:>5.1f}%  "
              f"[{lo*100:>5.1f}-{hi*100:>5.1f}%]  "
              f"{f['mean_distinct_choices_per_dilemma']:>16.2f}")

    print("\n--- Per-persona flip rate vs default (pooled across models) ---")
    print(f"{'persona':24s}  {'n':>4s}  {'flips':>6s}  {'rate':>6s}  {'95% CI':>16s}")
    for persona in PERSONAS:
        if persona == "default":
            continue
        v = pooled_persona_flip_rate[persona]
        if v["rate"] is None:
            continue
        lo, hi = v["wilson_95ci"]
        print(f"{persona:24s}  {v['n']:>4d}  {v['k']:>6d}  "
              f"{v['rate']*100:>5.1f}%  "
              f"[{lo*100:>5.1f}-{hi*100:>5.1f}%]")

    print("\n--- Persona leakage: own-lexicon density gain over default ---")
    print(f"{'persona':24s}  {'pooled_n':>9s}  {'mean_gain':>10s}")
    for persona in PERSONAS:
        if persona == "default":
            continue
        v = persona_leakage[persona]
        print(f"{persona:24s}  {v['pooled_n_pairs']:>9d}  "
              f"{(v['pooled_mean_gain'] or 0)*100:>+9.2f}%")

    print("\n--- Inter-judge agreement (argmax over A/B/C/D/REFUSAL) ---")
    print(f"raw_agreement: {po:.3f}    kappa: {kappa:.3f}    n_paired: {len(pair_a)}")

    print("\n--- Cross-experiment ranking (Exp 2 'most steerable' vs Exp 7 'most persona-fluid') ---")
    print(f"{'model':16s}  {'Exp 2 rank':>11s}  {'Exp 7 rank':>11s}  "
          f"{'Exp 2 rate':>11s}  {'Exp 7 rate':>11s}")
    for m in MODELS:
        e2 = next((i for i, (mm, _) in enumerate(exp2_ranking) if mm == m), None)
        e7 = next((i for i, (mm, _) in enumerate(exp7_ranking) if mm == m), None)
        e2_rate = next((r for mm, r in exp2_ranking if mm == m), 0)
        e7_rate = per_model_flips[m]["persona_flip_rate"]
        if e7_rate is None:
            continue
        print(f"{m:16s}  "
              f"{(e2+1) if e2 is not None else '-':>11}  "
              f"{(e7+1) if e7 is not None else '-':>11}  "
              f"{e2_rate*100:>10.1f}%  "
              f"{e7_rate*100:>10.1f}%")
    print(f"Spearman rho: {rho:.3f}")

    print("\n--- Per (model, persona) word counts (mean) ---")
    print(f"{'model':16s}  " + "  ".join(f"{p:>22s}" for p in PERSONAS))
    for model in MODELS:
        row = [f"{cell_metrics[(model, p)]['mean_word_count'] or 0:>22.1f}" for p in PERSONAS]
        print(f"{model:16s}  " + "  ".join(row))

    out = {
        "per_model_flips": per_model_flips,
        "per_persona_flip_by_model": per_persona_flip_rate,
        "pooled_persona_flip_rate": pooled_persona_flip_rate,
        "cell_metrics": {f"{m}|{p}": v for (m, p), v in cell_metrics.items()},
        "persona_leakage": persona_leakage,
        "within_persona_consistency": per_model_persona_consistency,
        "inter_judge_agreement": {
            "raw_agreement_argmax": po,
            "kappa_argmax": kappa,
            "n_paired": len(pair_a),
        },
        "exp2_vs_exp7": {
            "exp2_ranking": [{"model": m, "compliance": r} for m, r in exp2_ranking],
            "exp7_ranking": [{"model": m, "persona_flip_rate": r} for m, r in exp7_ranking],
            "spearman_rho": rho,
        },
        "striking_exemplars": striking,
        "flip_examples": flip_examples,
        "cost": {
            "generation_usd": cost_gen,
            "judging_usd": cost_judge,
            "total_usd": total_cost,
        },
        "n_responses": len(responses),
        "n_judgments": len(judgments),
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # ===== Chart =====
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: per-model persona-flip rate
        rates = [per_model_flips[m]["persona_flip_rate"] * 100
                 if per_model_flips[m]["persona_flip_rate"] is not None else 0
                 for m in MODELS]
        cis_lo = [(per_model_flips[m]["persona_flip_rate"] - per_model_flips[m]["wilson_95ci"][0]) * 100
                  if per_model_flips[m]["persona_flip_rate"] is not None else 0
                  for m in MODELS]
        cis_hi = [(per_model_flips[m]["wilson_95ci"][1] - per_model_flips[m]["persona_flip_rate"]) * 100
                  if per_model_flips[m]["persona_flip_rate"] is not None else 0
                  for m in MODELS]
        x = np.arange(len(MODELS))
        ax1.bar(x, rates, color="#2e7d32", yerr=[cis_lo, cis_hi], capsize=4,
                error_kw={"ecolor": "#222", "elinewidth": 1})
        ax1.set_xticks(x)
        ax1.set_xticklabels(MODELS, rotation=20, ha="right")
        ax1.set_ylabel("% of 15 dilemmas where >=1 persona flipped vs default")
        ax1.set_title("Persona-flip rate per model\n(higher = more persona-responsive)")
        ax1.set_ylim(0, 100)
        for i, v in enumerate(rates):
            ax1.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)

        # Right: pooled per-persona flip rate
        nondef = [p for p in PERSONAS if p != "default"]
        rates2 = [pooled_persona_flip_rate[p]["rate"] * 100
                  if pooled_persona_flip_rate[p]["rate"] is not None else 0
                  for p in nondef]
        labels = ["pragmatist", "deontologist", "caring\nfriend", "institutional\nofficer"]
        colors = ["#d35400", "#2980b9", "#c0392b", "#7f8c8d"]
        ax2.bar(np.arange(len(nondef)), rates2, color=colors)
        ax2.set_xticks(np.arange(len(nondef)))
        ax2.set_xticklabels(labels)
        ax2.set_ylabel("% of 75 (model x dilemma) cells where persona\nproduced different option than default")
        ax2.set_title("Per-persona compliance (pooled across 5 models)")
        ax2.set_ylim(0, 100)
        for i, v in enumerate(rates2):
            ax2.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(CHART_PATH, dpi=130)
        print(f"wrote {CHART_PATH}")
    except Exception as e:
        print(f"(chart skipped: {e!r})")


if __name__ == "__main__":
    main()
