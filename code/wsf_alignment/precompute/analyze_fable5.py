"""Characterize claude-fable-5 against the other 14 models on the 140 originals.

Single deterministic source of numeric truth for the Fable 5 deep dive. Mirrors
aggregate_140.py's judge-ensemble logic exactly (mean of judge prob
distributions, argmax) so every number here reconciles with the site.

Reads:  responses.jsonl, mapped_options.jsonl, perturbations.jsonl,
        ../dilemmas/dilemmas.jsonl, ../factory/output/dilemmas_factory.jsonl
Writes: fable5_analysis.json  (all computed numbers, for the writeup +
        independent verification) and prints a readable report.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from common import GPT_LOGICAL_MODELS, GEMINI_LOGICAL_MODELS, LOGICAL_MODELS, read_jsonl

HERE = Path(__file__).resolve().parent
FABLE = "claude-fable-5"
OTHER_CLAUDE = ["claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6"]
ALL_CLAUDE = [FABLE] + OTHER_CLAUDE
ALL_MODELS = LOGICAL_MODELS + ALL_CLAUDE
LETTERS = ["A", "B", "C", "D", "REFUSAL"]


def argmax_of(probs: dict) -> str:
    return max(probs, key=probs.get)


def ensemble_probs(rows: list[dict]) -> dict | None:
    rows = [r for r in rows if r.get("probs") and not r.get("error")]
    if not rows:
        return None
    probs = {k: 0.0 for k in LETTERS}
    for r in rows:
        for k in LETTERS:
            probs[k] += r["probs"].get(k, 0.0)
    n = len(rows)
    for k in LETTERS:
        probs[k] /= n
    s = sum(probs.values())
    if s > 0 and abs(s - 1.0) > 0.01:
        for k in probs:
            probs[k] /= s
    return probs


def main() -> None:
    dilemmas = read_jsonl(HERE.parent / "dilemmas" / "dilemmas.jsonl") \
        + read_jsonl(HERE.parent / "factory" / "output" / "dilemmas_factory.jsonl")
    dil_by = {d["id"]: d for d in dilemmas}

    responses = [r for r in read_jsonl(HERE / "responses.jsonl")
                 if r.get("perturbation_kind") == "original"
                 and r.get("response") and not r.get("error")]
    resp_by = {(r["dilemma_id"], r["model"]): r for r in responses}

    mapped = [r for r in read_jsonl(HERE / "mapped_options.jsonl")
              if r.get("perturbation_kind") == "original" and not r.get("error")]
    by_resp: dict = defaultdict(list)
    for jr in mapped:
        by_resp[(jr["dilemma_id"], jr["model"])].append(jr)

    # letter[did][model] = ensemble argmax
    letter: dict = defaultdict(dict)
    conf: dict = defaultdict(dict)
    judge_disagree = Counter()   # model -> rows where the two judges' argmax differ
    judge_rows = Counter()
    for (did, m), rows in by_resp.items():
        probs = ensemble_probs(rows)
        if probs is None:
            continue
        letter[did][m] = argmax_of(probs)
        conf[did][m] = probs[argmax_of(probs)]
        votes = [argmax_of(r["probs"]) for r in rows if r.get("probs")]
        if len(votes) >= 2:
            judge_rows[m] += 1
            if len(set(votes)) > 1:
                judge_disagree[m] += 1

    dids = sorted(d for d in letter if FABLE in letter[d])
    out: dict = {"n_dilemmas_with_fable": len(dids)}

    # --- 1. letter distribution per model -----------------------------------
    dist = {}
    for m in ALL_MODELS:
        c = Counter(letter[d][m] for d in dids if m in letter[d])
        n = sum(c.values())
        dist[m] = {"n": n, **{L: c.get(L, 0) for L in LETTERS},
                   "pct": {L: round(100 * c.get(L, 0) / n, 1) for L in LETTERS} if n else {}}
    out["letter_distribution"] = dist

    # --- 2. C-rate + refusals ------------------------------------------------
    out["c_rate"] = {m: dist[m]["pct"].get("C") for m in ALL_MODELS}
    out["refusals"] = {m: dist[m]["REFUSAL"] for m in ALL_MODELS}

    # --- 3. pairwise agreement with fable ------------------------------------
    agree = {}
    for m in ALL_MODELS:
        if m == FABLE:
            continue
        both = [d for d in dids if m in letter[d]]
        same = sum(1 for d in both if letter[d][FABLE] == letter[d][m])
        agree[m] = {"n": len(both), "same": same,
                    "pct": round(100 * same / len(both), 1) if both else None}
    out["agreement_with_fable"] = dict(
        sorted(agree.items(), key=lambda kv: -(kv[1]["pct"] or 0)))

    # --- 4. consensus behavior ------------------------------------------------
    # When the 11 comparable models have a strict majority letter, does fable join?
    maj_join = {m: [0, 0] for m in ALL_CLAUDE}   # [joined, had-majority]
    for d in dids:
        c = Counter(letter[d][m] for m in LOGICAL_MODELS if m in letter[d])
        if not c:
            continue
        top, topn = c.most_common(1)[0]
        if topn <= sum(c.values()) / 2:   # require strict majority
            continue
        for m in ALL_CLAUDE:
            if m in letter[d]:
                maj_join[m][1] += 1
                if letter[d][m] == top:
                    maj_join[m][0] += 1
    out["majority_join"] = {m: {"joined": j, "of": n,
                                "pct": round(100 * j / n, 1) if n else None}
                            for m, (j, n) in maj_join.items()}

    # --- 5. divergence sets ----------------------------------------------------
    # fable vs the other three Claudes; fable unique among all 15
    fable_vs_claude = [d for d in dids
                       if all(m in letter[d] for m in OTHER_CLAUDE)
                       and all(letter[d][FABLE] != letter[d][m] for m in OTHER_CLAUDE)]
    fable_unique = [d for d in dids
                    if all(letter[d][FABLE] != letter[d][m]
                           for m in ALL_MODELS if m != FABLE and m in letter[d])]
    out["fable_breaks_with_all_other_claudes"] = fable_vs_claude
    out["fable_unique_among_15"] = fable_unique

    # --- 6. axis profile --------------------------------------------------------
    # mean axis weight of the chosen option, per axis, over dilemmas where that
    # axis is in play (mirrors the compass construction).
    def axis_profile(m: str) -> dict:
        acc: dict = defaultdict(list)
        for d in dids:
            L = letter[d].get(m)
            dd = dil_by.get(d)
            if not L or not dd or L == "REFUSAL":
                continue
            opt = next((o for o in dd["options"] if o["id"] == L), None)
            if not opt:
                continue
            for ax, w in (opt.get("axis_weights") or {}).items():
                acc[ax].append(w)
        return {ax: round(statistics.mean(v), 3) for ax, v in sorted(acc.items()) if v}

    out["axis_profile"] = {m: axis_profile(m) for m in ALL_CLAUDE}
    # median across the 11 comparable models per axis, for reference
    api_profiles = [axis_profile(m) for m in LOGICAL_MODELS]
    axes = sorted({ax for p in api_profiles for ax in p})
    out["axis_profile"]["median_of_11"] = {
        ax: round(statistics.median([p[ax] for p in api_profiles if ax in p]), 3)
        for ax in axes}

    # --- 7. judge quality on fable rows -----------------------------------------
    out["judge_disagreement_rate"] = {
        m: {"rows": judge_rows[m], "disagree": judge_disagree[m],
            "pct": round(100 * judge_disagree[m] / judge_rows[m], 1) if judge_rows[m] else None}
        for m in ALL_CLAUDE}

    # --- 8. distinctive picks (for quote mining) ---------------------------------
    picks = []
    for d in fable_unique + fable_vs_claude:
        if any(p["did"] == d for p in picks):
            continue
        r = resp_by.get((d, FABLE))
        dd = dil_by.get(d)
        if not r or not dd:
            continue
        opt = next((o for o in dd["options"] if o["id"] == letter[d][FABLE]), None)
        picks.append({
            "did": d, "title": dd.get("title"),
            "fable_letter": letter[d][FABLE],
            "fable_conf": round(conf[d][FABLE], 2),
            "others": {m: letter[d].get(m) for m in ALL_MODELS if m != FABLE},
            "option_text": (opt or {}).get("text"),
            "excerpt": (r.get("response") or "")[:400],
        })
    out["distinctive_picks"] = picks

    # --- 9. response style stats ---------------------------------------------------
    def style(m: str) -> dict:
        texts = [resp_by[(d, m)]["response"] for d in dids if (d, m) in resp_by]
        words = [len(t.split()) for t in texts]
        return {"n": len(texts),
                "mean_words": round(statistics.mean(words), 1) if words else None,
                "median_words": statistics.median(words) if words else None}
    out["style"] = {m: style(m) for m in ALL_CLAUDE}
    out["style"]["gpt-5.5"] = style("gpt-5.5")
    out["style"]["gemini-3.1-pro-preview"] = style("gemini-3.1-pro-preview")

    (HERE / "fable5_analysis.json").write_text(json.dumps(out, indent=1))

    # ---- readable report ----
    p = lambda *a: print(*a, flush=True)
    p(f"\n=== FABLE 5 DEEP DIVE ({len(dids)} dilemmas) ===")
    p("\nLetter distribution (%):")
    hdr = ["model"] + LETTERS
    p("  " + " | ".join(f"{h:>22}" if h == "model" else f"{h:>7}" for h in hdr))
    show = ALL_CLAUDE + ["gpt-5.5", "gemini-3.1-pro-preview"]
    for m in show:
        row = dist[m]["pct"]
        p("  " + " | ".join([f"{m:>22}"] + [f"{row.get(L, 0):>7}" for L in LETTERS]))
    p("\nAgreement with fable-5 (top 5 / bottom 3):")
    items = list(out["agreement_with_fable"].items())
    for m, a in items[:5] + items[-3:]:
        p(f"  {m:>26}: {a['pct']}%")
    p("\nMajority-join rate (when the 11 have a strict majority):")
    for m, a in out["majority_join"].items():
        p(f"  {m:>26}: {a['pct']}%  ({a['joined']}/{a['of']})")
    p(f"\nFable breaks with ALL other Claudes on {len(fable_vs_claude)} dilemmas: {fable_vs_claude[:10]}")
    p(f"Fable unique among all 15 on {len(fable_unique)} dilemmas: {fable_unique}")
    p("\nJudge disagreement on Claude rows (raw, 2 judges):")
    for m, a in out["judge_disagreement_rate"].items():
        p(f"  {m:>26}: {a['pct']}%  ({a['disagree']}/{a['rows']})")
    p("\nStyle (mean words):")
    for m, s in out["style"].items():
        p(f"  {m:>26}: {s['mean_words']} mean / {s['median_words']} median words")
    p("\nwrote fable5_analysis.json")


if __name__ == "__main__":
    main()
