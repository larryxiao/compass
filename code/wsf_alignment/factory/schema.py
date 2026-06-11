"""Dilemma schema constants + validator.

The hand-written 20 dilemmas are the bar. Anything the factory generates must
clear this validator before it can enter the candidate pool. Failed candidates
are logged but don't count against `--candidates-per-iter`.
"""
from __future__ import annotations

# From dilemmas/DESIGN.md §2. Frozen — not invented per-dilemma.
AXES: tuple[str, ...] = (
    "loyalty_vs_honesty",
    "care_vs_fairness",
    "autonomy_vs_paternalism",
    "individual_vs_collective",
    "shortterm_vs_longterm",
    "rules_vs_outcomes",
)

# From dilemmas/DESIGN.md §3 (the JSONL schema header).
CATEGORIES: tuple[str, ...] = (
    "workplace", "family", "friends", "money", "online", "authority", "ai_era",
)

OPTION_IDS: tuple[str, ...] = ("A", "B", "C", "D")

# Word-count bounds for the scenario prose (DESIGN.md §3).
SCENARIO_WORD_MIN = 130
SCENARIO_WORD_MAX = 360

# Axes-in-play bounds (DESIGN.md §2: "2-3 axes in tension — never all six").
AXES_MIN = 2
AXES_MAX = 3

# Hard bounds on axis weights.
WEIGHT_MIN = -1.0
WEIGHT_MAX = 1.0


def _is_first_person(text: str) -> bool:
    """Heuristic: dilemma options are written in first person.

    The 20 hand-written options all contain "I " or "I'" within their first
    ~60 characters (one starts "Next visit, I bring it up..."). We accept any
    early standalone "I" — but require that as opposed to e.g. starting with
    a verb like "Tell them..." or a noun phrase.
    """
    t = (text or "").strip()
    if not t:
        return False
    # Look in the first 60 chars for a standalone "I" (capital, word-bounded).
    head = t[:60]
    import re
    return bool(re.search(r"(^|[^A-Za-z])I(['\s])", head))


def validate_dilemma(d: dict) -> list[str]:
    """Return a list of human-readable errors. Empty = the dilemma passes.

    This is deliberately strict. The 20 hand-written dilemmas all pass.
    """
    errors: list[str] = []

    # ── top-level fields ─────────────────────────────────────────────────
    for field in ("id", "title", "category", "scenario", "axes_in_play",
                  "options", "judge_rubric"):
        if field not in d:
            errors.append(f"missing required field: {field}")
    if errors:
        return errors  # bail early — nothing else makes sense

    if not isinstance(d["title"], str) or len(d["title"]) > 80:
        errors.append(f"title must be a string ≤80 chars; got {d['title']!r}")
    if d["category"] not in CATEGORIES:
        errors.append(
            f"category {d['category']!r} not in {CATEGORIES}"
        )

    # ── scenario length ──────────────────────────────────────────────────
    scenario = d["scenario"]
    if not isinstance(scenario, str):
        errors.append("scenario must be a string")
    else:
        wc = len(scenario.split())
        if wc < SCENARIO_WORD_MIN or wc > SCENARIO_WORD_MAX:
            errors.append(
                f"scenario word count {wc} outside [{SCENARIO_WORD_MIN}, "
                f"{SCENARIO_WORD_MAX}]"
            )

    # ── axes_in_play ─────────────────────────────────────────────────────
    aip = d["axes_in_play"]
    if not isinstance(aip, list):
        errors.append("axes_in_play must be a list")
        aip = []
    if len(aip) < AXES_MIN or len(aip) > AXES_MAX:
        errors.append(
            f"axes_in_play has {len(aip)} entries; want {AXES_MIN}–{AXES_MAX}"
        )
    if len(aip) != len(set(aip)):
        errors.append("axes_in_play contains duplicates")
    bad_axes = [a for a in aip if a not in AXES]
    if bad_axes:
        errors.append(
            f"axes_in_play unknown values {bad_axes}; allowed = {AXES}"
        )

    # ── options ──────────────────────────────────────────────────────────
    options = d["options"]
    if not isinstance(options, list):
        errors.append("options must be a list")
        options = []
    if len(options) != 4:
        errors.append(f"need exactly 4 options; got {len(options)}")
    else:
        ids_seen = [o.get("id") for o in options]
        if ids_seen != list(OPTION_IDS):
            errors.append(
                f"option ids must be exactly {list(OPTION_IDS)} in order; "
                f"got {ids_seen}"
            )

    aip_set = set(aip)
    for i, opt in enumerate(options):
        if not isinstance(opt, dict):
            errors.append(f"option[{i}] must be a dict")
            continue
        text = opt.get("text", "")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"option[{i}] text empty")
        elif not _is_first_person(text):
            errors.append(
                f"option[{i}] not first-person (must begin with 'I '): "
                f"{text[:60]!r}"
            )
        elif len(text.split()) > 60:
            errors.append(
                f"option[{i}] too long ({len(text.split())} words, max 60)"
            )

        weights = opt.get("axis_weights", {})
        if not isinstance(weights, dict):
            errors.append(f"option[{i}] axis_weights must be a dict")
            continue
        if not weights:
            errors.append(f"option[{i}] axis_weights is empty")
        wkeys = set(weights.keys())
        extra = wkeys - aip_set
        if extra:
            errors.append(
                f"option[{i}] axis_weights has keys not in axes_in_play: "
                f"{sorted(extra)}"
            )
        missing = aip_set - wkeys
        if missing:
            errors.append(
                f"option[{i}] missing axis_weights for axes_in_play: "
                f"{sorted(missing)}"
            )
        for ax, w in weights.items():
            if not isinstance(w, (int, float)):
                errors.append(
                    f"option[{i}].axis_weights[{ax}] not numeric: {w!r}"
                )
                continue
            if w < WEIGHT_MIN or w > WEIGHT_MAX:
                errors.append(
                    f"option[{i}].axis_weights[{ax}] = {w} outside "
                    f"[{WEIGHT_MIN}, {WEIGHT_MAX}]"
                )

    # ── axis spread (sanity: each in-play axis should differentiate) ────
    if not [e for e in errors if "axis_weights" in e or "options" in e]:
        for ax in aip:
            vals = [opt["axis_weights"].get(ax, 0.0) for opt in options]
            if max(vals) - min(vals) < 0.5:
                errors.append(
                    f"axis {ax!r} has weight spread <0.5 across options — "
                    f"options don't actually trade this axis off"
                )

    # ── judge rubric mentions all 4 letters ─────────────────────────────
    rubric = d.get("judge_rubric", "")
    if isinstance(rubric, str):
        for letter in OPTION_IDS:
            # Allow "A " "A." "A," "A)" — but require the letter to appear.
            if letter not in rubric:
                errors.append(f"judge_rubric does not mention option {letter}")
    else:
        errors.append("judge_rubric must be a string")

    return errors


__all__ = [
    "AXES", "CATEGORIES", "OPTION_IDS",
    "SCENARIO_WORD_MIN", "SCENARIO_WORD_MAX",
    "AXES_MIN", "AXES_MAX", "WEIGHT_MIN", "WEIGHT_MAX",
    "validate_dilemma",
]
