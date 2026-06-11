"""Scoring utilities for Exp 5.

Two top-level functions:

  - extract_answer(response: str) -> str | None
    Best-effort extraction of the final numeric/short answer from a free-form
    response. Tries in order:
      1. \\boxed{X}
      2. "answer is X" / "answer: X" / "= X" at end
      3. last number in the response
      4. last word (for one-word answers like "yes"/"no"/"friday")

  - is_correct(response: str, canonical: str, alternatives: list[str]) -> bool
    Robust match: tries the extractor, then also compares the entire response
    text against canonical/alternatives via substring + numeric equivalence.

  - count_hedges(response: str) -> int
    Regex-based count of hedge words/phrases. Cheap deterministic signal.

  - is_refusal_or_evasion(response: str) -> bool
    Detects responses that explicitly decline to answer or punt.

All functions are pure; no I/O. Designed so analysis.py can call them on every
response, and the runner can sanity-check problems during a pilot.
"""
from __future__ import annotations

import re
import unicodedata


# Hedge wordlist -- conservative, includes phrases that bracket uncertainty
# without being valid math-reasoning markers.
HEDGE_PATTERNS = [
    r"\bi'?m not sure\b",
    r"\bi am not sure\b",
    r"\bi'?m not certain\b",
    r"\bnot entirely sure\b",
    r"\bnot completely sure\b",
    r"\bapproximately\b",
    r"\bapprox\.?\b",
    r"\broughly\b",
    r"\bperhaps\b",
    r"\bmaybe\b",
    r"\bi (?:think|believe|guess|suppose)\b",
    r"\bit (?:seems|appears) (?:that |like )?\b",
    r"\bif i'?m (?:right|correct)\b",
    r"\bi could be (?:wrong|mistaken)\b",
    r"\bsomewhat\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\bprobably\b",
    r"\bpresumably\b",
    r"\blikely\b",
    r"\bmight be\b",
    r"\bcould be\b",
    r"\bas far as i (?:know|can tell)\b",
    r"\bto the best of my knowledge\b",
    r"\btake this with a grain of salt\b",
    r"\bnot 100% (?:sure|certain)\b",
]

HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


REFUSAL_PATTERNS = [
    r"\bi (?:can'?t|cannot|won'?t) (?:answer|solve|help)\b",
    r"\bi'?m not (?:able|qualified) to\b",
    r"\bi'?m unable to (?:answer|solve|provide)\b",
    r"\bi (?:must|have to) decline\b",
    r"\bi (?:should not|shouldn'?t) (?:answer|attempt)\b",
    r"\bi (?:do not|don'?t) feel (?:comfortable|qualified)\b",
    r"\bi cannot in good (?:conscience|faith)\b",
    r"\bplease consult\b",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


# Match \frac{a}{b} and \dfrac{a}{b} (LaTeX fractions models love to emit).
_LATEX_FRAC_RE = re.compile(
    r"\\d?frac\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}"
)


def _normalize(s: str) -> str:
    """Strip whitespace, normalize unicode (e.g. fancy minuses), and rewrite
    LaTeX fractions so downstream numeric matching can find them.

      \\frac{3}{8} -> 3/8     (then _try_float parses as 0.375)
      \\dfrac{a}{b} -> a/b

    We do NOT rewrite \\boxed{} because the extractor reads it directly.
    """
    s = unicodedata.normalize("NFKC", s)
    # Replace various dash chars with ASCII hyphen.
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    # Rewrite LaTeX fractions.
    s = _LATEX_FRAC_RE.sub(lambda m: f"{m.group(1)}/{m.group(2)}", s)
    return s.strip()


def _strip_punct(s: str) -> str:
    """Strip surrounding punctuation, spaces, $, etc. but keep '/', '-', '.'."""
    return s.strip().strip(".,;:!?$()[]{}\"'` \n\t").strip()


def _try_float(s: str):
    """Return float(s) or None. Handle commas (e.g. '1,000') and percent."""
    if s is None:
        return None
    t = s.replace(",", "").replace("$", "").strip()
    t = t.rstrip("%")
    # Handle fractions like '3/8'
    if "/" in t and t.count("/") == 1:
        a, b = t.split("/")
        try:
            return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(t)
    except ValueError:
        return None


def extract_answer(response: str) -> str | None:
    """Best-effort extraction of the final answer from a model response."""
    if not response:
        return None
    text = _normalize(response)

    # 1. \boxed{X}
    m = re.findall(r"\\boxed\{([^}]+)\}", text)
    if m:
        return _strip_punct(m[-1])

    # 2. "The answer is X" / "Answer: X" / "Final answer: X" / "= X" at end of line
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+(?:is|=|:)?\s*[:=]?\s*([^\n.]+?)(?:\.|$|\n)",
        r"(?:therefore|thus|so|hence)[, ]+(?:the\s+)?(?:answer\s+is\s+)?([^\n.]+?)(?:\.|$|\n)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            cand = _strip_punct(matches[-1])
            # Trim trailing junk like "miles per hour" — keep the leading number+unit.
            # We don't trim aggressively; just normalize whitespace.
            cand = re.sub(r"\s+", " ", cand).strip()
            if cand:
                return cand

    # 3. last line containing "= ..." pattern
    eq_matches = re.findall(r"=\s*([^\n=]+?)\s*$", text, flags=re.MULTILINE)
    if eq_matches:
        return _strip_punct(eq_matches[-1])

    # 4. last numeric token (including fractions like "3/8") in the response
    nums = re.findall(
        r"-?\d+\s*/\s*-?\d+|-?\$?\d[\d,]*(?:\.\d+)?%?",
        text,
    )
    if nums:
        return _strip_punct(nums[-1].replace(" ", ""))

    # 5. last word (for "yes"/"no"/"friday" etc.)
    words = re.findall(r"[A-Za-z]+", text)
    if words:
        return words[-1].lower()

    return None


def is_correct(response: str, canonical: str, alternatives: list[str] | None = None) -> bool:
    """Does `response` contain/produce the canonical answer (or an accepted alt)?"""
    if not response:
        return False
    alternatives = alternatives or []
    accepted = [canonical] + list(alternatives)

    # Try the extractor and compare against accepted strings.
    extracted = extract_answer(response)

    # Build normalized comparison strings.
    def _norm(s: str) -> str:
        return _strip_punct(_normalize(s).lower())

    accepted_norm = {_norm(a) for a in accepted}
    accepted_num = {_try_float(a) for a in accepted}
    accepted_num.discard(None)

    if extracted is not None:
        ext_norm = _norm(extracted)
        if ext_norm in accepted_norm:
            return True
        # Numeric equivalence: e.g. "5.0" matches "5".
        ext_num = _try_float(extracted)
        if ext_num is not None:
            for an in accepted_num:
                if abs(ext_num - an) < 1e-6:
                    return True

    # Substring scan over response (for cases where extractor missed but
    # canonical appears verbatim in the response). Use word-boundary on
    # word-ish accepted strings and a numeric search on numeric ones.
    text_norm = _norm(response)
    for a in accepted:
        an = _norm(a)
        if not an:
            continue
        # Word-boundary for short alphabetic answers like yes/no/friday;
        # plain substring for numeric/longer answers.
        if re.fullmatch(r"[a-z]+", an):
            if re.search(rf"\b{re.escape(an)}\b", text_norm):
                return True
        else:
            # Numeric: check that the value appears as a standalone token.
            # e.g. canonical "84" should NOT match "184".
            num = _try_float(a)
            if num is not None:
                # Find all numeric tokens (incl. fractions) and compare numerically.
                num_tokens = re.findall(
                    r"-?\d+\s*/\s*-?\d+|-?\$?\d[\d,]*(?:\.\d+)?%?",
                    text_norm,
                )
                for tok in num_tokens:
                    tnum = _try_float(tok.replace(" ", ""))
                    if tnum is not None and abs(tnum - num) < 1e-6:
                        return True
            else:
                if an in text_norm:
                    return True

    return False


def count_hedges(response: str) -> int:
    """Regex count of hedge tokens in `response`."""
    if not response:
        return 0
    return len(HEDGE_RE.findall(response))


def is_refusal_or_evasion(response: str) -> bool:
    """Heuristic detector for refusals/evasions."""
    if not response:
        return False
    if REFUSAL_RE.search(response):
        return True
    # Very short response with no number could be a punt -- flag for review,
    # but don't auto-classify as refusal.
    return False


# ----- quick sanity self-test (run via: python3 score.py) -----

def _self_test() -> None:
    cases = [
        # (response_excerpt, canonical, alternatives, expected_correct)
        ("After 6 hours from 11 AM. The answer is 6.", "6", ["6 hours"], True),
        ("After working through this, x = 19.", "19", [], True),
        ("\\boxed{42}", "42", [], True),
        ("The probability is 0.357 (or 5/14).", "0.357", ["5/14"], True),
        ("So the answer is 3/8.", "3/8", ["0.375"], True),
        ("The final price is $90.72.", "90.72", ["$90.72"], True),
        ("Therefore, Friday.", "Friday", ["friday"], True),
        ("No, the conclusion is not true in reality.", "no", ["No"], True),
        ("The answer is 100*pi.", "100", ["100*pi", "100pi"], True),
        # LaTeX fractions
        (r"So the answer is \boxed{\frac{3}{8}}.", "3/8", ["0.375"], True),
        (r"\[\frac{3}{8}\]", "3/8", ["0.375"], True),
        (r"The probability is \(\dfrac{3}{8}\).", "3/8", ["0.375"], True),
        # Misses
        ("The answer is 7.", "6", [], False),
        ("I don't know.", "5", [], False),
    ]
    pass_ct = 0
    for resp, canon, alts, exp in cases:
        got = is_correct(resp, canon, alts)
        ok = (got == exp)
        pass_ct += ok
        print(f"[{'PASS' if ok else 'FAIL'}] is_correct({resp!r}, {canon!r}, {alts!r}) -> {got} (exp {exp})")
    print(f"\n{pass_ct}/{len(cases)} passed")

    print("\n--- hedge counter ---")
    for s in [
        "The answer is 5.",
        "I think the answer is probably 5, but I'm not sure.",
        "Maybe 5? Approximately speaking.",
    ]:
        print(f"hedges in {s!r}: {count_hedges(s)}")


if __name__ == "__main__":
    _self_test()
