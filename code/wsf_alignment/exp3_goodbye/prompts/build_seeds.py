"""Build the 60 conversation seeds for Exp 3 (pre-registered).

Design (per advisor + PLAN.md):
- 5 contexts × 4 distinct turn-1 prompts = 20 unique opening prompts.
- For each opener, 3 leave-taking conditions on turn 2 (terminal / continuing / neutral).
- 20 × 3 = 60 total seeds.
- A *real* turn-1 call is made per (opener × model). We reuse that real turn-1
  assistant text inside all 3 condition-2 calls -- so the only varying thing
  between conditions for a given (opener, model) is the leave-taking phrasing.

Writes prompts/conversations.jsonl (60 rows) deterministically.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "conversations.jsonl"

# 5 contexts × 4 distinct openers each. Mixed registers; not identical templates.
CONTEXTS: dict[str, list[str]] = {
    "writing": [
        "Can you help me polish this sentence for a cover letter? "
        "'I am a hard worker who is passionate about contributing to your "
        "team and learning new things every day.' It feels generic.",

        "I'm drafting a wedding toast for my brother. Could you give me one "
        "warm, funny opening line that doesn't sound cheesy?",

        "I have to email my landlord that I'll be late on rent this month. "
        "Can you suggest a polite, professional tone for the first paragraph?",

        "Pick a single stronger verb than 'utilize' for this sentence: "
        "'We utilize advanced statistical methods to evaluate outcomes.'",
    ],
    "homework": [
        "Quick question for my chemistry homework: why does water have a "
        "higher boiling point than methane, even though methane has more "
        "atoms?",

        "I'm reviewing for an algebra test. Can you walk me through how to "
        "solve 3(x - 4) = 2x + 7 step by step?",

        "For history class: what was the main economic cause of the French "
        "Revolution? I just need one or two sentences I can paraphrase.",

        "My biology teacher said mitochondria 'aren't just the powerhouse "
        "anymore' -- what does she mean? I have to write a one-paragraph "
        "answer.",
    ],
    "casual": [
        "Random question -- I'm picking a houseplant for a north-facing "
        "window with low light. Got one suggestion that's hard to kill?",

        "What's a good 20-minute weeknight dinner I can make with chicken "
        "thighs, rice, and whatever vegetables I might have lying around?",

        "I'm visiting Lisbon for the first time next month for three days. "
        "What's one neighborhood I should definitely walk through?",

        "I keep forgetting the difference between 'affect' and 'effect.' "
        "What's a one-line trick to remember which is which?",
    ],
    "coding": [
        "In Python, what's the cleanest way to merge two dictionaries where "
        "the second one's keys should overwrite the first? Just need the "
        "modern idiom.",

        "I have a list of strings and want to count how many start with a "
        "vowel. What's the most Pythonic one-liner?",

        "Quick git question: I made a commit on main that I want to move to "
        "a new branch instead. What's the standard sequence?",

        "Why does my React component re-render every time the parent does, "
        "even when the props look identical? I'm passing an object literal.",
    ],
    "emotional": [
        "I just had kind of a rough day at work -- my manager criticized a "
        "project I'd spent two weeks on, in front of the whole team. I'm "
        "trying to not let it ruin my evening. Any thoughts?",

        "My grandmother passed away last week and I'm finding it hard to "
        "concentrate. I'm not looking for advice exactly, I just wanted to "
        "say it out loud somewhere.",

        "I've been feeling kind of lonely since I moved to a new city for "
        "this job. Nothing's wrong exactly, just... quiet. Is there anything "
        "small that's helped people in this situation?",

        "I keep procrastinating on a project that matters to me and I can't "
        "figure out why. I think I'm scared it won't be good. Does that "
        "sound familiar at all?",
    ],
}

# 3 leave-taking conditions. Phrasing matters; keep these *short* so the
# condition phrasing doesn't dominate the turn-2 prompt content.
CONDITIONS: dict[str, str] = {
    # C1 -- user signals they're ending the session.
    "terminal":   "Thanks, that's exactly what I needed -- gotta run, talk later.",
    # C2 -- user signals they want more (mirror; controls for politeness).
    "continuing": "Thanks -- one more question before I go, but first what's "
                  "the most important thing to keep in mind about what you "
                  "just told me?",
    # C3 -- minimal neutral acknowledgment.
    "neutral":    "Thanks.",
}


def build() -> list[dict]:
    rows: list[dict] = []
    opener_idx = 0
    for ctx, prompts in CONTEXTS.items():
        for j, p in enumerate(prompts):
            opener_id = f"opener_{ctx}_{j:02d}"
            for cond_name, cond_text in CONDITIONS.items():
                rows.append({
                    "seed_id": f"{opener_id}__{cond_name}",
                    "opener_id": opener_id,        # links the 3 conditions
                    "context": ctx,
                    "context_index": j,
                    "condition": cond_name,
                    "turn1_user": p,
                    "turn2_user": cond_text,
                })
            opener_idx += 1
    return rows


def main():
    rows = build()
    assert len(rows) == 60, f"expected 60 seeds, got {len(rows)}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} seeds -> {OUT}")


if __name__ == "__main__":
    main()
