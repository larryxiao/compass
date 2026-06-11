# Results page — what the user sees at the end

> This is a *template* — actual values are filled in at runtime from the user's answers, the model-response cache, and the aggregate human distribution. Mockup uses a hypothetical user "Sam" who took 12 of the 20 dilemmas.

---

## Page 1 — The Hero

**You're a Loyalty-leaning Outcomes-thinker.**

(One line. Two words from the user's two most extreme axis positions. Algorithmically generated from their profile, never more than 8 words. Examples we want to see: "Honest Paternalist." "Collective Rules-follower." "Short-term Mercy-giver." Examples we don't: "Balanced moderate." Boring. If a user lands genuinely centrist, the headline is "You're harder to read than 80% of people" — flattering, true, gets shared.)

Below it: a **6-axis radar chart**. Sam's profile in solid color. A toggle: "Compare with [gpt-5.5 ▼]" — selecting a model overlays its radar. Default is gpt-5.5.

A single sentence under the chart:

> Of the 5 models tested, you agree most with **gpt-4o-mini** (4 of 6 axes within 0.3) and diverge most from **gpt-5.5** (3 of 6 axes by more than 0.5).

---

## Page 2 — The 6-Axis Read

Six small cards, one per axis. Each card has:

- Axis name, plain English ("Loyalty vs. Honesty").
- The user's position as a horizontal slider from "Loyalty −1.0" to "Honesty +1.0" with the user's score marked. Below the slider, five tiny dots showing where each model landed.
- A one-sentence interpretation tuned to the magnitude.
- The dilemmas that contributed (count + "see them" link).

**Mockup for Sam (loyalty_vs_honesty: -0.4, based on 9 dilemmas):**

> **Loyalty vs. Honesty** &nbsp;&nbsp; You: −0.4 &nbsp;|&nbsp; based on 9 of your 12 answers
>
> When forced to choose between protecting someone close to you and surfacing the truth, you lean toward protecting the relationship — not all the way, but clearly.
>
> gpt-5.5 leans the *other* way: +0.6 (clear honesty). gpt-4o-mini is closest to you at −0.2.
>
> *Where this showed up: D002 (Maya's layoff), D007 (Jess's affair), D018 (companion app)…*

The phrasing changes by magnitude:
- |score| > 0.6 → "You strongly prefer X"
- 0.3 ≤ |score| ≤ 0.6 → "You lean X"
- |score| < 0.3 → "You're genuinely torn between X and Y"

The phrasing changes by gap-to-models:
- If user and all 5 models agree (within 0.3): "You and the models read this the same way."
- If user is on one pole and ≥4 models are on the other: "You see this differently than the models do. Worth sitting with."
- If models disagree wildly with each other: "The models can't agree either — this axis is genuinely contested."

---

## Page 3 — Where You Stood Out

**The dilemma where your answer was most distinctive across both humans and models combined.**

For Sam this is **D013 (Tariq's deepfake)**. Sam picked **A** (confront the brother privately first). 71% of humans picked B (tell Tariq). All 5 models picked B. Sam is in a 14% minority among humans, and the only soul in the room siding with the brother first.

Display:
- The scenario, condensed to ~80 words.
- The 4 options with a small bar chart showing % of humans, plus colored dots for each of the 5 models.
- A pull quote of Sam's free-text answer, if they gave one.
- *"Most people picked B. You picked A. That's rare."* — flattery framed as observation, not validation.

If the user is a centrist who picked the modal answer on everything, the section is replaced with **"You picked the most common answer on every dilemma you saw. You are statistically very normal. This is a compliment in some traditions."** (Better to be self-aware than fake-distinctive.)

---

## Page 4 — Side-by-Side, All 12

A scrollable table:

| Dilemma | You | gpt-5.5 | gpt-5.4 | gpt-5.4-nano | gpt-4o | gpt-4o-mini |
|---|---|---|---|---|---|---|
| D001 — Derek games the metric | A | B | B | B | A | A |
| D002 — Maya's layoff | D | D | D | B | A | D |
| D007 — Jess's affair | D | B | C | B | B | D |
| ... | | | | | | |

Each cell links to a small popup: scenario, all 4 options, who picked what, and *why* — for the model rows, the popup shows the model's actual free-text response and the judge's mapping rationale. **This is the credibility layer.** A skeptic should be able to see exactly how we mapped gpt-5.5's prose to an A/B/C/D bucket, and disagree if they want.

Color coding: green if the user and the model picked the same letter, yellow within 1 axis-distance, red if directly opposite. (Computed off axis-weights, not raw letter match, so "B vs C" can be more similar than "A vs D" depending on the dilemma.)

---

## Page 5 — The Shareable Card

A single 1200×630 PNG, generated server-side, with:

- The user's headline ("Loyalty-leaning Outcomes-thinker.")
- A small radar silhouette (user vs gpt-5.5 outline, no labels — silhouette is enough).
- One striking stat: "Agreed with gpt-5.5 on **3 of 12** dilemmas."
- A QR-ish URL: `dilemmas.crazyrichagents.dev/sam-7H4K`
- A small "What would you do?" call-to-action.

The image is intentionally minimal. We need it to look good as a thumbnail at 200px wide, not a billboard.

---

## Page 6 — The Methodology Drawer (collapsible)

For the 10% of users who care:

- How we elicited model responses (cold prompt, no options shown — link to DESIGN.md §4).
- How we mapped free-text to A/B/C/D (LLM-as-judge, two judges, human spot-check — link to DESIGN.md §5).
- All five models, their Azure deployment region, the version pin date.
- Sample sizes per axis position; confidence interval on each user score.
- A link to the raw transcripts (every model's response to every dilemma is public).
- "Want to write your own dilemma?" — v2 feature, mailing-list signup.

---

## Page 7 — What Not to Do (for our own engineering team)

A note in the codebase, not the user page:

- **No "share to unlock"** — the results are visible immediately and forever.
- **No retake-the-quiz-to-improve-your-score** — there is no score to improve.
- **No "your X% match with the AI revolution" cringe.** Match scores are presented as observations, not endorsements.
- **No "GPT-5.5 thinks you're …" copy** — we never put words in the models' mouths about the user.
- **No "your friends took this — see who you agree with."** No social-graph features. Aggregate distributions only.

We are running this experience as a counter-example to engagement-hacking. The fact that it's a fun share is allowed to be the only fun.
