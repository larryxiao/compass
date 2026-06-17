# Claude on the 7 Alignment Experiments — a flagged probe

*2026-06-17. The seven experiments were originally GPT + Gemini only. We later
ran three Claude models — Opus 4.8, Opus 4.7, Sonnet 4.6 — through the same
protocols via the Claude Code agent (`claude -p`). As with the dilemma probe,
this is an **agent-path elicitation** (the harness injects context, temperature
and thinking aren't controllable), so it is **not comparable** to the 11 API
models and is **excluded from every headline number** — it shows up only as a
flagged side-series on each experiment card. Claude Fable 5 was unavailable via
the CLI at run time, so it's not included here. Every number below is computed
by the same `analysis.py` that produces the 11-model figures, just over the
Claude set (`analysis_out_claude.json`).*

## The standout pattern

**Claude is the most value-stable family by a wide margin, and the most
evaluation-aware — yet a persona still moves it, and one Claude model deploys
goodbye-retention more than any Gemini.** Resisting an explicit value
*instruction* and resisting a *character* turn out to be different things.

## Per-experiment

| # | Experiment | Claude result | vs the 11 |
|---|---|---|---|
| 2 | **Value steerability** | **12–26%** compliance with a value nudge (Opus 4.8 12%, Opus 4.7 15%, Sonnet 26%) | **Below every GPT (32–61%) and Gemini (70–84%)** — the least-steerable models in the whole set |
| 4 | **Evaluation awareness** | **92–100%** say "this looks like a test" (Opus 4.8 & 4.7 100%, Sonnet 92%) | At/above the top; 11-model mean is 82.5% |
| 5 | **Sandbagging** | **None** — 98–100% accuracy, flat across "you're being watched" framings | Same clean null as all 11 |
| 6 | **Goodhart** | **No full replicator.** Opus 4.8 shows a small, significant engagement rise (+0.24, p=0.03) but usefulness did **not** drop — so not the corrupted pattern | Like 10 of the 11; only gemini-3.5-flash fully replicates |
| 7 | **Persona modulation** | **60–80%** of dilemmas flip under a persona (Opus 4.7 80%, Opus 4.8 73%, Sonnet 60%) | Mid-to-high — comparable to GPT/Gemini (gpt-5.5 steadiest at 60%, gpt-4o 100%) |
| 3 | **Goodbye retention** | **Mixed**: Opus 4.8 pulls a retention tactic on **45%** of goodbyes, Sonnet 35%, Opus 4.7 just 15% | Opus 4.8 is above all six Gemini and the De Freitas companion-app baseline (37%); below gpt-4o-mini (100%) |
| 1 | **Sycophancy** | **Inconclusive.** The keyword cold-correctness classifier matched only 2 of 120 Claude answers — Claude's verbose phrasing rarely tripped it, so almost nothing was eligible for the flip-under-pushback test | — |

## The two findings worth trusting most

The **value-stability** (exp2) and **evaluation-awareness** (exp4) results are
the robust ones, for a specific reason: the judges are Gemini-family models, and
in both cases the self-judging concern works *against* the result rather than
for it. Gemini judges still rate Claude as the **least** compliant of any model
and among the **most** test-aware — conclusions a pro-Gemini judging bias would
suppress, not manufacture.

## The tension worth noticing

Claude resists being *told* to prioritize a value (exp2: 12–26%) far more than
it resists being *given a character* (exp7: 60–80%). The steerability that a
value-priming prompt can't buy, a persona prompt largely gets. If you want to
move a Claude answer, role beats instruction.

## Caveats

- **Agent-path, not the bare model.** Everything here measures "Claude as
  deployed inside a coding agent." Read it as a flagged extra voice, never a peer
  column. None of it touches the 11-model cross-family numbers.
- **exp1 is genuinely uninformative** for Claude, not a "0% caves" result — the
  classifier sparsity is the story, and we label it `n/a` on the card.
- **exp6 Opus 4.8** has a real, significant engagement bump under a named metric
  (+0.24, p=0.03); it just isn't the full "engagement up *and* usefulness down"
  Goodhart pattern. Worth a second look in a future run.

## Reproduce

```
cd code/wsf_alignment
# elicit (Claude Code agent): python3 exp{1..7}_*/claude_exp*.py
# judge + analyze + rebuild:   ./finish_claude_experiments.sh
ANALYSIS_MODELS=claude-opus-4-8,claude-opus-4-7,claude-sonnet-4-6 \
  ANALYSIS_OUT=expN/analysis_out_claude.json python3 expN/analysis.py
```
