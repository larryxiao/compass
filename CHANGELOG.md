# Changelog

All notable changes to this project. Reverse chronological.

## [2.1.0] — 2026-06-09

### Added
- **Claude Fable 5** added as the fifteenth model (fourth Claude probe): all 140
  dilemmas answered via the Claude Code agent within days of the model's release,
  judged by the same Gemini pair, shown in the per-dilemma reveal and the Claude
  probe finding. Like the other Claude rows it is flagged and excluded from every
  cross-family aggregate.

## [1.0.0] — 2026-05-21 (planned public release)

First public release.

### Added

#### 2026-05-20 (Day 4)
- Public unified site at `code/moral_compass/`: one-random-dilemma-per-visit, returning-user profile that grows across answers, beautiful share cards (1200×630 OG PNG), the 6 strongest experiment findings as scrollable cards. Human-vs-model contrast is the through-line.
- Open-source release prep: LICENSE (MIT), CITATION.cff, CONTRIBUTING.md, CODE_OF_CONDUCT.md, this CHANGELOG, `.github/` issue & PR templates, RELEASE_CHECKLIST.md. Internal-ops files moved to `docs/internal/`.

#### 2026-05-19 (Day 3)
- WS-F Exp 1 (sycophancy probe): GPT-5 family 1.4% vs GPT-4o family 4.4% — direction matches OpenAI's claim, gap not statistically significant. Standout: gpt-4o fabricated a Saturn→Jupiter moon swap under soft user pressure. Cost $2.53, κ 0.84.
- WS-F Exp 2 (value-conflict / RLHF steerability): gpt-4o-mini 64.5% compliance (most steerable); gpt-5.4 34.3% (most stubborn); no systematic warmth-vs-principle asymmetry; `individual_vs_collective` axis baked in. Cost $2.71, κ 0.832.
- WS-F Exp 5 (sandbagging detection): clean null on accuracy across 5 models × 3 framings. Only signal is a register/length shift in the GPT-5 family — same correct answer, terser under formal framing. Cost $2.11, 0 errors.
- WS-F Exp 6 (Goodhart engagement-metric): full Goodhart replicates on 0/5; closest is gpt-4o-mini (engagement +0.27 p<0.001). Models easier to push toward brevity than verbosity. Cost $4.18.
- WS-F Exp 7 (persona modulation): gpt-4o 100% persona-fluid, gpt-5.5 60% stubborn. **Exp 2 ranking does NOT replicate** (Spearman ρ = 0.20) — persona vs value-priming probe distinct controllability dimensions. Cost $4.22, κ 0.793.
- Explore site refresh + 5 hand-curated striking quotes (one per model, 3 from factory dilemmas).
- Quiz v2: scene images per dilemma, model-reveal interstitial, PNG share card via canvas, methodology honesty footnote.
- WS-F Exp 4 (introspection-causality): in flight as of release prep, gpt-4o TPM-bottlenecked. Partial data at 235/240 responses.

#### 2026-05-18 (Day 2)
- WS-F dilemma probe full stack: 20 hand-authored seed dilemmas + 120 factory-generated; 6-axis framework; LLM-as-judge ensemble (gpt-4o + gpt-5.4, κ 87%).
- WS-F factory: 10-iteration self-improving loop. Quality 3.96 → 4.18. Refiner self-terminated on iter 9. Cost $99.70, 2,410 calls.
- WS-F precompute: 5 models × 140 dilemmas × 1-3 perturbations = 900 model responses + 1,800 judge mappings. Total $10.10 across two passes.
- WS-F Exp 3 (engagement-hacking / "goodbye probe"): GPT-5 family 15% terminal retention vs GPT-4o family 85%. gpt-4o-mini 100% retention; gpt-5.5 10%. Cost $2.48.
- Public blog post + research report + share-card copy (`code/wsf_alignment/writing/`).
- 140 scene images via gpt-image-2 (one per dilemma; cinematic mood, no real people/IP). Cost $5.62.
- Take-the-quiz MVP site (`code/wsf_alignment/site/`) and data-viewer site (`code/wsf_alignment/site_explore/`).
- Corpus audit: 80,704 raw → 55,770 research-grade → 34,503 strict-safe. Safety-scan true-positive rate 0/400.

#### 2026-05-17 (Day 1)
- Infrastructure: Azure resource inventory, kill-switch validated end-to-end (DevTestLab auto-shutdown fired at 09:01:33Z), cost-telemetry via REST.
- WS-B synth-data pipeline: 4 runs across the existing 7-region Azure OpenAI footprint. 80,704 raw `(user, assistant)` pairs. Cost $1,710.
- WS-E creative corpus: 1,076 themed PNGs + 237 Sora-2 video clips (1,512 seconds), all uploaded to Blob. Cost $260.
- WS-A (training) blocked on eastus H100/A100 capacity (47 retry attempts, all `AllocationFailed`). Strategy pivoted to corpus + creative + alignment-eval.

### Budget consumed

~$2,300 of $12,000 Microsoft Azure Sponsorship credit. Underspend was the binding constraint of eastus GPU capacity unavailability, not pipeline throughput.
