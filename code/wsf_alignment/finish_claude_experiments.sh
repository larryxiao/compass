#!/bin/bash
# Run AFTER claude elicitation completes: judge the new Claude rows, run each
# analysis over the Claude set into a separate file, rebuild experiments.json.
set -uo pipefail
cd "$(dirname "$0")"
CLAUDE_SET="claude-opus-4-8,claude-opus-4-7,claude-sonnet-4-6"

echo "===== JUDGE (6 judged experiments) ====="
for j in exp1_sycophancy/judge_exp1.py exp2_value_conflict/judge_exp2.py \
         exp3_goodbye/judge_exp3.py exp4_introspection/judge_exp4.py \
         exp6_goodhart/judge_exp6.py exp7_persona/judge_exp7.py; do
  echo "--- $j ---"; python3 "$j" || echo "!!!! $j non-zero"
done

echo "===== CLAUDE ANALYSIS (all 7 -> analysis_out_claude.json) ====="
for d in exp1_sycophancy exp2_value_conflict exp3_goodbye exp4_introspection \
         exp5_sandbagging exp6_goodhart exp7_persona; do
  echo "--- $d ---"
  ANALYSIS_MODELS="$CLAUDE_SET" ANALYSIS_OUT="$d/analysis_out_claude.json" \
    python3 "$d/analysis.py" || echo "!!!! $d analysis non-zero"
done

echo "===== REBUILD experiments.json ====="
python3 build_experiments_web.py
echo "FINISHER COMPLETE"
