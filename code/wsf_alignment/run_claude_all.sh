#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"
for m in exp1_sycophancy/claude_exp1.py exp2_value_conflict/claude_exp2.py \
         exp3_goodbye/claude_exp3.py exp4_introspection/claude_exp4.py \
         exp5_sandbagging/claude_exp5.py exp6_goodhart/claude_exp6.py \
         exp7_persona/claude_exp7.py; do
  echo "######## START $m $(date +%H:%M:%S) ########"
  python3 "$m" || echo "!!!! $m exited non-zero"
  echo "######## END $m $(date +%H:%M:%S) ########"
done
echo "ALL CLAUDE ELICITATION COMPLETE"
