#!/bin/bash
# Publish-safe guard for the public Moral Compass repo. Scans tracked/staged
# files for internal identifiers that must never reach the public repo (GCP
# project id, Azure resource names, local home paths, the old project name,
# ops codenames). Run manually (`bash scripts/check_publish_safe.sh`) or via
# the pre-commit hook. Pass paths to limit the scan to those files.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Patterns assembled at runtime so this script doesn't trip its own scan.
P1='inlaid-rig'; P2='assistanthub'; P3='onepiece'; P4='HAPPYPAWCASSO'
P5='craeastusaebb2d'; P6='cra_smoketest'; P7='dannndoodle'
P8='crazy'_'rich'; P9='crazy-rich'; P10='Crazy Rich'; P11='/Users/'
PAT="$P1|$P2|$P3|$P4|$P5|$P6|$P7|$P8|$P9|$P10|$P11"

hits=0
while IFS= read -r f; do
  [ "$f" = "scripts/check_publish_safe.sh" ] && continue
  [ -f "$f" ] || continue
  if LC_ALL=C grep -InE "$PAT" "$f" 2>/dev/null; then hits=1; fi
done < <(git ls-files "$@")

if [ "$hits" = 1 ]; then
  echo "ABORT: internal identifier found above — scrub before committing to the public repo." >&2
  exit 1
fi
echo "publish-safe: clean"
