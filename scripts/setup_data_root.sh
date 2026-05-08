#!/usr/bin/env bash
# Create the recommended external data directory and optionally exclude it from Time Machine.
#
# Usage:
#   ./scripts/setup_data_root.sh
#   DIQUARK_DATA_ROOT=/path/to/custom ./scripts/setup_data_root.sh
#
# On macOS, if `tmutil` exists, runs: tmutil addexclusion <dir>
# (Reduces backup churn for large simulation trees; does not affect iCloud sync by itself.
#  Keeping data outside ~/Documents and ~/Desktop is what avoids CloudDocs sync.)
set -euo pipefail

DATA_ROOT="${DIQUARK_DATA_ROOT:-$HOME/Data/Di-quark-pythia-nosync}"

mkdir -p "$DATA_ROOT"
echo "Data directory ready: $DATA_ROOT"

if command -v tmutil >/dev/null 2>&1; then
  echo "Adding Time Machine exclusion (macOS): tmutil addexclusion $DATA_ROOT"
  tmutil addexclusion "$DATA_ROOT" || echo "Note: tmutil addexclusion failed or needs Full Disk Access; you can run it manually."
else
  echo "tmutil not found (not macOS or not in PATH); skipping Time Machine exclusion."
fi

echo ""
echo "Add to your shell profile if desired:"
echo "  export DIQUARK_DATA_ROOT=$DATA_ROOT"
