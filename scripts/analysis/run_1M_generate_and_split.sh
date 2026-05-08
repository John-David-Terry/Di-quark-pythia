#!/usr/bin/env bash
# Generate 1M accepted struck-u DIS events (Breit CSV, no in-generator kick), then split:
#   ~90% → unchanged/   (exact copy from full CSV)
#   ~10% → altered/     (breit_px_kick_only: ±Δpx on struck quark + diquark partner)
# Observables are NOT run; use jet_hadron_observables_split_pi_pm.py when ready.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

BENCH="${ROOT}/outputs/dis_isr_benchmark_1M"
SPLIT="${BENCH}/split_90_10"
LOG="${BENCH}/generate_split_1M.log"
PY="${PY:-python3.11}"

mkdir -p "$BENCH"

{
  echo "=== START $(date -u) host=$(hostname) ==="
  echo "PY=${PY}"

  if [[ -f "${BENCH}/dis_isr_full_event_record.csv" ]] && [[ ! -f "${BENCH}/dis_isr_event_metadata.csv" ]]; then
    echo "Removing incomplete generation (CSV without metadata)."
    rm -f "${BENCH}/dis_isr_full_event_record.csv"
  fi

  echo "--- GENERATE 1M $(date -u) ---"
  /usr/bin/time -p "${PY}" scripts/analysis/generate_dis_isr_parton_dataset.py \
    --n-accepted 1000000 \
    --kick-fraction 0 \
    --output-dir "${BENCH}"

  echo "--- SPLIT ~10% altered $(date -u) ---"
  rm -rf "${SPLIT}"
  /usr/bin/time -p "${PY}" scripts/analysis/split_dis_sample_diquark_kick.py \
    --out-root "${SPLIT}" \
    --max-events 1000000 \
    --mode breit_px_kick_only \
    --full-event-csv "${BENCH}/dis_isr_full_event_record.csv" \
    --metadata-csv "${BENCH}/dis_isr_event_metadata.csv"

  echo "=== DONE $(date -u) ==="
} 2>&1 | tee -a "${LOG}"
