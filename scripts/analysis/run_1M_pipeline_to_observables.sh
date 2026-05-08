#!/usr/bin/env bash
# Sequential: generate 1M Breit-frame DIS events → 90/10 split → π± observables → plots.
#
# Transverse kick (~10%): applied in split_dis_sample_diquark_kick.py (--mode breit_px_kick_only)
# on the Breit CSV (±Δpx on struck quark + diquark partner), with 90% unchanged copies for twins.
# Generator uses --kick-fraction 0 so split owns the kick and unkicked baselines stay in full CSV.
#
# Observables use --csv-momenta-frame breit (no second build_LT on CSV rows).
#
# Logs wall times and UTC timestamps. Run with nohup for long jobs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

BENCH="${ROOT}/outputs/dis_isr_benchmark_1M"
SPLIT="${BENCH}/split_90_10"
LOG="${BENCH}/pipeline_1M_wallclock.log"
PY="${PY:-python3.11}"

mkdir -p "$BENCH"

{
  echo "=== PIPELINE START $(date -u) host=$(hostname) ==="

  # Clean stale partial generation (no metadata = incomplete)
  if [[ -f "${BENCH}/dis_isr_full_event_record.csv" ]] && [[ ! -f "${BENCH}/dis_isr_event_metadata.csv" ]]; then
    echo "Removing incomplete generation outputs (missing metadata)."
    rm -f "${BENCH}/dis_isr_full_event_record.csv"
  fi

  echo "--- GENERATE 1M accepted (Breit CSV, no in-generator kick) $(date -u) ---"
  /usr/bin/time -p "${PY}" scripts/analysis/generate_dis_isr_parton_dataset.py \
    --n-accepted 1000000 \
    --kick-fraction 0 \
    --output-dir "${BENCH}"
  echo "--- GENERATE DONE $(date -u) ---"

  echo "--- SPLIT 1M (~10% Breit-frame px kick, no diquark surgery) $(date -u) ---"
  rm -rf "${SPLIT}"
  /usr/bin/time -p "${PY}" scripts/analysis/split_dis_sample_diquark_kick.py \
    --out-root "${SPLIT}" \
    --max-events 1000000 \
    --mode breit_px_kick_only \
    --full-event-csv "${BENCH}/dis_isr_full_event_record.csv" \
    --metadata-csv "${BENCH}/dis_isr_event_metadata.csv"
  echo "--- SPLIT DONE $(date -u) ---"

  mkdir -p "${BENCH}/figures_observables"
  echo "--- OBSERVABLES (all altered pairs) $(date -u) ---"
  /usr/bin/time -p "${PY}" scripts/analysis/jet_hadron_observables_split_pi_pm.py \
    --split-root "${SPLIT}" \
    --metadata-csv "${BENCH}/dis_isr_event_metadata.csv" \
    --full-event-csv "${BENCH}/dis_isr_full_event_record.csv" \
    --csv-momenta-frame breit \
    --out-csv "${BENCH}/jet_hadron_pi_pm_observables.csv" \
    --figure-dir "${BENCH}/figures_observables"
  echo "--- OBSERVABLES DONE $(date -u) ---"

  mkdir -p "${BENCH}/figures_from_csv"
  echo "--- PLOT FROM CSV $(date -u) ---"
  /usr/bin/time -p "${PY}" scripts/analysis/plot_jet_hadron_pi_pm_observables.py \
    --csv "${BENCH}/jet_hadron_pi_pm_observables.csv" \
    --figure-dir "${BENCH}/figures_from_csv"
  echo "--- PLOT DONE $(date -u) ---"

  echo "=== PIPELINE END $(date -u) ==="
} 2>&1 | tee -a "${LOG}"
