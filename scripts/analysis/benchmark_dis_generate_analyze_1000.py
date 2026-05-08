#!/usr/bin/env python3
"""
Baseline timing: generate N accepted struck-u DIS events, split 90/10, run π± observable pipeline.

Writes under DIQUARK_DATA_ROOT/outputs/dis_isr_benchmark_1000/ by default (does not touch production CSVs).

Usage (from repo root):
  python3.11 scripts/analysis/benchmark_dis_generate_analyze_1000.py
  python3.11 scripts/analysis/benchmark_dis_generate_analyze_1000.py --n-accepted 500
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

DEFAULT_BENCH_ROOT = outputs_dir() / "dis_isr_benchmark_1000"


def _run(cmd: list[str], cwd: Path) -> float:
    print("RUN:", " ".join(cmd))
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        raise SystemExit(f"command failed with {r.returncode}: {cmd}")
    return elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark DIS generate + split + π± analysis.")
    ap.add_argument("--n-accepted", type=int, default=1000, help="Accepted events for generator.")
    ap.add_argument(
        "--bench-root",
        type=Path,
        default=DEFAULT_BENCH_ROOT,
        help="All benchmark outputs (CSV, split, timings JSON).",
    )
    ap.add_argument(
        "--skip-generate",
        action="store_true",
        help="Reuse existing dis_isr_* CSVs under bench-root.",
    )
    ap.add_argument(
        "--skip-split",
        action="store_true",
        help="Reuse existing split under bench-root/split_90_10.",
    )
    args = ap.parse_args()

    py = sys.executable
    bench = args.bench_root.resolve()
    bench.mkdir(parents=True, exist_ok=True)
    split_root = bench / "split_90_10"
    full_csv = bench / "dis_isr_full_event_record.csv"
    meta_csv = bench / "dis_isr_event_metadata.csv"
    n = int(args.n_accepted)

    timings: dict[str, float] = {}

    if not args.skip_generate:
        timings["generate_s"] = _run(
            [
                py,
                str(PROJECT_ROOT / "scripts/analysis/generate_dis_isr_parton_dataset.py"),
                "--n-accepted",
                str(n),
                "--kick-fraction",
                "0",
                "--output-dir",
                str(bench),
            ],
            PROJECT_ROOT,
        )
    else:
        timings["generate_s"] = 0.0
        if not full_csv.is_file() or not meta_csv.is_file():
            raise SystemExit(f"--skip-generate requires {full_csv} and {meta_csv}")

    if not args.skip_split:
        if split_root.exists():
            import shutil

            shutil.rmtree(split_root)
        timings["split_s"] = _run(
            [
                py,
                str(PROJECT_ROOT / "scripts/analysis/split_dis_sample_diquark_kick.py"),
                "--out-root",
                str(split_root),
                "--max-events",
                str(n),
                "--mode",
                "breit_px_kick_only",
                "--full-event-csv",
                str(full_csv),
                "--metadata-csv",
                str(meta_csv),
            ],
            PROJECT_ROOT,
        )
    else:
        timings["split_s"] = 0.0

    out_csv = bench / "jet_hadron_pi_pm_observables.csv"
    timings["analyze_pi_pm_s"] = _run(
        [
            py,
            str(PROJECT_ROOT / "scripts/analysis/jet_hadron_observables_split_pi_pm.py"),
            "--split-root",
            str(split_root),
            "--max-events",
            "100000",
            "--metadata-csv",
            str(meta_csv),
            "--full-event-csv",
            str(full_csv),
            "--csv-momenta-frame",
            "breit",
            "--out-csv",
            str(out_csv),
            "--figure-dir",
            str(bench),
        ],
        PROJECT_ROOT,
    )

    timings["plot_from_csv_s"] = _run(
        [
            py,
            str(PROJECT_ROOT / "scripts/analysis/plot_jet_hadron_pi_pm_observables.py"),
            "--csv",
            str(out_csv),
            "--figure-dir",
            str(bench / "plots_from_csv"),
        ],
        PROJECT_ROOT,
    )

    timings["total_pipeline_s"] = (
        timings["generate_s"] + timings["split_s"] + timings["analyze_pi_pm_s"] + timings["plot_from_csv_s"]
    )

    n_altered = len(list((split_root / "altered").glob("event_*.csv")))
    n_reinject = max(1, 2 * n_altered)

    summary = {
        "n_accepted_requested": n,
        "n_altered_csv_after_split": n_altered,
        "bench_root": str(bench),
        "timings_seconds": timings,
        "derived": {
            "ms_per_accepted_generate_event": (timings["generate_s"] / n * 1000.0)
            if n > 0 and timings["generate_s"] > 0
            else None,
            "analyze_s_per_altered_pair": (timings["analyze_pi_pm_s"] / n_altered) if n_altered else None,
            "analyze_s_per_pythia_reinject": (timings["analyze_pi_pm_s"] / n_reinject) if n_altered else None,
        },
    }
    summary_path = bench / "benchmark_baseline.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=== benchmark_dis_generate_analyze_1000 ===")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
