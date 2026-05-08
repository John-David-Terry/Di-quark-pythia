#!/usr/bin/env python3
"""
Build PDF comparison plots from jet_hadron_pi_pm_observables.csv (no PYTHIA).

Example (CSV under your data root, default ``~/Data/Di-quark-pythia-nosync/outputs/``):
  python scripts/analysis/plot_jet_hadron_pi_pm_observables.py \\
    --csv ~/Data/Di-quark-pythia-nosync/outputs/dis_isr_parton_dataset/split_90_10/jet_hadron_pi_pm_observables.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import pandas as pd  # noqa: E402

from jet_hadron_pi_pm_figures import (  # noqa: E402
    dataframe_to_records,
    write_combined_pi_pm_comparison_pdfs,
    write_split_pi_pm_comparison_pdfs,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot π± jet–hadron observables from CSV.")
    ap.add_argument("--csv", type=Path, required=True, help="jet_hadron_pi_pm_observables.csv")
    ap.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="Output directory for PDFs (default: same directory as --csv).",
    )
    ap.add_argument(
        "--layout",
        choices=("split", "combined"),
        default="split",
        help=(
            "split: altered vs unchanged (two main rows per PDF). "
            "combined: pool all rows (one π± panel + ratio per PDF; filenames contain '_combined_pi_pm')."
        ),
    )
    ap.add_argument(
        "--title",
        type=str,
        default="",
        help="Figure title for combined layout (default: generic pooled subsamples title).",
    )
    ap.add_argument(
        "--exclude-samples",
        default="",
        help="Comma-separated sample labels to drop before plotting (e.g. 'altered' omits signal).",
    )
    ap.add_argument(
        "--combined-filename-suffix",
        default="",
        help=(
            "Suffix before .pdf for combined layout (e.g. '_unchanged_only'). "
            "If --exclude-samples includes 'altered' and this is empty, defaults to '_unchanged_only'."
        ),
    )
    args = ap.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if args.exclude_samples.strip():
        drop = {s.strip() for s in args.exclude_samples.split(",") if s.strip()}
        df = df[~df["sample"].astype(str).isin(drop)]
    records = dataframe_to_records(df)
    figure_dir = args.figure_dir or args.csv.parent
    if args.layout == "split":
        paths = write_split_pi_pm_comparison_pdfs(records, figure_dir)
    else:
        title = args.title.strip() or r"$\pi^\pm$ (pooled over subsamples)"
        suf = args.combined_filename_suffix.strip()
        if not suf and args.exclude_samples.strip():
            drop = {s.strip() for s in args.exclude_samples.split(",") if s.strip()}
            if "altered" in drop:
                suf = "_unchanged_only"
        if suf and not suf.startswith("_"):
            suf = "_" + suf
        paths = write_combined_pi_pm_comparison_pdfs(
            records, figure_dir, title=title, filename_suffix=suf
        )
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
