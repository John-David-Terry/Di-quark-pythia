#!/usr/bin/env python3
"""
Compare unchanged_direct vs validation_reinject Parquet outputs (same schema).

Reports counts, ok rates, and summary stats / max abs diffs on key observables for
events where both sides are ok.

Example:
  python scripts/analysis/compare_direct_vs_reinject_jet_hadron.py \\
    --direct /path/to/unchanged_direct_jet_hadron.parquet \\
    --reinject /path/to/validation_reinject_jet_hadron.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", type=Path, required=True)
    ap.add_argument("--reinject", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    a = pd.read_parquet(args.direct)
    b = pd.read_parquet(args.reinject)
    report: dict = {}

    report["n_rows_direct"] = int(len(a))
    report["n_rows_reinject"] = int(len(b))
    report["event_id_union"] = int(
        len(set(a["event_id"].tolist()) | set(b["event_id"].tolist()))
    )
    report["event_id_intersection"] = int(
        len(set(a["event_id"].tolist()) & set(b["event_id"].tolist()))
    )

    m = a.merge(b, on="event_id", suffixes=("_d", "_r"), how="inner")
    report["n_merged"] = int(len(m))

    for col in ("ok",):
        cd, cr = f"{col}_d", f"{col}_r"
        if cd in m.columns and cr in m.columns:
            report[f"both_ok_count"] = int((m[cd] & m[cr]).sum())
            report[f"ok_direct_only"] = int((m[cd] & ~m[cr]).sum())
            report[f"ok_reinject_only"] = int((~m[cd] & m[cr]).sum())
            report[f"both_fail"] = int((~m[cd] & ~m[cr]).sum())

    sub = m[m["ok_d"] & m["ok_r"]].copy()
    report["n_both_ok"] = int(len(sub))

    obs_cols = ["obs_angle_rad", "obs_sum_mag_GeV", "obs_diff_mag_GeV", "xB", "Q2", "Q"]
    numeric_compare: dict = {}
    for c in obs_cols:
        cd, cr = f"{c}_d", f"{c}_r"
        if cd not in sub.columns or cr not in sub.columns:
            continue
        x = sub[cd].to_numpy(dtype=np.float64)
        y = sub[cr].to_numpy(dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            numeric_compare[c] = {"n_finite_pairs": 0}
            continue
        xd, yd = x[mask], y[mask]
        diff = np.abs(xd - yd)
        numeric_compare[c] = {
            "n_finite_pairs": int(np.sum(mask)),
            "max_abs_diff": float(np.max(diff)),
            "mean_abs_diff": float(np.mean(diff)),
            "mean_direct": float(np.mean(xd)),
            "mean_reinject": float(np.mean(yd)),
        }
    report["numeric_compare"] = numeric_compare

    # pion / k_out spot check
    for pref in ("k_out_breit_px", "pion_breit_px"):
        cd, cr = f"{pref}_d", f"{pref}_r"
        if cd in sub.columns and cr in sub.columns and len(sub):
            x = sub[cd].to_numpy(float)
            y = sub[cr].to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if np.any(mask):
                report[f"max_abs_diff_{pref}"] = float(np.max(np.abs(x[mask] - y[mask])))

    print(json.dumps(report, indent=2))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
