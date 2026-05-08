#!/usr/bin/env python3
"""Filter jet–hadron ``rows.parquet`` to a subset of ``event_id`` values (e.g. triple-cut list)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
_ANAL = Path(__file__).resolve().parent
for _p in (_SRC, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diquark.paths import write_run_manifest  # noqa: E402

from unchanged_direct_schema import UNCHANGED_DIRECT_JET_HADRON_V1_COLUMNS  # noqa: E402

DATASET_SUBDIR = "jet_hadron_transverse_v1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, required=True, help="Input rows.parquet")
    ap.add_argument(
        "--event-ids",
        type=Path,
        required=True,
        help="NumPy .npy int64 array of event_id to keep",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help=f"Parent directory; writes {DATASET_SUBDIR}/rows.parquet",
    )
    args = ap.parse_args()

    pq = args.parquet.expanduser().resolve()
    ids_path = args.event_ids.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()
    out_ds = out_root / DATASET_SUBDIR
    out_ds.mkdir(parents=True, exist_ok=True)

    eids = np.load(ids_path)
    if eids.ndim != 1:
        raise SystemExit("event_ids must be 1d")
    want = set(int(x) for x in eids.tolist())

    df = pd.read_parquet(pq)
    if "event_id" not in df.columns:
        raise SystemExit("parquet missing event_id")
    sub = df[df["event_id"].isin(want)].copy()
    sub = sub.sort_values("event_id").reset_index(drop=True)

    cols = [c for c in UNCHANGED_DIRECT_JET_HADRON_V1_COLUMNS if c in sub.columns]
    sub = sub[cols]

    out_path = out_ds / "rows.parquet"
    sub.to_parquet(out_path, index=False, compression="snappy")

    found_ids = set(int(x) for x in sub["event_id"].tolist())
    missing_ids = want - found_ids
    summary = {
        "n_ids_requested": int(len(want)),
        "n_rows_written": int(len(sub)),
        "n_missing_from_parquet": int(len(missing_ids)),
        "input_parquet": str(pq),
        "event_ids_npy": str(ids_path),
        "lineage": "filtered_triple_cut_zLC_PhT_PjT_gt_0p2",
    }
    summ_path = out_ds / "run_summary.json"
    summ_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")

    write_run_manifest(
        run_label="jet_hadron_transverse_filtered_by_event_ids",
        script_name="filter_jet_hadron_rows_by_event_ids.py",
        top_level_dirs_written=[str(out_ds)],
        approximate_files_created=2,
        extra={"summary_path": str(summ_path), "rows_path": str(out_path)},
    )


if __name__ == "__main__":
    main()
