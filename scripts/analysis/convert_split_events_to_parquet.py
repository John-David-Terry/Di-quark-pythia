#!/usr/bin/env python3
"""
One-time conversion: per-event CSVs under split_90_10/{altered,unchanged}/ -> Parquet shards.

Output layout (under <data-dir>/split_90_10_parquet/ by default):
  altered/shard_NNNNNN.parquet, altered/manifest.csv
  unchanged/shard_NNNNNN.parquet, unchanged/manifest.csv

manifest.csv columns: event_id, arm, shard

Original CSVs are not modified. Requires pyarrow (pandas Parquet engine).
"""

from __future__ import annotations

import argparse
import heapq
from pathlib import Path
from typing import List

import pandas as pd

_DEFAULT_OUT_NAME = "split_90_10_parquet"


def _sorted_event_csv_paths(arm_dir: Path, max_events: int = 0) -> List[Path]:
    if not arm_dir.is_dir():
        return []

    def ok(p: Path) -> bool:
        try:
            return p.is_file() and p.suffix == ".csv" and p.name.startswith("event_")
        except OSError:
            return False

    gen = (p for p in arm_dir.iterdir() if ok(p))
    if max_events > 0:
        return heapq.nsmallest(max_events, gen, key=lambda x: x.name)
    out = list(gen)
    out.sort(key=lambda x: x.name)
    return out


def _event_id_from_csv_path(p: Path) -> int:
    return int(p.stem.split("_")[1])


def convert_arm(
    arm: str,
    arm_dir: Path,
    out_arm_dir: Path,
    shard_size: int,
    max_events_per_arm: int = 0,
) -> int:
    paths = _sorted_event_csv_paths(arm_dir, max_events=max_events_per_arm)
    if not paths:
        print(f"  {arm}: no event CSVs under {arm_dir}", flush=True)
        out_arm_dir.mkdir(parents=True, exist_ok=True)
        # empty manifest
        pd.DataFrame(columns=["event_id", "arm", "shard"]).to_csv(
            out_arm_dir / "manifest.csv", index=False
        )
        return 0

    out_arm_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: List[dict] = []
    n_events = 0
    shard_idx = 0
    for i in range(0, len(paths), shard_size):
        chunk = paths[i : i + shard_size]
        dfs: List[pd.DataFrame] = []
        for p in chunk:
            eid = _event_id_from_csv_path(p)
            df = pd.read_csv(p)
            dfs.append(df)
            manifest_rows.append(
                {"event_id": eid, "arm": arm, "shard": f"shard_{shard_idx:06d}.parquet"}
            )
            n_events += 1
        big = pd.concat(dfs, ignore_index=True)
        shard_path = out_arm_dir / f"shard_{shard_idx:06d}.parquet"
        big.to_parquet(shard_path, index=False, engine="pyarrow")
        print(
            f"  {arm} shard_{shard_idx:06d}: {len(chunk)} events, {len(big)} rows -> {shard_path.name}",
            flush=True,
        )
        shard_idx += 1

    pd.DataFrame(manifest_rows).sort_values("event_id").to_csv(
        out_arm_dir / "manifest.csv", index=False
    )
    return n_events


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert split_90_10 per-event CSVs to Parquet shards + manifests."
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root containing split_90_10/",
    )
    ap.add_argument(
        "--out-name",
        type=str,
        default=_DEFAULT_OUT_NAME,
        help=f"Output directory name under data-dir (default: {_DEFAULT_OUT_NAME})",
    )
    ap.add_argument(
        "--shard-size",
        type=int,
        default=2000,
        help="Number of events per Parquet shard (default: 2000).",
    )
    ap.add_argument(
        "--max-events-per-arm",
        type=int,
        default=0,
        metavar="N",
        help="If >0, only convert the first N event CSVs per arm (sorted by filename). 0 = all.",
    )
    ap.add_argument(
        "--arm",
        choices=("both", "altered", "unchanged"),
        default="both",
        help="Which arm(s) to convert.",
    )
    args = ap.parse_args()
    data_dir = args.data_dir.resolve()
    split_root = data_dir / "split_90_10"
    if not split_root.is_dir():
        raise SystemExit(f"split_90_10 not found under {data_dir}")

    out_root = data_dir / args.out_name
    if args.shard_size < 1:
        raise SystemExit("--shard-size must be >= 1")

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required for Parquet output. Install with: pip install pyarrow"
        ) from exc

    print(f"data_dir={data_dir}\nout_root={out_root}\nshard_size={args.shard_size}", flush=True)
    total = 0
    max_pa = int(args.max_events_per_arm)
    if args.arm in ("both", "altered"):
        n = convert_arm(
            "altered",
            split_root / "altered",
            out_root / "altered",
            args.shard_size,
            max_events_per_arm=max_pa,
        )
        total += n
    if args.arm in ("both", "unchanged"):
        n = convert_arm(
            "unchanged",
            split_root / "unchanged",
            out_root / "unchanged",
            args.shard_size,
            max_events_per_arm=max_pa,
        )
        total += n
    print(f"Done. Total events written: {total}", flush=True)


if __name__ == "__main__":
    main()
