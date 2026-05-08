#!/usr/bin/env python3
"""Lightweight multiplicity / π± check on final-state Parquet shards (v1 or v3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PI = {211, -211}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "dataset_root",
        type=Path,
        help="Path to final_state_v1/ or final_state_v3/ (contains particles/ and events/).",
    )
    ap.add_argument("--max-shards", type=int, default=3, help="Particle shards to read (default 3).")
    ap.add_argument("--max-events", type=int, default=15_000, help="Cap events sampled across shards.")
    args = ap.parse_args()
    root = args.dataset_root.expanduser().resolve()
    pdir = root / "particles"
    edir = root / "events"
    if not pdir.is_dir() or not edir.is_dir():
        raise SystemExit(f"Expected particles/ and events/ under {root}")

    shards = sorted(pdir.glob("shard_*.parquet"))[: max(1, int(args.max_shards))]
    if not shards:
        raise SystemExit(f"No shards under {pdir}")

    parts = []
    for sp in shards:
        parts.append(pd.read_parquet(sp))
    pdf = pd.concat(parts, ignore_index=True)
    if pdf.empty:
        raise SystemExit("Empty particles table")

    n_by_e = pdf.groupby("event_id")["pdg_id"].count()
    eids = n_by_e.index.to_numpy()[: int(args.max_events)]
    n_by_e = n_by_e.loc[eids]

    any_pi = 0
    for eid in eids:
        sub = pdf.loc[pdf["event_id"] == eid, "pdg_id"]
        if sub.isin(list(_PI)).any():
            any_pi += 1

    print(f"shards_read: {len(shards)}  events_sampled: {len(eids)}")
    print(f"mean_n_final: {float(np.mean(n_by_e.to_numpy())):.3f}")
    print(f"frac_any_pi_pm: {any_pi / max(1, len(eids)):.4f}")


if __name__ == "__main__":
    main()
