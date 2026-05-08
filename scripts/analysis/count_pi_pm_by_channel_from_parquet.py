#!/usr/bin/env python3
"""
Join altered_metadata (split_channel A/B/C) with reinjected final_state_v1 hadrons and
summarize π⁺ (211) vs π⁻ (-211) multiplicities per channel.

Typical inputs:
  --altered-root  .../altered_100k_parquet  (altered_metadata/shard_*.parquet)
  --final-root    .../final_state_v1       (events + particles shards, ok==1)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Tuple

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_channel_map(altered_root: Path) -> Dict[int, str]:
    """event_id -> channel letter for alteration_succeeded rows with nonempty split_channel."""
    mdir = altered_root / "altered_metadata"
    if not mdir.is_dir():
        raise FileNotFoundError(f"missing altered_metadata under {altered_root}")
    out: Dict[int, str] = {}
    for pq in sorted(mdir.glob("shard_*.parquet")):
        df = pd.read_parquet(
            pq, columns=["event_id", "split_channel", "alteration_succeeded"]
        )
        sub = df[(df["alteration_succeeded"] == 1) & (df["split_channel"].isin(["A", "B", "C"]))]
        for eid, ch in zip(sub["event_id"].astype(int), sub["split_channel"].astype(str)):
            out[int(eid)] = str(ch)
    return out


def _load_ok_events(final_root: Path) -> Dict[int, int]:
    """event_id -> ok (1/0) from final_state events shards."""
    edir = final_root / "events"
    if not edir.is_dir():
        raise FileNotFoundError(f"missing events under {final_root}")
    ok_map: Dict[int, int] = {}
    for pq in sorted(edir.glob("shard_*.parquet")):
        df = pd.read_parquet(pq, columns=["event_id", "ok"])
        for eid, ok in zip(df["event_id"].astype(int), df["ok"].astype(int)):
            ok_map[int(eid)] = int(ok)
    return ok_map


def _pion_counts_per_event(final_root: Path) -> Dict[int, Tuple[int, int]]:
    """event_id -> (n_pi_plus, n_pi_minus) over all hadrons in particles shards."""
    pdir = final_root / "particles"
    if not pdir.is_dir():
        raise FileNotFoundError(f"missing particles under {final_root}")
    acc: DefaultDict[int, List[int]] = defaultdict(lambda: [0, 0])
    for pq in sorted(pdir.glob("shard_*.parquet")):
        df = pd.read_parquet(pq, columns=["event_id", "pdg_id"])
        g = df.groupby("event_id")["pdg_id"]
        for eid, series in g:
            eid_i = int(eid)
            s = series.astype(int)
            acc[eid_i][0] += int((s == 211).sum())
            acc[eid_i][1] += int((s == -211).sum())
    return {k: (v[0], v[1]) for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="π⁺/π⁻ counts per split channel from Parquet.")
    ap.add_argument("--altered-root", type=Path, required=True, help="altered_100k_parquet directory")
    ap.add_argument("--final-root", type=Path, required=True, help="final_state_v1 directory")
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write summary JSON.",
    )
    args = ap.parse_args()

    altered_root = args.altered_root.expanduser().resolve()
    final_root = args.final_root.expanduser().resolve()

    ch_map = _load_channel_map(altered_root)
    ok_map = _load_ok_events(final_root)
    pion_n = _pion_counts_per_event(final_root)

    by_ch: Dict[str, Dict[str, Any]] = {
        "A": {"n_events": 0, "n_pi_plus": 0, "n_pi_minus": 0, "n_pi_total": 0},
        "B": {"n_events": 0, "n_pi_plus": 0, "n_pi_minus": 0, "n_pi_total": 0},
    }
    skipped_not_ok = 0
    skipped_no_hadrons = 0

    for eid, ch in sorted(ch_map.items()):
        if ch not in ("A", "B"):
            continue
        if ok_map.get(eid, 0) != 1:
            skipped_not_ok += 1
            continue
        if eid not in pion_n:
            skipped_no_hadrons += 1
            continue
        np_p, np_m = pion_n[eid]
        by_ch[ch]["n_events"] += 1
        by_ch[ch]["n_pi_plus"] += np_p
        by_ch[ch]["n_pi_minus"] += np_m
        by_ch[ch]["n_pi_total"] += np_p + np_m

    summary: Dict[str, Any] = {
        "altered_root": str(altered_root),
        "final_root": str(final_root),
        "events_with_channel_metadata": len(ch_map),
        "skipped_reinject_not_ok": skipped_not_ok,
        "skipped_no_particle_rows": skipped_no_hadrons,
        "per_channel": {},
    }

    for ch in ("A", "B"):
        row = by_ch[ch]
        ne = max(1, int(row["n_events"]))
        summary["per_channel"][ch] = {
            **row,
            "mean_pi_plus_per_event": float(row["n_pi_plus"]) / ne,
            "mean_pi_minus_per_event": float(row["n_pi_minus"]) / ne,
            "mean_pi_charge_excess_per_event": float(row["n_pi_plus"] - row["n_pi_minus"]) / ne,
            "label": "(ud)->[d]+(uu)+ubar" if ch == "A" else "(ud)->[d]+(ud)+dbar",
        }

    text = json.dumps(summary, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
