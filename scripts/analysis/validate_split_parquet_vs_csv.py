#!/usr/bin/env python3
"""
Compare ``process_one_event_split`` outcomes for the same events loaded from:
  (A) legacy ``dis_isr_full_event_record.csv`` + metadata CSV
  (B) ``editable_source_v1`` Parquet (via ``load_editable_event_dataframe``)

Same ``--seed`` / ``--mode`` / ``--delta-px`` / ``--max-events`` as a split run.
Exits 0 only if branches and particle tables match for every checked event.
"""

from __future__ import annotations

import argparse
import math
import numbers
import random
import sys
from pathlib import Path
from typing import Any, Set

import pandas as pd

_ANAL = Path(__file__).resolve().parent
if str(_ANAL) not in sys.path:
    sys.path.insert(0, str(_ANAL))

from editable_source_parquet import load_editable_event_dataframe  # noqa: E402
from split_dis_sample_diquark_kick import process_one_event_split  # noqa: E402


def _fork_rng(rng: random.Random) -> random.Random:
    r = random.Random()
    r.setstate(rng.getstate())
    return r


def _meta_equivalent(a: Any, b: Any, *, rtol: float = 1e-14, atol: float = 1e-14) -> bool:
    if (
        isinstance(a, numbers.Real)
        and isinstance(b, numbers.Real)
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    ):
        fa, fb = float(a), float(b)
        return fa == fb or math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(_meta_equivalent(a[k], b[k], rtol=rtol, atol=atol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_meta_equivalent(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b))
    return a == b


def main() -> None:
    ap = argparse.ArgumentParser(description="Parquet vs CSV parity for diquark split core.")
    ap.add_argument("--full-event-csv", type=Path, required=True)
    ap.add_argument("--metadata-csv", type=Path, required=True)
    ap.add_argument(
        "--editable-parent",
        type=Path,
        required=True,
        help="Parent of editable_source_v1/ (same events as CSV).",
    )
    ap.add_argument("--max-events", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delta-px", type=float, default=0.4)
    ap.add_argument(
        "--mode",
        choices=("diquark_kick", "breit_px_kick_only"),
        default="diquark_kick",
    )
    ap.add_argument(
        "--events-per-shard",
        type=int,
        default=10_000,
        help="Must match editable source run_summary if loading by shard.",
    )
    args = ap.parse_args()

    ev = pd.read_csv(args.full_event_csv).sort_values(["event_id", "particle_index"])
    md = pd.read_csv(args.metadata_csv).sort_values("event_id")
    md_map = md.set_index("event_id")

    all_ids = sorted(ev["event_id"].unique().tolist())[: max(0, int(args.max_events))]
    rng = random.Random(int(args.seed))
    ids_shuffled = list(all_ids)
    rng.shuffle(ids_shuffled)
    n_alter = int(round(0.1 * len(ids_shuffled)))
    alter_set: Set[int] = set(ids_shuffled[:n_alter])

    failures: list[str] = []
    rng = random.Random(int(args.seed))
    for eid in all_ids:
        g_csv = ev[ev["event_id"] == eid].copy()
        try:
            g_pq = load_editable_event_dataframe(
                args.editable_parent.resolve(),
                int(eid),
                events_per_shard=int(args.events_per_shard),
            )
        except Exception as exc:
            failures.append(f"event {eid}: parquet load failed: {exc}")
            continue
        g_csv = g_csv.sort_values("particle_index").reset_index(drop=True)
        g_pq = g_pq.sort_values("particle_index").reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(g_csv, g_pq, check_dtype=False, check_like=True)
        except AssertionError:
            failures.append(f"event {eid}: input DataFrame mismatch CSV vs Parquet")

        md_row = md_map.loc[eid] if eid in md_map.index else None
        in_alter = eid in alter_set

        r_csv = _fork_rng(rng)
        r1 = process_one_event_split(
            g_csv,
            int(eid),
            event_in_alter_pool=in_alter,
            md_row=md_row,
            mode=str(args.mode),
            delta_px=float(args.delta_px),
            rng=r_csv,
        )
        r_pq = _fork_rng(rng)
        r2 = process_one_event_split(
            g_pq,
            int(eid),
            event_in_alter_pool=in_alter,
            md_row=md_row,
            mode=str(args.mode),
            delta_px=float(args.delta_px),
            rng=r_pq,
        )
        rng.setstate(r_csv.getstate())
        p1, b1, m1, _ = r1
        p2, b2, m2, _ = r2
        if b1 != b2:
            failures.append(f"event {eid}: branch {b1} vs {b2}")
            continue
        try:
            pd.testing.assert_frame_equal(
                p1.reset_index(drop=True),
                p2.reset_index(drop=True),
                check_dtype=False,
                check_like=True,
            )
        except AssertionError:
            failures.append(f"event {eid}: output particles differ")
        if (m1 is None) != (m2 is None):
            failures.append(f"event {eid}: altered_meta presence mismatch")
        elif m1 is not None and m2 is not None:
            if not _meta_equivalent(m1, m2):
                failures.append(f"event {eid}: altered_meta mismatch")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(" ", f)
        raise SystemExit(1)
    print(f"ok: compared {len(all_ids)} events (CSV vs Parquet input + process_one_event_split).")


if __name__ == "__main__":
    main()
