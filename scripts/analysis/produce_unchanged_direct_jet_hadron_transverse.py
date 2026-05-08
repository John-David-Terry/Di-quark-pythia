#!/usr/bin/env python3
"""
Build unchanged_direct Parquet for jet–hadron transverse observables from
``dis_isr_full_event_record.csv`` (Breit-frame, post-hadronization).

No PYTHIA calls — reads final hadrons from the saved record.

Example:
  python scripts/analysis/produce_unchanged_direct_jet_hadron_transverse.py \\
    --full-event-csv /path/to/dis_isr_full_event_record.csv \\
    --metadata-csv /path/to/dis_isr_event_metadata.csv \\
    --event-ids-file /path/to/ids.txt \\
    --out /path/to/unchanged_direct_jet_hadron.parquet
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
_ANAL = Path(__file__).resolve().parent
for _p in (_SRC, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from unchanged_direct_jet_hadron_core import row_from_full_event_breit  # noqa: E402
from unchanged_direct_schema import UNCHANGED_DIRECT_JET_HADRON_V1_COLUMNS  # noqa: E402

_FULL_USECOLS = [
    "event_id",
    "particle_index",
    "pdg_id",
    "status",
    "isFinal",
    "mother1",
    "mother2",
    "daughter1",
    "daughter2",
    "col",
    "acol",
    "px",
    "py",
    "pz",
    "E",
]


def _load_event_ids_set(path: Path) -> Set[int]:
    out: Set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(int(line))
    return out


def _gather_full_events(
    full_csv: Path, want: Set[int], chunksize: int
) -> Dict[int, List[pd.DataFrame]]:
    buckets: Dict[int, List[pd.DataFrame]] = defaultdict(list)
    for chunk in pd.read_csv(
        full_csv,
        usecols=_FULL_USECOLS,
        dtype={"event_id": "int64"},
        chunksize=chunksize,
    ):
        sub = chunk[chunk["event_id"].isin(want)]
        if sub.empty:
            continue
        for eid, g in sub.groupby("event_id", sort=False):
            buckets[int(eid)].append(g)
    return {eid: parts for eid, parts in buckets.items() if eid in want}


def main() -> None:
    ap = argparse.ArgumentParser(description="unchanged_direct jet–hadron artifact from full event CSV.")
    ap.add_argument("--full-event-csv", type=Path, required=True)
    ap.add_argument("--metadata-csv", type=Path, required=True)
    ap.add_argument("--event-ids-file", type=Path, required=True, help="One event_id per line.")
    ap.add_argument("--out", type=Path, required=True, help="Output .parquet path.")
    ap.add_argument("--chunksize", type=int, default=500_000)
    args = ap.parse_args()

    want = _load_event_ids_set(args.event_ids_file)
    if not want:
        raise SystemExit("No event IDs in --event-ids-file.")

    md = pd.read_csv(
        args.metadata_csv,
        usecols=["event_id", "struck_outgoing_index", "struck_incoming_index"],
        dtype={
            "event_id": "int64",
            "struck_outgoing_index": "int64",
            "struck_incoming_index": "int64",
        },
    ).set_index("event_id")

    parts = _gather_full_events(args.full_event_csv.resolve(), want, args.chunksize)
    rows: List[dict] = []
    for eid in sorted(want):
        if eid not in parts:
            rows.append(
                {
                    "event_id": eid,
                    "arm": "unchanged",
                    "source_lineage": "full_event_record_breit",
                    "ok": False,
                    "failure_reason": "missing_from_full_event_csv",
                    "xB": float("nan"),
                    "Q2": float("nan"),
                    "Q": float("nan"),
                    "k_out_breit_E": float("nan"),
                    "k_out_breit_px": float("nan"),
                    "k_out_breit_py": float("nan"),
                    "k_out_breit_pz": float("nan"),
                    "pion_pdg": 0,
                    "pion_breit_E": float("nan"),
                    "pion_breit_px": float("nan"),
                    "pion_breit_py": float("nan"),
                    "pion_breit_pz": float("nan"),
                    "obs_angle_rad": float("nan"),
                    "obs_sum_mag_GeV": float("nan"),
                    "obs_diff_mag_GeV": float("nan"),
                    "n_final_hadrons_used": 0,
                }
            )
            continue
        df_ev = pd.concat(parts[eid], ignore_index=True)
        rows.append(
            row_from_full_event_breit(
                df_ev,
                eid,
                md,
                arm="unchanged",
                source_lineage="full_event_record_breit",
            )
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df[[c for c in UNCHANGED_DIRECT_JET_HADRON_V1_COLUMNS if c in out_df.columns]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"Wrote {len(out_df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
