#!/usr/bin/env python3
"""
validation_reinject: same reinject/hadronization path as split_kinematics_extract for
unchanged parton CSVs, then emit the unchanged_direct v1 schema (+ observables).

Use the same event_id list as unchanged_direct for apples-to-apples comparison.

Example:
  python scripts/analysis/produce_validation_reinject_jet_hadron_transverse.py \\
    --split-root /path/to/split_90_10 \\
    --metadata-csv /path/to/dis_isr_event_metadata.csv \\
    --event-ids-file /path/to/ids.txt \\
    --full-event-csv /path/to/dis_isr_full_event_record.csv \\
    --out /path/to/validation_reinject_jet_hadron.parquet

If ``unchanged/event_XXXXXX.csv`` is missing (common for 90/10 splits), pass the same
``--full-event-csv`` used by ``jet_hadron_observables_split_pi_pm.py`` so the unchanged
parton table is sliced from the full record (same ``event_id``).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
_ANAL = Path(__file__).resolve().parent
for _p in (_SRC, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load_ske():
    path = _ANAL / "split_kinematics_extract.py"
    spec = importlib.util.spec_from_file_location("_ske_v", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_ske = _load_ske()

from jet_hadron_observables_split_pi_pm import build_pythia_reinjector  # noqa: E402

from unchanged_direct_jet_hadron_core import row_from_parton_csv_reinject_fixed  # noqa: E402
from unchanged_direct_schema import UNCHANGED_DIRECT_JET_HADRON_V1_COLUMNS  # noqa: E402


def _load_event_ids_set(path: Path) -> Set[int]:
    out: Set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(int(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="validation_reinject jet–hadron artifact (parton CSV + PYTHIA).")
    ap.add_argument("--split-root", type=Path, required=True, help="Directory with altered/ and unchanged/.")
    ap.add_argument("--metadata-csv", type=Path, required=True)
    ap.add_argument("--event-ids-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--full-event-csv",
        type=Path,
        default=None,
        help="Optional: slice unchanged parton rows when unchanged/event_*.csv is absent.",
    )
    args = ap.parse_args()

    want = sorted(_load_event_ids_set(args.event_ids_file))
    if not want:
        raise SystemExit("No event IDs in --event-ids-file.")

    usecols = list(_ske._EVENT_CSV_USECOLS)
    fe_sub: Optional[pd.DataFrame] = None
    if args.full_event_csv is not None:
        fe_sub = pd.read_csv(
            args.full_event_csv.resolve(),
            usecols=usecols,
            dtype={"event_id": "int64"},
        )
        fe_sub = fe_sub[fe_sub["event_id"].isin(want)].sort_values(
            ["event_id", "particle_index"]
        )

    md = pd.read_csv(
        args.metadata_csv,
        usecols=["event_id", "struck_outgoing_index", "struck_incoming_index"],
        dtype={
            "event_id": "int64",
            "struck_outgoing_index": "int64",
            "struck_incoming_index": "int64",
        },
    ).set_index("event_id")

    unchanged_dir = args.split_root.resolve() / "unchanged"
    p = build_pythia_reinjector()
    rows: List[dict] = []
    for eid in want:
        csv_path = unchanged_dir / f"event_{eid:06d}.csv"
        if csv_path.is_file():
            df = _ske._read_event_csv(csv_path)
        elif fe_sub is not None:
            g = fe_sub[fe_sub["event_id"] == eid]
            if g.empty:
                rows.append(
                    {
                        "event_id": eid,
                        "arm": "validation_reinject",
                        "source_lineage": "reinject_parton_csv",
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
            df = (
                g[list(_ske._EVENT_CSV_USECOLS)]
                .copy()
                .sort_values("particle_index")
                .reset_index(drop=True)
            )
        else:
            rows.append(
                {
                    "event_id": eid,
                    "arm": "validation_reinject",
                    "source_lineage": "reinject_parton_csv",
                    "ok": False,
                    "failure_reason": "missing_parton_csv",
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
        rows.append(
            row_from_parton_csv_reinject_fixed(
                df,
                eid,
                md,
                None,
                "unchanged",
                p,
                arm="validation_reinject",
                source_lineage="reinject_parton_csv",
            )
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df[[c for c in UNCHANGED_DIRECT_JET_HADRON_V1_COLUMNS if c in out_df.columns]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"Wrote {len(out_df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
