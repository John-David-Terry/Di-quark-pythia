#!/usr/bin/env python3
"""
Reinject altered Parquet events into PYTHIA using the existing split reinjection path.

Input:
  <altered-parent>/altered_100k_parquet/
    particles/shard_*.parquet
    events/shard_*.parquet
    altered_metadata/shard_*.parquet

Output (same shard layout style as background final-state dataset):
  <out-parent>/final_state_v1/
    particles/shard_XXXXXX.parquet
    events/shard_XXXXXX.parquet
    manifest.parquet
    run_summary.json

Only events with alteration_succeeded == 1 are processed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tqdm is required; install from requirements.txt") from exc

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
_ANAL = Path(__file__).resolve().parent
for _p in (_SRC, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jet_hadron_observables_split_pi_pm import (  # noqa: E402
    build_pythia_reinjector,
    final_colored_partons,
    run_pythia_reinject_collect,
)

DATASET_ROOT = "final_state_v1"
ARM_NAME = "altered_reinject"


def default_input_parent() -> Path:
    return Path.home() / "Data" / "dis_isr_editable_altered_100k"


def default_output_parent() -> Path:
    return Path.home() / "Data" / "dis_isr_altered_reinject_100k"


def _load_success_event_ids(altered_root: Path) -> Set[int]:
    mdir = altered_root / "altered_metadata"
    out: Set[int] = set()
    for pq in sorted(mdir.glob("shard_*.parquet")):
        df = pd.read_parquet(pq, columns=["event_id", "alteration_succeeded"])
        sub = df[df["alteration_succeeded"] == 1]
        if sub.empty:
            continue
        out.update(int(x) for x in sub["event_id"].tolist())
    return out


def _cast_particles_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {
            "event_id": "int64",
            "particle_index": "int32",
            "pdg_id": "int32",
            "px": "float64",
            "py": "float64",
            "pz": "float64",
            "E": "float64",
        }
    )


def _cast_events_df(df: pd.DataFrame) -> pd.DataFrame:
    base: Dict[str, str] = {
        "event_id": "int64",
        "arm": "string",
        "ok": "int32",
        "failure_reason": "string",
        "n_final": "int32",
        "Q2": "float64",
        "xB": "float64",
        "x": "float64",
        "Q": "float64",
        "qT": "float64",
        "y": "float64",
        "S": "float64",
        "phiq": "float64",
        "weight": "float64",
    }
    return df.astype(base)


class _ShardWriter:
    def __init__(self, out_parent: Path, events_per_shard: int) -> None:
        self.root = out_parent / DATASET_ROOT
        (self.root / "particles").mkdir(parents=True, exist_ok=True)
        (self.root / "events").mkdir(parents=True, exist_ok=True)
        self.events_per_shard = max(1, int(events_per_shard))
        self._p_buf: List[Dict[str, Any]] = []
        self._e_buf: List[Dict[str, Any]] = []
        self._shard_idx = 0
        self.manifest_rows: List[Dict[str, Any]] = []

    def append(self, event_row: Dict[str, Any], particle_rows: List[Dict[str, Any]]) -> None:
        self._e_buf.append(event_row)
        self._p_buf.extend(particle_rows)
        if len(self._e_buf) >= self.events_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self._e_buf:
            return
        edf = _cast_events_df(pd.DataFrame(self._e_buf))
        pdf = _cast_particles_df(pd.DataFrame(self._p_buf))
        si = self._shard_idx
        ppath = self.root / "particles" / f"shard_{si:06d}.parquet"
        epath = self.root / "events" / f"shard_{si:06d}.parquet"
        pdf.to_parquet(ppath, index=False, compression="snappy", engine="pyarrow")
        edf.to_parquet(epath, index=False, compression="snappy", engine="pyarrow")

        fe = int(edf["event_id"].min())
        le = int(edf["event_id"].max())
        ne = int(len(edf))
        self.manifest_rows.append(
            {
                "shard_path": f"particles/shard_{si:06d}.parquet",
                "dataset": "particles",
                "first_event_id": fe,
                "last_event_id": le,
                "n_events": ne,
                "n_rows": int(len(pdf)),
            }
        )
        self.manifest_rows.append(
            {
                "shard_path": f"events/shard_{si:06d}.parquet",
                "dataset": "events",
                "first_event_id": fe,
                "last_event_id": le,
                "n_events": ne,
                "n_rows": ne,
            }
        )
        self._shard_idx += 1
        self._p_buf = []
        self._e_buf = []

    def finalize(self) -> None:
        self.flush()
        man = pd.DataFrame(self.manifest_rows)
        man.to_parquet(self.root / "manifest.parquet", index=False, compression="snappy", engine="pyarrow")


def _event_row_base(md_row: Optional[pd.Series], event_id: int) -> Dict[str, Any]:
    def _f(col: str) -> float:
        if md_row is None or col not in md_row.index:
            return float("nan")
        return float(md_row[col])

    row: Dict[str, Any] = {
        "event_id": int(event_id),
        "arm": ARM_NAME,
        "ok": 0,
        "failure_reason": "",
        "n_final": 0,
        "Q2": _f("Q2"),
        "xB": _f("xB"),
        "x": _f("x"),
        "Q": _f("Q"),
        "qT": _f("qT"),
        "y": _f("y"),
        "S": _f("S"),
        "phiq": _f("phiq"),
        "weight": _f("weight"),
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Reinject altered_100k_parquet events into PYTHIA.")
    ap.add_argument("--input-parent", type=Path, default=None, help="Parent containing altered_100k_parquet/")
    ap.add_argument("--out-parent", type=Path, default=None, help="Parent for final_state_v1 output")
    ap.add_argument("--events-per-shard", type=int, default=10_000)
    ap.add_argument("--max-events", type=int, default=0, help="If >0, cap number of successful altered events to process.")
    args = ap.parse_args()

    in_parent = (args.input_parent or default_input_parent()).resolve()
    altered_root = in_parent / "altered_100k_parquet"
    if not altered_root.is_dir():
        raise SystemExit(f"missing altered input root: {altered_root}")

    out_parent = (args.out_parent or default_output_parent()).resolve()
    writer = _ShardWriter(out_parent, int(args.events_per_shard))

    success_ids = _load_success_event_ids(altered_root)
    if int(args.max_events) > 0:
        keep = sorted(success_ids)[: int(args.max_events)]
        success_ids = set(keep)
    total = len(success_ids)
    if total == 0:
        raise SystemExit("no events with alteration_succeeded == 1")

    pythia = build_pythia_reinjector()
    lt_identity = np.eye(4, dtype=np.float64)

    stats: Dict[str, Any] = {
        "total_attempted": 0,
        "total_succeeded": 0,
        "total_failed": 0,
        "failure_reasons": {},
        "input_root": str(altered_root),
        "output_root": str(out_parent / DATASET_ROOT),
        "arm": ARM_NAME,
        "events_per_shard": int(args.events_per_shard),
    }
    failure_counts: Counter[str] = Counter()

    particles_dir = altered_root / "particles"
    t0 = time.perf_counter()
    with tqdm(total=total, desc="Reinjecting altered events", unit="ev", mininterval=0.2) as pbar:
        for pshard in sorted(particles_dir.glob("shard_*.parquet")):
            eshard = altered_root / "events" / pshard.name
            if not eshard.is_file():
                raise FileNotFoundError(f"missing matching events shard for {pshard}")
            pdf = pd.read_parquet(pshard)
            edf = pd.read_parquet(eshard).set_index("event_id")
            for event_id, g in pdf.groupby("event_id", sort=True):
                eid = int(event_id)
                if eid not in success_ids:
                    continue
                stats["total_attempted"] += 1
                row = _event_row_base(edf.loc[eid] if eid in edf.index else None, eid)

                partons = final_colored_partons(g.sort_values("particle_index").reset_index(drop=True))
                if partons.empty:
                    row["ok"] = 0
                    row["failure_reason"] = "no_final_colored_partons"
                    failure_counts[row["failure_reason"]] += 1
                    writer.append(row, [])
                    stats["total_failed"] += 1
                    pbar.update(1)
                    continue

                ok, err, hadrons = run_pythia_reinject_collect(
                    pythia,
                    partons,
                    lt_identity,
                    True,  # source parquet is already Breit frame
                )
                if not ok:
                    row["ok"] = 0
                    row["failure_reason"] = str(err) if str(err) else "pythia_next_failed"
                    failure_counts[row["failure_reason"]] += 1
                    writer.append(row, [])
                    stats["total_failed"] += 1
                    pbar.update(1)
                    continue

                out_parts: List[Dict[str, Any]] = []
                for i, (pid, p4) in enumerate(hadrons):
                    out_parts.append(
                        {
                            "event_id": eid,
                            "particle_index": int(i),
                            "pdg_id": int(pid),
                            "px": float(p4[1]),
                            "py": float(p4[2]),
                            "pz": float(p4[3]),
                            "E": float(p4[0]),
                        }
                    )
                row["ok"] = 1
                row["failure_reason"] = ""
                row["n_final"] = int(len(out_parts))
                writer.append(row, out_parts)
                stats["total_succeeded"] += 1
                pbar.update(1)

    writer.finalize()
    elapsed = time.perf_counter() - t0
    stats["failure_reasons"] = dict(sorted(failure_counts.items(), key=lambda kv: kv[0]))
    stats["elapsed_s"] = float(elapsed)
    stats["events_per_s"] = float(stats["total_attempted"] / elapsed) if elapsed > 0 else 0.0
    stats["n_output_shards"] = int(len(writer.manifest_rows) // 2)
    run_summary = out_parent / DATASET_ROOT / "run_summary.json"
    run_summary.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

