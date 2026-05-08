#!/usr/bin/env python3
"""
Parquet-native 90/10 diquark split: read ``editable_source_v1`` **shard-by-shard**, write
``split_90_10_parquet`` with low file count (no giant CSV, no per-event CSV output).

Physics matches ``split_dis_sample_diquark_kick.process_one_event_split`` (same RNG / alter
pool / surgery / kicks as the legacy CSV driver).

Loaders: ``split_output_parquet.load_split_particles_dataframe``, etc.

For the **900k background + 100k editable** campaign, the 100k pool should **not** be
re-split 90/10 here; use ``transform_editable_all_altered_parquet.py`` to emit
``altered_100k_parquet`` (every event in the alter pool) instead.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
_ANAL = Path(__file__).resolve().parent
for _p in (_SRC, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from editable_source_parquet import (  # noqa: E402
    EDITABLE_SOURCE_V1_DIRNAME,
    EVENT_TABLE_COLUMNS,
    PARTICLE_CSV_COLUMN_ORDER,
)

from split_dis_sample_diquark_kick import process_one_event_split  # noqa: E402

from split_output_parquet import SPLIT_SUBDIR  # noqa: E402


def default_editable_parent() -> Path:
    return Path.home() / "Data" / "dis_isr_editable_source_100k"


def _cast_particles_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {
            "event_id": "int64",
            "particle_index": "int64",
            "pdg_id": "int32",
            "status": "int32",
            "mother1": "int32",
            "mother2": "int32",
            "daughter1": "int32",
            "daughter2": "int32",
            "col": "int32",
            "acol": "int32",
            "px": "float64",
            "py": "float64",
            "pz": "float64",
            "E": "float64",
            "m": "float64",
            "pT": "float64",
            "eta": "float64",
            "phi": "float64",
            "isFinal": "int32",
        }
    )


def _cast_events_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {
            "event_id": "int64",
            "Q2": "float64",
            "xB": "float64",
            "x": "float64",
            "Q": "float64",
            "qT": "float64",
            "y": "float64",
            "S": "float64",
            "phiq": "float64",
            "struck_incoming_index": "int32",
            "struck_outgoing_index": "int32",
            "transverse_kick_applied": "int32",
            "kick_kx_gev": "float64",
            "kick_ky_gev": "float64",
            "n_particles": "int32",
            "accepted": "int32",
            "weight": "float64",
        }
    )


def _flatten_altered_meta(event_id: int, meta: Dict[str, Any]) -> Dict[str, Any]:
    npd = meta.get("new_particle_pdgs") or {}
    mode = str(meta.get("mode", "diquark_kick"))
    row: Dict[str, Any] = {
        "event_id": int(event_id),
        "meta_json": json.dumps(meta, sort_keys=True),
        "split_mode": mode,
    }
    if mode == "breit_px_kick_only":
        row["split_channel"] = ""
        row["channel_description"] = ""
        row["struck_quark_index"] = int(meta.get("struck_quark_index", -1))
        row["kick_partner_particle_index"] = int(meta.get("kick_partner_particle_index", -1))
        row["kick_delta_px_gev"] = float(meta.get("kick_delta_px_gev", float("nan")))
        row["original_diquark_index"] = -1
        row["stripped_quark_index"] = -1
        row["created_antiquark_index"] = -1
        row["daughter_diquark_index"] = -1
        row["color_tag_A"] = -1
        row["color_tag_B"] = -1
        row["pdg_stripped"] = 0
        row["pdg_antiquark"] = 0
        row["pdg_daughter_diquark"] = 0
    else:
        row["split_channel"] = str(meta.get("split_channel", ""))
        row["channel_description"] = str(meta.get("channel_description", ""))
        row["struck_quark_index"] = int(meta.get("struck_quark_index", -1))
        row["kick_partner_particle_index"] = -1
        row["kick_delta_px_gev"] = float(meta.get("kick_delta_px_gev", float("nan")))
        row["original_diquark_index"] = int(meta.get("original_diquark_index", -1))
        row["stripped_quark_index"] = int(meta.get("stripped_quark_index", -1))
        row["created_antiquark_index"] = int(meta.get("created_antiquark_index", -1))
        row["daughter_diquark_index"] = int(meta.get("daughter_diquark_index", -1))
        row["color_tag_A"] = int(meta.get("color_tag_A", -1))
        row["color_tag_B"] = int(meta.get("color_tag_B", -1))
        row["pdg_stripped"] = int(npd.get("stripped", 0))
        row["pdg_antiquark"] = int(npd.get("antiquark", 0))
        row["pdg_daughter_diquark"] = int(npd.get("daughter_diquark", 0))
    return row


class _BranchWriter:
    def __init__(
        self,
        out_parent: Path,
        branch: str,
        events_per_shard: int,
        write_altered_meta: bool,
    ) -> None:
        self.out_parent = out_parent
        self.branch = branch
        self.events_per_shard = max(1, int(events_per_shard))
        self.write_altered_meta = write_altered_meta
        self.root = out_parent / SPLIT_SUBDIR / branch
        (self.root / "particles").mkdir(parents=True, exist_ok=True)
        (self.root / "events").mkdir(parents=True, exist_ok=True)
        if write_altered_meta:
            (self.root / "altered_metadata").mkdir(parents=True, exist_ok=True)
        self.manifest_rows: List[Dict[str, Any]] = []
        self._shard_idx = 0
        self._p_buf: List[pd.DataFrame] = []
        self._e_buf: List[Dict[str, Any]] = []
        self._m_buf: List[Dict[str, Any]] = []
        self._ev_count = 0

    def _flush(self) -> None:
        if not self._e_buf:
            return
        pdf = pd.concat(self._p_buf, ignore_index=True)
        edf = pd.DataFrame(self._e_buf)
        pdf = _cast_particles_df(pdf[PARTICLE_CSV_COLUMN_ORDER])
        edf = _cast_events_df(edf[EVENT_TABLE_COLUMNS])
        si = self._shard_idx
        ppath = self.root / "particles" / f"shard_{si:06d}.parquet"
        epub = self.root / "events" / f"shard_{si:06d}.parquet"
        pdf.to_parquet(ppath, index=False, compression="snappy", engine="pyarrow")
        edf.to_parquet(epub, index=False, compression="snappy", engine="pyarrow")
        fe = int(edf["event_id"].min())
        le = int(edf["event_id"].max())
        ne = int(len(edf))
        self.manifest_rows.append(
            {
                "dataset": "particles",
                "shard_path": f"particles/shard_{si:06d}.parquet",
                "first_event_id": fe,
                "last_event_id": le,
                "n_events": ne,
                "n_rows": int(len(pdf)),
            }
        )
        self.manifest_rows.append(
            {
                "dataset": "events",
                "shard_path": f"events/shard_{si:06d}.parquet",
                "first_event_id": fe,
                "last_event_id": le,
                "n_events": ne,
                "n_rows": ne,
            }
        )
        if self.write_altered_meta and self._m_buf:
            mdf = pd.DataFrame(self._m_buf)
            mpath = self.root / "altered_metadata" / f"shard_{si:06d}.parquet"
            mdf.to_parquet(mpath, index=False, compression="snappy", engine="pyarrow")
            self.manifest_rows.append(
                {
                    "dataset": "altered_metadata",
                    "shard_path": f"altered_metadata/shard_{si:06d}.parquet",
                    "first_event_id": fe,
                    "last_event_id": le,
                    "n_events": ne,
                    "n_rows": int(len(mdf)),
                }
            )
        self._shard_idx += 1
        self._p_buf = []
        self._e_buf = []
        self._m_buf = []
        self._ev_count = 0

    def append(
        self,
        particles: pd.DataFrame,
        event_row: Dict[str, Any],
        altered_meta: Optional[Dict[str, Any]],
    ) -> None:
        self._p_buf.append(particles[PARTICLE_CSV_COLUMN_ORDER].copy())
        self._e_buf.append(event_row)
        if self.write_altered_meta and altered_meta is not None:
            eid = int(event_row["event_id"])
            self._m_buf.append(_flatten_altered_meta(eid, altered_meta))
        self._ev_count += 1
        if self._ev_count >= self.events_per_shard:
            self._flush()

    def finalize(self) -> None:
        self._flush()


def _read_editable_run_summary(editable_v1: Path) -> Tuple[int, int]:
    p = editable_v1 / "run_summary.json"
    if not p.is_file():
        raise FileNotFoundError(f"missing {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    n = int(d["n_accepted"])
    eps = int(d.get("events_per_shard", 10_000))
    return n, eps


def _iter_input_shard_pairs(editable_v1: Path) -> List[Tuple[Path, Path]]:
    pdir = editable_v1 / "particles"
    pairs: List[Tuple[Path, Path]] = []
    for pp in sorted(pdir.glob("shard_*.parquet")):
        stem = pp.stem
        ep = editable_v1 / "events" / f"{stem}.parquet"
        if not ep.is_file():
            raise FileNotFoundError(f"missing matching events shard for {pp.name}")
        pairs.append((pp, ep))
    if not pairs:
        raise FileNotFoundError(f"no particle shards under {pdir}")
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="90/10 diquark split: editable_source_v1 Parquet → split_90_10_parquet."
    )
    ap.add_argument(
        "--editable-parent",
        type=Path,
        default=None,
        help=f"Directory containing {EDITABLE_SOURCE_V1_DIRNAME}/ (default: {default_editable_parent()})",
    )
    ap.add_argument(
        "--out-parent",
        type=Path,
        default=None,
        help="Parent directory for output split_90_10_parquet/ (default: same as --editable-parent).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delta-px", type=float, default=0.4)
    ap.add_argument(
        "--events-per-output-shard",
        type=int,
        default=10_000,
        help="Flush each output branch after this many stored events (default 10000).",
    )
    ap.add_argument("--max-events", type=int, default=0, help="If >0, cap input event_ids processed.")
    ap.add_argument(
        "--mode",
        choices=("diquark_kick", "breit_px_kick_only"),
        default="diquark_kick",
    )
    args = ap.parse_args()

    ep = (args.editable_parent or default_editable_parent()).resolve()
    editable_v1 = ep / EDITABLE_SOURCE_V1_DIRNAME
    if not editable_v1.is_dir():
        raise SystemExit(f"editable dataset not found: {editable_v1}")

    out_parent = (args.out_parent or ep).resolve()
    split_base = out_parent / SPLIT_SUBDIR
    split_base.mkdir(parents=True, exist_ok=True)

    n_accepted, _src_eps = _read_editable_run_summary(editable_v1)
    all_ids = list(range(n_accepted))
    if int(args.max_events) > 0:
        all_ids = all_ids[: int(args.max_events)]

    import random

    rng = random.Random(int(args.seed))
    ids_shuffled = list(all_ids)
    rng.shuffle(ids_shuffled)
    n_alter = int(round(0.1 * len(ids_shuffled)))
    alter_set = set(ids_shuffled[:n_alter])

    stats: Dict[str, Any] = {
        "mode": args.mode,
        "seed": args.seed,
        "delta_px": args.delta_px,
        "total_ids": len(all_ids),
        "planned_altered": n_alter,
        "planned_unchanged": len(all_ids) - n_alter,
        "written_unchanged": 0,
        "written_altered": 0,
        "altered_channel_C": 0,
        "fallback_unchanged_no_ud_diquark": 0,
        "fallback_unchanged_struck_fail": 0,
        "fallback_unchanged_validation_fail": 0,
    }

    w_un = _BranchWriter(
        out_parent, "unchanged", int(args.events_per_output_shard), write_altered_meta=False
    )
    w_alt = _BranchWriter(
        out_parent, "altered", int(args.events_per_output_shard), write_altered_meta=True
    )

    t0 = time.perf_counter()
    pairs = _iter_input_shard_pairs(editable_v1)
    id_allowed = set(all_ids)

    def _event_row_dict(md_row: Optional[pd.Series], eid: int, n_part: int) -> Dict[str, Any]:
        if md_row is None:
            raise ValueError(f"missing events-table row for event_id={eid}")
        d: Dict[str, Any] = {"event_id": int(eid), "n_particles": int(n_part)}
        for c in EVENT_TABLE_COLUMNS:
            if c in ("event_id", "n_particles"):
                continue
            if c not in md_row.index:
                raise ValueError(f"events shard missing column {c!r} for event_id={eid}")
            d[c] = md_row[c]
        return d

    for pp, epth in pairs:
        pdf = pd.read_parquet(pp)
        edf = pd.read_parquet(epth)
        edf_map = edf.set_index("event_id")
        for event_id, g in pdf.groupby("event_id", sort=True):
            eid = int(event_id)
            if eid not in id_allowed:
                continue
            g = g.sort_values("particle_index")
            md_row = edf_map.loc[eid] if eid in edf_map.index else None
            particles_out, branch, altered_meta, delta = process_one_event_split(
                g,
                eid,
                event_in_alter_pool=(eid in alter_set),
                md_row=md_row,
                mode=str(args.mode),
                delta_px=float(args.delta_px),
                rng=rng,
            )
            for k, v in delta.items():
                stats[k] = stats.get(k, 0) + v
            event_row = _event_row_dict(md_row, eid, len(particles_out))
            if branch == "unchanged":
                w_un.append(particles_out, event_row, None)
            else:
                w_alt.append(particles_out, event_row, altered_meta)

    w_un.finalize()
    w_alt.finalize()

    for w in (w_un, w_alt):
        man_df = pd.DataFrame(w.manifest_rows)
        man_path = w.root / "manifest.parquet"
        man_df.to_parquet(man_path, index=False, compression="snappy", engine="pyarrow")

    stats["total_input_events"] = int(stats["total_ids"])
    stats["attempted_altered"] = int(stats["planned_altered"])
    stats["failed_altered"] = int(stats["attempted_altered"] - stats["written_altered"])
    stats["elapsed_s"] = float(time.perf_counter() - t0)
    stats["editable_source"] = str(editable_v1)
    stats["output_parent"] = str(out_parent)
    stats["events_per_output_shard"] = int(args.events_per_output_shard)
    stats["branch_shard_settings"] = {
        "unchanged": {"events_per_shard": int(args.events_per_output_shard)},
        "altered": {"events_per_shard": int(args.events_per_output_shard)},
    }

    summary_path = split_base / "split_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
