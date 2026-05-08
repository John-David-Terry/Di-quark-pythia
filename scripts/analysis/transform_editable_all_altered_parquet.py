#!/usr/bin/env python3
"""
Apply diquark split + px kick to **every** event in ``editable_source_v1`` (no 90/10 inside
the 100k pool). Intended workflow:

- **900k** background = unchanged control (separate dataset).
- **100k** editable source → **this script** → **100k** rows in ``altered_100k_parquet/``,
  each processed with ``event_in_alter_pool=True`` (physics fallbacks still possible).

Output: ``<out-parent>/altered_100k_parquet/`` with sharded Parquet + ``tqdm`` progress.

**Hard topology:** every altered event uses channel **C**,
``(ud) -> [d] + (us) + sbar`` (PYTHIA ``su_0`` = 3201 + ``sbar``).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "tqdm is required: pip install tqdm (see requirements.txt)"
    ) from exc

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

OUTPUT_SUBDIR = "altered_100k_parquet"


def default_editable_parent() -> Path:
    return Path.home() / "Data" / "dis_isr_editable_source_100k"


def default_out_parent() -> Path:
    return Path.home() / "Data" / "dis_isr_editable_altered_100k"


def _read_editable_run_summary(editable_v1: Path) -> int:
    p = editable_v1 / "run_summary.json"
    if not p.is_file():
        raise FileNotFoundError(f"missing {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    return int(d["n_accepted"])


def _iter_input_shard_pairs(editable_v1: Path) -> List[Tuple[Path, Path]]:
    pdir = editable_v1 / "particles"
    pairs: List[Tuple[Path, Path]] = []
    for pp in sorted(pdir.glob("shard_*.parquet")):
        ep = editable_v1 / "events" / f"{pp.stem}.parquet"
        if not ep.is_file():
            raise FileNotFoundError(f"missing matching events shard for {pp.name}")
        pairs.append((pp, ep))
    if not pairs:
        raise FileNotFoundError(f"no particle shards under {pdir}")
    return pairs


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


def _flatten_altered_meta_success(event_id: int, meta: Dict[str, Any]) -> Dict[str, Any]:
    npd = meta.get("new_particle_pdgs") or {}
    mode = str(meta.get("mode", "diquark_kick"))
    row: Dict[str, Any] = {
        "event_id": int(event_id),
        "meta_json": json.dumps(meta, sort_keys=True),
        "split_mode": mode,
        "alteration_succeeded": 1,
        "failure_reason": "",
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


def _failure_reason_from_delta(delta: Dict[str, int]) -> str:
    if delta.get("written_altered"):
        return ""
    if delta.get("fallback_unchanged_struck_fail"):
        return "struck_unresolved"
    if delta.get("fallback_unchanged_no_ud_diquark"):
        return "no_ud_diquark"
    if delta.get("fallback_unchanged_validation_fail"):
        return "validation_fail"
    return "unknown"


def _flatten_altered_meta_failure(event_id: int, reason: str) -> Dict[str, Any]:
    return {
        "event_id": int(event_id),
        "meta_json": "",
        "split_mode": "",
        "split_channel": "",
        "channel_description": "",
        "struck_quark_index": -1,
        "kick_partner_particle_index": -1,
        "kick_delta_px_gev": float("nan"),
        "original_diquark_index": -1,
        "stripped_quark_index": -1,
        "created_antiquark_index": -1,
        "daughter_diquark_index": -1,
        "color_tag_A": -1,
        "color_tag_B": -1,
        "pdg_stripped": 0,
        "pdg_antiquark": 0,
        "pdg_daughter_diquark": 0,
        "alteration_succeeded": 0,
        "failure_reason": reason,
    }


ALTERED_META_COLUMNS = list(_flatten_altered_meta_failure(0, "").keys())


class _AlteredAllWriter:
    def __init__(self, out_parent: Path, events_per_shard: int) -> None:
        self.root = out_parent / OUTPUT_SUBDIR
        (self.root / "particles").mkdir(parents=True, exist_ok=True)
        (self.root / "events").mkdir(parents=True, exist_ok=True)
        (self.root / "altered_metadata").mkdir(parents=True, exist_ok=True)
        self.events_per_shard = max(1, int(events_per_shard))
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
        mdf = pd.DataFrame(self._m_buf)
        for col in ALTERED_META_COLUMNS:
            if col not in mdf.columns:
                mdf[col] = pd.NA
        mdf = mdf[ALTERED_META_COLUMNS]
        pdf = _cast_particles_df(pdf[PARTICLE_CSV_COLUMN_ORDER])
        edf = _cast_events_df(edf[EVENT_TABLE_COLUMNS])
        si = self._shard_idx
        pdf.to_parquet(
            self.root / "particles" / f"shard_{si:06d}.parquet",
            index=False,
            compression="snappy",
            engine="pyarrow",
        )
        edf.to_parquet(
            self.root / "events" / f"shard_{si:06d}.parquet",
            index=False,
            compression="snappy",
            engine="pyarrow",
        )
        mdf.to_parquet(
            self.root / "altered_metadata" / f"shard_{si:06d}.parquet",
            index=False,
            compression="snappy",
            engine="pyarrow",
        )
        fe = int(edf["event_id"].min())
        le = int(edf["event_id"].max())
        ne = int(len(edf))
        for dataset, nrows in (
            ("particles", int(len(pdf))),
            ("events", ne),
            ("altered_metadata", int(len(mdf))),
        ):
            subdir = "altered_metadata" if dataset == "altered_metadata" else dataset
            self.manifest_rows.append(
                {
                    "dataset": dataset,
                    "shard_path": f"{subdir}/shard_{si:06d}.parquet",
                    "first_event_id": fe,
                    "last_event_id": le,
                    "n_events": ne,
                    "n_rows": nrows,
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
        meta_row: Dict[str, Any],
    ) -> None:
        self._p_buf.append(particles[PARTICLE_CSV_COLUMN_ORDER].copy())
        self._e_buf.append(event_row)
        self._m_buf.append(meta_row)
        self._ev_count += 1
        if self._ev_count >= self.events_per_shard:
            self._flush()

    def finalize(self) -> None:
        self._flush()
        man = pd.DataFrame(self.manifest_rows)
        man.to_parquet(
            self.root / "manifest.parquet", index=False, compression="snappy", engine="pyarrow"
        )


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Transform all editable_source_v1 events through diquark+kick (no 90/10)."
    )
    ap.add_argument(
        "--editable-parent",
        type=Path,
        default=None,
        help=f"Parent of {EDITABLE_SOURCE_V1_DIRNAME}/ (default: {default_editable_parent()})",
    )
    ap.add_argument(
        "--out-parent",
        type=Path,
        default=None,
        help=f"Parent for output {OUTPUT_SUBDIR}/ (default: {default_out_parent()})",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reserved; split topology is fixed).")
    ap.add_argument("--delta-px", type=float, default=0.4)
    ap.add_argument("--events-per-shard", type=int, default=10_000)
    ap.add_argument("--max-events", type=int, default=0, help="If >0, only process event_id in 0..N-1.")
    ap.add_argument(
        "--target-altered-success",
        type=int,
        default=0,
        help="If >0, stop after writing this many successful alterations (skip writing failures). "
        "Scans editable shards in order until the count is reached.",
    )
    ap.add_argument(
        "--mode",
        choices=("diquark_kick", "breit_px_kick_only"),
        default="diquark_kick",
    )
    args = ap.parse_args()

    ep = (args.editable_parent or default_editable_parent()).resolve()
    editable_v1 = ep / EDITABLE_SOURCE_V1_DIRNAME
    if not editable_v1.is_dir():
        raise SystemExit(f"missing editable dataset: {editable_v1}")

    out_parent = (args.out_parent or default_out_parent()).resolve()
    n_accepted = _read_editable_run_summary(editable_v1)
    all_ids = list(range(n_accepted))
    if int(args.max_events) > 0:
        all_ids = all_ids[: int(args.max_events)]
    id_allowed: Set[int] = set(all_ids)

    rng = random.Random(int(args.seed))
    writer = _AlteredAllWriter(out_parent, int(args.events_per_shard))

    stats: Dict[str, Any] = {
        "mode": args.mode,
        "seed": args.seed,
        "delta_px": args.delta_px,
        "total_processed": 0,
        "written_altered": 0,
        "fallback_unchanged_no_ud_diquark": 0,
        "fallback_unchanged_struck_fail": 0,
        "fallback_unchanged_validation_fail": 0,
        "altered_channel_C": 0,
    }

    pairs = _iter_input_shard_pairs(editable_v1)
    t0 = time.perf_counter()
    last_report = t0
    target_ok = int(args.target_altered_success)
    progress_total = target_ok if target_ok > 0 else len(id_allowed)
    altered_success_written = 0
    stop = False

    with tqdm(total=progress_total, desc="Alter all", unit="ev", mininterval=0.3) as pbar:
        for pp, epth in pairs:
            if stop:
                break
            pdf = pd.read_parquet(pp)
            edf = pd.read_parquet(epth)
            edf_map = edf.set_index("event_id")
            for event_id, g in pdf.groupby("event_id", sort=True):
                if stop:
                    break
                eid = int(event_id)
                if eid not in id_allowed:
                    continue
                g = g.sort_values("particle_index")
                md_row = edf_map.loc[eid] if eid in edf_map.index else None
                particles_out, branch, altered_meta, delta = process_one_event_split(
                    g,
                    eid,
                    event_in_alter_pool=True,
                    md_row=md_row,
                    mode=str(args.mode),
                    delta_px=float(args.delta_px),
                    rng=rng,
                    forced_split_channel=None,
                )
                for k, v in delta.items():
                    stats[k] = stats.get(k, 0) + v

                if branch == "altered" and altered_meta is not None:
                    mrow = _flatten_altered_meta_success(eid, altered_meta)
                else:
                    mrow = _flatten_altered_meta_failure(
                        eid, _failure_reason_from_delta(delta)
                    )

                stats["total_processed"] += 1

                if target_ok > 0:
                    if branch == "altered" and altered_meta is not None:
                        event_row = _event_row_dict(md_row, eid, len(particles_out))
                        writer.append(particles_out, event_row, mrow)
                        altered_success_written += 1
                        pbar.update(1)
                        if altered_success_written >= target_ok:
                            stop = True
                else:
                    event_row = _event_row_dict(md_row, eid, len(particles_out))
                    writer.append(particles_out, event_row, mrow)
                    pbar.update(1)

                now = time.perf_counter()
                if now - last_report >= 0.5:
                    dt = now - t0
                    rate = stats["total_processed"] / dt if dt > 0 else 0.0
                    pbar.set_postfix(rate=f"{rate:.1f} ev/s", refresh=False)
                    last_report = now

    writer.finalize()

    if target_ok > 0 and altered_success_written < target_ok:
        raise SystemExit(
            f"--target-altered-success={target_ok} but only {altered_success_written} "
            f"successful alterations were collected after scanning {stats['total_processed']} "
            f"input events (increase editable pool or relax topology)."
        )

    elapsed = time.perf_counter() - t0
    stats["elapsed_s"] = float(elapsed)
    if target_ok > 0:
        stats["target_altered_success"] = int(target_ok)
        stats["altered_success_written"] = int(altered_success_written)
    stats["editable_source"] = str(editable_v1)
    stats["output_root"] = str(out_parent / OUTPUT_SUBDIR)
    stats["events_per_shard"] = int(args.events_per_shard)
    stats["output_subdir"] = OUTPUT_SUBDIR
    stats["altered_meta_columns"] = ALTERED_META_COLUMNS
    summary_path = out_parent / OUTPUT_SUBDIR / "run_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
