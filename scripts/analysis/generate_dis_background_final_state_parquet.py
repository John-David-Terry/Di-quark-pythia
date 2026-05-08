#!/usr/bin/env python3
"""
Generate DIS background events and store **only** final-state particles in sharded Parquet
(no per-event CSVs).

Layout under ``--output-dir``::

    final_state_v1/   (default ``--final-state-variant v1``)
      particles/shard_000000.parquet
      events/shard_000000.parquet
      manifest.parquet

    final_state_v3/   (``--final-state-variant v3``, recommended for k_out)
      Same hadronization as v1: **one** ``next()`` with ``HadronLevel:all = on``. After each
      accepted event, read the hard subprocess outgoing quark from ``pythia.process`` (status23,
      quark PDG), save lab four-momentum, build Breit ``LT`` from the hadronic ``event``, and
      store ``k_out_breit_*``. **Do not use** the old v2 two-step ``forceHadronLevel`` pattern;
      it under-hadronizes events.

      ``struck_incoming_index`` / ``struck_outgoing_index`` are indices in ``pythia.process``.

    final_state_v2/ (removed from CLI — broken under-hadronization; regenerate as v3.)

Default ``--output-dir``: ``~/Data/dis_isr_background_final_state`` (v1),
``~/Data/dis_isr_background_final_state_v3`` (v3).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import pythia8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pythia8 import failed: {exc}") from exc

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
_ANAL = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ANAL) not in sys.path:
    sys.path.insert(0, str(_ANAL))

from diquark.analyze_events_raw import FLIP_Z_PTREL, flip_z  # noqa: E402

from generate_dis_isr_parton_dataset import (  # noqa: E402
    hard_subprocess_incoming_quark_process_index,
    hard_subprocess_outgoing_quark_lab_p4_and_index,
    is_finite_particle,
    pick_incoming_quark_index,
    try_build_lt_from_event,
)

ARM_BACKGROUND = 0
EVENTS_PER_SHARD = 10_000
DATASET_ROOT_V1 = "final_state_v1"
DATASET_ROOT_V3 = "final_state_v3"


def default_output_dir() -> Path:
    return Path.home() / "Data" / "dis_isr_background_final_state"


def default_output_dir_v3() -> Path:
    return Path.home() / "Data" / "dis_isr_background_final_state_v3"


def build_pythia_background(seed: int, *, hadron_level: bool = True) -> pythia8.Pythia:
    """PYTHIA for DIS with ISR+FSR; production uses hadron_level=True (v1/v3)."""
    p = pythia8.Pythia()
    p.readString("Beams:idA = 11")
    p.readString("Beams:idB = 2212")
    p.readString("Beams:eA = 18.0")
    p.readString("Beams:eB = 275.0")
    p.readString("Beams:frameType = 2")

    p.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    p.readString("PhaseSpace:Q2Min = 16.0")

    p.readString("ProcessLevel:all = on")
    p.readString("PDF:lepton = off")
    p.readString("PartonLevel:ISR = on")
    p.readString("PartonLevel:FSR = on")
    p.readString("PartonLevel:MPI = off")
    p.readString("PartonLevel:Remnants = on")
    if hadron_level:
        p.readString("HadronLevel:all = on")
    else:
        p.readString("HadronLevel:all = off")

    p.readString(f"Random:seed = {int(seed)}")
    p.readString("Random:setSeed = on")
    p.readString("Print:quiet = on")

    if not p.init():
        raise RuntimeError("PYTHIA init failed in build_pythia_background")
    return p


def _try_event_weight(pythia: pythia8.Pythia) -> float:
    try:
        info = pythia.infoPython()
        for name in ("weight", "getWeight", "weightSum"):
            if hasattr(info, name):
                w = getattr(info, name)
                if callable(w):
                    return float(w())
                return float(w)
    except Exception:
        pass
    return float("nan")


def _try_q2_xb(pythia: pythia8.Pythia) -> Tuple[float, float]:
    try:
        info = pythia.infoPython()
        q2 = float(info.Q2Fac())
        xb = float(info.x2())
        return q2, xb
    except Exception:
        return float("nan"), float("nan")


def collect_final_state_breit(
    ev: pythia8.Event,
    LT: np.ndarray,
    event_id: int,
) -> List[Dict[str, Any]]:
    """Final particles only; momenta in DIS Breit-like frame (same as full-event CSV)."""
    rows: List[Dict[str, Any]] = []
    dense = 0
    for i in range(ev.size()):
        p = ev[i]
        if not p.isFinal():
            continue
        pid = int(p.id())
        px = float(p.px())
        py = float(p.py())
        pz = float(p.pz())
        E = float(p.e())
        if E <= 0:
            continue
        m = float(p.m())
        if not is_finite_particle(px, py, pz, E, m):
            continue
        p4_lab = np.array([E, px, py, pz], dtype=np.float64)
        p4_lab = flip_z(p4_lab, FLIP_Z_PTREL)
        p4_b = LT @ p4_lab
        Eb = float(p4_b[0])
        pxb = float(p4_b[1])
        pyb = float(p4_b[2])
        pzb = float(p4_b[3])
        if Eb <= 0:
            continue
        m2 = Eb * Eb - pxb * pxb - pyb * pyb - pzb * pzb
        if m2 < 0 and m2 > -1e-6:
            m2 = 0.0
        mb = math.sqrt(max(0.0, m2))
        if not is_finite_particle(pxb, pyb, pzb, Eb, mb):
            continue
        rows.append(
            {
                "event_id": int(event_id),
                "particle_index": int(dense),
                "pdg_id": int(pid),
                "px": pxb,
                "py": pyb,
                "pz": pzb,
                "E": Eb,
            }
        )
        dense += 1
    return rows


def _cast_schema(particles_df: pd.DataFrame, events_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    p = particles_df.astype(
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
    e = events_df.astype(
        {
            "event_id": "int64",
            "arm": "int32",
            "n_final": "int32",
            "Q2": "float64",
            "xB": "float64",
            "weight": "float64",
        }
    )
    return p, e


def _cast_schema_with_kout(particles_df: pd.DataFrame, events_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    p, e = _cast_schema(particles_df, events_df)
    extra = {
        "k_out_breit_px": "float64",
        "k_out_breit_py": "float64",
        "k_out_breit_pz": "float64",
        "k_out_breit_E": "float64",
        "struck_incoming_index": "int32",
        "struck_outgoing_index": "int32",
    }
    for col, typ in extra.items():
        e[col] = e[col].astype(typ)
    return p, e


def write_shard_pair(
    root: Path,
    dataset_root: str,
    shard_idx: int,
    particles_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    schema: str = "v1",
) -> Tuple[str, str, int, int, int, int, int]:
    """
    Write particles/events shards. Returns manifest tuple:
    (particles_rel, events_rel, first_eid, last_eid, n_events, n_particle_rows, n_event_rows)
    """
    pdir = root / dataset_root / "particles"
    edir = root / dataset_root / "events"
    pdir.mkdir(parents=True, exist_ok=True)
    edir.mkdir(parents=True, exist_ok=True)
    ppath = pdir / f"shard_{shard_idx:06d}.parquet"
    epath = edir / f"shard_{shard_idx:06d}.parquet"
    if schema in ("v2", "v3"):
        particles_df, events_df = _cast_schema_with_kout(particles_df, events_df)
    else:
        particles_df, events_df = _cast_schema(particles_df, events_df)
    particles_df.to_parquet(ppath, index=False, compression="snappy", engine="pyarrow")
    events_df.to_parquet(epath, index=False, compression="snappy", engine="pyarrow")
    fe = int(events_df["event_id"].min())
    le = int(events_df["event_id"].max())
    ne = int(len(events_df))
    npr = int(len(particles_df))
    prel = f"particles/shard_{shard_idx:06d}.parquet"
    erel = f"events/shard_{shard_idx:06d}.parquet"
    return prel, erel, fe, le, ne, npr, ne


def run_validation(root: Path, dataset_root: str = DATASET_ROOT_V1) -> Dict[str, Any]:
    """Check event counts vs particle row counts and Parquet readability."""
    base = root / dataset_root
    report: Dict[str, Any] = {"ok": True, "errors": []}
    if not base.is_dir():
        report["ok"] = False
        report["errors"].append(f"missing dataset root {base}")
        return report
    manifest_path = base / "manifest.parquet"
    if not manifest_path.is_file():
        report["ok"] = False
        report["errors"].append(f"missing {manifest_path}")
        return report

    man = pd.read_parquet(manifest_path)
    n_final_by_event: Dict[int, int] = {}
    counts: Dict[int, int] = {}

    for _, row in man.iterrows():
        ds = str(row["dataset"])
        rel = str(row["shard_path"])
        path = base / rel.replace("\\", "/")
        if not path.is_file():
            report["ok"] = False
            report["errors"].append(f"missing shard {path}")
            continue
        df = pd.read_parquet(path)
        if ds == "events":
            for _, er in df.iterrows():
                eid = int(er["event_id"])
                n_final_by_event[eid] = int(er["n_final"])
        elif ds == "particles":
            pass

    pdir = base / "particles"
    if pdir.is_dir():
        for pq in sorted(pdir.glob("shard_*.parquet")):
            df = pd.read_parquet(pq)
            vc = df.groupby("event_id").size()
            for eid, c in vc.items():
                counts[int(eid)] = counts.get(int(eid), 0) + int(c)
        for eid, nf in n_final_by_event.items():
            c = counts.get(eid)
            if c is None:
                report["ok"] = False
                report["errors"].append(f"event_id {eid} in events but no particles")
            elif c != nf:
                report["ok"] = False
                report["errors"].append(
                    f"event_id {eid}: n_final={nf} but particle rows={c}"
                )
        for eid in counts:
            if eid not in n_final_by_event:
                report["ok"] = False
                report["errors"].append(f"event_id {eid} in particles but not events")

        for pq in sorted(pdir.glob("shard_*.parquet")):
            df = pd.read_parquet(pq)
            for eid, g in df.groupby("event_id", sort=False):
                g = g.sort_values("particle_index")
                idx = np.asarray(g["particle_index"], dtype=np.int64)
                exp = np.arange(idx.size, dtype=np.int64)
                if idx.size == 0:
                    continue
                if not np.array_equal(idx, exp):
                    report["ok"] = False
                    report["errors"].append(
                        f"event_id {int(eid)}: particle_index not dense 0..n-1"
                    )
    else:
        report["ok"] = False
        report["errors"].append(f"missing particles dir {pdir}")

    report["total_events_manifest_sum"] = int(man[man["dataset"] == "events"]["n_events"].sum())
    report["total_particle_rows_manifest_sum"] = int(
        man[man["dataset"] == "particles"]["n_rows"].sum()
    )
    report["n_final_keys"] = len(n_final_by_event)
    report["particle_indexed_events"] = len(counts)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate DIS background; write final-state-only sharded Parquet."
    )
    ap.add_argument(
        "--final-state-variant",
        choices=("v1", "v3"),
        default="v1",
        help=(
            "v1: final state only, no k_out. v3: same single next() as v1 + hard k_out from "
            "pythia.process after hadronization (final_state_v3/)."
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Parent directory (creates final_state_v1/ or final_state_v3/). "
        f"Default: {default_output_dir()} (v1) or {default_output_dir_v3()} (v3).",
    )
    ap.add_argument(
        "--n-accepted",
        type=int,
        default=900_000,
        help="Target accepted struck-u events with valid Breit frame (default 900000).",
    )
    ap.add_argument(
        "--events-per-shard",
        type=int,
        default=EVENTS_PER_SHARD,
        help=f"Events per Parquet shard (default {EVENTS_PER_SHARD}).",
    )
    ap.add_argument("--seed", type=int, default=12345, help="PYTHIA Random:seed.")
    ap.add_argument(
        "--run-manifest-json",
        type=Path,
        default=None,
        help="Write a small JSON summary next to the dataset (path, shards, bytes).",
    )
    ap.add_argument(
        "--validate-only",
        type=Path,
        default=None,
        metavar="OUTPUT_DIR",
        help="Only validate an existing tree (parent of final_state_v1/ or final_state_v3/). "
        "If only one of those exists under OUTPUT_DIR, it is selected automatically; if both "
        "exist, use --final-state-variant to choose.",
    )
    ap.add_argument(
        "--auto-validate-max",
        type=int,
        default=10_000,
        help="If n_accepted <= this value, run validation checks after generation (default 10000).",
    )
    args = ap.parse_args()

    variant = str(args.final_state_variant)
    dataset_root_name = DATASET_ROOT_V3 if variant == "v3" else DATASET_ROOT_V1
    schema_tag = "v3" if variant == "v3" else "v1"

    if args.validate_only is not None:
        vroot = args.validate_only.expanduser().resolve()
        has_v3 = (vroot / DATASET_ROOT_V3).is_dir()
        has_v1 = (vroot / DATASET_ROOT_V1).is_dir()
        if has_v3 and has_v1:
            dname = DATASET_ROOT_V3 if variant == "v3" else DATASET_ROOT_V1
        elif has_v3:
            dname = DATASET_ROOT_V3
        elif has_v1:
            dname = DATASET_ROOT_V1
        else:
            dname = dataset_root_name
        rep = run_validation(vroot, dataset_root=dname)
        print(json.dumps(rep, indent=2))
        raise SystemExit(0 if rep["ok"] else 1)

    if args.output_dir is not None:
        out_root = args.output_dir.expanduser().resolve()
    elif variant == "v3":
        out_root = default_output_dir_v3().resolve()
    else:
        out_root = default_output_dir().resolve()

    n_target = int(args.n_accepted)
    events_per_shard = max(1, int(args.events_per_shard))

    fs_root = out_root / dataset_root_name
    (fs_root / "particles").mkdir(parents=True, exist_ok=True)
    (fs_root / "events").mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []

    particle_buffer: List[Dict[str, Any]] = []
    event_buffer: List[Dict[str, Any]] = []

    total_generated = 0
    accepted = 0
    breit_rejections = 0
    hard_subprocess_miss = 0
    t0 = time.perf_counter()

    def flush_shard() -> None:
        nonlocal particle_buffer, event_buffer, manifest_rows
        if not event_buffer:
            return
        first_eid = int(event_buffer[0]["event_id"])
        shard_idx = first_eid // events_per_shard
        pdf = pd.DataFrame(particle_buffer)
        edf = pd.DataFrame(event_buffer)
        prel, erel, fe, le, ne, npr, ner = write_shard_pair(
            out_root,
            dataset_root_name,
            shard_idx,
            pdf,
            edf,
            schema=schema_tag,
        )
        manifest_rows.append(
            {
                "shard_path": prel,
                "dataset": "particles",
                "first_event_id": fe,
                "last_event_id": le,
                "n_events": ne,
                "n_rows": npr,
            }
        )
        manifest_rows.append(
            {
                "shard_path": erel,
                "dataset": "events",
                "first_event_id": fe,
                "last_event_id": le,
                "n_events": ne,
                "n_rows": ner,
            }
        )
        particle_buffer = []
        event_buffer = []

    _pbar_desc = f"Generating DIS background ({variant})"
    pbar = tqdm(total=n_target, desc=_pbar_desc, unit="evt", mininterval=0.3, smoothing=0.05)
    try:
        if variant == "v1":
            pythia = build_pythia_background(int(args.seed), hadron_level=True)
            ev = pythia.event
            while accepted < n_target:
                if not pythia.next():
                    continue
                total_generated += 1

                inc_idx = pick_incoming_quark_index(ev)
                if inc_idx is None:
                    continue
                if abs(int(ev[inc_idx].id())) != 2:
                    continue

                LT = try_build_lt_from_event(ev)
                if LT is None:
                    breit_rejections += 1
                    continue

                event_id = accepted
                q2, xb = _try_q2_xb(pythia)
                w = _try_event_weight(pythia)

                parts = collect_final_state_breit(ev, LT, event_id)
                n_final = len(parts)
                particle_buffer.extend(parts)
                event_buffer.append(
                    {
                        "event_id": int(event_id),
                        "arm": int(ARM_BACKGROUND),
                        "n_final": int(n_final),
                        "Q2": float(q2),
                        "xB": float(xb),
                        "weight": float(w),
                    }
                )
                accepted += 1
                pbar.update(1)

                if len(event_buffer) >= events_per_shard:
                    flush_shard()
        else:
            # v3: one full next() with hadronization; read hard k_out from pythia.process after.
            pythia = build_pythia_background(int(args.seed), hadron_level=True)
            ev = pythia.event
            while accepted < n_target:
                if not pythia.next():
                    continue
                total_generated += 1

                inc_idx = pick_incoming_quark_index(ev)
                if inc_idx is None:
                    continue
                if abs(int(ev[inc_idx].id())) != 2:
                    continue

                proc = pythia.process
                iq = hard_subprocess_incoming_quark_process_index(proc)
                p4_lab, oq = hard_subprocess_outgoing_quark_lab_p4_and_index(proc)
                if p4_lab is None or oq < 0:
                    hard_subprocess_miss += 1
                    continue

                LT = try_build_lt_from_event(ev)
                if LT is None:
                    breit_rejections += 1
                    continue

                p4_b = LT @ flip_z(p4_lab, FLIP_Z_PTREL)
                k_E = float(p4_b[0])
                k_px = float(p4_b[1])
                k_py = float(p4_b[2])
                k_pz = float(p4_b[3])

                event_id = accepted
                q2, xb = _try_q2_xb(pythia)
                w = _try_event_weight(pythia)

                parts = collect_final_state_breit(ev, LT, event_id)
                n_final = len(parts)
                particle_buffer.extend(parts)
                event_buffer.append(
                    {
                        "event_id": int(event_id),
                        "arm": int(ARM_BACKGROUND),
                        "n_final": int(n_final),
                        "Q2": float(q2),
                        "xB": float(xb),
                        "weight": float(w),
                        "k_out_breit_px": k_px,
                        "k_out_breit_py": k_py,
                        "k_out_breit_pz": k_pz,
                        "k_out_breit_E": k_E,
                        "struck_incoming_index": int(iq),
                        "struck_outgoing_index": int(oq),
                    }
                )
                accepted += 1
                pbar.update(1)

                if len(event_buffer) >= events_per_shard:
                    flush_shard()
    finally:
        pbar.close()

    flush_shard()

    man_df = pd.DataFrame(manifest_rows)
    man_path = fs_root / "manifest.parquet"
    man_df.to_parquet(man_path, index=False, compression="snappy", engine="pyarrow")

    elapsed = time.perf_counter() - t0
    shard_files = sorted((fs_root / "particles").glob("shard_*.parquet"))
    n_shards = len(shard_files)
    sizes = [p.stat().st_size for p in shard_files]
    sizes += [p.stat().st_size for p in sorted((fs_root / "events").glob("shard_*.parquet"))]
    sizes.append(man_path.stat().st_size)
    total_bytes = sum(sizes)
    avg_mb = (total_bytes / max(1, len(sizes))) / (1024 * 1024)

    summary: Dict[str, Any] = {
        "output_dir": str(out_root),
        "dataset_root": str(fs_root),
        "final_state_variant": variant,
        "n_accepted": accepted,
        "n_generated_tried": total_generated,
        "breit_rejections": breit_rejections,
        "events_per_shard": events_per_shard,
        "n_particle_shards": n_shards,
        "n_manifest_rows": int(len(manifest_rows)),
        "total_disk_bytes": int(total_bytes),
        "avg_file_mb": float(avg_mb),
        "elapsed_s": float(elapsed),
        "acceptance_fraction": float(accepted / total_generated) if total_generated else 0.0,
    }
    if variant == "v3":
        summary["hard_subprocess_miss"] = int(hard_subprocess_miss)

    run_summary_path = fs_root / "run_summary.json"
    run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["run_summary_path"] = str(run_summary_path)
    summary["total_disk_bytes_incl_run_summary"] = int(total_bytes + run_summary_path.stat().st_size)

    print(json.dumps(summary, indent=2))

    if args.run_manifest_json:
        args.run_manifest_json.parent.mkdir(parents=True, exist_ok=True)
        args.run_manifest_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if n_target <= int(args.auto_validate_max):
        rep = run_validation(out_root, dataset_root=dataset_root_name)
        print("validation:", json.dumps(rep, indent=2))
        if not rep["ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
