#!/usr/bin/env python3
"""
Deprecated: use ``split_kinematics_extract.py`` for the full 100k split (unified job queue,
no twin logic). This script kept for older split-by-id workflows.

Multiprocess extraction of jet (outgoing struck quark) and leading-pion kinematics after
reinjection, without computing full transverse observables.

Each worker process:
  - builds ONE ``pythia8.Pythia`` instance (reused for all events in that shard — avoids
    per-event init overhead and reduces load vs naive parallel copies);
  - optionally lowers scheduling priority (``nice``) to be gentler on laptops;
  - processes a disjoint subset of ``altered/`` and ``unchanged/`` event CSVs (default:
    split total 100k into 5 × (2k altered + 18k unchanged)).

Output columns (Breit frame for post-injection quantities; lab for k_out from CSV before boost):
  - event_id, sample, worker_id
  - k_out in lab (from CSV) and in Breit (jet definition, same as analyze_jet_hadron_transverse_observables)
  - leading charged pion (|pdg|==211) in target hemisphere in Breit: pdg, E, px, py, pz
  - Bjorken x, Q^2, Q = sqrt(Q^2)

Merge worker CSVs for downstream offline observable construction.

Run from repo root:
  python scripts/analysis/jet_hadron_kinematics_extract_mp.py \\
    --split-root ... --workers 5 --csv-momenta-frame breit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pythia8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pythia8 import failed: {exc}") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from diquark.paths import outputs_dir  # noqa: E402

from jet_hadron_observables_split_pi_pm import (  # noqa: E402
    build_pythia_reinjector,
    color_balance_ok,
    extract_beams,
    final_colored_partons,
    flip_z,
    FLIP_Z_PTREL,
    Qmax_ptrel,
    Qmin_ptrel,
    resolve_struck_index,
    row_p4,
    run_pythia_reinject_collect,
    xmax_ptrel,
    xmin_ptrel,
)
from diquark.analyze_events_raw import build_LT  # noqa: E402

DEFAULT_SPLIT_ROOT = outputs_dir() / "dis_isr_parton_dataset" / "split_90_10"
DEFAULT_METADATA = outputs_dir() / "dis_isr_parton_dataset" / "dis_isr_event_metadata.csv"
DEFAULT_FULL_EVENT_CSV = outputs_dir() / "dis_isr_parton_dataset" / "dis_isr_full_event_record.csv"


def _event_ids_from_glob(d: Path, pattern: str) -> List[int]:
    out: List[int] = []
    for p in sorted(d.glob(pattern)):
        stem = p.stem
        try:
            out.append(int(stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return out


def leading_target_pion_breit(
    hadrons: List[Tuple[int, np.ndarray]],
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """Highest-energy |pdg|==211 hadron with pz>0 in Breit (post-injection frame)."""
    best_e = -1.0
    best_pid: Optional[int] = None
    best_p4: Optional[np.ndarray] = None
    for pid, p4 in hadrons:
        if abs(int(pid)) != 211:
            continue
        p4 = np.asarray(p4, dtype=np.float64)
        E, pz = float(p4[0]), float(p4[3])
        if pz <= 0:
            continue
        if E > best_e:
            best_e = E
            best_pid = int(pid)
            best_p4 = p4.copy()
    return best_pid, best_p4


def extract_kinematics_row(
    df: pd.DataFrame,
    struck_idx: int,
    sample: str,
    worker_id: int,
    p: pythia8.Pythia,
    csv_momenta_breit: bool,
) -> Dict[str, Any]:
    """One row: jet k_out (lab+breit), leading pion Breit, x, Q2, Q, status flags."""
    df = df.sort_values("particle_index").reset_index(drop=True)
    ev_id = int(df["event_id"].iloc[0])
    base: Dict[str, Any] = {
        "event_id": ev_id,
        "sample": sample,
        "worker_id": worker_id,
        "ok": False,
        "reason": "",
    }

    e_in, e_sc, p_in, beam_msg = extract_beams(df)
    if e_in is None:
        base["reason"] = beam_msg
        return base

    k_row = df[df["particle_index"] == struck_idx]
    if len(k_row) != 1:
        base["reason"] = "k_out_row"
        return base
    k_out = row_p4(k_row.iloc[0])

    if csv_momenta_breit:
        e_in_ev, e_sc_ev, p_in_ev = e_in, e_sc, p_in
        k_out_ev = k_out
    else:
        e_in_ev = flip_z(e_in, FLIP_Z_PTREL)
        e_sc_ev = flip_z(e_sc, FLIP_Z_PTREL)
        p_in_ev = flip_z(p_in, FLIP_Z_PTREL)
        k_out_ev = flip_z(k_out, FLIP_Z_PTREL)

    qmu = e_in_ev - e_sc_ev
    Q2 = -(qmu[0] * qmu[0] - qmu[1] * qmu[1] - qmu[2] * qmu[2] - qmu[3] * qmu[3])
    if Q2 <= 0:
        base["reason"] = "Q2"
        return base
    Q = float(np.sqrt(Q2))
    qT = float(np.hypot(qmu[1], qmu[2]))
    p_dot_q = float(
        p_in_ev[0] * qmu[0] - p_in_ev[1] * qmu[1] - p_in_ev[2] * qmu[2] - p_in_ev[3] * qmu[3]
    )
    if p_dot_q == 0:
        base["reason"] = "pdotq"
        return base
    x = Q2 / (2.0 * p_dot_q)
    if not (xmin_ptrel <= x <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel):
        base["reason"] = "xQ_window"
        return base

    Ee = float(e_in_ev[0])
    Ep = float(p_in_ev[0])
    S = 4.0 * Ee * Ep
    y = Q2 / (S * x)
    phiq = float(np.arctan2(qmu[2], qmu[1]))
    if csv_momenta_breit:
        LT = np.eye(4, dtype=np.float64)
    else:
        LT = build_LT(Ee, Ep, qmu, x, y, qT, phiq, S)
        if LT is None:
            base["reason"] = "build_LT"
            return base

    partons = final_colored_partons(df)
    c_ok, c_msg = color_balance_ok(partons)
    if not c_ok:
        base["reason"] = "color"
        base["color_message"] = c_msg
        return base

    py_ok, py_err, hadrons = run_pythia_reinject_collect(p, partons, LT, csv_momenta_breit)
    if not py_ok:
        base["reason"] = "pythia"
        base["pythia_error"] = py_err
        return base

    boost = lambda v: LT @ np.asarray(v, dtype=float)
    p_breit = boost(p_in_ev)
    if float(p_breit[0] + p_breit[3]) <= 0:
        base["reason"] = "P_plus"
        return base

    k_out_breit = boost(k_out_ev)
    # Stored CSV frame: Breit-like when --csv-momenta-frame breit (same components as breit jet);
    # lab after flip_z when CSV is lab frame.
    base["k_out_stored_E"] = float(k_out_ev[0])
    base["k_out_stored_px"] = float(k_out_ev[1])
    base["k_out_stored_py"] = float(k_out_ev[2])
    base["k_out_stored_pz"] = float(k_out_ev[3])
    base["k_out_breit_E"] = float(k_out_breit[0])
    base["k_out_breit_px"] = float(k_out_breit[1])
    base["k_out_breit_py"] = float(k_out_breit[2])
    base["k_out_breit_pz"] = float(k_out_breit[3])

    base["x"] = float(x)
    base["Q2"] = float(Q2)
    base["Q"] = float(Q)

    pi_pid, pi_p4 = leading_target_pion_breit(hadrons)
    if pi_pid is None or pi_p4 is None:
        base["reason"] = "no_pion_candidate"
        base["n_final_hadrons"] = len(hadrons)
        return base

    base["pion_pdg"] = pi_pid
    base["pion_breit_E"] = float(pi_p4[0])
    base["pion_breit_px"] = float(pi_p4[1])
    base["pion_breit_py"] = float(pi_p4[2])
    base["pion_breit_pz"] = float(pi_p4[3])
    base["n_final_hadrons"] = len(hadrons)
    base["ok"] = True
    base["reason"] = "ok"
    return base


def _worker_run(payload: Tuple[int, List[int], List[int], Dict[str, Any]]) -> Path:
    worker_id, altered_ids, unchanged_ids, cfg = payload
    if cfg.get("nice_increment", 0) > 0:
        try:
            os.nice(int(cfg["nice_increment"]))
        except OSError:
            pass

    split_root = Path(cfg["split_root"])
    altered_dir = split_root / "altered"
    unchanged_dir = split_root / "unchanged"
    meta_csv = Path(cfg["metadata_csv"])
    md_map: Optional[pd.DataFrame] = None
    if meta_csv.is_file():
        md = pd.read_csv(meta_csv).sort_values("event_id").reset_index(drop=True)
        md_map = md.set_index("event_id")

    csv_momenta_breit = bool(cfg["csv_momenta_breit"])
    out_path = Path(cfg["out_path"])

    p = build_pythia_reinjector()
    rows: List[Dict[str, Any]] = []

    def handle_one(event_id: int, sample: str, csv_path: Path) -> None:
        meta_path = altered_dir / f"event_{event_id:06d}.meta.json" if sample == "altered" else None
        df = pd.read_csv(csv_path).sort_values("particle_index").reset_index(drop=True)
        ev0 = int(df["event_id"].iloc[0])
        idx, _ = resolve_struck_index(df, ev0, meta_path, md_map)
        if idx is None:
            rows.append(
                {
                    "event_id": event_id,
                    "sample": sample,
                    "worker_id": worker_id,
                    "ok": False,
                    "reason": "struck_unresolved",
                }
            )
            return
        rows.append(extract_kinematics_row(df, idx, sample, worker_id, p, csv_momenta_breit))

    for eid in altered_ids:
        handle_one(eid, "altered", altered_dir / f"event_{int(eid):06d}.csv")

    for eid in unchanged_ids:
        handle_one(eid, "unchanged", unchanged_dir / f"event_{int(eid):06d}.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract jet k_out + leading pion kinematics (multiprocess, reused PYTHIA per worker)."
    )
    ap.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    ap.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    ap.add_argument("--workers", type=int, default=5, help="Parallel worker processes (default 5).")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for worker CSVs (default: split-root/kinematics_extract).",
    )
    ap.add_argument(
        "--csv-momenta-frame",
        choices=("lab", "breit"),
        default="breit",
        help="Must match how split CSVs were produced.",
    )
    ap.add_argument(
        "--nice-increment",
        type=int,
        default=10,
        help="Add to process nice in each worker (0=disabled). Lower priority on laptops.",
    )
    ap.add_argument(
        "--max-altered",
        type=int,
        default=0,
        help="If >0, only use first N altered IDs (for testing).",
    )
    ap.add_argument(
        "--max-unchanged",
        type=int,
        default=0,
        help="If >0, only use first N unchanged IDs (for testing).",
    )
    args = ap.parse_args()

    import multiprocessing as mp

    split_root = args.split_root.resolve()
    altered_dir = split_root / "altered"
    unchanged_dir = split_root / "unchanged"
    if not altered_dir.is_dir():
        raise SystemExit(f"altered dir not found: {altered_dir}")
    if not unchanged_dir.is_dir():
        raise SystemExit(f"unchanged dir not found: {unchanged_dir}")

    altered_ids = _event_ids_from_glob(altered_dir, "event_*.csv")
    unchanged_ids = _event_ids_from_glob(unchanged_dir, "event_*.csv")
    if args.max_altered > 0:
        altered_ids = altered_ids[: args.max_altered]
    if args.max_unchanged > 0:
        unchanged_ids = unchanged_ids[: args.max_unchanged]

    n_w = max(1, int(args.workers))
    alt_chunks = np.array_split(np.array(sorted(altered_ids), dtype=np.int64), n_w)
    un_chunks = np.array_split(np.array(sorted(unchanged_ids), dtype=np.int64), n_w)

    out_dir = (args.out_dir or (split_root / "kinematics_extract")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_momenta_breit = args.csv_momenta_frame == "breit"

    payloads: List[Tuple[int, List[int], List[int], Dict[str, Any]]] = []
    for w in range(n_w):
        a_list = [int(x) for x in alt_chunks[w].tolist()]
        u_list = [int(x) for x in un_chunks[w].tolist()]
        cfg = {
            "split_root": str(split_root),
            "metadata_csv": str(Path(args.metadata_csv).resolve()),
            "csv_momenta_breit": csv_momenta_breit,
            "nice_increment": int(args.nice_increment),
            "out_path": str(out_dir / f"kinematics_worker_{w:02d}.csv"),
        }
        payloads.append((w, a_list, u_list, cfg))

    print(
        f"split_root={split_root}\n"
        f"workers={n_w}  altered_total={len(altered_ids)}  unchanged_total={len(unchanged_ids)}\n"
        f"per-worker altered counts: {[len(c) for c in alt_chunks]}\n"
        f"per-worker unchanged counts: {[len(c) for c in un_chunks]}\n"
        f"out_dir={out_dir}"
    )

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_w) as pool:
        paths = pool.map(_worker_run, payloads)

    merged = out_dir / "kinematics_merged.csv"
    parts = [pd.read_csv(p) for p in paths]
    pd.concat(parts, ignore_index=True).sort_values(["worker_id", "sample", "event_id"]).to_csv(
        merged, index=False
    )
    print(f"Wrote {paths}")
    print(f"Merged: {merged}")


if __name__ == "__main__":
    main()
