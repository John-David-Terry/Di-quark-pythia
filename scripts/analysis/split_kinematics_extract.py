#!/usr/bin/env python3
"""
Pure kinematics extraction for a full 90/10 split (unchanged + altered event CSVs).

For **each** ``event_*.csv`` under ``split_90_10/unchanged/`` and ``split_90_10/altered/``:
  1. Read the CSV (same format as ``generate_dis_isr_parton_dataset.py``).
  2. Resolve the struck outgoing quark index (metadata CSV and optional ``altered/*.meta.json``).
  3. Reinject final colored partons into PYTHIA, ``next()`` (FSR + hadronization).
  4. Save one row: ``k_out`` in Breit, leading charged pion in Breit, ``xB``, ``Q``, ``Q2``,
     ``pythia_ok``, failure reason.

No twin pairing, no full-event-record fallback, no π± PDFs or transverse observables.

Defaults resolve paths via ``diquark.paths.outputs_dir()`` — set ``--data-dir`` to the directory
that **contains** ``split_90_10/`` and ``dis_isr_event_metadata.csv`` (e.g.
``.../outputs/dis_isr_parton_dataset`` or ``.../outputs/dis_isr_benchmark_1M``).

Example:
  python scripts/analysis/split_kinematics_extract.py \\
    --data-dir /path/to/outputs/dis_isr_benchmark_1M \\
    --workers 5 --csv-momenta-frame breit

Progress: tqdm bar on stderr (``pip install tqdm``); use ``--no-progress`` to disable.

Benchmark timing (e.g. 900 unchanged + 100 altered, one core):
  python scripts/analysis/split_kinematics_extract.py \\
    --data-dir ... --workers 1 --nice-increment 0 \\
    --bench-unchanged 900 --bench-altered 100 --out-dir .../bench_run

Parquet input (after ``scripts/analysis/convert_split_events_to_parquet.py``):
  python scripts/analysis/split_kinematics_extract.py \\
    --data-dir ... --input-format parquet --workers 1 \\
    --parquet-split-root .../split_90_10_parquet

``--input-format auto`` selects Parquet when both ``altered/manifest.csv`` and
``unchanged/manifest.csv`` exist under ``--parquet-split-root``. Requires pyarrow.

Parquet I/O: ``--parquet-read-mode shard`` (default) reads each shard once and slices
by ``event_id`` in memory; use ``filtered`` for per-event filtered reads (debug).
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import json
import os
from collections import Counter
import multiprocessing as mp
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

# (path, sample, parquet_event_id): third element set for Parquet shards; None for per-event CSV.
SplitJob = Tuple[str, str, Optional[int]]

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
    extract_beams_arrays,
    final_colored_partons,
    flip_z,
    FLIP_Z_PTREL,
    Qmax_ptrel,
    Qmin_ptrel,
    resolve_struck_index,
    run_pythia_reinject_collect,
    xmax_ptrel,
    xmin_ptrel,
)
from diquark.analyze_events_raw import build_LT  # noqa: E402


# Columns required by ``extract_one_event``, ``resolve_struck_index`` / struck reco, and
# ``final_colored_partons`` (see ``row_is_final_colored_qcd`` → daughter1/daughter2).
# Omitting unused columns (m, pT, eta, phi, mother1, …) cuts parse work. We use the C
# engine and default inference: ints → int64, momenta → float64 (same as a full read of
# these columns). Passing ``dtype=`` to ``read_csv`` was measured ~2× slower per file on
# typical event CSV sizes here.
_EVENT_CSV_USECOLS: Tuple[str, ...] = (
    "event_id",
    "particle_index",
    "pdg_id",
    "status",
    "isFinal",
    "daughter1",
    "daughter2",
    "col",
    "acol",
    "px",
    "py",
    "pz",
    "E",
)


def _read_event_csv(path_or_buf: Any) -> pd.DataFrame:
    """Parse one event CSV: only columns needed downstream, C engine."""
    return pd.read_csv(
        path_or_buf,
        usecols=list(_EVENT_CSV_USECOLS),
        engine="c",
    )


def _read_event_parquet(shard_path: Path, event_id: int) -> pd.DataFrame:
    """Load one event's rows from a shard via pyarrow predicate pushdown; same columns as CSV path."""
    df = pd.read_parquet(
        shard_path,
        engine="pyarrow",
        filters=[("event_id", "==", int(event_id))],
        columns=list(_EVENT_CSV_USECOLS),
    )
    if df.empty:
        raise ValueError(f"no rows for event_id={event_id} in {shard_path}")
    return df.copy()


_PERF = time.perf_counter


class _Seg:
    """Wall-time accumulator between cut() calls (optional dict)."""

    __slots__ = ("acc", "t")

    def __init__(self, acc: Optional[MutableMapping[str, float]]):
        self.acc = acc
        self.t = _PERF()

    def cut(self, key: str) -> None:
        if self.acc is None:
            self.t = _PERF()
            return
        n = _PERF()
        self.acc[key] = self.acc.get(key, 0.0) + (n - self.t)
        self.t = n

    def sync(self) -> None:
        self.t = _PERF()


# Per-event bucket keys accumulated in the worker (plus total_event set per event).
# For Parquet shard mode, ``parquet_read`` / ``parquet_group_or_index_build`` are amortized
# across events in the shard; ``event_extract_from_shard`` is measured per event (iloc slice).
_TIMING_BUCKET_KEYS = (
    "csv_read",
    "parquet_read",
    "parquet_group_or_index_build",
    "event_extract_from_shard",
    "struck_resolution",
    "kinematics_setup",
    "pythia_append",
    "pythia_next",
    "hadron_scan",
    "row_write",
)

_TIMING_STAGE_KEYS = _TIMING_BUCKET_KEYS  # alias for merge / reporting

# Optional sub-buckets under csv_read (filled only when --profile-timing).
_TIMING_CSV_SUB_KEYS = ("file_open_read", "pandas_parse")

# Sub-buckets under kinematics_setup (filled only when --profile-timing).
_TIMING_KIN_SUB_KEYS = (
    "beam_dis_selection",
    "breit_setup",
    "final_parton_selection",
    "color_balance_check",
    "reinject_prep",
)


def _ks_begin(timings: Optional[MutableMapping[str, float]]) -> float:
    return _PERF() if timings is not None else 0.0


def _ks_add(
    timings: Optional[MutableMapping[str, float]], t0: float, sub_key: str
) -> None:
    """Add wall time to ``sub_key`` and to top-level ``kinematics_setup``."""
    if timings is None:
        return
    dt = _PERF() - t0
    timings[sub_key] = timings.get(sub_key, 0.0) + dt
    timings["kinematics_setup"] = timings.get("kinematics_setup", 0.0) + dt


def _timing_acc_init() -> Dict[str, Any]:
    acc: Dict[str, Any] = {
        "n_events_seen": 0,
        "n_events_ok": 0,
        "total_seconds": 0.0,
    }
    for k in _TIMING_STAGE_KEYS:
        acc[f"{k}_seconds"] = 0.0
    for k in _TIMING_CSV_SUB_KEYS:
        acc[f"{k}_seconds"] = 0.0
    for k in _TIMING_KIN_SUB_KEYS:
        acc[f"{k}_seconds"] = 0.0
    return acc


def _timing_acc_add_event(
    acc: Dict[str, Any],
    per_event: MutableMapping[str, float],
    ok: bool,
) -> None:
    acc["n_events_seen"] += 1
    if ok:
        acc["n_events_ok"] += 1
    acc["total_seconds"] += float(per_event.get("total_event", 0.0))
    for k in _TIMING_STAGE_KEYS:
        acc[f"{k}_seconds"] += float(per_event.get(k, 0.0))
    for k in _TIMING_CSV_SUB_KEYS:
        acc[f"{k}_seconds"] += float(per_event.get(k, 0.0))
    for k in _TIMING_KIN_SUB_KEYS:
        acc[f"{k}_seconds"] += float(per_event.get(k, 0.0))


def _write_timing_worker_json(out_dir: Path, worker_id: int, acc: Dict[str, Any]) -> None:
    n = int(acc["n_events_seen"])
    payload: Dict[str, Any] = {
        "worker_id": worker_id,
        "n_events_seen": n,
        "n_events_ok": int(acc["n_events_ok"]),
        "total_seconds": float(acc["total_seconds"]),
    }
    for k in _TIMING_STAGE_KEYS:
        key = f"{k}_seconds"
        payload[key] = float(acc[key])
        payload[f"{k}_avg_seconds"] = float(acc[key]) / n if n else 0.0
    for k in _TIMING_CSV_SUB_KEYS:
        key = f"{k}_seconds"
        payload[key] = float(acc[key])
        payload[f"{k}_avg_seconds"] = float(acc[key]) / n if n else 0.0
    for k in _TIMING_KIN_SUB_KEYS:
        key = f"{k}_seconds"
        payload[key] = float(acc[key])
        payload[f"{k}_avg_seconds"] = float(acc[key]) / n if n else 0.0
    payload["total_event_avg_seconds"] = float(acc["total_seconds"]) / n if n else 0.0
    path = out_dir / f"timing_worker_{worker_id:02d}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def merge_timing_reports(out_dir: Path) -> Optional[Dict[str, Any]]:
    paths = sorted(out_dir.glob("timing_worker_*.json"))
    if not paths:
        return None
    acc = _timing_acc_init()
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        acc["n_events_seen"] += int(d["n_events_seen"])
        acc["n_events_ok"] += int(d["n_events_ok"])
        acc["total_seconds"] += float(d["total_seconds"])
        for k in _TIMING_STAGE_KEYS:
            acc[f"{k}_seconds"] += float(d.get(f"{k}_seconds", 0.0))
        for k in _TIMING_CSV_SUB_KEYS:
            acc[f"{k}_seconds"] += float(d.get(f"{k}_seconds", 0.0))
        for k in _TIMING_KIN_SUB_KEYS:
            acc[f"{k}_seconds"] += float(d.get(f"{k}_seconds", 0.0))
    n = int(acc["n_events_seen"])
    stage_sum = sum(float(acc[f"{k}_seconds"]) for k in _TIMING_STAGE_KEYS)
    summary: Dict[str, Any] = {
        "n_events_seen": n,
        "n_events_ok": int(acc["n_events_ok"]),
        "total_seconds": float(acc["total_seconds"]),
        "tracked_stage_seconds": stage_sum,
    }
    for k in _TIMING_STAGE_KEYS:
        sk = f"{k}_seconds"
        summary[sk] = float(acc[sk])
        summary[f"{k}_avg_seconds"] = float(acc[sk]) / n if n else 0.0
        summary[f"{k}_pct_of_stages"] = (
            100.0 * float(acc[sk]) / stage_sum if stage_sum > 0 else 0.0
        )
    for k in _TIMING_CSV_SUB_KEYS:
        sk = f"{k}_seconds"
        summary[sk] = float(acc[sk])
        summary[f"{k}_avg_seconds"] = float(acc[sk]) / n if n else 0.0
    cr = float(acc["csv_read_seconds"])
    for k in _TIMING_CSV_SUB_KEYS:
        sk = f"{k}_seconds"
        summary[f"{k}_pct_of_csv_read"] = (
            100.0 * float(acc[sk]) / cr if cr > 0 else 0.0
        )
    ts_loop = float(summary["tracked_stage_seconds"])
    for k in _TIMING_KIN_SUB_KEYS:
        sk = f"{k}_seconds"
        summary[sk] = float(acc[sk])
        summary[f"{k}_avg_seconds"] = float(acc[sk]) / n if n else 0.0
        summary[f"{k}_pct_of_tracked_loop"] = (
            100.0 * float(acc[sk]) / ts_loop if ts_loop > 0 else 0.0
        )
    summary["total_event_avg_seconds"] = float(acc["total_seconds"]) / n if n else 0.0
    (out_dir / "timing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def merge_slow_event_reports(out_dir: Path, n_requested: int) -> None:
    if n_requested <= 0:
        return
    paths = sorted(out_dir.glob("slow_events_worker_*.json"))
    events: List[Dict[str, Any]] = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        events.extend(data.get("events", []))
    events.sort(key=lambda r: -float(r["total_event_seconds"]))
    top = events[:n_requested]
    out = {"n_requested": n_requested, "events": top}
    (out_dir / "slow_events.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )


def leading_target_pion_breit(
    hadrons: List[Tuple[int, np.ndarray]],
) -> Tuple[Optional[int], Optional[np.ndarray]]:
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


def extract_one_event(
    df: pd.DataFrame,
    struck_idx: int,
    sample: str,
    worker_id: int,
    p: pythia8.Pythia,
    csv_momenta_breit: bool,
    timings: Optional[MutableMapping[str, float]] = None,
) -> Dict[str, Any]:
    seg = _Seg(timings)
    t_beam0 = _ks_begin(timings)
    # ``df`` is already sorted by ``particle_index`` in the worker (same as before).
    ev_id = int(df["event_id"].to_numpy()[0])
    row: Dict[str, Any] = {
        "event_id": ev_id,
        "sample": sample,
        "worker_id": worker_id,
        "pythia_ok": False,
        "ok": False,
        "reason": "",
        "failure_reason": "",
    }

    pdg = df["pdg_id"].to_numpy()
    status = df["status"].to_numpy()
    is_final = df["isFinal"].to_numpy()
    E = df["E"].to_numpy(dtype=np.float64)
    px = df["px"].to_numpy(dtype=np.float64)
    py = df["py"].to_numpy(dtype=np.float64)
    pz = df["pz"].to_numpy(dtype=np.float64)
    pidx = df["particle_index"].to_numpy()

    e_in, e_sc, p_in, beam_msg = extract_beams_arrays(
        pdg, status, is_final, E, px, py, pz
    )
    if e_in is None:
        _ks_add(timings, t_beam0, "beam_dis_selection")
        row["reason"] = beam_msg
        row["failure_reason"] = beam_msg
        return row

    k_hits = np.flatnonzero(pidx == struck_idx)
    if k_hits.size != 1:
        _ks_add(timings, t_beam0, "beam_dis_selection")
        row["reason"] = "k_out_row"
        row["failure_reason"] = "k_out_row"
        return row
    ik = int(k_hits[0])
    k_out = np.array([E[ik], px[ik], py[ik], pz[ik]], dtype=np.float64)
    _ks_add(timings, t_beam0, "beam_dis_selection")

    t_bz0 = _ks_begin(timings)
    if csv_momenta_breit:
        e_in_ev, e_sc_ev, p_in_ev = e_in, e_sc, p_in
        k_out_ev = k_out
    else:
        e_in_ev = flip_z(e_in, FLIP_Z_PTREL)
        e_sc_ev = flip_z(e_sc, FLIP_Z_PTREL)
        p_in_ev = flip_z(p_in, FLIP_Z_PTREL)
        k_out_ev = flip_z(k_out, FLIP_Z_PTREL)
    _ks_add(timings, t_bz0, "breit_setup")

    t_dis0 = _ks_begin(timings)
    qmu = e_in_ev - e_sc_ev
    Q2 = -(qmu[0] * qmu[0] - qmu[1] * qmu[1] - qmu[2] * qmu[2] - qmu[3] * qmu[3])
    if Q2 <= 0:
        _ks_add(timings, t_dis0, "beam_dis_selection")
        row["reason"] = "Q2"
        row["failure_reason"] = "Q2"
        return row
    Q = float(np.sqrt(Q2))
    qT = float(np.hypot(qmu[1], qmu[2]))
    p_dot_q = float(
        p_in_ev[0] * qmu[0] - p_in_ev[1] * qmu[1] - p_in_ev[2] * qmu[2] - p_in_ev[3] * qmu[3]
    )
    if p_dot_q == 0:
        _ks_add(timings, t_dis0, "beam_dis_selection")
        row["reason"] = "pdotq"
        row["failure_reason"] = "pdotq"
        return row
    xB = Q2 / (2.0 * p_dot_q)
    if not (xmin_ptrel <= xB <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel):
        _ks_add(timings, t_dis0, "beam_dis_selection")
        row["reason"] = "xQ_window"
        row["failure_reason"] = "xQ_window"
        return row

    Ee = float(e_in_ev[0])
    Ep = float(p_in_ev[0])
    S = 4.0 * Ee * Ep
    y = Q2 / (S * xB)
    phiq = float(np.arctan2(qmu[2], qmu[1]))
    _ks_add(timings, t_dis0, "beam_dis_selection")

    t_lt0 = _ks_begin(timings)
    if csv_momenta_breit:
        LT = np.eye(4, dtype=np.float64)
    else:
        LT = build_LT(Ee, Ep, qmu, xB, y, qT, phiq, S)
        if LT is None:
            _ks_add(timings, t_lt0, "breit_setup")
            row["reason"] = "build_LT"
            row["failure_reason"] = "build_LT"
            return row
    _ks_add(timings, t_lt0, "breit_setup")

    t_fp0 = _ks_begin(timings)
    partons = final_colored_partons(df)
    _ks_add(timings, t_fp0, "final_parton_selection")

    t_col0 = _ks_begin(timings)
    c_ok, c_msg = color_balance_ok(partons)
    if not c_ok:
        _ks_add(timings, t_col0, "color_balance_check")
        row["reason"] = "color"
        row["failure_reason"] = "color"
        row["color_message"] = c_msg
        return row
    _ks_add(timings, t_col0, "color_balance_check")

    py_phases: Dict[str, float] = {}
    py_ok, py_err, hadrons = run_pythia_reinject_collect(
        p,
        partons,
        LT,
        csv_momenta_breit,
        phase_times=py_phases if timings is not None else None,
    )
    if timings is not None:
        for k, v in py_phases.items():
            timings[k] = timings.get(k, 0.0) + float(v)
            if k == "reinject_prep":
                timings["kinematics_setup"] = timings.get("kinematics_setup", 0.0) + float(
                    v
                )

    seg.sync()
    row["pythia_ok"] = bool(py_ok)
    if not py_ok:
        row["reason"] = "pythia"
        row["failure_reason"] = "pythia"
        row["pythia_error"] = py_err
        seg.cut("row_write")
        return row

    t_post0 = _ks_begin(timings)
    boost = lambda v: LT @ np.asarray(v, dtype=float)
    p_breit = boost(p_in_ev)
    if float(p_breit[0] + p_breit[3]) <= 0:
        _ks_add(timings, t_post0, "breit_setup")
        row["reason"] = "P_plus"
        row["failure_reason"] = "P_plus"
        return row

    k_out_breit = boost(k_out_ev)
    row["k_out_stored_E"] = float(k_out_ev[0])
    row["k_out_stored_px"] = float(k_out_ev[1])
    row["k_out_stored_py"] = float(k_out_ev[2])
    row["k_out_stored_pz"] = float(k_out_ev[3])
    row["k_out_breit_E"] = float(k_out_breit[0])
    row["k_out_breit_px"] = float(k_out_breit[1])
    row["k_out_breit_py"] = float(k_out_breit[2])
    row["k_out_breit_pz"] = float(k_out_breit[3])
    row["xB"] = float(xB)
    row["Q2"] = float(Q2)
    row["Q"] = float(Q)
    _ks_add(timings, t_post0, "breit_setup")

    pi_pid, pi_p4 = leading_target_pion_breit(hadrons)
    seg.cut("hadron_scan")
    if pi_pid is None or pi_p4 is None:
        row["reason"] = "no_pion_candidate"
        row["failure_reason"] = "no_pion_candidate"
        row["n_final_hadrons"] = len(hadrons)
        seg.cut("row_write")
        return row

    row["pion_pdg"] = pi_pid
    row["pion_breit_E"] = float(pi_p4[0])
    row["pion_breit_px"] = float(pi_p4[1])
    row["pion_breit_py"] = float(pi_p4[2])
    row["pion_breit_pz"] = float(pi_p4[3])
    row["n_final_hadrons"] = len(hadrons)
    row["ok"] = True
    row["reason"] = "ok"
    row["failure_reason"] = ""
    seg.cut("row_write")
    return row


def _unique_sorted_event_ids(meta_csv: Path, max_events: int = 0) -> List[int]:
    """Sorted unique ``event_id`` values from ``dis_isr_event_metadata.csv``.

    For ``--max-events`` smoke runs, only the first ~200k metadata rows are read so we
    do not parse 1M rows before processing a handful of events. Full runs scan the file
    in chunks and build the full sorted unique list.
    """
    if max_events > 0:
        df = pd.read_csv(
            meta_csv,
            usecols=["event_id"],
            dtype={"event_id": "int64"},
            nrows=200_000,
        )
        return np.unique(df["event_id"].to_numpy()).tolist()
    parts: List[np.ndarray] = []
    for chunk in pd.read_csv(
        meta_csv, usecols=["event_id"], dtype={"event_id": "int64"}, chunksize=500_000
    ):
        parts.append(chunk["event_id"].to_numpy())
    if not parts:
        return []
    u = np.unique(np.concatenate(parts))
    return u.tolist()


def _path_exists_quick(p: Path) -> bool:
    """Like ``Path.is_file`` but maps common FS errors to False (or re-raises)."""
    try:
        return p.is_file()
    except OSError as exc:
        # Sandbox / cloud / interrupted syscalls: caller may fall back to glob.
        if getattr(exc, "errno", None) in (60, 89, 5):
            raise
        return False


def _nsmallest_event_csv_paths(dirp: Path, n: int) -> List[Path]:
    """Smallest ``n`` ``event_*.csv`` paths by filename without sorting the full directory."""

    def ok(p: Path) -> bool:
        return p.is_file() and p.name.startswith("event_") and p.suffix == ".csv"

    if n <= 0:
        return []
    return heapq.nsmallest(n, (p for p in dirp.iterdir() if ok(p)), key=lambda p: p.name)


def _try_probe_first_n_event_csvs(dirp: Path, n: int) -> Optional[List[Path]]:
    """Fast path: assume ``event_%06d.csv`` and lexicographic == numeric order.

    Probe ``event_000001.csv``, ``event_000002.csv``, … until ``n`` existing files are
    collected. At most ``max(10*n, 10000)`` indices are tried; return ``None`` if
    fewer than ``n`` hits within that bound (caller should fall back to directory scan).
    """
    if n <= 0:
        return []
    max_probes = max(10 * n, 10_000)
    out: List[Path] = []
    for i in range(1, max_probes + 1):
        p = dirp / f"event_{i:06d}.csv"
        if _path_exists_quick(p):
            out.append(p)
            if len(out) >= n:
                return out
    return None


def _collect_benchmark_event_paths(
    dirp: Path, n: int, timings: MutableMapping[str, float]
) -> List[Path]:
    """First ``n`` smallest event CSV paths; probe fast path then optional ``iterdir`` fallback."""
    t_fast0 = _PERF()
    probed = _try_probe_first_n_event_csvs(dirp, n)
    timings["benchmark_job_list_fast_path"] = float(
        timings.get("benchmark_job_list_fast_path", 0.0)
    ) + (_PERF() - t_fast0)
    if probed is not None:
        return probed
    t_fb0 = _PERF()
    out = _nsmallest_event_csv_paths(dirp, n)
    timings["benchmark_job_list_fallback"] = float(
        timings.get("benchmark_job_list_fallback", 0.0)
    ) + (_PERF() - t_fb0)
    return out


def _build_job_list_glob(
    split_root: Path, max_events: int = 0
) -> List[SplitJob]:
    """Fallback: round-robin altered/unchanged paths.

    Full runs use ``glob`` + sort. For ``max_events > 0`` (smoke), only the smallest
    ``ceil(max_events/2)`` paths per side are collected via ``heapq.nsmallest`` over
    ``iterdir`` so we never materialize/sort ~1e6 filenames.
    """
    altered_dir = split_root / "altered"
    unchanged_dir = split_root / "unchanged"
    if max_events > 0:
        half = (max_events + 1) // 2
        altered = _nsmallest_event_csv_paths(altered_dir, half)
        unchanged = _nsmallest_event_csv_paths(unchanged_dir, half)
    else:
        altered = sorted(altered_dir.glob("event_*.csv"))
        unchanged = sorted(unchanged_dir.glob("event_*.csv"))
    jobs: List[SplitJob] = []
    ia, iu = 0, 0
    while ia < len(altered) or iu < len(unchanged):
        if ia < len(altered):
            jobs.append((str(altered[ia].resolve()), "altered", None))
            ia += 1
        if iu < len(unchanged):
            jobs.append((str(unchanged[iu].resolve()), "unchanged", None))
            iu += 1
    if max_events > 0 and len(jobs) > max_events:
        jobs = jobs[:max_events]
    return jobs


def _build_job_list(
    split_root: Path, meta_csv: Path, max_events: int = 0
) -> List[SplitJob]:
    """(absolute csv path, sample) for every ``event_*.csv`` present in altered/ or unchanged/.

    Event IDs are taken from ``dis_isr_event_metadata.csv`` (sorted). For each ID we
    ``stat`` ``altered/event_XXXXXX.csv`` then ``unchanged/...`` if needed — this avoids
    ``listdir``/``glob`` on directories with ~1e6 entries (can hang on some filesystems).

    Altered and unchanged path lists are merged in **round-robin** order (altered,
    unchanged, altered, ...) until one side is exhausted, then the remainder is appended,
    so worker shards and ``--max-events`` slices stay mixed.

    If per-path ``stat`` fails (e.g. errno 89), falls back to ``_build_job_list_glob``.
    """
    altered_dir = split_root / "altered"
    unchanged_dir = split_root / "unchanged"
    all_ids = _unique_sorted_event_ids(meta_csv, max_events)
    altered_paths: List[str] = []
    unchanged_paths: List[str] = []
    half_need = (max_events + 1) // 2 if max_events > 0 else 0
    try:
        for i, eid in enumerate(all_ids):
            if i > 0 and i % 200_000 == 0:
                print(f"  job list scan: {i}/{len(all_ids)} event_ids", flush=True)
            pa = altered_dir / f"event_{int(eid):06d}.csv"
            pu = unchanged_dir / f"event_{int(eid):06d}.csv"
            if _path_exists_quick(pa):
                altered_paths.append(str(pa.resolve()))
            elif _path_exists_quick(pu):
                unchanged_paths.append(str(pu.resolve()))
            if (
                max_events > 0
                and len(altered_paths) >= half_need
                and len(unchanged_paths) >= half_need
            ):
                altered_paths = altered_paths[:half_need]
                unchanged_paths = unchanged_paths[:half_need]
                break
    except OSError as exc:
        print(
            f"Warning: stat-based job list failed ({exc}); falling back to glob/sort.",
            flush=True,
        )
        return _build_job_list_glob(split_root, max_events)

    jobs: List[SplitJob] = []
    ia, iu = 0, 0
    while ia < len(altered_paths) or iu < len(unchanged_paths):
        if ia < len(altered_paths):
            jobs.append((altered_paths[ia], "altered", None))
            ia += 1
        if iu < len(unchanged_paths):
            jobs.append((unchanged_paths[iu], "unchanged", None))
            iu += 1
    if max_events > 0 and len(jobs) > max_events:
        jobs = jobs[:max_events]
    return jobs


def _build_job_list_benchmark(
    split_root: Path, n_altered: int, n_unchanged: int
) -> Tuple[List[SplitJob], Dict[str, float]]:
    """First ``n_altered`` / ``n_unchanged`` ``event_*.csv`` paths (lexicographic), round-robin merged.

    Uses sequential ``event_%06d.csv`` probing when possible; falls back to full-directory
    scan per side if not enough files are found within the probe bound.
    """
    altered_dir = split_root / "altered"
    unchanged_dir = split_root / "unchanged"
    timings: Dict[str, float] = {
        "benchmark_job_list_fast_path": 0.0,
        "benchmark_job_list_fallback": 0.0,
    }
    altered = _collect_benchmark_event_paths(altered_dir, max(0, n_altered), timings)
    unchanged = _collect_benchmark_event_paths(unchanged_dir, max(0, n_unchanged), timings)
    jobs: List[SplitJob] = []
    ia, iu = 0, 0
    while ia < len(altered) or iu < len(unchanged):
        if ia < len(altered):
            jobs.append((str(altered[ia].resolve()), "altered", None))
            ia += 1
        if iu < len(unchanged):
            jobs.append((str(unchanged[iu].resolve()), "unchanged", None))
            iu += 1
    return jobs, timings


def _parquet_manifest_shard_map(manifest_csv: Path) -> Dict[int, str]:
    """event_id -> absolute path to shard parquet."""
    if not manifest_csv.is_file():
        return {}
    df = pd.read_csv(manifest_csv, dtype={"event_id": "int64"})
    root = manifest_csv.parent
    out: Dict[int, str] = {}
    for _, row in df.iterrows():
        out[int(row["event_id"])] = str((root / str(row["shard"])).resolve())
    return out


def _build_job_list_parquet(
    parquet_root: Path, meta_csv: Path, max_events: int = 0
) -> List[SplitJob]:
    """Same ordering rules as ``_build_job_list`` but jobs point at Parquet shards + explicit event_id."""
    alt_map = _parquet_manifest_shard_map(parquet_root / "altered" / "manifest.csv")
    un_map = _parquet_manifest_shard_map(parquet_root / "unchanged" / "manifest.csv")
    all_ids = _unique_sorted_event_ids(meta_csv, max_events)
    altered_paths: List[Tuple[str, int]] = []
    unchanged_paths: List[Tuple[str, int]] = []
    half_need = (max_events + 1) // 2 if max_events > 0 else 0
    for i, eid in enumerate(all_ids):
        if i > 0 and i % 200_000 == 0:
            print(f"  job list scan (parquet): {i}/{len(all_ids)} event_ids", flush=True)
        eid_i = int(eid)
        if eid_i in alt_map:
            altered_paths.append((alt_map[eid_i], eid_i))
        elif eid_i in un_map:
            unchanged_paths.append((un_map[eid_i], eid_i))
        if (
            max_events > 0
            and len(altered_paths) >= half_need
            and len(unchanged_paths) >= half_need
        ):
            altered_paths = altered_paths[:half_need]
            unchanged_paths = unchanged_paths[:half_need]
            break

    jobs: List[SplitJob] = []
    ia, iu = 0, 0
    while ia < len(altered_paths) or iu < len(unchanged_paths):
        if ia < len(altered_paths):
            sp, eid = altered_paths[ia]
            jobs.append((sp, "altered", eid))
            ia += 1
        if iu < len(unchanged_paths):
            sp, eid = unchanged_paths[iu]
            jobs.append((sp, "unchanged", eid))
            iu += 1
    if max_events > 0 and len(jobs) > max_events:
        jobs = jobs[:max_events]
    return jobs


def _build_job_list_benchmark_parquet(
    parquet_root: Path, n_altered: int, n_unchanged: int
) -> Tuple[List[SplitJob], Dict[str, float]]:
    """First ``n_altered`` / ``n_unchanged`` events by ``event_id`` from each arm's manifest, round-robin."""
    timings: Dict[str, float] = {
        "benchmark_job_list_fast_path": 0.0,
        "benchmark_job_list_fallback": 0.0,
    }
    t0 = _PERF()
    altered_m = parquet_root / "altered" / "manifest.csv"
    unchanged_m = parquet_root / "unchanged" / "manifest.csv"
    altered_jobs: List[SplitJob] = []
    unchanged_jobs: List[SplitJob] = []
    if altered_m.is_file() and n_altered > 0:
        adf = pd.read_csv(altered_m, dtype={"event_id": "int64"})
        adf = adf.sort_values("event_id").head(int(n_altered))
        for _, row in adf.iterrows():
            sp = (parquet_root / "altered" / str(row["shard"])).resolve()
            altered_jobs.append((str(sp), "altered", int(row["event_id"])))
    if unchanged_m.is_file() and n_unchanged > 0:
        udf = pd.read_csv(unchanged_m, dtype={"event_id": "int64"})
        udf = udf.sort_values("event_id").head(int(n_unchanged))
        for _, row in udf.iterrows():
            sp = (parquet_root / "unchanged" / str(row["shard"])).resolve()
            unchanged_jobs.append((str(sp), "unchanged", int(row["event_id"])))
    jobs: List[SplitJob] = []
    ia, iu = 0, 0
    while ia < len(altered_jobs) or iu < len(unchanged_jobs):
        if ia < len(altered_jobs):
            jobs.append(altered_jobs[ia])
            ia += 1
        if iu < len(unchanged_jobs):
            jobs.append(unchanged_jobs[iu])
            iu += 1
    timings["benchmark_job_list_fast_path"] = _PERF() - t0
    return jobs, timings


def _event_id_from_csv_path(path_str: str) -> int:
    stem = Path(path_str).stem
    return int(stem.split("_")[1])


def _event_id_from_job(job: SplitJob) -> int:
    _path_str, _sample, peid = job
    if peid is not None:
        return int(peid)
    return _event_id_from_csv_path(_path_str)


def _drain_progress_queue(progress_q: Any, total: int, mininterval: float = 0.3) -> None:
    """Consume one message per finished event and drive tqdm (stderr)."""
    try:
        from tqdm import tqdm
    except ImportError:
        for _ in range(total):
            progress_q.get()
        return
    with tqdm(
        total=total,
        unit="evt",
        mininterval=mininterval,
        file=sys.stderr,
        dynamic_ncols=True,
    ) as pbar:
        for _ in range(total):
            progress_q.get()
            pbar.update(1)


def _metadata_subset_fingerprint(
    data_dir: Path, meta_csv: Path, event_ids: set[int], extra: str = ""
) -> str:
    """Stable id for caching ``_metadata_subset.csv`` for a given job set."""
    key = (
        f"{data_dir.resolve()}|{meta_csv.resolve()}|{extra}|"
        + ",".join(str(e) for e in sorted(event_ids))
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _write_metadata_subset(
    full_meta: Path, event_ids: set[int], out_path: Path, chunk_rows: int = 250_000
) -> int:
    """Scan ``full_meta`` once (chunked), keep rows whose ``event_id`` is in ``event_ids``."""
    usecols = ["event_id", "struck_outgoing_index", "struck_incoming_index"]
    dtype = {
        "event_id": "int64",
        "struck_outgoing_index": "int64",
        "struck_incoming_index": "int64",
    }
    chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        full_meta, usecols=usecols, dtype=dtype, chunksize=chunk_rows
    ):
        sub = chunk[chunk["event_id"].isin(event_ids)]
        if len(sub):
            chunks.append(sub)
    if not chunks:
        raise SystemExit(
            "No metadata rows matched job event_ids (empty subset). Check metadata CSV vs split."
        )
    merged = pd.concat(chunks, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return int(len(merged))


def _worker_run(payload: Tuple[int, List[SplitJob], Dict[str, Any]]) -> Path:
    worker_id, jobs, cfg = payload
    if int(cfg.get("nice_increment", 0)) > 0:
        try:
            os.nice(int(cfg["nice_increment"]))
        except OSError:
            pass

    altered_dir = Path(cfg["altered_meta_dir"])
    md_map: Optional[pd.DataFrame] = None
    mpath = Path(cfg["metadata_csv"])
    if mpath.is_file():
        md = pd.read_csv(
            mpath,
            usecols=["event_id", "struck_outgoing_index", "struck_incoming_index"],
            dtype={
                "event_id": "int64",
                "struck_outgoing_index": "int64",
                "struck_incoming_index": "int64",
            },
        ).sort_values("event_id")
        md_map = md.set_index("event_id")

    csv_momenta_breit = bool(cfg["csv_momenta_breit"])
    verbose = bool(cfg.get("verbose", False))
    out_path = Path(cfg["out_path"])
    out_dir = out_path.parent
    profile_timing = bool(cfg.get("profile_timing", False))
    profile_slow_n = int(cfg.get("profile_slow_events", 0) or 0)
    p = build_pythia_reinjector()
    rows: List[Dict[str, Any]] = []
    progress_q = cfg.get("progress_q")
    timing_acc = _timing_acc_init() if profile_timing else None
    slow_heap: List[Tuple[float, Dict[str, Any]]] = []

    def _push_slow(total_ev: float, rec: Dict[str, Any], tmap: Dict[str, float]) -> None:
        if profile_slow_n <= 0:
            return
        fail = rec.get("failure_reason") or rec.get("reason") or ""
        slow_rec: Dict[str, Any] = {
            "event_id": int(rec.get("event_id", -1)),
            "sample": str(rec.get("sample", "")),
            "csv_path": str(rec.get("csv_path", "")),
            "total_event_seconds": float(total_ev),
            "pythia_ok": bool(rec.get("pythia_ok", False)),
            "failure_reason": fail,
        }
        for k in _TIMING_STAGE_KEYS:
            slow_rec[k] = float(tmap.get(k, 0.0))
        for k in _TIMING_CSV_SUB_KEYS:
            slow_rec[k] = float(tmap.get(k, 0.0))
        for k in _TIMING_KIN_SUB_KEYS:
            slow_rec[k] = float(tmap.get(k, 0.0))
        if len(slow_heap) < profile_slow_n:
            heapq.heappush(slow_heap, (total_ev, slow_rec))
        elif total_ev > slow_heap[0][0]:
            heapq.heapreplace(slow_heap, (total_ev, slow_rec))

    def _emit_loaded_event(
        ji: int,
        job: SplitJob,
        df: pd.DataFrame,
        event_id: int,
        meta_stem: str,
        trace_path: str,
        meta_path: Optional[Path],
        timings: Optional[Dict[str, float]],
        t_evt0: float,
    ) -> None:
        _path_str, sample, _parquet_eid = job
        if profile_timing and timings is not None:
            t0 = _PERF()
        ev0 = int(df["event_id"].iloc[0])
        idx, _ = resolve_struck_index(df, ev0, meta_path, md_map)
        if profile_timing and timings is not None:
            timings["struck_resolution"] = _PERF() - t0

        if idx is None:
            rec = {
                "event_id": event_id,
                "sample": sample,
                "worker_id": worker_id,
                "pythia_ok": False,
                "ok": False,
                "reason": "struck_unresolved",
                "failure_reason": "struck_unresolved",
                "csv_path": trace_path,
            }
            if profile_timing and timings is not None:
                timings["total_event"] = _PERF() - t_evt0
                _timing_acc_add_event(timing_acc, timings, False)
                _push_slow(timings["total_event"], rec, timings)
            rows.append(rec)
            return

        rec = extract_one_event(
            df,
            idx,
            sample,
            worker_id,
            p,
            csv_momenta_breit,
            timings=timings,
        )
        rec["csv_path"] = trace_path
        if profile_timing and timings is not None:
            timings["total_event"] = _PERF() - t_evt0
            _timing_acc_add_event(timing_acc, timings, bool(rec.get("ok", False)))
            _push_slow(timings["total_event"], rec, timings)
        rows.append(rec)

    use_parquet_shard = (
        str(cfg.get("parquet_read_mode", "shard")) == "shard"
        and bool(jobs)
        and all(j[2] is not None for j in jobs)
    )

    if use_parquet_shard:
        # Per-shard event counts for amortizing read/group time; global job order preserved
        # so PYTHIA sees the same event sequence as the filtered path.
        shard_nevents: Dict[str, int] = {}
        for _p, _s, peid in jobs:
            if peid is not None:
                shard_nevents[_p] = shard_nevents.get(_p, 0) + 1
        shard_remaining: Counter[str] = Counter(shard_nevents)
        cache_df: Dict[str, pd.DataFrame] = {}
        cache_idx: Dict[str, Dict[int, Any]] = {}
        shard_amort: Dict[str, Tuple[float, float]] = {}

        for ji, job in enumerate(jobs):
            try:
                path_str, sample, parquet_eid = job
                assert parquet_eid is not None
                event_id = int(parquet_eid)
                meta_stem = f"event_{event_id:06d}"
                trace_path = f"{path_str}#event_id={event_id}"
                meta_path: Optional[Path] = None
                if sample == "altered":
                    cand = altered_dir / f"{meta_stem}.meta.json"
                    if cand.is_file():
                        meta_path = cand
                if verbose:
                    print(
                        f"worker {worker_id}: event {ji + 1}/{len(jobs)} {sample} {trace_path}",
                        flush=True,
                    )
                t_evt0 = _PERF()
                timings: Optional[Dict[str, float]] = None
                if profile_timing:
                    timings = {k: 0.0 for k in _TIMING_BUCKET_KEYS}
                    for _k in _TIMING_CSV_SUB_KEYS:
                        timings[_k] = 0.0
                    for _k in _TIMING_KIN_SUB_KEYS:
                        timings[_k] = 0.0
                    timings["total_event"] = 0.0
                    timings["csv_read"] = 0.0
                    timings["file_open_read"] = 0.0
                    timings["pandas_parse"] = 0.0

                try:
                    if path_str not in cache_df:
                        sp = Path(path_str)
                        nsh = shard_nevents[path_str]
                        if profile_timing:
                            t_r0 = _PERF()
                            cache_df[path_str] = pd.read_parquet(
                                sp,
                                engine="pyarrow",
                                columns=list(_EVENT_CSV_USECOLS),
                            )
                            read_dt = _PERF() - t_r0
                            t_g0 = _PERF()
                            raw_idx = cache_df[path_str].groupby(
                                "event_id", sort=False
                            ).indices
                            cache_idx[path_str] = {
                                int(k): v for k, v in raw_idx.items()
                            }
                            del raw_idx
                            group_dt = _PERF() - t_g0
                            shard_amort[path_str] = (read_dt / nsh, group_dt / nsh)
                        else:
                            cache_df[path_str] = pd.read_parquet(
                                sp,
                                engine="pyarrow",
                                columns=list(_EVENT_CSV_USECOLS),
                            )
                            raw_idx = cache_df[path_str].groupby(
                                "event_id", sort=False
                            ).indices
                            cache_idx[path_str] = {
                                int(k): v for k, v in raw_idx.items()
                            }
                            del raw_idx
                            shard_amort[path_str] = (0.0, 0.0)

                    amort_read, amort_group = shard_amort[path_str]
                    if profile_timing and timings is not None:
                        timings["parquet_read"] = amort_read
                        timings["parquet_group_or_index_build"] = amort_group

                    df_full = cache_df[path_str]
                    idx_by_eid = cache_idx[path_str]
                    pos = idx_by_eid.get(event_id)
                    if pos is None or len(pos) == 0:
                        rec = {
                            "event_id": event_id,
                            "sample": sample,
                            "worker_id": worker_id,
                            "pythia_ok": False,
                            "ok": False,
                            "reason": "parquet_missing_event_id",
                            "failure_reason": "parquet_missing_event_id",
                            "csv_path": trace_path,
                        }
                        if profile_timing and timings is not None:
                            timings["event_extract_from_shard"] = 0.0
                            timings["total_event"] = _PERF() - t_evt0
                            _timing_acc_add_event(timing_acc, timings, False)
                            _push_slow(timings["total_event"], rec, timings)
                        rows.append(rec)
                    else:
                        if profile_timing and timings is not None:
                            t_ex0 = _PERF()
                            df_ev = df_full.iloc[pos].copy()
                            timings["event_extract_from_shard"] = _PERF() - t_ex0
                        else:
                            df_ev = df_full.iloc[pos].copy()

                        _emit_loaded_event(
                            ji,
                            job,
                            df_ev,
                            event_id,
                            meta_stem,
                            trace_path,
                            meta_path,
                            timings,
                            t_evt0,
                        )
                finally:
                    shard_remaining[path_str] -= 1
                    if shard_remaining[path_str] <= 0:
                        cache_df.pop(path_str, None)
                        cache_idx.pop(path_str, None)
                        shard_amort.pop(path_str, None)
            finally:
                if progress_q is not None:
                    progress_q.put(1)
    else:
        for ji, job in enumerate(jobs):
            try:
                path_str, sample, parquet_eid = job
                if verbose:
                    print(
                        f"worker {worker_id}: event {ji + 1}/{len(jobs)} {sample} {path_str}",
                        flush=True,
                    )
                t_evt0 = _PERF()
                timings: Optional[Dict[str, float]] = None
                if profile_timing:
                    timings = {k: 0.0 for k in _TIMING_BUCKET_KEYS}
                    for _k in _TIMING_CSV_SUB_KEYS:
                        timings[_k] = 0.0
                    for _k in _TIMING_KIN_SUB_KEYS:
                        timings[_k] = 0.0
                    timings["total_event"] = 0.0

                csv_path = Path(path_str)
                if parquet_eid is not None:
                    event_id = int(parquet_eid)
                    meta_stem = f"event_{event_id:06d}"
                else:
                    meta_stem = csv_path.stem
                    try:
                        event_id = int(meta_stem.split("_")[1])
                    except (IndexError, ValueError):
                        rec = {
                            "event_id": -1,
                            "sample": sample,
                            "worker_id": worker_id,
                            "pythia_ok": False,
                            "ok": False,
                            "reason": "bad_filename",
                            "failure_reason": "bad_filename",
                            "csv_path": path_str,
                        }
                        if profile_timing and timings is not None:
                            timings["total_event"] = _PERF() - t_evt0
                            _timing_acc_add_event(timing_acc, timings, False)
                            _push_slow(timings["total_event"], rec, timings)
                        rows.append(rec)
                        continue

                meta_path: Optional[Path] = None
                if sample == "altered":
                    cand = altered_dir / f"{meta_stem}.meta.json"
                    if cand.is_file():
                        meta_path = cand

                trace_path = (
                    path_str if parquet_eid is None else f"{path_str}#event_id={event_id}"
                )

                if parquet_eid is not None:
                    if profile_timing and timings is not None:
                        t0 = _PERF()
                        df = _read_event_parquet(csv_path, event_id)
                        t1 = _PERF()
                        timings["parquet_read"] = t1 - t0
                        timings["file_open_read"] = 0.0
                        timings["pandas_parse"] = 0.0
                        timings["csv_read"] = 0.0
                        timings["parquet_group_or_index_build"] = 0.0
                        timings["event_extract_from_shard"] = 0.0
                    else:
                        df = _read_event_parquet(csv_path, event_id)
                elif profile_timing and timings is not None:
                    t0 = _PERF()
                    raw = csv_path.read_bytes()
                    t1 = _PERF()
                    df = _read_event_csv(io.BytesIO(raw))
                    t2 = _PERF()
                    timings["file_open_read"] = t1 - t0
                    timings["pandas_parse"] = t2 - t1
                    timings["csv_read"] = t2 - t0
                    timings["parquet_read"] = 0.0
                    timings["parquet_group_or_index_build"] = 0.0
                    timings["event_extract_from_shard"] = 0.0
                else:
                    df = _read_event_csv(csv_path)

                _emit_loaded_event(
                    ji,
                    job,
                    df,
                    event_id,
                    meta_stem,
                    trace_path,
                    meta_path,
                    timings,
                    t_evt0,
                )
            finally:
                if progress_q is not None:
                    progress_q.put(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    if profile_timing and timing_acc is not None:
        _write_timing_worker_json(out_dir, worker_id, timing_acc)
    if profile_slow_n > 0 and slow_heap:
        slow_sorted = sorted(slow_heap, key=lambda x: -x[0])
        slow_payload = {
            "worker_id": worker_id,
            "events": [r for _, r in slow_sorted],
        }
        (out_dir / f"slow_events_worker_{worker_id:02d}.json").write_text(
            json.dumps(slow_payload, indent=2) + "\n", encoding="utf-8"
        )
    return out_path


def main() -> None:
    default_data = outputs_dir() / "dis_isr_parton_dataset"
    ap = argparse.ArgumentParser(
        description="Kinematics-only extraction: all altered + unchanged split CSVs, 5 workers, no plots."
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=default_data,
        help=f"Directory containing split_90_10/ and dis_isr_event_metadata.csv (default: {default_data})",
    )
    ap.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Override path to dis_isr_event_metadata.csv (default: <data-dir>/dis_isr_event_metadata.csv)",
    )
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-dir>/kinematics_extract_full)",
    )
    ap.add_argument("--csv-momenta-frame", choices=("lab", "breit"), default="breit")
    ap.add_argument("--nice-increment", type=int, default=5)
    ap.add_argument("--max-events", type=int, default=0, help="If >0, only first N jobs (debug).")
    ap.add_argument(
        "--bench-unchanged",
        type=int,
        default=0,
        metavar="N",
        help="With --bench-altered, build a benchmark list of N smallest unchanged + M smallest altered (round-robin).",
    )
    ap.add_argument(
        "--bench-altered",
        type=int,
        default=0,
        metavar="M",
        help="With --bench-unchanged, see --bench-unchanged.",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm event-level progress bar.",
    )
    ap.add_argument(
        "--force-progress",
        action="store_true",
        help="Show the progress bar even when stderr is not a TTY.",
    )
    ap.add_argument(
        "--format",
        choices=("csv", "parquet"),
        default="csv",
        help="Merged output format (workers always write CSV shards).",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Log each event path as workers process (noisy; for debugging).",
    )
    ap.add_argument(
        "--profile-timing",
        action="store_true",
        help=(
            "Per-event wall times (perf_counter): write timing_worker_WW.json per worker "
            "and merge to timing_summary.json."
        ),
    )
    ap.add_argument(
        "--profile-slow-events",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Keep the N slowest events by total_event wall time; workers write "
            "slow_events_worker_WW.json, merged to slow_events.json."
        ),
    )
    ap.add_argument(
        "--input-format",
        choices=("csv", "parquet", "auto"),
        default="csv",
        help=(
            "Event source: per-event CSVs under split_90_10, or Parquet shards + manifests "
            "under --parquet-split-root. 'auto' uses Parquet when both arm manifests exist."
        ),
    )
    ap.add_argument(
        "--parquet-split-root",
        type=Path,
        default=None,
        help="Parquet dataset root (default: <data-dir>/split_90_10_parquet).",
    )
    ap.add_argument(
        "--parquet-read-mode",
        choices=("shard", "filtered"),
        default="shard",
        help=(
            "Parquet only: 'shard' reads each shard once then slices by event_id in memory "
            "(default). 'filtered' uses one filtered Parquet read per event (debug / baseline)."
        ),
    )
    args = ap.parse_args()
    if int(args.profile_slow_events) > 0 and not bool(args.profile_timing):
        print(
            "Note: --profile-slow-events implies --profile-timing; enabling timing collection.",
            flush=True,
        )
        args.profile_timing = True

    data_dir = args.data_dir.resolve()
    split_root = data_dir / "split_90_10"
    parquet_root = (args.parquet_split_root or (data_dir / "split_90_10_parquet")).resolve()
    meta_csv = (args.metadata_csv or (data_dir / "dis_isr_event_metadata.csv")).resolve()
    altered_dir = split_root / "altered"
    unchanged_dir = split_root / "unchanged"

    if args.input_format == "auto":
        p_alt_m = parquet_root / "altered" / "manifest.csv"
        p_un_m = parquet_root / "unchanged" / "manifest.csv"
        input_eff = "parquet" if (p_alt_m.is_file() and p_un_m.is_file()) else "csv"
        print(f"input-format auto -> {input_eff}", flush=True)
    else:
        input_eff = str(args.input_format)

    if not meta_csv.is_file():
        raise SystemExit(f"metadata CSV not found: {meta_csv}")

    if input_eff == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:
            raise SystemExit("Parquet input requires pyarrow (pip install pyarrow).") from exc
        if not (parquet_root / "altered").is_dir() or not (parquet_root / "unchanged").is_dir():
            raise SystemExit(
                f"Parquet input requires {parquet_root}/altered and {parquet_root}/unchanged."
            )
        for rel in ("altered/manifest.csv", "unchanged/manifest.csv"):
            mf = parquet_root / rel
            if not mf.is_file():
                raise SystemExit(f"Parquet manifest not found: {mf}")
        if not altered_dir.is_dir():
            raise SystemExit(
                f"Parquet mode still needs split_90_10/altered/ for .meta.json sidecars: {altered_dir}"
            )
    else:
        if not altered_dir.is_dir():
            raise SystemExit(f"altered dir not found: {altered_dir}")
        if not unchanged_dir.is_dir():
            raise SystemExit(f"unchanged dir not found: {unchanged_dir}")

    t_wall0 = _PERF()

    t_job0 = _PERF()
    bench_job_list_timings: Dict[str, float] = {
        "benchmark_job_list_fast_path": 0.0,
        "benchmark_job_list_fallback": 0.0,
    }
    if args.bench_unchanged > 0 or args.bench_altered > 0:
        if args.bench_unchanged <= 0 or args.bench_altered <= 0:
            raise SystemExit(
                "Benchmark mode requires both --bench-unchanged N and --bench-altered M with N,M > 0."
            )
        if args.max_events > 0:
            print("Note: --max-events ignored when --bench-* is set.", flush=True)
        if input_eff == "parquet":
            jobs, bench_job_list_timings = _build_job_list_benchmark_parquet(
                parquet_root, args.bench_altered, args.bench_unchanged
            )
        else:
            jobs, bench_job_list_timings = _build_job_list_benchmark(
                split_root, args.bench_altered, args.bench_unchanged
            )
    else:
        if input_eff == "parquet":
            jobs = _build_job_list_parquet(parquet_root, meta_csv, args.max_events)
        else:
            jobs = _build_job_list(split_root, meta_csv, args.max_events)
    job_list_build_seconds = _PERF() - t_job0

    n_w = max(1, int(args.workers))
    if not jobs:
        raise SystemExit("No jobs built (check split CSVs / Parquet manifests vs metadata).")

    idx_chunks = np.array_split(np.arange(len(jobs), dtype=np.int64), n_w)
    out_dir = (args.out_dir or (data_dir / "kinematics_extract_full")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    subset_path = out_dir / "_metadata_subset.csv"
    fp_path = out_dir / "_metadata_subset.fingerprint"
    t_meta0 = _PERF()
    job_event_ids: set[int] = set()
    for job in jobs:
        try:
            job_event_ids.add(_event_id_from_job(job))
        except (IndexError, ValueError):
            continue
    fp_extra = (
        f"input={input_eff}|pqroot={parquet_root if input_eff == 'parquet' else ''}"
        f"|pqread={args.parquet_read_mode if input_eff == 'parquet' else ''}"
    )
    meta_fp = _metadata_subset_fingerprint(data_dir, meta_csv, job_event_ids, extra=fp_extra)
    metadata_subset_cache_hit = (
        subset_path.is_file()
        and fp_path.is_file()
        and fp_path.read_text(encoding="utf-8").strip() == meta_fp
    )
    if metadata_subset_cache_hit:
        n_meta = int(len(pd.read_csv(subset_path, usecols=["event_id"])))
        print(
            f"metadata_subset: reused cache rows={n_meta} path={subset_path}",
            flush=True,
        )
    else:
        n_meta = _write_metadata_subset(meta_csv, job_event_ids, subset_path)
        fp_path.write_text(meta_fp + "\n", encoding="utf-8")
        print(f"metadata_subset: rows={n_meta} path={subset_path}", flush=True)
    metadata_subset_build_seconds = _PERF() - t_meta0

    csv_momenta_breit = args.csv_momenta_frame == "breit"
    cfg_base: Dict[str, Any] = {
        "altered_meta_dir": str(altered_dir),
        "metadata_csv": str(subset_path),
        "csv_momenta_breit": csv_momenta_breit,
        "nice_increment": int(args.nice_increment),
        "verbose": bool(args.verbose),
        "profile_timing": bool(args.profile_timing),
        "profile_slow_events": int(args.profile_slow_events),
        "input_format": input_eff,
        "parquet_read_mode": (
            str(args.parquet_read_mode) if input_eff == "parquet" else "n/a"
        ),
    }

    payloads: List[Tuple[int, List[SplitJob], Dict[str, Any]]] = []
    for w in range(n_w):
        ix = idx_chunks[w]
        if len(ix) == 0:
            continue
        chunk_jobs = [jobs[int(i)] for i in ix]
        cfg = {**cfg_base, "out_path": str(out_dir / f"kinematics_worker_{w:02d}.csv")}
        payloads.append((w, chunk_jobs, cfg))

    print(
        f"data_dir={data_dir}\n"
        f"input_format={input_eff}\n"
        f"split_root={split_root}\n"
        + (
            f"parquet_split_root={parquet_root}\n"
            f"parquet_read_mode={args.parquet_read_mode}\n"
            if input_eff == "parquet"
            else ""
        )
        + f"metadata={meta_csv}\n"
        f"total_jobs={len(jobs)}  workers={n_w}  non_empty_shards={len(payloads)}\n"
        f"out_dir={out_dir}"
    )

    want_progress = not args.no_progress and (
        args.force_progress or sys.stderr.isatty()
    )
    tqdm_ok = False
    if want_progress:
        try:
            import tqdm as _tqdm_check  # noqa: F401

            tqdm_ok = True
        except ImportError:
            print(
                "Install tqdm for a progress bar: pip install tqdm",
                flush=True,
            )

    ctx = mp.get_context("spawn")
    progress_thread: Optional[threading.Thread] = None
    if want_progress and tqdm_ok and len(jobs) > 0:
        pq = ctx.Queue()
        cfg_base["progress_q"] = pq
        payloads = [
            (wid, cj, {**cfg_base, "out_path": str(out_dir / f"kinematics_worker_{wid:02d}.csv")})
            for wid, cj, _ in payloads
        ]
        progress_thread = threading.Thread(
            target=_drain_progress_queue,
            args=(pq, len(jobs)),
            daemon=True,
        )
        progress_thread.start()

    t_worker0 = _PERF()
    try:
        if len(payloads) == 1:
            paths = [_worker_run(payloads[0])]
        else:
            with ctx.Pool(processes=min(n_w, len(payloads))) as pool:
                paths = pool.map(_worker_run, payloads)
    finally:
        if progress_thread is not None:
            progress_thread.join()
    worker_processing_seconds = _PERF() - t_worker0

    if args.profile_timing:
        summary = merge_timing_reports(out_dir)
        if summary is not None:
            n = int(summary["n_events_seen"])
            ts = float(summary["total_seconds"])
            print("=== timing summary (merged) ===", flush=True)
            print(f"total events processed: {n}", flush=True)
            print(f"total wall time tracked (sum of per-event total): {ts:.6f} s", flush=True)
            print(
                f"sum of stage buckets: {float(summary['tracked_stage_seconds']):.6f} s",
                flush=True,
            )
            print("percent of time in (vs sum of stage buckets):", flush=True)
            for k in _TIMING_STAGE_KEYS:
                pct = float(summary[f"{k}_pct_of_stages"])
                av = float(summary[f"{k}_avg_seconds"])
                print(f"  {k}: {pct:.2f}%  (avg {av:.6e} s/event)", flush=True)
            print(
                f"  total_event avg: {float(summary['total_event_avg_seconds']):.6e} s/event",
                flush=True,
            )
            ts_loop = float(summary["tracked_stage_seconds"])
            if float(summary.get("csv_read_seconds", 0.0)) > 0.0:
                print("  csv_read breakdown (% of csv_read wall time):", flush=True)
                for k in _TIMING_CSV_SUB_KEYS:
                    print(
                        f"    {k}: {float(summary[f'{k}_pct_of_csv_read']):.2f}%  "
                        f"(avg {float(summary[f'{k}_avg_seconds']):.6e} s/event)",
                        flush=True,
                    )
            if float(summary.get("parquet_read_seconds", 0.0)) > 0.0:
                pr = float(summary["parquet_read_seconds"])
                print(
                    f"  parquet_read: {100.0 * pr / ts_loop:.2f}% of tracked loop  "
                    f"({pr:.6f} s total, avg {pr / n:.6e} s/event)",
                    flush=True,
                )
            if float(summary.get("parquet_group_or_index_build_seconds", 0.0)) > 0.0:
                pg = float(summary["parquet_group_or_index_build_seconds"])
                print(
                    f"  parquet_group_or_index_build: {100.0 * pg / ts_loop:.2f}% of tracked loop  "
                    f"({pg:.6f} s total, avg {pg / n:.6e} s/event)",
                    flush=True,
                )
            if float(summary.get("event_extract_from_shard_seconds", 0.0)) > 0.0:
                ex = float(summary["event_extract_from_shard_seconds"])
                print(
                    f"  event_extract_from_shard: {100.0 * ex / ts_loop:.2f}% of tracked loop  "
                    f"({ex:.6f} s total, avg {ex / n:.6e} s/event)",
                    flush=True,
                )
            ks_tot = float(summary["kinematics_setup_seconds"])
            kin_sum = sum(float(summary[f"{k}_seconds"]) for k in _TIMING_KIN_SUB_KEYS)
            print(
                "kinematics_setup sub-buckets (seconds, % of tracked loop, avg s/event):",
                flush=True,
            )
            print(
                f"  kinematics_setup (sum of subs): {kin_sum:.6f} s  "
                f"({100.0 * kin_sum / ts_loop:.2f}% of loop)  "
                f"(avg {kin_sum / n:.6e} s/event)",
                flush=True,
            )
            print(
                f"  kinematics_setup (top-level timer): {ks_tot:.6f} s  "
                f"({100.0 * ks_tot / ts_loop:.2f}% of loop)  "
                f"(avg {ks_tot / n:.6e} s/event)",
                flush=True,
            )
            for k in _TIMING_KIN_SUB_KEYS:
                sk = f"{k}_seconds"
                sec = float(summary[sk])
                pct = float(summary[f"{k}_pct_of_tracked_loop"])
                av = float(summary[f"{k}_avg_seconds"])
                print(
                    f"  {k}: {sec:.6f} s  ({pct:.2f}% of loop)  (avg {av:.6e} s/event)",
                    flush=True,
                )
            if ks_tot > 0.0:
                print(
                    f"  (sanity: sum(subs)/kinematics_setup = {100.0 * kin_sum / ks_tot:.2f}%)",
                    flush=True,
                )
            best_ks_sub = max(
                _TIMING_KIN_SUB_KEYS,
                key=lambda kk: float(summary[f"{kk}_seconds"]),
            )
            print(
                f"kinematics_setup hotspot (largest sub-bucket): {best_ks_sub} "
                f"({float(summary[f'{best_ks_sub}_pct_of_tracked_loop']):.2f}% of tracked loop)",
                flush=True,
            )
            best_k = max(
                _TIMING_STAGE_KEYS,
                key=lambda kk: float(summary[f"{kk}_seconds"]),
            )
            print(
                f"bottleneck (largest top-level stage): {best_k} "
                f"({float(summary[f'{best_k}_pct_of_stages']):.2f}% of stage time)",
                flush=True,
            )
        merge_slow_event_reports(out_dir, int(args.profile_slow_events))
        if int(args.profile_slow_events) > 0:
            sp = out_dir / "slow_events.json"
            if sp.is_file():
                slow_data = json.loads(sp.read_text(encoding="utf-8"))
                evs = slow_data.get("events", [])
                print("=== top 10 slowest events (by total_event_seconds) ===", flush=True)
                for i, ev in enumerate(evs[:10], 1):
                    eid = ev.get("event_id")
                    tot = float(ev.get("total_event_seconds", 0.0))
                    print(
                        f"  {i}. event_id={eid} sample={ev.get('sample')} "
                        f"total_event={tot:.6f}s pythia_ok={ev.get('pythia_ok')}",
                        flush=True,
                    )

    t_merge0 = _PERF()
    if len(paths) == 1:
        merged = pd.read_csv(paths[0])
    else:
        parts = [pd.read_csv(p) for p in paths]
        merged = pd.concat(parts, ignore_index=True)
    merged = merged.sort_values(["sample", "event_id"], kind="mergesort")
    if args.format == "parquet":
        merged_path = out_dir / "kinematics_merged.parquet"
        try:
            merged.to_parquet(merged_path, index=False)
        except ImportError as exc:
            raise SystemExit(
                "Parquet output requires pyarrow (pip install pyarrow). Use --format csv or install pyarrow."
            ) from exc
        print(f"Merged: {merged_path}")
    else:
        merged_path = out_dir / "kinematics_merged.csv"
        merged.to_csv(merged_path, index=False)
        print(f"Merged: {merged_path}")
    merge_outputs_seconds = _PERF() - t_merge0
    print(f"Worker shards: {paths}")

    t_wall1 = _PERF()
    total_process_wall_seconds = t_wall1 - t_wall0
    process_payload = {
        "total_process_wall_seconds": float(total_process_wall_seconds),
        "job_list_build_seconds": float(job_list_build_seconds),
        "benchmark_job_list_fast_path_seconds": float(
            bench_job_list_timings.get("benchmark_job_list_fast_path", 0.0)
        ),
        "benchmark_job_list_fallback_seconds": float(
            bench_job_list_timings.get("benchmark_job_list_fallback", 0.0)
        ),
        "metadata_subset_build_seconds": float(metadata_subset_build_seconds),
        "metadata_subset_cache_hit": bool(metadata_subset_cache_hit),
        "worker_processing_seconds": float(worker_processing_seconds),
        "merge_outputs_seconds": float(merge_outputs_seconds),
        "n_jobs": int(len(jobs)),
        "n_workers": int(n_w),
        "input_format": input_eff,
        "parquet_split_root": str(parquet_root) if input_eff == "parquet" else "",
        "parquet_read_mode": (
            str(args.parquet_read_mode) if input_eff == "parquet" else ""
        ),
    }
    (out_dir / "process_timing_summary.json").write_text(
        json.dumps(process_payload, indent=2) + "\n", encoding="utf-8"
    )
    print("=== process wall time (main) ===", flush=True)
    print(f"total_process_wall: {total_process_wall_seconds:.6f} s", flush=True)
    print(f"job_list_build: {job_list_build_seconds:.6f} s", flush=True)
    if int(args.bench_unchanged) > 0 and int(args.bench_altered) > 0:
        print(
            f"  benchmark_job_list_fast_path: "
            f"{float(bench_job_list_timings.get('benchmark_job_list_fast_path', 0.0)):.6f} s",
            flush=True,
        )
        print(
            f"  benchmark_job_list_fallback: "
            f"{float(bench_job_list_timings.get('benchmark_job_list_fallback', 0.0)):.6f} s",
            flush=True,
        )
    print(f"metadata_subset_build: {metadata_subset_build_seconds:.6f} s", flush=True)
    print(f"worker_processing: {worker_processing_seconds:.6f} s", flush=True)
    print(f"merge_outputs: {merge_outputs_seconds:.6f} s", flush=True)


if __name__ == "__main__":
    main()
