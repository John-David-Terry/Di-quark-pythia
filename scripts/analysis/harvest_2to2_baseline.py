#!/usr/bin/env python3
"""
Harvest a 2->2 DIS baseline library for later 2->4 studies.

High-level overview
--------------------
Goal
  Build a baseline dataset of realistic 2->2 DIS event kinematics (struck-u and struck-d)
  generated with PYTHIA using the same beam and Q2Min phase space as our shard generator.

For each accepted event (accepted means: we can robustly identify the scattered electron
and the outgoing hard struck quark, and we classify the struck flavor as u or d),
we store:
  1) scattered final-state electron 4-momentum and derived variables
  2) outgoing hard struck quark 4-momentum and derived variables
  3) hardest forward down quark ("forward" defined by eta > eta_forward_cut) and derived variables
     - main selector: largest pplus = E + pz
     - secondary selector: largest pT (stored for later cross-check plots)
  4) event-level DIS kinematics: Q2, xB, y, W2 when computable

Files produced
  outputs/baseline_2to2/
    struck_u_baseline.csv
    struck_d_baseline.csv
    combined_baseline.csv
    summary.json
    README_summary.md

Low-level execution plan (what this script actually does)
------------------------------------------------------------
1) Initialize PYTHIA with the requested DIS setup:
     - Beams:idA=11, idB=2212, eA=18, eB=275, frameType=2
     - WeakBosonExchange:ff2ff(t:gmZ)=on
     - PDF:lepton=off
     - PhaseSpace:Q2Min=16
     - PartonLevel:ISR=on, FSR=on, MPI=off
     - HadronLevel:all=on
     - ColourReconnection:reconnect=off
     - HardQCD:all=off
2) Loop over PYTHIA events until we reach targets for accepted struck-u and struck-d.
3) For each event, identify the scattered electron:
     - Prefer status==44; else choose the highest-energy electron with status>0.
4) Identify incoming struck quark (hard-process tag):
     - status==-21 and abs(id) in {1..5}.
     - If multiple exist, choose the one with the largest energy and count a diagnostic.
5) Identify outgoing hard struck quark:
     - Candidate quarks with abs(id)==abs(incoming_id) whose ancestry chain contains
       the incoming struck-quark index.
     - Priority: abs(status)==23 first (choose max pplus), else 63<=abs(status)<=68 (max pplus),
       else fall back to max E.
6) Identify the hardest forward d-quark(s):
     - Consider only particles with id==1 (d-quark) and apply forward cut eta>eta_forward_cut.
     - Candidate status tiers: abs(status)==23; else 63<=abs(status)<=68; else status!=0.
     - Choose by largest pplus (main) and by largest pT (secondary).
     - If no forward d candidates exist, store null/NaN values and has_forward_d=0.
7) Compute event DIS kinematics (Q2, xB, y, W2) from incoming/outgoing electron and incoming proton.
8) Write rows incrementally to CSV in chunks to avoid losing progress.
9) At the end, write combined_baseline.csv and summary.json + README_summary.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pythia8


import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

DEFAULT_OUTDIR = outputs_dir() / "baseline_2to2"

QUARK_ABS_IDS = {1, 2, 3, 4, 5}
TARGET_FLAVORS = {"u", "d"}


def p4_dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Minkowski dot product with (+,-,-,-) signature: a.E*b.E - a.p*b.p"""
    return float(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3])


def pT_from(px: float, py: float) -> float:
    return math.hypot(px, py)


def eta_from(E: float, pz: float, pt: float, eps: float = 1e-12) -> Optional[float]:
    # eta = 0.5 log((E+pz)/(E-pz))
    if pt < eps:
        # If pt is ~0, eta is ill-defined; treat as None to avoid blowing up.
        return None
    denom = E - pz
    numer = E + pz
    if denom <= 0 or numer <= 0:
        return None
    return 0.5 * math.log(numer / denom)


def phi_from(px: float, py: float) -> float:
    return math.atan2(py, px)


def safe_float(x: Optional[float]) -> float:
    return float("nan") if x is None else float(x)


def pplus_pminus(E: float, pz: float) -> Tuple[float, float]:
    return (E + pz, E - pz)


@dataclass
class DisKinematics:
    Q2: float
    xB: float
    y: float
    W2: float


def compute_dis_kinematics(
    e_in: Sequence[float],
    e_out: Sequence[float],
    p_in: Sequence[float],
) -> Optional[DisKinematics]:
    """
    Compute Q2, xB, y, W2 using:
      q = k - k'
      Q2 = -q^2
      xB = Q2 / (2 p.q)
      y = (p.q) / (p.k)
      W2 = (p+q)^2
    All as invariants with Minkowski metric (+,-,-,-).
    """
    # q = k - k'
    q = (e_in[0] - e_out[0], e_in[1] - e_out[1], e_in[2] - e_out[2], e_in[3] - e_out[3])
    q2 = p4_dot(q, q)
    Q2 = -q2
    if not math.isfinite(Q2):
        return None
    if Q2 < 0:
        # Numerical noise; clamp to 0 for downstream hist plots.
        Q2 = 0.0

    p_dot_q = p4_dot(p_in, q)
    p_dot_k = p4_dot(p_in, e_in)
    if abs(p_dot_q) < 1e-12 or abs(p_dot_k) < 1e-12:
        return DisKinematics(Q2=Q2, xB=float("nan"), y=float("nan"), W2=float("nan"))

    xB = Q2 / (2.0 * p_dot_q)
    y = p_dot_q / p_dot_k

    # W2 = (p+q)^2
    pq = (p_in[0] + q[0], p_in[1] + q[1], p_in[2] + q[2], p_in[3] + q[3])
    W2 = p4_dot(pq, pq)

    return DisKinematics(Q2=float(Q2), xB=float(xB), y=float(y), W2=float(W2))


def get_scattered_electron(ev: pythia8.Event) -> Optional[int]:
    """
    Return index of scattered electron (id=11) with status>0.
    Prefer status==44; else choose the highest-energy electron among status>0.
    """
    best_44: Optional[int] = None
    best_energy: float = -1.0
    best_idx: Optional[int] = None
    for i in range(ev.size()):
        p = ev[i]
        if p.id() != 11:
            continue
        if p.status() <= 0:
            continue
        if p.status() == 44:
            best_44 = i
            break
        if p.e() > best_energy:
            best_energy = float(p.e())
            best_idx = i
    return best_44 if best_44 is not None else best_idx


def find_incoming_beams(ev: pythia8.Event) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[Tuple[float, float, float, float]]]:
    """
    Find incoming electron and proton by ids and status.
    - electron: id=11 status==-12
    - proton: id=2212 status<0
    Returns (e_in, p_in) each as (E,px,py,pz) or None if not found.
    """
    e_in: Optional[Tuple[float, float, float, float]] = None
    p_in: Optional[Tuple[float, float, float, float]] = None
    for i in range(ev.size()):
        p = ev[i]
        if p.id() == 11 and p.status() == -12:
            e_in = (float(p.e()), float(p.px()), float(p.py()), float(p.pz()))
        if p.id() == 2212 and p.status() < 0:
            p_in = (float(p.e()), float(p.px()), float(p.py()), float(p.pz()))
    return e_in, p_in


def get_ancestors(ev: pythia8.Event, start: int, max_depth: int = 80) -> Set[int]:
    """Indices reachable by following mother1/mother2 links backward."""
    seen: Set[int] = set()
    queue: List[int] = [start]
    for _ in range(max_depth):
        if not queue:
            break
        idx = queue.pop(0)
        if idx in seen or idx <= 0 or idx >= ev.size():
            continue
        seen.add(idx)
        p = ev[idx]
        m1, m2 = p.mother1(), p.mother2()
        if m1 > 0:
            queue.append(m1)
        if m2 > 0 and m2 != m1:
            queue.append(m2)
    return seen


def find_incoming_struck_quark(ev: pythia8.Event) -> Tuple[Optional[int], bool]:
    """
    Find incoming hard struck quark tag:
      - status == -21
      - abs(id) in {1..5}

    Returns (index, ambiguous_assignment_flag).
    If multiple exist, pick the one with largest energy and mark ambiguous.
    """
    candidates: List[int] = []
    for i in range(ev.size()):
        p = ev[i]
        if p.status() == -21 and abs(p.id()) in QUARK_ABS_IDS:
            candidates.append(i)
    if not candidates:
        return None, False
    if len(candidates) > 1:
        # Pick the highest-energy one; still count diagnostic upstream.
        best = max(candidates, key=lambda j: ev[j].e())
        return best, True
    return candidates[0], False


def find_outgoing_struck_quark(ev: pythia8.Event, incoming_q_idx: int) -> Optional[int]:
    """
    Identify the outgoing hard struck quark corresponding to incoming_q_idx.

    Method:
      - Consider all quarks with abs(id) == abs(id_in) whose ancestry chain contains incoming_q_idx.
      - Prefer those with abs(status) == 23 (max pplus), else those with 63<=abs(status)<=68 (max pplus),
        else fallback to max E.
    """
    in_id = abs(ev[incoming_q_idx].id())
    anc_in = incoming_q_idx  # just a name

    best_status23: Optional[int] = None
    best_status23_pplus = -1e300

    best_shower: Optional[int] = None
    best_shower_pplus = -1e300

    best_fallback: Optional[int] = None
    best_fallback_E = -1e300

    for i in range(ev.size()):
        if i == incoming_q_idx:
            continue
        p = ev[i]
        if abs(p.id()) != in_id:
            continue
        # Ancestry contains incoming_q_idx?
        if anc_in not in get_ancestors(ev, i):
            continue

        E = float(p.e())
        px, py, pz = float(p.px()), float(p.py()), float(p.pz())
        pt = pT_from(px, py)
        pplus, _ = pplus_pminus(E, pz)

        abs_status = abs(p.status())
        if abs_status == 23:
            if pplus > best_status23_pplus:
                best_status23_pplus = pplus
                best_status23 = i
        elif 63 <= abs_status <= 68:
            if pplus > best_shower_pplus:
                best_shower_pplus = pplus
                best_shower = i
        else:
            if E > best_fallback_E:
                best_fallback_E = E
                best_fallback = i

    if best_status23 is not None:
        return best_status23
    if best_shower is not None:
        return best_shower
    return best_fallback


def collect_down_candidates(ev: pythia8.Event, eta_min: float) -> List[int]:
    """
    Collect down-quark candidates (id==1) passing eta > eta_min.

    Status tiering matches the general intent of other scripts:
      - prefer abs(status)==23
      - else 63<=abs(status)<=68
      - else any status!=0 quark
    """
    candidates23: List[int] = []
    candidates_shower: List[int] = []
    candidates_any: List[int] = []

    for i in range(ev.size()):
        p = ev[i]
        if p.id() != 1:
            continue
        abs_status = abs(p.status())
        # Build eta
        px, py, pz, E = float(p.px()), float(p.py()), float(p.pz()), float(p.e())
        pt = pT_from(px, py)
        eta = eta_from(E, pz, pt)
        if eta is None:
            continue
        if eta <= eta_min:
            continue
        if abs_status == 23:
            candidates23.append(i)
        elif 63 <= abs_status <= 68:
            candidates_shower.append(i)
        elif p.status() != 0:
            candidates_any.append(i)

    return candidates23 if candidates23 else (candidates_shower if candidates_shower else candidates_any)


def particle_row_fields(px: float, py: float, pz: float, E: float, mass: float) -> Dict[str, float]:
    pt = pT_from(px, py)
    phi = phi_from(px, py)
    eta = eta_from(E, pz, pt)
    pplus, pminus = pplus_pminus(E, pz)
    return {
        "px": float(px),
        "py": float(py),
        "pz": float(pz),
        "E": float(E),
        "m": float(mass),
        "pT": float(pt),
        "phi": float(phi),
        "eta": safe_float(eta),
        "pplus": float(pplus),
        "pminus": float(pminus),
    }


def extract_particle_4vec(ev: pythia8.Event, idx: int) -> Tuple[float, float, float, float, float]:
    p = ev[idx]
    return float(p.px()), float(p.py()), float(p.pz()), float(p.e()), float(p.m())


def build_pythia() -> pythia8.Pythia:
    p = pythia8.Pythia()
    p.readString("Beams:idA = 11")
    p.readString("Beams:idB = 2212")
    p.readString("Beams:eA = 18")
    p.readString("Beams:eB = 275")
    p.readString("Beams:frameType = 2")
    p.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    p.readString("HardQCD:all = off")
    p.readString("PDF:lepton = off")
    p.readString("PhaseSpace:Q2Min = 16")
    p.readString("PartonLevel:ISR = on")
    p.readString("PartonLevel:FSR = on")
    p.readString("PartonLevel:MPI = off")
    p.readString("HadronLevel:all = on")
    p.readString("ColourReconnection:reconnect = off")
    p.readString("Random:setSeed = on")
    p.readString("Random:seed = 123456")
    p.init()
    return p


def write_rows_append_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Append rows to CSV, writing header only when file does not exist."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    header = not path.exists()
    df.to_csv(path, mode=mode, header=header, index=False, float_format="%.10g")


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest 2->2 DIS baseline for struck-u and struck-d.")
    parser.add_argument("--target-u", type=int, default=100000)
    parser.add_argument("--target-d", type=int, default=100000)
    parser.add_argument("--eta-forward-cut", type=float, default=0.0, help="Forward region cut: eta > eta_forward_cut for d selection.")
    parser.add_argument("--chunk-accepted", type=int, default=20000, help="Flush accepted events to CSV every N accepted (combined accepted).")
    parser.add_argument("--progress-every", type=int, default=50000, help="Print progress every N accepted events per flavor.")
    parser.add_argument("--test", action="store_true", help="Run a small smoke test.")
    parser.add_argument("--source-label", type=str, default="ISRFSR_ON", help="Saved into summary for provenance.")
    args = parser.parse_args()

    target_u = 5 if args.test else args.target_u
    target_d = 5 if args.test else args.target_d

    outdir = DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    u_csv = outdir / "struck_u_baseline.csv"
    d_csv = outdir / "struck_d_baseline.csv"
    combined_csv = outdir / "combined_baseline.csv"
    plots_dir = outdir / "validation_plots"

    # Fieldnames (fixed schema so incremental writes remain consistent)
    fieldnames = [
        # Event metadata
        "event_id",
        "source_config_label",
        "struck_flavor",
        "has_forward_d",
        "Q2",
        "xB",
        "y",
        "W2",
        # Electron
        "ele_px", "ele_py", "ele_pz", "ele_E", "ele_pT", "ele_eta", "ele_phi", "ele_pplus", "ele_pminus",
        # Struck quark (hard outgoing)
        "struck_px", "struck_py", "struck_pz", "struck_E", "struck_m", "struck_pT", "struck_eta", "struck_phi", "struck_pplus", "struck_pminus",
        # Hardest forward d by pplus
        "d_pplus_px", "d_pplus_py", "d_pplus_pz", "d_pplus_E", "d_pplus_m", "d_pplus_pT", "d_pplus_eta", "d_pplus_phi", "d_pplus_pplus", "d_pplus_pminus",
        # Hardest forward d by pT (alternative selector)
        "d_pt_px", "d_pt_py", "d_pt_pz", "d_pt_E", "d_pt_m", "d_pt_pT", "d_pt_eta", "d_pt_phi", "d_pt_pplus", "d_pt_pminus",
    ]

    # Reset output CSVs for this run (so partial runs don't mix schema)
    if u_csv.exists():
        u_csv.unlink()
    if d_csv.exists():
        d_csv.unlink()
    if combined_csv.exists():
        combined_csv.unlink()

    pythia = build_pythia()

    counters: Dict[str, Any] = {
        "target_u": target_u,
        "target_d": target_d,
        "accepted_u": 0,
        "accepted_d": 0,
        "generated_total": 0,
        "accepted_total": 0,
        "events_no_electron": 0,
        "events_no_incoming_struck": 0,
        "ambiguous_struck_assignment": 0,
        "events_no_outgoing_struck": 0,
        "events_wrong_struck_flavor": 0,
        "n_forward_d_u": 0,
        "n_forward_d_d": 0,
    }

    chunk_u: List[Dict[str, Any]] = []
    chunk_d: List[Dict[str, Any]] = []
    accepted_since_flush = 0

    event_id = 0
    t0 = time.time()
    next_progress_u = args.progress_every
    next_progress_d = args.progress_every

    while counters["accepted_u"] < target_u or counters["accepted_d"] < target_d:
        if not pythia.next():
            continue
        counters["generated_total"] += 1

        ev = pythia.event
        # Scattered electron
        ele_idx = get_scattered_electron(ev)
        if ele_idx is None:
            counters["events_no_electron"] += 1
            continue

        # Incoming beams for DIS kinematics
        e_in, p_in = find_incoming_beams(ev)
        if e_in is None or p_in is None:
            continue

        # Incoming struck quark
        incoming_q_idx, ambiguous = find_incoming_struck_quark(ev)
        if incoming_q_idx is None:
            counters["events_no_incoming_struck"] += 1
            continue
        if ambiguous:
            counters["ambiguous_struck_assignment"] += 1

        incoming_q_id_abs = abs(int(ev[incoming_q_idx].id()))
        if incoming_q_id_abs == 2:
            struck_flavor = "u"
        elif incoming_q_id_abs == 1:
            struck_flavor = "d"
        else:
            counters["events_wrong_struck_flavor"] += 1
            continue

        # Only accept if target not reached
        if struck_flavor == "u" and counters["accepted_u"] >= target_u:
            continue
        if struck_flavor == "d" and counters["accepted_d"] >= target_d:
            continue

        # Outgoing hard struck quark
        out_struck_idx = find_outgoing_struck_quark(ev, incoming_q_idx)
        if out_struck_idx is None:
            counters["events_no_outgoing_struck"] += 1
            continue

        # Forward hardest d quark selection
        d_candidates = collect_down_candidates(ev, eta_min=args.eta_forward_cut)

        has_forward_d = 1 if d_candidates else 0
        d_pplus_idx: Optional[int] = None
        d_pt_idx: Optional[int] = None
        if d_candidates:
            d_pplus_idx = max(d_candidates, key=lambda j: float(ev[j].e() + ev[j].pz()))
            d_pt_idx = max(d_candidates, key=lambda j: float(pT_from(float(ev[j].px()), float(ev[j].py()))))

        # Build row
        event_id += 1
        Q2 = xB = y = W2 = float("nan")
        kin = compute_dis_kinematics(e_in=e_in, e_out=(
            float(ev[ele_idx].e()), float(ev[ele_idx].px()), float(ev[ele_idx].py()), float(ev[ele_idx].pz())
        ), p_in=p_in)
        if kin is not None:
            Q2, xB, y, W2 = kin.Q2, kin.xB, kin.y, kin.W2

        ele_px, ele_py, ele_pz, ele_E, _ = extract_particle_4vec(ev, ele_idx)
        ele_row = particle_row_fields(ele_px, ele_py, ele_pz, ele_E, 0.0)

        struck_px, struck_py, struck_pz, struck_E, struck_m = extract_particle_4vec(ev, out_struck_idx)
        struck_row = particle_row_fields(struck_px, struck_py, struck_pz, struck_E, struck_m)

        if d_pplus_idx is not None:
            dpx, dpy, dpz, dE, dm = extract_particle_4vec(ev, d_pplus_idx)
            d_pplus_row = particle_row_fields(dpx, dpy, dpz, dE, dm)
        else:
            d_pplus_row = {k: float("nan") for k in ["px", "py", "pz", "E", "m", "pT", "eta", "phi", "pplus", "pminus"]}

        if d_pt_idx is not None:
            dpx2, dpy2, dpz2, dE2, dm2 = extract_particle_4vec(ev, d_pt_idx)
            d_pt_row = particle_row_fields(dpx2, dpy2, dpz2, dE2, dm2)
        else:
            d_pt_row = {k: float("nan") for k in ["px", "py", "pz", "E", "m", "pT", "eta", "phi", "pplus", "pminus"]}

        row: Dict[str, Any] = {
            "event_id": event_id,
            "source_config_label": args.source_label,
            "struck_flavor": struck_flavor,
            "has_forward_d": has_forward_d,
            "Q2": Q2,
            "xB": xB,
            "y": y,
            "W2": W2,
            # electron
            "ele_px": ele_row["px"],
            "ele_py": ele_row["py"],
            "ele_pz": ele_row["pz"],
            "ele_E": ele_row["E"],
            "ele_pT": ele_row["pT"],
            "ele_eta": ele_row["eta"],
            "ele_phi": ele_row["phi"],
            "ele_pplus": ele_row["pplus"],
            "ele_pminus": ele_row["pminus"],
            # struck
            "struck_px": struck_row["px"],
            "struck_py": struck_row["py"],
            "struck_pz": struck_row["pz"],
            "struck_E": struck_row["E"],
            "struck_m": struck_row["m"],
            "struck_pT": struck_row["pT"],
            "struck_eta": struck_row["eta"],
            "struck_phi": struck_row["phi"],
            "struck_pplus": struck_row["pplus"],
            "struck_pminus": struck_row["pminus"],
            # d by pplus
            "d_pplus_px": d_pplus_row["px"],
            "d_pplus_py": d_pplus_row["py"],
            "d_pplus_pz": d_pplus_row["pz"],
            "d_pplus_E": d_pplus_row["E"],
            "d_pplus_m": d_pplus_row["m"],
            "d_pplus_pT": d_pplus_row["pT"],
            "d_pplus_eta": d_pplus_row["eta"],
            "d_pplus_phi": d_pplus_row["phi"],
            "d_pplus_pplus": d_pplus_row["pplus"],
            "d_pplus_pminus": d_pplus_row["pminus"],
            # d by pt
            "d_pt_px": d_pt_row["px"],
            "d_pt_py": d_pt_row["py"],
            "d_pt_pz": d_pt_row["pz"],
            "d_pt_E": d_pt_row["E"],
            "d_pt_m": d_pt_row["m"],
            "d_pt_pT": d_pt_row["pT"],
            "d_pt_eta": d_pt_row["eta"],
            "d_pt_phi": d_pt_row["phi"],
            "d_pt_pplus": d_pt_row["pplus"],
            "d_pt_pminus": d_pt_row["pminus"],
        }

        if struck_flavor == "u":
            chunk_u.append(row)
            counters["accepted_u"] += 1
            counters["n_forward_d_u"] += int(has_forward_d)
        else:
            chunk_d.append(row)
            counters["accepted_d"] += 1
            counters["n_forward_d_d"] += int(has_forward_d)
        counters["accepted_total"] += 1
        accepted_since_flush += 1

        # Flush in chunks
        if accepted_since_flush >= args.chunk_accepted:
            write_rows_append_csv(u_csv, chunk_u, fieldnames)
            write_rows_append_csv(d_csv, chunk_d, fieldnames)
            chunk_u.clear()
            chunk_d.clear()
            accepted_since_flush = 0

        do_progress = False
        if counters["accepted_u"] >= next_progress_u:
            do_progress = True
            while counters["accepted_u"] >= next_progress_u:
                next_progress_u += args.progress_every
        if counters["accepted_d"] >= next_progress_d:
            do_progress = True
            while counters["accepted_d"] >= next_progress_d:
                next_progress_d += args.progress_every

        if do_progress:
            elapsed = time.time() - t0
            acc_u = counters["accepted_u"]
            acc_d = counters["accepted_d"]
            rem_u = max(0, target_u - acc_u)
            rem_d = max(0, target_d - acc_d)
            rate_u = acc_u / elapsed if elapsed > 0 else 0.0
            rate_d = acc_d / elapsed if elapsed > 0 else 0.0
            eta_u = (rem_u / rate_u) if rate_u > 0 else float("inf")
            eta_d = (rem_d / rate_d) if rate_d > 0 else float("inf")
            eta_remaining = max(eta_u, eta_d)
            frac_u = (counters["n_forward_d_u"] / acc_u) if acc_u > 0 else float("nan")
            frac_d = (counters["n_forward_d_d"] / acc_d) if acc_d > 0 else float("nan")
            print(
                f"[progress] gen={counters['generated_total']} "
                f"acc_u={acc_u}/{target_u} acc_d={acc_d}/{target_d} "
                f"fwd_d_u={counters['n_forward_d_u']} fwd_d_d={counters['n_forward_d_d']} "
                f"frac_u={frac_u:.6f} frac_d={frac_d:.6f} "
                f"elapsed_s={elapsed:.1f} eta_remaining_s={eta_remaining:.1f}",
                flush=True,
            )

    # Flush remaining
    if chunk_u:
        write_rows_append_csv(u_csv, chunk_u, fieldnames)
    if chunk_d:
        write_rows_append_csv(d_csv, chunk_d, fieldnames)

    # Write combined
    df_u = pd.read_csv(u_csv)
    df_d = pd.read_csv(d_csv)
    df_comb = pd.concat([df_u, df_d], ignore_index=True)
    df_comb.to_csv(combined_csv, index=False)

    counters["eta_forward_cut"] = args.eta_forward_cut
    counters["outdir"] = str(outdir)
    counters["n_u_rows"] = int(df_u.shape[0])
    counters["n_d_rows"] = int(df_d.shape[0])
    counters["n_combined_rows"] = int(df_comb.shape[0])

    (outdir / "summary.json").write_text(json.dumps(counters, indent=2), encoding="utf-8")

    readme = outdir / "README_summary.md"
    readme.write_text(
        "\n".join(
            [
                "# 2->2 DIS baseline harvesting",
                "",
                "Settings:",
                "- Process: DIS (WeakBosonExchange:ff2ff(t:gmZ)=on), HardQCD:all=off",
                f"- Beams: eA=18 GeV, eB=275 GeV, frameType=2 (ep setup)",
                "- PDF:lepton=off, PhaseSpace:Q2Min=16",
                "- ISR=on, FSR=on, MPI=off, HadronLevel:all=on",
                "- ColourReconnection:reconnect=off",
                "",
                "Identification logic:",
                "- Scattered electron: prefer status==44 else highest-energy e- with status>0.",
                "- Incoming struck quark tag: status==-21 and abs(id) in {1..5}. If multiple, pick max E and count diagnostic.",
                "- Outgoing hard struck quark: abs(id)==abs(incoming_id) and ancestry chain contains incoming tag index.",
                "  Priority: abs(status)==23 (max pplus) else 63<=abs(status)<=68 (max pplus) else max E.",
                "- Hardest forward d quark candidates: particle id==1, forward cut eta>eta_forward_cut.",
                "  Candidate tiers: abs(status)==23 else 63<=abs(status)<=68 else any status!=0. Select by max pplus (main) and max pT (secondary).",
                "",
                "Outputs:",
                f"- {u_csv}",
                f"- {d_csv}",
                f"- {combined_csv}",
                "- {outdir}/validation_plots/* (produced by validate_2to2_baseline.py)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("Harvest complete.", flush=True)
    print(json.dumps({k: counters[k] for k in ["accepted_u", "accepted_d", "generated_total", "eta_forward_cut"]}, indent=2))


if __name__ == "__main__":
    main()

