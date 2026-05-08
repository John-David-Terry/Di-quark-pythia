#!/usr/bin/env python3
"""
Generate a mixed sample of kicked 2->4 DIS LHE events for struck u and struck d quarks.

Base:
  - Reuse the DIS setup and 2->4 construction logic from build_single_2to4_lhe_from_pythia_event.py
  - Do NOT redesign topology or colour flow; only add a production loop + kT injections.

Targets:
  - 2000 accepted struck-u events
  - 2000 accepted struck-d events

For each accepted event:
  1) Identify struck flavour (u or d) from incoming struck quark.
  2) Build the 2->4 hard event (same structure as the working PoP).
  3) Identify which two outgoing quarks to modify:
       - struck u sample: outgoing pair [u, d] plus dbar; modify the (u, d) pair.
       - struck d sample: outgoing pair [d, d] plus dbar; modify the two d quarks.
  4) Inject transverse momenta into those two quarks only (incoming remain beam-collinear).
     Struck-u: always SRC-like correlated pair (+q on u, -q on d in the transverse plane).
     Struck-d: uncorrelated kicks on the two d quarks.
  5) Recompute energies on shell.
  6) Write one LHE per accepted event plus a metadata JSON.
  7) For each accepted event, also write paired unkicked/kicked copies under
     outputs/popf_lhe_single_event_2to4_{unkicked,kicked}/ for reinjection comparisons.

Output:
  outputs/src_mixture_validation/
    struck_u/
      lhe/event_u_XXXXXX.lhe
      metadata/event_u_XXXXXX.json
    struck_d/
      lhe/event_d_XXXXXX.lhe
      metadata/event_d_XXXXXX.json
    run_summary.json
  outputs/popf_lhe_single_event_2to4_unkicked/   (single_event_2to4_unkicked.lhe or event_NNNNNN_unkicked.lhe)
  outputs/popf_lhe_single_event_2to4_kicked/    (single_event_2to4_kicked.lhe or event_NNNNNN_kicked.lhe)

Run:
  - Use python3.11 (the environment where pythia8 works), e.g.:
      /usr/local/bin/python3.11 scripts/analysis/generate_src_mixture_lhe_samples.py
  - For a small smoke test (5 u + 5 d accepted events):
      /usr/local/bin/python3.11 scripts/analysis/generate_src_mixture_lhe_samples.py --test
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pythia8

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir


OUTDIR_BASE = outputs_dir() / "src_mixture_validation"
U_LHE_DIR = OUTDIR_BASE / "struck_u" / "lhe"
U_META_DIR = OUTDIR_BASE / "struck_u" / "metadata"
D_LHE_DIR = OUTDIR_BASE / "struck_d" / "lhe"
D_META_DIR = OUTDIR_BASE / "struck_d" / "metadata"

RUN_SUMMARY_PATH = OUTDIR_BASE / "run_summary.json"

# Paired hard events for PYTHIA reinjection / hadronization A/B tests: same event, unkicked vs kicked.
# We want event-by-event comparisons between kicked and unkicked events after PYTHIA reinjects and
# hadronizes them, so we need both versions of the same hard event.
POPF_UNKICKED_DIR = outputs_dir() / "popf_lhe_single_event_2to4_unkicked"
POPF_KICKED_DIR = outputs_dir() / "popf_lhe_single_event_2to4_kicked"

for d in [U_LHE_DIR, U_META_DIR, D_LHE_DIR, D_META_DIR, POPF_UNKICKED_DIR, POPF_KICKED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

QUARK_ABS_IDS = {1, 2, 3, 4, 5}


@dataclass
class KTConfig:
    sigma_uncorr: float = 0.25  # GeV
    q0: float = 0.50            # GeV (central SRC magnitude)
    sigma_q: float = 0.08       # GeV
    sigma_src_residual: float = 0.05  # GeV


@dataclass
class EventMomentumSummary:
    total_px_before: float
    total_py_before: float
    total_pz_before: float
    total_E_before: float
    total_px_after: float
    total_py_after: float
    total_pz_after: float
    total_E_after: float
    delta_px: float
    delta_py: float
    delta_pz: float
    delta_E: float


def build_pythia() -> pythia8.Pythia:
    p = pythia8.Pythia()
    p.readString("Beams:idA = 11")
    p.readString("Beams:idB = 2212")
    p.readString("Beams:eA = 18")
    p.readString("Beams:eB = 275")
    p.readString("Beams:frameType = 2")
    p.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    p.readString("PDF:lepton = off")
    p.readString("PhaseSpace:Q2Min = 16")
    p.readString("PartonLevel:ISR = on")
    p.readString("PartonLevel:FSR = off")
    p.readString("PartonLevel:MPI = off")
    p.readString("HadronLevel:all = off")
    p.readString("Random:setSeed = on")
    p.readString("Random:seed = 123456")
    p.init()
    return p


def particle_record(ev: pythia8.Event, i: int) -> Dict[str, Any]:
    p = ev[i]
    return {
        "index": i,
        "pdg_id": int(p.id()),
        "status": int(p.status()),
        "mother1": int(p.mother1()),
        "mother2": int(p.mother2()),
        "daughter1": int(p.daughter1()),
        "daughter2": int(p.daughter2()),
        "color": int(p.col()),
        "anticolor": int(p.acol()),
        "px": float(p.px()),
        "py": float(p.py()),
        "pz": float(p.pz()),
        "E": float(p.e()),
        "m": float(p.m()),
    }


def get_ancestors(ev: pythia8.Event, start: int, max_depth: int = 50) -> Set[int]:
    seen: Set[int] = set()
    queue = [start]
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


def ancestry_chain(ev: pythia8.Event, start: int, max_steps: int = 50) -> List[int]:
    path: List[int] = []
    idx = start
    for _ in range(max_steps):
        if idx <= 0 or idx >= ev.size():
            break
        path.append(idx)
        p = ev[idx]
        m1, m2 = p.mother1(), p.mother2()
        if m1 > 0:
            idx = m1
        elif m2 > 0:
            idx = m2
        else:
            break
    return path


def lowest_common_ancestor(ev: pythia8.Event, i: int, j: int) -> Optional[int]:
    a_i = get_ancestors(ev, i)
    a_j = get_ancestors(ev, j)
    common = a_i & a_j
    return min(common) if common else None


def select_2to4_topology(ev: pythia8.Event) -> Optional[Dict[str, Any]]:
    """
    Mirror the selection logic from build_single_2to4_lhe_from_pythia_event.py:
      - incoming e, incoming quark (status -21, abs(id) in 1..5)
      - outgoing struck quark with ancestry to incoming quark
      - partner quark q' + matching antiquark qbar'
    Returns a dict with selected particle records (from event) and indices.
    """
    # Final-state quarks and antiquarks
    final_quarks: List[int] = []
    final_antiquarks: List[int] = []
    final_electrons: List[int] = []
    for i in range(ev.size()):
        p = ev[i]
        if not p.isFinal():
            continue
        aid = abs(p.id())
        if aid == 11:
            final_electrons.append(i)
        elif aid in QUARK_ABS_IDS:
            if p.id() > 0:
                final_quarks.append(i)
            else:
                final_antiquarks.append(i)

    if len(final_quarks) < 2 or len(final_antiquarks) < 1 or len(final_electrons) < 1:
        return None

    # Incoming hard: electron (id=11) and quark (abs(id) in 1..5), status -21
    in_e_idx: Optional[int] = None
    in_q_idx: Optional[int] = None
    for i in range(ev.size()):
        p = ev[i]
        if p.status() != -21:
            continue
        if p.id() == 11:
            in_e_idx = i
        elif abs(p.id()) in QUARK_ABS_IDS:
            in_q_idx = i
    if in_e_idx is None or in_q_idx is None:
        return None

    incoming_id = ev[in_q_idx].id()

    # Outgoing struck quark: final quark same |id| as incoming, ancestry contains incoming quark
    struck_idx: Optional[int] = None
    for i in final_quarks:
        if abs(ev[i].id()) != abs(incoming_id):
            continue
        if in_q_idx in get_ancestors(ev, i):
            struck_idx = i
            break
    if struck_idx is None:
        return None

    # Outgoing electron: any final electron (first one)
    out_e_idx = final_electrons[0]

    # Partner quark: first final quark different from struck and different flavour
    partner_q_idx: Optional[int] = None
    for i in final_quarks:
        if i == struck_idx:
            continue
        if abs(ev[i].id()) == abs(incoming_id):
            continue
        partner_q_idx = i
        break
    if partner_q_idx is None:
        return None

    partner_id = ev[partner_q_idx].id()

    # Partner antiquark: first final antiquark with id == -partner_id
    partner_qbar_idx: Optional[int] = None
    for i in final_antiquarks:
        if ev[i].id() == -partner_id:
            partner_qbar_idx = i
            break
    if partner_qbar_idx is None:
        return None

    event_records = [particle_record(ev, i) for i in range(ev.size())]

    return {
        "event_records": event_records,
        "in_e_idx": in_e_idx,
        "in_q_idx": in_q_idx,
        "out_e_idx": out_e_idx,
        "struck_idx": struck_idx,
        "partner_q_idx": partner_q_idx,
        "partner_qbar_idx": partner_qbar_idx,
    }


def find_matching_2to4_topology(
    ev: pythia8.Event,
    summary: Dict[str, Any],
) -> Optional[Tuple[List[List[float]], str, Tuple[int, int, int]]]:
    """
    Flavour-agnostic topology finder.

    Steps:
      1) Identify incoming electron, incoming struck quark, outgoing electron, outgoing struck quark
         using the same logic as select_2to4_topology (but without fixing partner flavour).
      2) Build candidate (q, qbar) pairs from remaining final-state partons:
           - remaining final-state quarks (excluding struck)
           - all final-state antiquarks
           - require qbar.id() == -q.id()
      3) For each candidate pair, build 2->4 LHE rows and classify via identify_sample_and_pair().
         Accept the first candidate matching either:
           - eu_to_e_ud_dbar (struck-u sample)
           - ed_to_e_dd_dbar (struck-d sample)

    Returns:
      (rows_24, sample_type, (idx_struck, idx_q, idx_qbar))
    where sample_type is "u" or "d" and indices are 1-based LHE entry indices.
    """
    # Final-state quarks and antiquarks
    final_quarks: List[int] = []
    final_antiquarks: List[int] = []
    final_electrons: List[int] = []
    for i in range(ev.size()):
        p = ev[i]
        if not p.isFinal():
            continue
        aid = abs(p.id())
        if aid == 11:
            final_electrons.append(i)
        elif aid in QUARK_ABS_IDS:
            if p.id() > 0:
                final_quarks.append(i)
            else:
                final_antiquarks.append(i)

    if len(final_quarks) < 2 or len(final_antiquarks) < 1 or len(final_electrons) < 1:
        return None

    # Incoming hard: electron (id=11) and quark (abs(id) in 1..5), status -21
    in_e_idx: Optional[int] = None
    in_q_idx: Optional[int] = None
    for i in range(ev.size()):
        p = ev[i]
        if p.status() != -21:
            continue
        if p.id() == 11:
            in_e_idx = i
        elif abs(p.id()) in QUARK_ABS_IDS:
            in_q_idx = i
    if in_e_idx is None or in_q_idx is None:
        return None

    incoming_id = ev[in_q_idx].id()

    # RAW HARD PROCESS counters
    if abs(incoming_id) == 2:
        summary["raw_struck_u"] += 1
    elif abs(incoming_id) == 1:
        summary["raw_struck_d"] += 1

    # Outgoing struck quark: final quark same |id| as incoming, ancestry contains incoming quark
    struck_idx: Optional[int] = None
    candidate_struck_indices: List[int] = []
    for i in final_quarks:
        if abs(ev[i].id()) != abs(incoming_id):
            continue
        if in_q_idx in get_ancestors(ev, i):
            candidate_struck_indices.append(i)

    if len(candidate_struck_indices) == 0:
        return None
    if len(candidate_struck_indices) > 1:
        summary["events_with_multiple_same_flavor_quarks"] += 1
        summary["ambiguous_struck_assignment"] += 1

    struck_idx = candidate_struck_indices[0]
    if struck_idx is None:
        return None

    # Outgoing electron: any final electron (first one)
    out_e_idx = final_electrons[0]

    # Remaining quarks (candidates for q') and all antiquarks (candidates for qbar')
    remaining_quarks = [i for i in final_quarks if i != struck_idx]
    event_records = [particle_record(ev, i) for i in range(ev.size())]

    # PAIR AVAILABILITY: count events with at least one d dbar pair
    has_ddbar_pair = False
    for qi in final_quarks:
        if ev[qi].id() != 1:
            continue
        for qbi in final_antiquarks:
            if ev[qbi].id() == -1:
                has_ddbar_pair = True
                break
        if has_ddbar_pair:
            break

    if has_ddbar_pair:
        summary["events_with_ddbar_pair"] += 1
        if abs(incoming_id) == 2:
            summary["struck_u_with_ddbar_pair"] += 1
        elif abs(incoming_id) == 1:
            summary["struck_d_with_ddbar_pair"] += 1

    candidate_pairs_count = 0
    topology_matched = False

    for q_idx in remaining_quarks:
        q_id = ev[q_idx].id()
        for qbar_idx in final_antiquarks:
            if ev[qbar_idx].id() != -q_id:
                continue
            candidate_pairs_count += 1

            selected = {
                "event_records": event_records,
                "in_e_idx": in_e_idx,
                "in_q_idx": in_q_idx,
                "out_e_idx": out_e_idx,
                "struck_idx": struck_idx,
                "partner_q_idx": q_idx,
                "partner_qbar_idx": qbar_idx,
            }
            rows_24 = build_2to4_lhe_rows(selected)
            sample_info = identify_sample_and_pair(rows_24)
            if sample_info is None:
                continue
            sample_type, (idx_struck_lhe, idx_q_lhe, idx_qbar_lhe) = sample_info
            # Topology matched
            topology_matched = True
            # Candidate multiplicity diagnostics
            if abs(incoming_id) == 2:
                summary["total_candidate_pairs_u"] += candidate_pairs_count
                summary["u_topology_success"] += 1
            elif abs(incoming_id) == 1:
                summary["total_candidate_pairs_d"] += candidate_pairs_count
                summary["d_topology_success"] += 1
            return rows_24, sample_type, (idx_struck_lhe, idx_q_lhe, idx_qbar_lhe)

    # No topology match
    if candidate_pairs_count > 0:
        if abs(incoming_id) == 2:
            summary["total_candidate_pairs_u"] += candidate_pairs_count
            summary["u_topology_fail"] += 1
        elif abs(incoming_id) == 1:
            summary["total_candidate_pairs_d"] += candidate_pairs_count
            summary["d_topology_fail"] += 1

    return None


def build_2to4_lhe_rows(selected: Dict[str, Any]) -> List[List[float]]:
    """
    Build the 6-particle 2->4 LHE rows in the same way as build_single_2to4_lhe_from_pythia_event.py.
    Returns list of rows:
      [idup, istup, m1, m2, col, acol, px, py, pz, E, m]
    """
    in_e = selected["event_records"][selected["in_e_idx"]]
    in_q = selected["event_records"][selected["in_q_idx"]]
    out_e = selected["event_records"][selected["out_e_idx"]]
    struck = selected["event_records"][selected["struck_idx"]]
    partner_q = selected["event_records"][selected["partner_q_idx"]]
    partner_qbar = selected["event_records"][selected["partner_qbar_idx"]]

    def p4(d: Dict[str, Any]) -> Tuple[float, float, float, float]:
        return float(d["px"]), float(d["py"]), float(d["pz"]), float(d["E"])

    out_e_p4 = p4(out_e)
    struck_p4 = p4(struck)
    partner_q_p4 = p4(partner_q)
    partner_qbar_p4 = p4(partner_qbar)
    in_e_p4 = p4(in_e)

    # Incoming quark from 4-momentum conservation; then enforce beam-collinear px=py=0.
    p_in_q_x = out_e_p4[0] + struck_p4[0] + partner_q_p4[0] + partner_qbar_p4[0] - in_e_p4[0]
    p_in_q_y = out_e_p4[1] + struck_p4[1] + partner_q_p4[1] + partner_qbar_p4[1] - in_e_p4[1]
    p_in_q_z = out_e_p4[2] + struck_p4[2] + partner_q_p4[2] + partner_qbar_p4[2] - in_e_p4[2]
    # Force px=py=0 and recompute E with original in_quark mass
    p_in_q_x, p_in_q_y = 0.0, 0.0
    m_in_q = float(in_q["m"])
    p_in_q_E = math.sqrt(m_in_q * m_in_q + p_in_q_z * p_in_q_z)

    # Masses: use source for struck and in_quark; partner q/qbar as in build_single_2to4...
    m_partner_q = float(partner_q["m"])
    m_partner_qbar = float(partner_qbar["m"])
    if abs(partner_q["pdg_id"]) in (1, 2, 3) and m_partner_q < 0.01:
        m_partner_q = 0.0
    if abs(partner_qbar["pdg_id"]) in (1, 2, 3) and m_partner_qbar < 0.01:
        m_partner_qbar = 0.0

    rows: List[List[float]] = [
        # 1: incoming electron
        [11, -1, 0, 0, 0, 0, 0.0, 0.0, float(in_e["pz"]), float(in_e["E"]), float(in_e["m"])],
        # 2: incoming struck quark
        [int(in_q["pdg_id"]), -1, 0, 0, 101, 0, p_in_q_x, p_in_q_y, p_in_q_z, p_in_q_E, float(in_q["m"])],
        # 3: outgoing electron
        [11, 1, 1, 2, 0, 0, out_e_p4[0], out_e_p4[1], out_e_p4[2], out_e_p4[3], float(out_e["m"])],
        # 4: outgoing struck quark
        [int(struck["pdg_id"]), 1, 1, 2, 101, 0, struck_p4[0], struck_p4[1], struck_p4[2], struck_p4[3], float(struck["m"])],
        # 5: outgoing partner quark q'
        [int(partner_q["pdg_id"]), 1, 1, 2, 102, 0, partner_q_p4[0], partner_q_p4[1], partner_q_p4[2], partner_q_p4[3], m_partner_q],
        # 6: outgoing partner antiquark qbar'
        [int(partner_qbar["pdg_id"]), 1, 1, 2, 0, 102, partner_qbar_p4[0], partner_qbar_p4[1], partner_qbar_p4[2], partner_qbar_p4[3], m_partner_qbar],
    ]
    return rows


def identify_sample_and_pair(rows: List[List[float]]) -> Optional[Tuple[str, Tuple[int, int, int]]]:
    """
    Given the 2->4 LHE rows (1..6), identify whether this event belongs to struck-u or struck-d sample,
    and which pair to modify.

    Returns:
      - ("u", (idx_struck, idx_q, idx_qbar)) or ("d", (idx_struck, idx_q, idx_qbar))
    where indices are 1-based LHE indices.

    Sanity checks:
      - struck u: incoming quark id=+2, outgoing quarks (entries 4,5) must be {u,d}={2,1}, and entry 6 must be dbar (-1).
      - struck d: incoming quark id=+1, outgoing quarks (4,5) must be {d,d}={1,1}, and entry 6 must be dbar (-1).
    """
    in_q_id = rows[1][0]  # entry 2
    id4 = rows[3][0]
    id5 = rows[4][0]
    id6 = rows[5][0]

    # Only consider quarks (positive IDs) in samples.
    if in_q_id == 2:
        # struck u
        if id6 == -1 and {id4, id5} == {1, 2}:
            return "u", (4, 5, 6)
        return None
    if in_q_id == 1:
        # struck d
        if id6 == -1 and id4 == 1 and id5 == 1:
            return "d", (4, 5, 6)
        return None
    return None


def sample_uncorrelated_kT(cfg: KTConfig) -> Tuple[float, float]:
    return random.gauss(0.0, cfg.sigma_uncorr), random.gauss(0.0, cfg.sigma_uncorr)


def sample_src_pair_kT(cfg: KTConfig) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    SRC-like back-to-back pair:
      q_vec = q * (cos(phi), sin(phi)), q ~ N(q0, sigma_q), phi uniform.
      kT1 = +q_vec + delta1, kT2 = -q_vec + delta2, delta ~ N(0, sigma_src_residual).
    """
    phi = random.uniform(0.0, 2.0 * math.pi)
    q = random.gauss(cfg.q0, cfg.sigma_q)
    qx = q * math.cos(phi)
    qy = q * math.sin(phi)
    dx1, dy1 = random.gauss(0.0, cfg.sigma_src_residual), random.gauss(0.0, cfg.sigma_src_residual)
    dx2, dy2 = random.gauss(0.0, cfg.sigma_src_residual), random.gauss(0.0, cfg.sigma_src_residual)
    k1 = (qx + dx1, qy + dy1)
    k2 = (-qx + dx2, -qy + dy2)
    return k1, k2


def apply_pair_kicks(
    rows: List[List[float]],
    idx_a: int,
    idx_b: int,
    sample_type: str,
    cfg: KTConfig,
) -> Tuple[str, Tuple[float, float], Tuple[float, float]]:
    """
    Apply transverse kicks to two outgoing quarks:
      - For sample_type == 'u':
          always SRC-like correlated pair (mapped so that u gets +q, d gets -q).
      - For sample_type == 'd':
          always uncorrelated.
    Returns (draw_type, (kx_a, ky_a), (kx_b, ky_b)).
    """
    draw_type = "uncorrelated"

    if sample_type == "u":
        draw_type = "src"
        (kx1, ky1), (kx2, ky2) = sample_src_pair_kT(cfg)
        # Map: outgoing u gets +q, d gets -q
        id_a = rows[idx_a - 1][0]
        id_b = rows[idx_b - 1][0]
        if id_a == 2 and id_b == 1:  # a=u, b=d
            kA = (kx1, ky1)
            kB = (kx2, ky2)
        elif id_a == 1 and id_b == 2:  # a=d, b=u
            kA = (kx2, ky2)
            kB = (kx1, ky1)
        else:
            # Fallback (should not happen if identify_sample_and_pair is correct)
            kA = (kx1, ky1)
            kB = (kx2, ky2)
        return draw_type, kA, kB

    # Uncorrelated for all d samples
    kA = sample_uncorrelated_kT(cfg)
    kB = sample_uncorrelated_kT(cfg)
    return draw_type, kA, kB


def compute_event_momentum_summary(rows_before: List[List[float]], rows_after: List[List[float]]) -> EventMomentumSummary:
    def sum4(rows: List[List[float]]) -> Tuple[float, float, float, float]:
        px = sum(r[6] for r in rows)
        py = sum(r[7] for r in rows)
        pz = sum(r[8] for r in rows)
        e = sum(r[9] for r in rows)
        return px, py, pz, e

    px_b, py_b, pz_b, e_b = sum4(rows_before)
    px_a, py_a, pz_a, e_a = sum4(rows_after)
    return EventMomentumSummary(
        total_px_before=px_b,
        total_py_before=py_b,
        total_pz_before=pz_b,
        total_E_before=e_b,
        total_px_after=px_a,
        total_py_after=py_a,
        total_pz_after=pz_a,
        total_E_after=e_a,
        delta_px=px_a - px_b,
        delta_py=py_a - py_b,
        delta_pz=pz_a - pz_b,
        delta_E=e_a - e_b,
    )


def popf_paired_lhe_paths(target_u: int, target_d: int, serial: int) -> Tuple[Path, Path]:
    """
    Paths for side-by-side unkicked vs kicked LHE files.
    If only one event is requested in total (target_u + target_d == 1), use fixed
    single_event_2to4_{unkicked,kicked}.lhe names; otherwise event_NNNNNN_{unkicked,kicked}.lhe.
    """
    if target_u + target_d == 1:
        return (
            POPF_UNKICKED_DIR / "single_event_2to4_unkicked.lhe",
            POPF_KICKED_DIR / "single_event_2to4_kicked.lhe",
        )
    return (
        POPF_UNKICKED_DIR / f"event_{serial:06d}_unkicked.lhe",
        POPF_KICKED_DIR / f"event_{serial:06d}_kicked.lhe",
    )


def write_single_event_lhe(
    rows: List[List[float]],
    out_path: Path,
) -> None:
    """
    Write a minimal single-event 2->4 LHE, with the same <init> block structure
    as in build_single_2to4_lhe_from_pythia_event.py.
    """
    idprup = 1
    xwgtup = 1.0
    scalup = 10.0
    aqedup = 1.0 / 137.0
    aqcdup = 0.118
    idbmup1, idbmup2 = 11, 2212
    ebmup1, ebmup2 = 18.0, 275.0
    pdfgup1, pdfgup2, pdfsup1, pdfsup2 = 0, 0, 0, 0
    idwtup, nprup = 3, 1
    xsecup, xerrup, xmaxup, lprup = 1.0, 0.0, 1.0, idprup

    nup = len(rows)

    with out_path.open("w") as f:
        f.write('<LesHouchesEvents version="1.0">\n')
        f.write("<header>\n  <!-- SRC mixture validation 2->4 LHE -->\n</header>\n")
        f.write("<init>\n")
        f.write(
            f" {idbmup1:8d} {idbmup2:8d} {ebmup1:14.7e} {ebmup2:14.7e} "
            f"{pdfgup1:4d} {pdfgup2:4d} {pdfsup1:4d} {pdfsup2:4d} {idwtup:4d} {nprup:4d}\n"
        )
        f.write(f" {xsecup:14.7e} {xerrup:14.7e} {xmaxup:14.7e} {lprup:4d}\n")
        f.write("</init>\n<event>\n")
        f.write(
            f" {nup:4d} {idprup:4d} {xwgtup:14.7e} {scalup:14.7e} "
            f"{aqedup:14.7e} {aqcdup:14.7e}\n"
        )
        for row in rows:
            idup, istup, m1, m2, col, acol, px, py, pz, e, m = row
            f.write(
                f" {idup:8d} {istup:4d}{m1:5d}{m2:5d}{col:5d}{acol:5d}"
                f" {px:14.7e} {py:14.7e} {pz:14.7e} {e:14.7e} {m:14.7e}"
                f" {0.0:10.3e} {9.0:10.3e}\n"
            )
        f.write("</event>\n</LesHouchesEvents>\n")


def write_event_metadata(
    out_path: Path,
    accepted_index: int,
    source_event_index: int,
    struck_flavor: str,
    sample_type: str,
    draw_type: str,
    pair_ids: Tuple[int, int],
    cfg: KTConfig,
    kT1: Tuple[float, float],
    kT2: Tuple[float, float],
    mom_sum: EventMomentumSummary,
    lhe_path: Path,
    lhe_path_popf_unkicked: Path,
    lhe_path_popf_kicked: Path,
) -> None:
    meta = {
        "accepted_index": accepted_index,
        "source_event_index": source_event_index,
        "struck_flavor": struck_flavor,
        "sample_type": sample_type,
        "draw_type": draw_type,
        "process_label": "eu_to_e_ud_dbar" if sample_type == "u" else "ed_to_e_dd_dbar",
        "modified_pair_pdgs": [pair_ids[0], pair_ids[1]],
        "sigma_uncorr": cfg.sigma_uncorr,
        "q0": cfg.q0,
        "sigma_q": cfg.sigma_q,
        "sigma_src_residual": cfg.sigma_src_residual,
        "injected_kT_1": list(kT1),
        "injected_kT_2": list(kT2),
        "event_momentum_summary": asdict(mom_sum),
        "lhe_path": str(lhe_path),
        "lhe_path_popf_unkicked": str(lhe_path_popf_unkicked),
        "lhe_path_popf_kicked": str(lhe_path_popf_kicked),
    }
    out_path.write_text(json.dumps(meta, indent=2))


def update_run_summary(summary: Dict[str, Any]) -> None:
    RUN_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SRC mixture 2->4 LHE samples for struck u/d.")
    parser.add_argument("--target-u", type=int, default=2000)
    parser.add_argument("--target-d", type=int, default=2000)
    parser.add_argument("--test", action="store_true", help="Run a small smoke test (5 u + 5 d).")
    args = parser.parse_args()

    target_u = args.target_u
    target_d = args.target_d
    if args.test:
        target_u = 5
        target_d = 5

    cfg = KTConfig()

    summary: Dict[str, Any] = {
        "target_u": target_u,
        "target_d": target_d,
        "accepted_u": 0,
        "accepted_d": 0,
        "generated_total": 0,
        "generated_dis": 0,
        "skipped_non_ud": 0,
        "skipped_failed_build": 0,
        "skipped_failed_pair_id": 0,
        "skipped_failed_lhe_write": 0,
        "n_u_uncorrelated": 0,
        "n_u_src": 0,
        "n_d_uncorrelated": 0,
        "generated_u_candidates": 0,
        "generated_d_candidates": 0,
        # Diagnostics for struck flavour and topology
        "raw_struck_u": 0,
        "raw_struck_d": 0,
        "events_with_ddbar_pair": 0,
        "struck_u_with_ddbar_pair": 0,
        "struck_d_with_ddbar_pair": 0,
        "total_candidate_pairs_u": 0,
        "total_candidate_pairs_d": 0,
        "u_topology_success": 0,
        "u_topology_fail": 0,
        "d_topology_success": 0,
        "d_topology_fail": 0,
        "events_with_multiple_same_flavor_quarks": 0,
        "ambiguous_struck_assignment": 0,
        # Averages will be filled at the end
        "avg_candidate_pairs_u": 0.0,
        "avg_candidate_pairs_d": 0.0,
        "popf_pair_serial": 0,
    }

    pythia = build_pythia()
    ev = pythia.event

    while summary["accepted_u"] < target_u or summary["accepted_d"] < target_d:
        if not pythia.next():
            summary["generated_total"] += 1
            update_run_summary(summary)
            continue

        summary["generated_total"] += 1
        summary["generated_dis"] += 1
        source_event_index = summary["generated_total"]

        # Find a 2->4 topology and flavour classification, without assuming partner flavour a priori.
        topo = find_matching_2to4_topology(ev, summary)
        if topo is None:
            summary["skipped_failed_build"] += 1
            continue

        rows_24, sample_type, (idx_struck, idx_q, idx_qbar) = topo

        if sample_type == "u":
            summary["generated_u_candidates"] += 1
        elif sample_type == "d":
            summary["generated_d_candidates"] += 1

        # Respect targets per flavour
        if sample_type == "u" and summary["accepted_u"] >= target_u:
            continue
        if sample_type == "d" and summary["accepted_d"] >= target_d:
            continue

        # Unmodified hard event (before kT): deep copy so unkicked LHE matches built 2->4 rows exactly.
        rows_original = copy.deepcopy(rows_24)

        # Apply kT kicks to the two quarks (idx_struck, idx_q)
        draw_type, (kx1, ky1), (kx2, ky2) = apply_pair_kicks(
            rows_24, idx_struck, idx_q, sample_type, cfg
        )

        # Kicked copy + on-shell energy recomputation (do not mutate rows_original)
        rows_before = copy.deepcopy(rows_original)
        rows_after = copy.deepcopy(rows_original)

        for idx_lhe, (kx, ky) in zip((idx_struck, idx_q), ((kx1, ky1), (kx2, ky2))):
            row = rows_after[idx_lhe - 1]
            px_old, py_old, pz, E_old, m = row[6], row[7], row[8], row[9], row[10]
            px_new = px_old + kx
            py_new = py_old + ky
            E_new = math.sqrt(px_new * px_new + py_new * py_new + pz * pz + m * m)
            row[6], row[7], row[9] = px_new, py_new, E_new

        mom_sum = compute_event_momentum_summary(rows_before, rows_after)

        # Decide file names and counters
        if sample_type == "u":
            summary["accepted_u"] += 1
            acc_idx = summary["accepted_u"]
            sub_lhe_dir, sub_meta_dir = U_LHE_DIR, U_META_DIR
            prefix = "event_u_"
            if draw_type == "src":
                summary["n_u_src"] += 1
            else:
                summary["n_u_uncorrelated"] += 1
            struck_flavor = "u"
        else:
            summary["accepted_d"] += 1
            acc_idx = summary["accepted_d"]
            sub_lhe_dir, sub_meta_dir = D_LHE_DIR, D_META_DIR
            prefix = "event_d_"
            summary["n_d_uncorrelated"] += 1
            struck_flavor = "d"

        fname = f"{prefix}{acc_idx:06d}"
        lhe_path = sub_lhe_dir / f"{fname}.lhe"
        meta_path = sub_meta_dir / f"{fname}.json"

        popf_serial = summary["popf_pair_serial"] + 1
        path_popf_unkicked, path_popf_kicked = popf_paired_lhe_paths(target_u, target_d, popf_serial)

        try:
            print(f"Writing unkicked LHE: {path_popf_unkicked}")
            write_single_event_lhe(rows_original, path_popf_unkicked)
            print(f"Writing kicked LHE:   {path_popf_kicked}")
            write_single_event_lhe(rows_after, path_popf_kicked)
            write_single_event_lhe(rows_after, lhe_path)
            summary["popf_pair_serial"] = popf_serial
        except Exception:
            summary["skipped_failed_lhe_write"] += 1
            continue

        # Metadata: PDGs for modified pair
        id_a = rows_24[idx_struck - 1][0]
        id_b = rows_24[idx_q - 1][0]

        write_event_metadata(
            meta_path,
            accepted_index=acc_idx,
            source_event_index=source_event_index,
            struck_flavor=struck_flavor,
            sample_type=sample_type,
            draw_type=draw_type,
            pair_ids=(id_a, id_b),
            cfg=cfg,
            kT1=(kx1, ky1),
            kT2=(kx2, ky2),
            mom_sum=mom_sum,
            lhe_path=lhe_path,
            lhe_path_popf_unkicked=path_popf_unkicked,
            lhe_path_popf_kicked=path_popf_kicked,
        )

        update_run_summary(summary)

        if acc_idx % 100 == 0 or (args.test and acc_idx <= 5):
            print(
                f"[progress] accepted_u={summary['accepted_u']}/{target_u} "
                f"accepted_d={summary['accepted_d']}/{target_d} "
                f"generated_total={summary['generated_total']}"
            )

    # Final summary print and averages
    if summary["raw_struck_u"] > 0:
        summary["avg_candidate_pairs_u"] = (
            float(summary["total_candidate_pairs_u"]) / float(summary["raw_struck_u"])
        )
    if summary["raw_struck_d"] > 0:
        summary["avg_candidate_pairs_d"] = (
            float(summary["total_candidate_pairs_d"]) / float(summary["raw_struck_d"])
        )
    update_run_summary(summary)

    # Final summary print
    print("Run complete.")
    print(
        f"accepted_u={summary['accepted_u']}/{target_u} "
        f"accepted_d={summary['accepted_d']}/{target_d} "
        f"generated_total={summary['generated_total']}"
    )


if __name__ == "__main__":
    main()

