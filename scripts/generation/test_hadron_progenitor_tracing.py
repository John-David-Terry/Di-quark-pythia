#!/usr/bin/env python3
"""
Proof-of-principle ancestry tracing for DIS hadrons in PYTHIA 8.

This script reuses the same general DIS setup as generate_events_raw.py:
  - E_e = 18 GeV, E_p = 275 GeV
  - WeakBosonExchange:ff2ff(t:gmZ) = on
  - PhaseSpace:Q2Min = 16 GeV^2
  - ISR/FSR ON or OFF depending on label

Instead of writing sharded final-state arrays, it inspects the full PYTHIA
event record and attempts to trace final-state hadrons (especially pi-)
relative to the DIS **branch structure**:
  - struck-branch (outgoing hard parton line)
  - remnant-side branch (beam remnant after the struck parton is removed)

The key question is whether a given hadron’s ancestry graph reaches the
struck-side branch, the remnant-side branch, both, or neither.

Breit-frame hemisphere analysis (optional):
  For each hadron, the script computes pz_breit and pz_lab and classifies
  is_forward_lab = (pz_lab > 0) and is_target_breit = (pz_breit > 0).
  This tests whether remnant-branch-reachable hadrons lie in the Breit target
  hemisphere (z > 0). Use --forward-lab-pz-only and/or --breit-target-only to restrict the sample.

Output:
  - JSONL file with one record per accepted event, containing:
      * DIS neighborhood / branch-seed summary
      * list of selected hadrons
      * ancestry chains and branch-reachability classification for each hadron
  - Human-readable summary printed at the end with validation statistics.

This is a proof-of-principle only. It does NOT modify events, perform
re-injection, or run at production scale.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pythia8

# ---------------------------------------------------------------------------
# Configuration (mirrors scripts/generation/generate_events_raw.py)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Import Breit-frame transform from analysis codebase
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
from diquark.paths import outputs_dir  # noqa: E402

try:
    from diquark.analyze_events_raw import build_LT
except ImportError:
    build_LT = None  # type: ignore[assignment]

E_E = 18.0
E_P = 275.0
Q2_MIN = 16.0

BASE_SEED = 12345


LABEL_CONFIGS: Dict[str, Dict] = {
    "ETA_ON_CRON": {
        "idA": 2212,
        "idB": 11,
        "eA": E_P,
        "eB": E_E,
        "isr_fsr_on": True,
        "colour_reconnect_on": True,
        "seed_offset": 100,
        "description": "ColourReconnection on, beams (p,e), ISR/FSR ON",
    },
    "ISRFSR_ON": {
        "idA": 11,
        "idB": 2212,
        "eA": E_E,
        "eB": E_P,
        "isr_fsr_on": True,
        "colour_reconnect_on": False,
        "seed_offset": 1,
        "description": "ISR/FSR ON, ColourReconnection OFF, beams (e,p)",
    },
    "ISRFSR_OFF": {
        "idA": 11,
        "idB": 2212,
        "eA": E_E,
        "eB": E_P,
        "isr_fsr_on": False,
        "colour_reconnect_on": False,
        "seed_offset": 0,
        "description": "ISR/FSR OFF, ColourReconnection OFF, beams (e,p)",
    },
}


# ---------------------------------------------------------------------------
# Helper dataclasses for structured output
# ---------------------------------------------------------------------------

@dataclass
class ParticleSnapshot:
    idx: int
    id: int
    status: int
    px: float
    py: float
    pz: float
    e: float
    m: float
    mother1: int
    mother2: int


@dataclass
class TraceStep:
    idx: int
    id: int
    status: int
    mothers: List[int]


@dataclass
class HadronTraceResult:
    hadron: ParticleSnapshot
    trace_chain: List[TraceStep]
    termination_label: str
    reached_struck_branch: bool
    reached_remnant_branch: bool
    struck_hits: List[int]
    remnant_hits: List[int]
    branch_classification: str
    ambiguity_notes: str
    # Breit-frame hemisphere analysis (set when LT is available)
    pz_lab: Optional[float] = None
    pz_breit: Optional[float] = None
    pT_breit: Optional[float] = None
    is_forward_lab: bool = False
    is_target_breit: bool = False


@dataclass
class EventTraceRecord:
    label: str
    event_number: int  # local counter within this run
    global_event_index: int  # index within PYTHIA stream for this label
    q2: Optional[float]
    x_bj: Optional[float]
    has_struck_branch: bool
    has_remnant_branch: bool
    struck_branch_seed_indices: List[int]
    remnant_branch_seed_indices: List[int]
    struck_branch_primary: Optional[ParticleSnapshot]
    scattered_electron: Optional[ParticleSnapshot]
    incoming_electron: Optional[ParticleSnapshot]
    incoming_proton: Optional[ParticleSnapshot]
    hadron_traces: List[HadronTraceResult]


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def p4_from_particle(p: pythia8.Particle) -> Tuple[float, float, float, float]:
    return float(p.e()), float(p.px()), float(p.py()), float(p.pz())


def minkowski_norm(e: float, px: float, py: float, pz: float) -> float:
    return e * e - px * px - py * py - pz * pz


def snapshot_particle(p: pythia8.Particle, idx: int) -> ParticleSnapshot:
    e, px, py, pz = p4_from_particle(p)
    m2 = minkowski_norm(e, px, py, pz)
    m = float(np.sign(m2) * np.sqrt(abs(m2))) if m2 != 0.0 else 0.0
    return ParticleSnapshot(
        idx=idx,
        id=int(p.id()),
        status=int(p.status()),
        px=px,
        py=py,
        pz=pz,
        e=e,
        m=m,
        mother1=int(p.mother1()),
        mother2=int(p.mother2()),
    )


# ---------------------------------------------------------------------------
# PYTHIA setup and DIS helpers
# ---------------------------------------------------------------------------

def setup_pythia(label: str) -> pythia8.Pythia:
    """Construct a PYTHIA instance mirroring generate_events_raw.py for a given label."""
    if label not in LABEL_CONFIGS:
        raise ValueError(f"Unknown label {label}. Valid: {list(LABEL_CONFIGS.keys())}")
    cfg = LABEL_CONFIGS[label]

    p = pythia8.Pythia()
    p.readString(f"Beams:idA = {cfg['idA']}")
    p.readString(f"Beams:idB = {cfg['idB']}")
    p.readString(f"Beams:eA = {cfg['eA']}")
    p.readString(f"Beams:eB = {cfg['eB']}")
    p.readString("Beams:frameType = 2")
    p.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    p.readString("HardQCD:all = off")
    p.readString("PDF:lepton = off")
    p.readString(f"PhaseSpace:Q2Min = {Q2_MIN}")

    p.readString("HadronLevel:all = on")
    p.readString(f"ColourReconnection:reconnect = {'on' if cfg['colour_reconnect_on'] else 'off'}")

    if cfg["isr_fsr_on"]:
        p.readString("PartonLevel:ISR = on")
        p.readString("PartonLevel:FSR = on")
    else:
        p.readString("PartonLevel:ISR = off")
        p.readString("PartonLevel:FSR = off")

    p.readString("Random:setSeed = on")
    p.readString(f"Random:seed = {BASE_SEED + cfg['seed_offset']}")

    p.init()
    return p


def find_incoming_beams(ev: pythia8.Event) -> Tuple[Optional[int], Optional[int]]:
    """Locate incoming electron and proton indices (mirrors generate_events_raw logic)."""
    e_idx = None
    p_idx = None
    for i in range(ev.size()):
        p = ev[i]
        if p.id() == 11 and p.status() == -12:
            e_idx = i
        if p.id() == 2212 and p.status() < 0:
            p_idx = i
    return e_idx, p_idx


def get_scattered_electron_idx(ev: pythia8.Event) -> Optional[int]:
    """Prefer status 44; else highest-energy final-state electron (status > 0)."""
    electrons: List[int] = []
    for i in range(ev.size()):
        p = ev[i]
        if p.id() == 11 and p.status() > 0:
            electrons.append(i)
    if not electrons:
        return None
    for i in electrons:
        if ev[i].status() == 44:
            return i
    # Fallback: highest-energy
    best_i = max(electrons, key=lambda idx: ev[idx].e())
    return best_i


def compute_q2_and_x(ev: pythia8.Event, e_in_idx: int, e_sc_idx: int, p_in_idx: int) -> Tuple[float, float]:
    """Compute Q^2 and Bjorken x from incoming/scattered electron and proton."""
    e_in = ev[e_in_idx]
    e_sc = ev[e_sc_idx]
    p_in = ev[p_in_idx]

    Ein, pxin, pyin, pzin = p4_from_particle(e_in)
    Esc, pxsc, pysc, pzsc = p4_from_particle(e_sc)
    Pin, ppx, ppy, ppz = p4_from_particle(p_in)

    q_e = Ein - Esc
    q_px = pxin - pxsc
    q_py = pyin - pysc
    q_pz = pzin - pzsc

    q2 = -minkowski_norm(q_e, q_px, q_py, q_pz)
    if q2 <= 0.0:
        return q2, float("nan")

    Pdotq = Pin * q_e - ppx * q_px - ppy * q_py - ppz * q_pz
    if Pdotq <= 0.0:
        return q2, float("nan")

    x = q2 / (2.0 * Pdotq)
    return q2, x


def build_breit_transform(
    e_in: np.ndarray,
    e_sc: np.ndarray,
    p_in: np.ndarray,
    debug: bool = False,
) -> Optional[np.ndarray]:
    """
    Build Lorentz transform to Breit frame from incoming/scattered electron and proton.

    Replicates the logic of build_LT in the analysis codebase:
      q = e_in - e_sc, Q² = -q², x = Q²/(2 P·q), S = 4 Ee Ep, y = Q²/(S x).
    In the Breit frame we expect q ≈ (0, 0, 0, -Q) (q along -z, q⁰ ≈ 0).

    Returns 4x4 numpy array LT such that v_breit = LT @ v_lab, or None if build fails.
    """
    if build_LT is None:
        if debug:
            print("  [build_breit_transform] build_LT not available (import failed).")
        return None
    e_in = np.asarray(e_in, dtype=float).ravel()
    e_sc = np.asarray(e_sc, dtype=float).ravel()
    p_in = np.asarray(p_in, dtype=float).ravel()
    if e_in.size != 4 or e_sc.size != 4 or p_in.size != 4:
        return None
    q = e_in - e_sc
    Q2 = -minkowski_norm(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    if Q2 <= 0:
        return None
    Q = np.sqrt(Q2)
    Pdotq = float(p_in[0] * q[0] - p_in[1] * q[1] - p_in[2] * q[2] - p_in[3] * q[3])
    if Pdotq <= 0:
        return None
    x = Q2 / (2.0 * Pdotq)
    Ee = float(e_in[0])
    Ep = float(p_in[0])
    S = 4.0 * Ee * Ep
    y = Q2 / (S * x)
    qT = float(np.hypot(q[1], q[2]))
    phiq = float(np.arctan2(q[2], q[1]))
    qmu = np.array([float(q[0]), float(q[1]), float(q[2]), float(q[3])])
    LT = build_LT(Ee, Ep, qmu, x, y, qT, phiq, S)
    if LT is None:
        return None
    # Internal checks: transform q to Breit and verify q_T ≈ 0, q⁰ ≈ 0
    q_breit = LT @ q
    q0_b = float(q_breit[0])
    qx_b = float(q_breit[1])
    qy_b = float(q_breit[2])
    qz_b = float(q_breit[3])
    qT_breit = np.hypot(qx_b, qy_b)
    if debug:
        if abs(q0_b) > 0.1 * Q or qT_breit > 0.1 * Q:
            print(
                f"  [build_breit_transform] WARNING: q_breit not longitudinal: "
                f"q0={q0_b:.4f} qT={qT_breit:.4f} Q={Q:.4f}"
            )
        # Minkowski norm of q should be -Q² in both frames
        q2_breit = minkowski_norm(q0_b, qx_b, qy_b, qz_b)
        if abs(q2_breit + Q2) > 0.01 * Q2:
            print(f"  [build_breit_transform] WARNING: q² not preserved: q²_breit={q2_breit:.4f} -Q²={-Q2:.4f}")
    return LT


def identify_struck_quark_candidates(ev: pythia8.Event) -> List[int]:
    """
    Identify hard DIS struck-quark candidates.

    Strategy (mirrors generate_events_raw.find_k_out, but returns all candidates):
      1. All quarks (|id| in {1..6}) with |status| == 23.
      2. Else, quarks with 63 <= |status| <= 68.
      3. Else, any non-zero-status quark, selecting the highest-energy one.

    We return a list of candidate indices. The first element is the primary
    candidate; additional elements indicate ambiguity.
    """
    quark_ids = {1, 2, 3, 4, 5, 6}

    # Step 1: |status| == 23
    candidates_23: List[int] = []
    for i in range(ev.size()):
        p = ev[i]
        if abs(p.id()) in quark_ids and abs(p.status()) == 23:
            candidates_23.append(i)
    if candidates_23:
        return candidates_23

    # Step 2: 63 <= |status| <= 68
    candidates_6x: List[int] = []
    for i in range(ev.size()):
        p = ev[i]
        if abs(p.id()) in quark_ids and 63 <= abs(p.status()) <= 68:
            candidates_6x.append(i)
    if candidates_6x:
        return candidates_6x

    # Step 3: any non-zero-status quark; mark highest-energy first
    best_idx = None
    bestE = -1.0
    all_quarks: List[int] = []
    for i in range(ev.size()):
        p = ev[i]
        if abs(p.id()) in quark_ids and p.status() != 0:
            all_quarks.append(i)
            if p.e() > bestE:
                bestE = p.e()
                best_idx = i
    if best_idx is None:
        return []
    # Put highest-energy first, then others
    reordered = [best_idx] + [i for i in all_quarks if i != best_idx]
    return reordered


# ---------------------------------------------------------------------------
# Ancestry tracing
# ---------------------------------------------------------------------------

TERMINATION_LABELS = {
    "reached_beam",
    "reached_remnant",
    "reached_string_or_junction",
    "no_mother",
    "ambiguous_branch",
    "max_depth",
    "cycle_detected",
    "other",
}


def mothers_of(p: pythia8.Particle) -> List[int]:
    """Return a cleaned list of mother indices for a particle."""
    m1 = int(p.mother1())
    m2 = int(p.mother2())
    moms: List[int] = []
    if m1 >= 0:
        moms.append(m1)
    if m2 >= 0 and m2 != m1:
        moms.append(m2)
    return moms


def classify_termination(ev: pythia8.Event, idx: int) -> str:
    """Classify why ancestry tracing terminated at this particle index."""
    if idx < 0 or idx >= ev.size():
        return "other"
    p = ev[idx]
    pid = int(p.id())
    status = int(p.status())

    # Beams (incoming lepton or proton, status < 0)
    if status < 0 and (pid == 11 or pid == 2212):
        return "reached_beam"

    # Remnant-like proton or nucleon
    if abs(pid) == 2212 and status > 0 and not p.isFinal():
        return "reached_remnant"

    # String / junction / system particles (PYTHIA codes: 92, 93, 94, etc.)
    if abs(pid) in {92, 93, 94}:
        return "reached_string_or_junction"

    moms = mothers_of(p)
    if not moms:
        return "no_mother"

    return "other"


def build_branch_nodes(
    ev: pythia8.Event,
    seed_indices: List[int],
    stop_at: Optional[Set[int]] = None,
) -> Set[int]:
    """
    Build the set of nodes belonging to a branch by walking mothers upward
    from a set of seed indices until reaching any index in stop_at.

    This is used to approximate the struck-side and remnant-side branches
    in the DIS neighborhood.
    """
    if stop_at is None:
        stop_at = set()
    branch: Set[int] = set()
    queue: List[int] = list(seed_indices)
    while queue:
        idx = queue.pop()
        if idx in branch or idx in stop_at or idx < 0 or idx >= ev.size():
            continue
        branch.add(idx)
        p = ev[idx]
        for m in mothers_of(p):
            if m not in branch and m not in stop_at:
                queue.append(m)
    return branch


def classify_branch_reachability(
    reached_struck: bool,
    reached_remnant: bool,
    termination_label: str,
) -> Tuple[str, str]:
    """
    Classify a hadron according to which branches its ancestry can reach.

    Returns (branch_classification, ambiguity_notes).
    """
    if reached_remnant and not reached_struck:
        return "remnant_branch_reachable", ""
    if reached_struck and not reached_remnant:
        return "struck_branch_reachable", ""
    if reached_struck and reached_remnant:
        return "both_branches_reachable", "ancestry intersects both struck and remnant branch nodes"

    # Neither branch reached: classify by termination label
    if termination_label == "reached_string_or_junction":
        return "string_only", "trace terminated in string/junction object"
    if termination_label in {"ambiguous_branch", "cycle_detected", "max_depth"}:
        return "ambiguous", f"trace terminated with label={termination_label}"
    return "neither_branch_reachable", f"no branch seeds reached; termination={termination_label}"


def identify_dis_neighborhood(
    ev: pythia8.Event,
    e_idx: int,
    p_idx: int,
    e_sc_idx: int,
) -> Tuple[List[int], List[int], Set[int], Set[int]]:
    """
    Identify a local DIS neighborhood and approximate struck/remnant branches.

    Strategy:
      - Use identify_struck_quark_candidates to get struck-branch seeds.
      - Use the incoming proton's daughters as starting points on the proton side.
      - From those daughters, define remnant-side seeds as daughters that are
        NOT on the struck-side branch (as inferred from mother chains).

    Returns:
      struck_seeds, remnant_seeds, struck_branch_nodes, remnant_branch_nodes
    """
    struck_seeds = identify_struck_quark_candidates(ev)

    # Build struck-branch node set (walk upwards until reaching the proton)
    struck_branch_nodes: Set[int] = set()
    if struck_seeds:
        struck_branch_nodes = build_branch_nodes(ev, struck_seeds, stop_at={p_idx})

    # Proton daughters: in PYTHIA, daughter1/daughter2 define a contiguous range
    remnant_seeds: List[int] = []
    p = ev[p_idx]
    d1 = int(p.daughter1())
    d2 = int(p.daughter2())
    proton_daughters: List[int] = []
    if d1 > 0 and d2 >= d1:
        proton_daughters = list(range(d1, d2 + 1))

    for d in proton_daughters:
        if d < 0 or d >= ev.size():
            continue
        # Anything not on the struck branch is a candidate remnant seed
        if d not in struck_branch_nodes:
            remnant_seeds.append(d)

    # Build remnant-branch nodes by walking upwards from remnant seeds
    remnant_branch_nodes: Set[int] = set()
    if remnant_seeds:
        remnant_branch_nodes = build_branch_nodes(ev, remnant_seeds, stop_at={p_idx})

    return struck_seeds, remnant_seeds, struck_branch_nodes, remnant_branch_nodes


def trace_ancestry(
    ev: pythia8.Event,
    hadron_idx: int,
    struck_branch_nodes: Set[int],
    remnant_branch_nodes: Set[int],
    max_depth: int = 100,
) -> HadronTraceResult:
    """
    Trace ancestry of a final-state hadron backward through mother links.

    We walk the mother graph in a breadth-first manner, keeping a visited set
    to avoid cycles. We stop when:
      - we hit beams / remnants / strings
      - there are no more mothers
      - depth exceeds max_depth
    We record whether the ancestry intersects the struck-side or remnant-side
    branch node sets at any point.
    """
    hadron_p = ev[hadron_idx]
    hadron_snap = snapshot_particle(hadron_p, hadron_idx)

    queue: List[Tuple[int, int]] = [(hadron_idx, 0)]  # (idx, depth)
    visited: Set[int] = {hadron_idx}
    trace_steps: List[TraceStep] = []
    termination_label = "other"
    struck_hits: List[int] = []
    remnant_hits: List[int] = []

    while queue:
        idx, depth = queue.pop(0)
        if depth > max_depth:
            termination_label = "max_depth"
            break

        p = ev[idx]
        moms = mothers_of(p)
        trace_steps.append(TraceStep(idx=idx, id=int(p.id()), status=int(p.status()), mothers=moms))

        # Record branch hits (do not terminate early; we want full ancestry)
        if idx in struck_branch_nodes and idx not in struck_hits:
            struck_hits.append(idx)
        if idx in remnant_branch_nodes and idx not in remnant_hits:
            remnant_hits.append(idx)

        # No mothers: terminate
        if not moms:
            termination_label = classify_termination(ev, idx)
            break

        # Branching logic
        if len(moms) > 2 and depth > 0:
            # Highly branching ancestry, treat as ambiguous for now.
            termination_label = "ambiguous_branch"
            break

        for m in moms:
            if m < 0 or m >= ev.size():
                # Invalid index; treat as termination.
                termination_label = "other"
                continue
            if m in visited:
                termination_label = "cycle_detected"
                continue
            visited.add(m)
            queue.append((m, depth + 1))

    # If no specific label assigned yet, classify based on final index in chain
    if termination_label == "other" and trace_steps:
        termination_label = classify_termination(ev, trace_steps[-1].idx)

    reached_struck = len(struck_hits) > 0
    reached_remnant = len(remnant_hits) > 0
    branch_class, ambiguity = classify_branch_reachability(
        reached_struck=reached_struck,
        reached_remnant=reached_remnant,
        termination_label=termination_label,
    )

    return HadronTraceResult(
        hadron=hadron_snap,
        trace_chain=trace_steps,
        termination_label=termination_label,
        reached_struck_branch=reached_struck,
        reached_remnant_branch=reached_remnant,
        struck_hits=struck_hits,
        remnant_hits=remnant_hits,
        branch_classification=branch_class,
        ambiguity_notes=ambiguity,
    )


# ---------------------------------------------------------------------------
# Event loop and tracing
# ---------------------------------------------------------------------------

def get_selected_hadrons(
    ev: pythia8.Event,
    trace_all_hadrons: bool,
    trace_charged_pions: bool,
) -> List[int]:
    """
    Select final-state hadrons to trace.

    Default: pi- only (id == -211).
    Options:
      - trace_charged_pions: all charged pions (id = ±211).
      - trace_all_hadrons: any final-state hadron.
    """
    indices: List[int] = []
    for i in range(ev.size()):
        p = ev[i]
        if not p.isFinal():
            continue

        pid = int(p.id())
        if trace_all_hadrons:
            if p.isHadron():
                indices.append(i)
        elif trace_charged_pions:
            if abs(pid) == 211:
                indices.append(i)
        else:
            if pid == -211:
                indices.append(i)
    return indices


def run_tracing_for_label(
    label: str,
    n_events_target: int,
    max_debug_events: int,
    out_path: Path,
    trace_all_hadrons: bool,
    trace_charged_pions: bool,
    forward_lab_pz_only: bool = False,
    breit_target_only: bool = False,
) -> None:
    """Main loop for a single label."""
    pythia = setup_pythia(label)
    ev = pythia.event

    n_generated = 0
    n_accepted = 0
    n_with_struck_branch = 0
    n_with_remnant_branch = 0
    n_pi_minus_examined = 0

    n_cycles_detected = 0
    n_invalid_mothers = 0

    # Branch-classification statistics
    branch_counts_all: Dict[str, int] = {}
    branch_counts_pi_minus: Dict[str, int] = {}

    # Hemisphere cross-tabulation: (branch_class, is_target_breit) and (branch_class, is_forward_lab)
    n_remnant_target = 0
    n_remnant_current = 0
    n_struck_target = 0
    n_struck_current = 0
    n_both_target = 0
    n_both_current = 0
    n_remnant_pi = 0
    n_remnant_pi_target = 0
    n_struck_pi = 0
    n_struck_pi_current = 0
    n_forward_pi = 0
    n_forward_pi_target_breit = 0

    # pz_breit histograms by branch (for summary stats)
    pz_breit_remnant: List[float] = []
    pz_breit_struck: List[float] = []
    pz_breit_both: List[float] = []

    # Breit transform sanity (debug)
    n_breit_fail = 0

    # Debug examples
    debug_examples: List[EventTraceRecord] = []

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fout:
        while n_accepted < n_events_target:
            if not pythia.next():
                continue
            n_generated += 1

            e_idx, p_idx = find_incoming_beams(ev)
            e_sc_idx = get_scattered_electron_idx(ev)
            if e_idx is None or p_idx is None or e_sc_idx is None:
                continue

            q2, xbj = compute_q2_and_x(ev, e_idx, e_sc_idx, p_idx)
            if q2 <= Q2_MIN or not np.isfinite(xbj) or xbj <= 0.0:
                continue

            # Build Breit transform from event 4-vectors
            e_in_4 = np.array([ev[e_idx].e(), ev[e_idx].px(), ev[e_idx].py(), ev[e_idx].pz()], dtype=float)
            e_sc_4 = np.array([ev[e_sc_idx].e(), ev[e_sc_idx].px(), ev[e_sc_idx].py(), ev[e_sc_idx].pz()], dtype=float)
            p_in_4 = np.array([ev[p_idx].e(), ev[p_idx].px(), ev[p_idx].py(), ev[p_idx].pz()], dtype=float)
            LT = build_breit_transform(e_in_4, e_sc_4, p_in_4, debug=(n_accepted < max_debug_events))
            if LT is None:
                n_breit_fail += 1

            # Identify DIS neighborhood and branch seeds
            struck_seeds, remnant_seeds, struck_nodes, remnant_nodes = identify_dis_neighborhood(
                ev, e_idx=e_idx, p_idx=p_idx, e_sc_idx=e_sc_idx
            )
            has_struck_branch = len(struck_seeds) > 0
            has_remnant_branch = len(remnant_seeds) > 0
            if has_struck_branch:
                n_with_struck_branch += 1
            if has_remnant_branch:
                n_with_remnant_branch += 1

            struck_primary = snapshot_particle(ev[struck_seeds[0]], struck_seeds[0]) if has_struck_branch else None

            rec = EventTraceRecord(
                label=label,
                event_number=n_accepted,
                global_event_index=n_generated,
                q2=float(q2),
                x_bj=float(xbj),
                has_struck_branch=has_struck_branch,
                has_remnant_branch=has_remnant_branch,
                struck_branch_seed_indices=struck_seeds,
                remnant_branch_seed_indices=remnant_seeds,
                struck_branch_primary=struck_primary,
                scattered_electron=snapshot_particle(ev[e_sc_idx], e_sc_idx),
                incoming_electron=snapshot_particle(ev[e_idx], e_idx),
                incoming_proton=snapshot_particle(ev[p_idx], p_idx),
                hadron_traces=[],
            )

            hadron_indices = get_selected_hadrons(
                ev,
                trace_all_hadrons=trace_all_hadrons,
                trace_charged_pions=trace_charged_pions,
            )

            for hidx in hadron_indices:
                p = ev[hidx]
                p4_lab = np.array([p.e(), p.px(), p.py(), p.pz()], dtype=float)
                pz_lab = float(p4_lab[3])
                pz_breit_val: Optional[float] = None
                pT_breit_val: Optional[float] = None
                is_target_breit = False
                if LT is not None:
                    p4_breit = LT @ p4_lab
                    pz_breit_val = float(p4_breit[3])
                    pT_breit_val = float(np.hypot(p4_breit[1], p4_breit[2]))
                    is_target_breit = pz_breit_val > 0
                is_forward_lab = pz_lab > 0

                if forward_lab_pz_only and not is_forward_lab:
                    continue
                if breit_target_only and LT is not None and not is_target_breit:
                    continue

                hres = trace_ancestry(ev, hidx, struck_nodes, remnant_nodes, max_depth=100)
                hres.pz_lab = pz_lab
                hres.pz_breit = pz_breit_val
                hres.pT_breit = pT_breit_val
                hres.is_forward_lab = is_forward_lab
                hres.is_target_breit = is_target_breit
                rec.hadron_traces.append(hres)

                if hres.termination_label == "cycle_detected":
                    n_cycles_detected += 1
                for step in hres.trace_chain:
                    for m in step.mothers:
                        if m < 0 or m >= ev.size():
                            n_invalid_mothers += 1

                classification = hres.branch_classification
                branch_counts_all[classification] = branch_counts_all.get(classification, 0) + 1
                if ev[hidx].id() == -211:
                    n_pi_minus_examined += 1
                    branch_counts_pi_minus[classification] = branch_counts_pi_minus.get(classification, 0) + 1

                # Cross-tabulation and hemisphere stats
                if hres.branch_classification == "remnant_branch_reachable":
                    if hres.is_target_breit:
                        n_remnant_target += 1
                        if pz_breit_val is not None:
                            pz_breit_remnant.append(pz_breit_val)
                    else:
                        n_remnant_current += 1
                    if ev[hidx].id() == -211:
                        n_remnant_pi += 1
                        if hres.is_target_breit:
                            n_remnant_pi_target += 1
                elif hres.branch_classification == "struck_branch_reachable":
                    if hres.is_target_breit:
                        n_struck_target += 1
                    else:
                        n_struck_current += 1
                    if pz_breit_val is not None:
                        pz_breit_struck.append(pz_breit_val)
                    if ev[hidx].id() == -211:
                        n_struck_pi += 1
                        if not hres.is_target_breit:
                            n_struck_pi_current += 1
                elif hres.branch_classification == "both_branches_reachable":
                    if hres.is_target_breit:
                        n_both_target += 1
                        if pz_breit_val is not None:
                            pz_breit_both.append(pz_breit_val)
                    else:
                        n_both_current += 1
                        if pz_breit_val is not None:
                            pz_breit_both.append(pz_breit_val)

                if ev[hidx].id() == -211 and is_forward_lab:
                    n_forward_pi += 1
                    if is_target_breit:
                        n_forward_pi_target_breit += 1

            # JSONL: include hemisphere fields per hadron
            def hadron_to_json(ht: HadronTraceResult) -> dict:
                d = {
                    "hadron": asdict(ht.hadron),
                    "trace_chain": [asdict(s) for s in ht.trace_chain],
                    "termination_label": ht.termination_label,
                    "reached_struck_branch": ht.reached_struck_branch,
                    "reached_remnant_branch": ht.reached_remnant_branch,
                    "struck_hits": ht.struck_hits,
                    "remnant_hits": ht.remnant_hits,
                    "branch_classification": ht.branch_classification,
                    "ambiguity_notes": ht.ambiguity_notes,
                    "pz_lab": ht.pz_lab,
                    "pz_breit": ht.pz_breit,
                    "pT_breit": ht.pT_breit,
                    "is_forward_lab": ht.is_forward_lab,
                    "is_target_breit": ht.is_target_breit,
                }
                return d

            json_rec = {
                "label": rec.label,
                "event_number": rec.event_number,
                "global_event_index": rec.global_event_index,
                "q2": rec.q2,
                "x_bj": rec.x_bj,
                "has_struck_branch": rec.has_struck_branch,
                "has_remnant_branch": rec.has_remnant_branch,
                "struck_branch_seed_indices": rec.struck_branch_seed_indices,
                "remnant_branch_seed_indices": rec.remnant_branch_seed_indices,
                "struck_branch_primary": asdict(rec.struck_branch_primary) if rec.struck_branch_primary is not None else None,
                "scattered_electron": asdict(rec.scattered_electron) if rec.scattered_electron is not None else None,
                "incoming_electron": asdict(rec.incoming_electron) if rec.incoming_electron is not None else None,
                "incoming_proton": asdict(rec.incoming_proton) if rec.incoming_proton is not None else None,
                "hadron_traces": [hadron_to_json(ht) for ht in rec.hadron_traces],
            }
            fout.write(json.dumps(json_rec) + "\n")

            # Collect some debug examples
            if len(debug_examples) < max_debug_events:
                debug_examples.append(rec)

            n_accepted += 1

    pythia.stat()

    # ------------------------------------------------------------------
    # Human-readable summary (Validation tests 1–6)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Label: {label}")
    print("=" * 70)
    print("Validation test 1: Event-record sanity")
    print(f"  Generated events: {n_generated}")
    print(f"  Accepted events (after Q^2 and x cuts): {n_accepted}")
    print(f"  Events with a plausible struck branch: {n_with_struck_branch}")
    print(f"  Events with a plausible remnant branch: {n_with_remnant_branch}")

    print("\nValidation test 2: Mother-chain consistency")
    print(f"  Cycles detected in ancestry: {n_cycles_detected}")
    print(f"  Invalid mother references encountered: {n_invalid_mothers}")

    print("\nValidation test 3: DIS neighborhood / branch seeds")
    print("  (Representative example events with local DIS neighborhood)")
    for rec in debug_examples[:min(5, len(debug_examples))]:
        print("-" * 70)
        print(f"Event {rec.event_number} (global index {rec.global_event_index}), label={rec.label}")
        print(f"  Struck-branch seeds: {rec.struck_branch_seed_indices}")
        print(f"  Remnant-branch seeds: {rec.remnant_branch_seed_indices}")
        if rec.struck_branch_primary is not None:
            sq = rec.struck_branch_primary
            print(
                f"  Primary struck-branch seed idx={sq.idx}, id={sq.id}, status={sq.status}, "
                f"E={sq.e:.3f}, px={sq.px:.3f}, py={sq.py:.3f}, pz={sq.pz:.3f}"
            )
        # Dump a small local neighborhood around the proton and seeds
        print("  Local DIS neighborhood (idx:id(status)->mothers,daughters):")
        interesting: Set[int] = set()
        interesting.add(rec.incoming_proton.idx if rec.incoming_proton is not None else -1)
        interesting.update(rec.struck_branch_seed_indices)
        interesting.update(rec.remnant_branch_seed_indices)
        # Add daughters of these nodes
        for idx in list(interesting):
            if idx < 0 or idx >= ev.size():
                continue
            p = ev[idx]
            d1 = int(p.daughter1())
            d2 = int(p.daughter2())
            if d1 > 0 and d2 >= d1:
                for d in range(d1, d2 + 1):
                    interesting.add(d)
        for idx in sorted(i for i in interesting if 0 <= i < ev.size()):
            p = ev[idx]
            moms = mothers_of(p)
            d1 = int(p.daughter1())
            d2 = int(p.daughter2())
            daughters = []
            if d1 > 0 and d2 >= d1:
                daughters = list(range(d1, d2 + 1))
            print(
                f"    {idx}: id={p.id()} status={p.status()} "
                f"mothers={moms if moms else '-'} daughters={daughters if daughters else '-'}"
            )
        # Show up to two hadrons with full ancestry
        shown = 0
        for hres in rec.hadron_traces:
            if shown >= 2:
                break
            h = hres.hadron
            print(f"  Hadron idx={h.idx}, id={h.id}, status={h.status}, E={h.e:.3f}, px={h.px:.3f}, py={h.py:.3f}, pz={h.pz:.3f}")
            print(
                f"    termination={hres.termination_label}, "
                f"branch_class={hres.branch_classification}, "
                f"reached_struck={hres.reached_struck_branch}, "
                f"reached_remnant={hres.reached_remnant_branch}"
            )
            print("    ancestry chain (idx:id(status)->mothers):")
            for step in hres.trace_chain:
                mothers_str = ",".join(str(m) for m in step.mothers) if step.mothers else "-"
                print(f"      {step.idx}: {step.id}({step.status}) <- [{mothers_str}]")
            shown += 1

    print("\nValidation test 4: Hadron branch reachability (all selected hadrons)")
    total_hadrons = sum(branch_counts_all.values())
    if total_hadrons > 0:
        for cls, cnt in sorted(branch_counts_all.items(), key=lambda kv: -kv[1]):
            frac = cnt / total_hadrons
            print(f"  {cls:24s}: {cnt:6d} ({frac:6.3f})")
    else:
        print("  No hadrons were selected for tracing.")

    print("\nValidation test 5: Forward-pion sanity study")
    print("  (Currently no explicit forward cut implemented; statistics are for all pi-.)")
    print(f"  Number of final-state pi- examined: {n_pi_minus_examined}")
    if n_pi_minus_examined > 0:
        for cls, cnt in sorted(branch_counts_pi_minus.items(), key=lambda kv: -kv[1]):
            frac = cnt / n_pi_minus_examined
            print(f"  {cls:24s}: {cnt:6d} ({frac:6.3f})")

    print("\nValidation test 6: Mother-link integrity")
    print(f"  Cycles detected in ancestry: {n_cycles_detected}")
    print(f"  Invalid mother references encountered: {n_invalid_mothers}")

    # --- Breit-frame and hemisphere validation ---
    print("\n" + "-" * 70)
    print("Breit-frame and hemisphere analysis")
    print("-" * 70)
    print("Validation 1 — Breit transform sanity")
    print(f"  Events with failed Breit transform (LT is None): {n_breit_fail} / {n_accepted}")
    if n_accepted > 0 and n_breit_fail > 0:
        print(f"  WARNING: {100.0 * n_breit_fail / n_accepted:.1f}% of events had no valid LT.")

    print("\nValidation 2 — Hemisphere consistency (expected: remnant in target, struck in current)")
    n_remnant = n_remnant_target + n_remnant_current
    n_struck = n_struck_target + n_struck_current
    if n_remnant > 0:
        frac_remnant_in_target = n_remnant_target / n_remnant
        print(f"  Fraction of remnant_branch_reachable hadrons with pz_breit > 0: {frac_remnant_in_target:.3f} ({n_remnant_target}/{n_remnant})")
    else:
        print("  No remnant_branch_reachable hadrons.")
    if n_struck > 0:
        frac_struck_in_current = n_struck_current / n_struck
        print(f"  Fraction of struck_branch_reachable hadrons with pz_breit < 0:   {frac_struck_in_current:.3f} ({n_struck_current}/{n_struck})")
    else:
        print("  No struck_branch_reachable hadrons.")

    print("\nCross-tabulation A — All hadrons (branch vs Breit hemisphere)")
    print(f"  remnant_branch_reachable & pz_breit>0: {n_remnant_target}")
    print(f"  remnant_branch_reachable & pz_breit<0: {n_remnant_current}")
    print(f"  struck_branch_reachable   & pz_breit>0: {n_struck_target}")
    print(f"  struck_branch_reachable   & pz_breit<0: {n_struck_current}")
    print(f"  both_branches_reachable   & pz_breit>0: {n_both_target}")
    print(f"  both_branches_reachable   & pz_breit<0: {n_both_current}")

    print("\nCross-tabulation B — pi- only")
    print(f"  remnant_branch_reachable pi- in target (pz_breit>0): {n_remnant_pi_target} / {n_remnant_pi}" + (f" ({100.0*n_remnant_pi_target/max(1,n_remnant_pi):.1f}%)" if n_remnant_pi else ""))
    print(f"  struck_branch_reachable   pi- in current (pz_breit<0): {n_struck_pi_current} / {n_struck_pi}" + (f" ({100.0*n_struck_pi_current/max(1,n_struck_pi):.1f}%)" if n_struck_pi else ""))

    print("\nCross-tabulation C — Forward pi- only (LAB pz_lab > 0)")
    if n_forward_pi > 0:
        frac_forward_in_target = n_forward_pi_target_breit / n_forward_pi
        print(f"  Forward pi- with pz_breit > 0: {n_forward_pi_target_breit} / {n_forward_pi} ({frac_forward_in_target:.3f})")
        print(f"  -> P(pz_breit>0 | pz_lab>0) for pi- ≈ {frac_forward_in_target:.3f} (LAB forward vs Breit target correlation)")
    else:
        print("  No forward pi- in selected sample.")

    print("\nHemisphere correlation statistics")
    if n_remnant > 0:
        print(f"  fraction_remnant_in_target = N(remnant & pz_breit>0) / N(remnant) = {n_remnant_target}/{n_remnant} = {n_remnant_target/max(1,n_remnant):.3f}")
    if n_struck > 0:
        print(f"  fraction_struck_in_current = N(struck & pz_breit<0) / N(struck) = {n_struck_current}/{n_struck} = {n_struck_current/max(1,n_struck):.3f}")

    print("\nValidation 3 — LAB vs Breit comparison (pi-)")
    if n_forward_pi > 0:
        print(f"  P(pz_lab > 0 | pz_breit > 0) for pi-: {n_forward_pi_target_breit}/{n_forward_pi} = {n_forward_pi_target_breit/max(1,n_forward_pi):.3f}")
        print("  (Forward LAB pi- that also lie in Breit target hemisphere)")

    if pz_breit_remnant or pz_breit_struck or pz_breit_both:
        print("\npz_breit distribution summary (by branch classification)")
        for name, vals in [("remnant_branch_reachable", pz_breit_remnant), ("struck_branch_reachable", pz_breit_struck), ("both_branches_reachable", pz_breit_both)]:
            if vals:
                arr = np.array(vals)
                print(f"  {name}: n={len(arr)}, mean={float(np.mean(arr)):.4f}, std={float(np.std(arr)):.4f}, min={float(np.min(arr)):.4f}, max={float(np.max(arr)):.4f}")

    print("\nJSONL output written to:", str(out_path))
    print("  (Each hadron record includes pz_lab, pz_breit, is_forward_lab, is_target_breit.)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Proof-of-principle ancestry tracing of final-state hadrons relative to "
            "DIS struck and remnant branches using the full PYTHIA 8 event record."
        )
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="ISRFSR_OFF",
        help=f"Comma-separated labels to run (default: ISRFSR_OFF). Choices: {list(LABEL_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=100,
        help="Number of accepted events (after DIS cuts) to trace per label (default: 100).",
    )
    parser.add_argument(
        "--max-debug-events",
        type=int,
        default=5,
        help="Maximum number of events to include in detailed debug printout (default: 5).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(outputs_dir() / "hadron_progenitor_traces.jsonl"),
        help="Output JSONL path (default: under DIQUARK_DATA_ROOT/outputs/).",
    )
    parser.add_argument(
        "--trace-all-hadrons",
        action="store_true",
        help="If set, trace all final-state hadrons instead of just pions.",
    )
    parser.add_argument(
        "--trace-charged-pions",
        action="store_true",
        help="If set, trace all charged pions (id = ±211) instead of just pi- (id = -211).",
    )
    parser.add_argument(
        "--forward-lab-pz-only",
        action="store_true",
        help="If set, restrict hadrons to LAB forward hemisphere (pz_lab > 0).",
    )
    parser.add_argument(
        "--breit-target-only",
        action="store_true",
        help="If set, restrict hadrons to Breit target hemisphere (pz_breit > 0).",
    )

    args = parser.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    unknown = [l for l in labels if l not in LABEL_CONFIGS]
    if unknown:
        parser.error(f"Unknown label(s): {unknown}. Valid: {list(LABEL_CONFIGS.keys())}")

    out_path = Path(args.out)

    for label in labels:
        print("\n" + "=" * 70)
        print(f"Running ancestry tracing for label: {label}")
        print("=" * 70)
        run_tracing_for_label(
            label=label,
            n_events_target=args.n_events,
            max_debug_events=args.max_debug_events,
            out_path=out_path if len(labels) == 1 else out_path.with_name(f"{out_path.stem}_{label}{out_path.suffix}"),
            trace_all_hadrons=args.trace_all_hadrons,
            trace_charged_pions=args.trace_charged_pions,
            forward_lab_pz_only=args.forward_lab_pz_only,
            breit_target_only=args.breit_target_only,
        )


if __name__ == "__main__":
    main()

