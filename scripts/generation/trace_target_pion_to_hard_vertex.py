#!/usr/bin/env python3
"""
Trace the hardest pi- in the Breit target hemisphere and determine which
proton-side remnant partons are on an ancestry path to that pion.

Strategy:
  - Build Breit frame; tag hardest π⁻ in target hemisphere (pz_breit > 0, max E_breit).
  - Identify hard-interaction set (struck candidates + mothers/daughters) and a
    remnant_candidate_set (proton-side leftover partons, first-pass operational definition).
  - BFS backward from the tagged pion; track intersections with both sets.
  - Record: reached_remnant_candidates, reached_hard_interaction, which nodes hit,
    first-hit step, classification (remnant_only / hard_only / both / neither),
    and one reconstructed path to first remnant hit and first hard hit.

Caveats (first-pass implementation):
  - Remnant-candidate set is an operational definition for diagnostics, not a
    unique physical parentage. It identifies whether the tagged π⁻ ancestor graph
    intersects the candidate remnant-side parton set and/or the hard set.
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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
try:
    from diquark.analyze_events_raw import build_LT
except ImportError:
    build_LT = None  # type: ignore[assignment]

# Config (mirrors generate_events_raw / test_hadron_progenitor_tracing)
E_E = 18.0
E_P = 275.0
Q2_MIN = 16.0
BASE_SEED = 12345

LABEL_CONFIGS: Dict[str, Dict] = {
    "ISRFSR_ON": {"idA": 11, "idB": 2212, "eA": E_E, "eB": E_P, "isr_fsr_on": True, "colour_reconnect_on": False, "seed_offset": 1},
    "ISRFSR_OFF": {"idA": 11, "idB": 2212, "eA": E_E, "eB": E_P, "isr_fsr_on": False, "colour_reconnect_on": False, "seed_offset": 0},
    "ETA_ON_CRON": {"idA": 2212, "idB": 11, "eA": E_P, "eB": E_E, "isr_fsr_on": True, "colour_reconnect_on": True, "seed_offset": 100},
}


def p4_from_particle(p: pythia8.Particle) -> Tuple[float, float, float, float]:
    return float(p.e()), float(p.px()), float(p.py()), float(p.pz())


def minkowski_norm(e: float, px: float, py: float, pz: float) -> float:
    return e * e - px * px - py * py - pz * pz


def setup_pythia(label: str) -> pythia8.Pythia:
    if label not in LABEL_CONFIGS:
        raise ValueError(f"Unknown label {label}")
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
    e_idx, p_idx = None, None
    for i in range(ev.size()):
        p = ev[i]
        if p.id() == 11 and p.status() == -12:
            e_idx = i
        if p.id() == 2212 and p.status() < 0:
            p_idx = i
    return e_idx, p_idx


def get_scattered_electron_idx(ev: pythia8.Event) -> Optional[int]:
    candidates = [i for i in range(ev.size()) if ev[i].id() == 11 and ev[i].status() > 0]
    if not candidates:
        return None
    for i in candidates:
        if ev[i].status() == 44:
            return i
    return max(candidates, key=lambda i: ev[i].e())


def compute_q2_and_x(ev: pythia8.Event, e_in_idx: int, e_sc_idx: int, p_in_idx: int) -> Tuple[float, float]:
    e_in, e_sc, p_in = ev[e_in_idx], ev[e_sc_idx], ev[p_in_idx]
    Ein, pxin, pyin, pzin = p4_from_particle(e_in)
    Esc, pxsc, pysc, pzsc = p4_from_particle(e_sc)
    Pin, ppx, ppy, ppz = p4_from_particle(p_in)
    q_e = Ein - Esc
    q_px, q_py, q_pz = pxin - pxsc, pyin - pysc, pzin - pzsc
    q2 = -minkowski_norm(q_e, q_px, q_py, q_pz)
    if q2 <= 0:
        return q2, float("nan")
    Pdotq = Pin * q_e - ppx * q_px - ppy * q_py - ppz * q_pz
    if Pdotq <= 0:
        return q2, float("nan")
    return q2, q2 / (2.0 * Pdotq)


def build_breit_transform(e_in: np.ndarray, e_sc: np.ndarray, p_in: np.ndarray) -> Optional[np.ndarray]:
    if build_LT is None:
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
    Pdotq = float(p_in[0] * q[0] - p_in[1] * q[1] - p_in[2] * q[2] - p_in[3] * q[3])
    if Pdotq <= 0:
        return None
    x = Q2 / (2.0 * Pdotq)
    Ee, Ep = float(e_in[0]), float(p_in[0])
    S = 4.0 * Ee * Ep
    y = Q2 / (S * x)
    qT = float(np.hypot(q[1], q[2]))
    phiq = float(np.arctan2(q[2], q[1]))
    qmu = np.array([float(q[0]), float(q[1]), float(q[2]), float(q[3])])
    return build_LT(Ee, Ep, qmu, x, y, qT, phiq, S)


def identify_struck_quark_candidates(ev: pythia8.Event) -> List[int]:
    quark_ids = {1, 2, 3, 4, 5, 6}
    for i in range(ev.size()):
        p = ev[i]
        if abs(p.id()) in quark_ids and abs(p.status()) == 23:
            return [i]
    for i in range(ev.size()):
        p = ev[i]
        if abs(p.id()) in quark_ids and 63 <= abs(p.status()) <= 68:
            return [i]
    best_idx, bestE = None, -1.0
    all_q = []
    for i in range(ev.size()):
        p = ev[i]
        if abs(p.id()) in quark_ids and p.status() != 0:
            all_q.append(i)
            if p.e() > bestE:
                bestE = p.e()
                best_idx = i
    if best_idx is None:
        return []
    return [best_idx] + [i for i in all_q if i != best_idx]


identify_struck_candidates = identify_struck_quark_candidates  # alias for clarity


def mothers_of(p: pythia8.Particle) -> List[int]:
    m1, m2 = int(p.mother1()), int(p.mother2())
    out = []
    if m1 >= 0:
        out.append(m1)
    if m2 >= 0 and m2 != m1:
        out.append(m2)
    return out


def daughters_of(p: pythia8.Particle) -> List[int]:
    d1, d2 = int(p.daughter1()), int(p.daughter2())
    if d1 <= 0 or d2 < d1:
        return []
    return list(range(d1, d2 + 1))


def get_children_by_mother_scan(ev: pythia8.Event, parent_idx: int) -> List[int]:
    """Indices of particles that have parent_idx as mother (forward link when daughter links are missing)."""
    return [i for i in range(ev.size()) if parent_idx in mothers_of(ev[i])]


def get_descendants_forward(
    ev: pythia8.Event,
    seed_indices: List[int],
    exclude: Optional[Set[int]] = None,
    max_nodes: int = 2000,
) -> Set[int]:
    """BFS forward from seeds along daughter links. Returns set of reached indices."""
    exclude = exclude or set()
    out: Set[int] = set()
    queue = list(seed_indices)
    while queue and len(out) < max_nodes:
        idx = queue.pop(0)
        if idx in out or idx in exclude or idx < 0 or idx >= ev.size():
            continue
        out.add(idx)
        for d in daughters_of(ev[idx]):
            if d not in out:
                queue.append(d)
    return out


# Parton-like: quarks, antiquarks, gluons (PDG 1-6, 21; negative for antiparticles)
PARTON_PIDS: Set[int] = {1, 2, 3, 4, 5, 6, 21}
def is_parton_like(pid: int) -> bool:
    return abs(pid) in PARTON_PIDS


def identify_remnant_candidate_set(
    ev: pythia8.Event,
    proton_idx: int,
    hard_nodes: Set[int],
    struck_list: List[int],
) -> Tuple[Set[int], List[Dict]]:
    """
    Build a broad proton-side remnant candidate set: parton-like nodes that are
    descendants of the proton but not descendants of the struck branch.
    Returns (remnant_candidate_indices, remnant_candidate_nodes) where nodes
    list includes idx, pid, status, mothers, daughters, tag for inspection.
    First-pass operational definition; not a unique physical remnant.
    """
    # Proton-descendant seeds: use daughter links if present, else mother-scan (PYTHIA often leaves proton daughters empty)
    proton_daughters = daughters_of(ev[proton_idx])
    proton_seeds = proton_daughters if proton_daughters else get_children_by_mother_scan(ev, proton_idx)
    # All nodes reachable from proton (proton + seeds and their forward descendants)
    descendants_proton = {proton_idx} | get_descendants_forward(ev, proton_seeds, set())
    # All nodes reachable from struck by following daughters (struck branch)
    descendants_struck = get_descendants_forward(ev, struck_list, set())
    # Remnant-side: proton descendants not on struck branch, not in hard set
    remnant_indices: Set[int] = set()
    for i in descendants_proton:
        if i in hard_nodes or i in descendants_struck:
            continue
        if i < 0 or i >= ev.size():
            continue
        p = ev[i]
        if is_parton_like(p.id()):
            remnant_indices.add(i)

    # Build full node list for debugging: remnant candidates + excluded (struck, hard neighbor)
    node_list: List[Dict] = []
    seen = set()
    for i in sorted(descendants_proton):
        if i < 0 or i >= ev.size():
            continue
        p = ev[i]
        if not is_parton_like(p.id()):
            continue
        if i in seen:
            continue
        seen.add(i)
        moms = mothers_of(p)
        daus = daughters_of(p)
        if i in struck_list:
            tag = "excluded_struck"
        elif i in hard_nodes:
            tag = "excluded_hard_neighbor"
        elif i in remnant_indices:
            tag = "remnant_candidate"
        else:
            tag = "unknown"
        node_list.append({
            "idx": i,
            "pid": int(p.id()),
            "status": int(p.status()),
            "mothers": moms,
            "daughters": daus,
            "tag": tag,
        })
    return remnant_indices, node_list


def find_hardest_target_pi_minus(ev: pythia8.Event, LT: np.ndarray) -> Optional[Tuple[int, List[float], List[float], float, float]]:
    """
    Select the single hardest pi- (PID=-211) in the Breit target hemisphere (pz_breit > 0).
    Hardest = largest energy in the Breit frame.
    Returns (idx, p4_lab, p4_breit, energy_breit, pz_breit) or None.
    """
    best_idx = None
    best_E_breit = -1.0
    best_p4_lab = best_p4_breit = None
    for i in range(ev.size()):
        p = ev[i]
        if not p.isFinal() or p.id() != -211:
            continue
        p4 = np.array([p.e(), p.px(), p.py(), p.pz()], dtype=float)
        p4_b = LT @ p4
        pz_b = float(p4_b[3])
        if pz_b <= 0:
            continue
        E_b = float(p4_b[0])
        if E_b > best_E_breit:
            best_E_breit = E_b
            best_idx = i
            best_p4_lab = p4.tolist()
            best_p4_breit = p4_b.tolist()
    if best_idx is None:
        return None
    return (best_idx, best_p4_lab, best_p4_breit, best_E_breit, best_p4_breit[3])


def identify_hard_interaction_nodes(
    ev: pythia8.Event,
) -> Tuple[Set[int], List[Dict], List[Dict]]:
    """
    Build the hard-interaction node set: struck-quark candidates plus their
    immediate mothers and daughters (the photon–quark interaction neighborhood).
    Returns (hard_node_indices, node_summaries_with_tags, struck_candidate_summaries).
    """
    struck = identify_struck_quark_candidates(ev)
    hard: Set[int] = set(struck)
    for i in struck:
        p = ev[i]
        m1, m2 = int(p.mother1()), int(p.mother2())
        if m1 >= 0:
            hard.add(m1)
        if m2 >= 0:
            hard.add(m2)
        d1, d2 = int(p.daughter1()), int(p.daughter2())
        if d1 > 0 and d2 >= d1:
            for j in range(d1, d2 + 1):
                hard.add(j)
    # Summaries
    struck_summaries = []
    for i in struck:
        p = ev[i]
        e, px, py, pz = p4_from_particle(p)
        struck_summaries.append({"idx": i, "pid": int(p.id()), "status": int(p.status()), "e": e, "px": px, "py": py, "pz": pz})
    node_summaries = []
    for i in sorted(hard):
        if i < 0 or i >= ev.size():
            continue
        p = ev[i]
        e, px, py, pz = p4_from_particle(p)
        tags = []
        if i in struck:
            tags.append("struck_candidate")
        for s in struck:
            if s != i:
                p_s = ev[s]
                if i in mothers_of(p_s):
                    tags.append("mother_of_struck")
                d1, d2 = int(p_s.daughter1()), int(p_s.daughter2())
                if d1 > 0 and d2 >= d1 and d1 <= i <= d2:
                    tags.append("daughter_of_struck")
        if not tags and i in hard:
            tags.append("hard_interaction_node")
        # Photon-like: PID 22
        if abs(p.id()) == 22:
            tags.append("exchanged_boson_candidate")
        node_summaries.append({
            "idx": i,
            "pid": int(p.id()),
            "status": int(p.status()),
            "mother1": int(p.mother1()),
            "mother2": int(p.mother2()),
            "daughter1": int(p.daughter1()),
            "daughter2": int(p.daughter2()),
            "e": e, "px": px, "py": py, "pz": pz,
            "tags": list(set(tags)),
        })
    return hard, node_summaries, struck_summaries


def trace_backward_bfs(
    ev: pythia8.Event,
    start_idx: int,
    hard_nodes: Set[int],
    remnant_nodes: Set[int],
    max_depth: int = 100,
) -> Tuple[
    List[Dict],
    Dict[int, int],
    Set[int],
    Set[int],
    Optional[int],
    Optional[int],
    str,
]:
    """
    BFS backward from start_idx through mothers. Do not stop at first hard hit;
    track intersections with hard_nodes and remnant_nodes. Record parent pointers
    for path reconstruction.
    Returns (ancestry_trace, parent_map, reached_remnant_indices, reached_hard_indices,
             first_remnant_hit_step, first_hard_hit_step, stop_reason).
    """
    trace: List[Dict] = []
    visited: Set[int] = set()
    parent_map: Dict[int, int] = {}  # node -> parent index that discovered it
    queue: List[Tuple[int, int]] = [(start_idx, 0)]
    reached_remnant: Set[int] = set()
    reached_hard: Set[int] = set()
    first_remnant_hit_step: Optional[int] = None
    first_hard_hit_step: Optional[int] = None
    stop_reason = "completed"

    while queue:
        idx, depth = queue.pop(0)
        if idx in visited:
            stop_reason = "cycle_detected"
            break
        if depth > max_depth:
            stop_reason = "max_depth"
            break
        visited.add(idx)
        if idx < 0 or idx >= ev.size():
            stop_reason = "invalid_index"
            break

        p = ev[idx]
        moms = mothers_of(p)
        step_idx = len(trace)
        step = {
            "idx": idx,
            "pid": int(p.id()),
            "status": int(p.status()),
            "mothers": moms,
        }
        trace.append(step)

        if idx in remnant_nodes:
            reached_remnant.add(idx)
            if first_remnant_hit_step is None:
                first_remnant_hit_step = step_idx
        if idx in hard_nodes:
            reached_hard.add(idx)
            if first_hard_hit_step is None:
                first_hard_hit_step = step_idx

        if not moms:
            stop_reason = "no_mother"
            break
        if len(moms) > 2:
            stop_reason = "ambiguous_branch"
            break

        for m in moms:
            if m < 0 or m >= ev.size():
                stop_reason = "invalid_mother"
                break
            if m not in visited and m not in parent_map:
                parent_map[m] = idx
            if m not in visited:
                queue.append((m, depth + 1))

    return (
        trace,
        parent_map,
        reached_remnant,
        reached_hard,
        first_remnant_hit_step,
        first_hard_hit_step,
        stop_reason,
    )


def reconstruct_path(
    ev: pythia8.Event,
    parent_map: Dict[int, int],
    start_idx: int,
    hit_idx: int,
) -> List[Dict]:
    """Build one path from start_idx to hit_idx using parent_map (backward from hit to start)."""
    path: List[Dict] = []
    cur = hit_idx
    while True:
        if cur < 0 or cur >= ev.size():
            break
        p = ev[cur]
        path.append({
            "idx": cur,
            "pid": int(p.id()),
            "status": int(p.status()),
            "mothers": mothers_of(p),
        })
        if cur == start_idx:
            break
        cur = parent_map.get(cur)
        if cur is None:
            break
    path.reverse()
    return path


def classify_trace_result(reached_remnant: bool, reached_hard: bool) -> str:
    """Classification label: remnant_only, hard_only, both, neither."""
    if reached_remnant and reached_hard:
        return "both"
    if reached_remnant:
        return "remnant_only"
    if reached_hard:
        return "hard_only"
    return "neither"


def classify_remnant_paths(
    paths_to_remnant: List[List[Dict]],
    hard_nodes: Set[int],
) -> Tuple[bool, bool, bool, str, List[List[Dict]], List[List[Dict]]]:
    """
    For each path (pion -> ... -> remnant), test whether path[:-1] contains any hard node.
    Returns (has_remnant_path, has_avoiding_hard, has_through_hard, remnant_path_classification,
             paths_avoiding_hard, paths_through_hard).
    """
    paths_through: List[List[Dict]] = []
    paths_avoiding: List[List[Dict]] = []
    for path in paths_to_remnant:
        if not path:
            continue
        # Path goes tagged_pion -> ... -> remnant; check nodes before the final (remnant) node
        touches_hard = any(n["idx"] in hard_nodes for n in path[:-1])
        if touches_hard:
            paths_through.append(path)
        else:
            paths_avoiding.append(path)
    has_remnant = len(paths_to_remnant) > 0
    has_through = len(paths_through) > 0
    has_avoiding = len(paths_avoiding) > 0
    if not has_remnant:
        classification = "no_remnant_path"
    elif has_avoiding and has_through:
        classification = "both_types"
    elif has_avoiding:
        classification = "remnant_avoids_hard"
    else:
        classification = "remnant_via_hard"
    return has_remnant, has_avoiding, has_through, classification, paths_avoiding, paths_through


def write_remnant_ancestry_report(
    label: str,
    n_accepted: int,
    n_tagged: int,
    classification_counts: Dict[str, int],
    remnant_path_classification_counts: Dict[str, int],
    example_by_class: Dict[str, Optional[dict]],
    example_by_remnant_path: Dict[str, Optional[dict]],
    report_path: Path,
) -> None:
    """Write TARGET_PION_REMNANT_ANCESTRY_REPORT.md with summary and examples."""
    lines = [
        "# Target π⁻ remnant ancestry report",
        "",
        "Generated from `scripts/generation/trace_target_pion_to_hard_vertex.py` (remnant + hard set tracing).",
        "",
        "**Caveats:** Remnant-candidate set is a first-pass operational definition; this does not prove unique physical parentage. It identifies whether the tagged π⁻ ancestor graph intersects the candidate remnant-side parton set and/or the hard set.",
        "",
        "## Summary",
        "",
        f"- **Label:** {label}",
        f"- **Total accepted DIS events:** {n_accepted}",
        f"- **Total tagged π⁻ events (hardest in Breit target):** {n_tagged}",
        "",
        "### Classification counts",
        "",
        "| Classification | Count | Fraction |",
        "|-----------------|-------|----------|",
    ]
    for cls in ["remnant_only", "hard_only", "both", "neither"]:
        cnt = classification_counts.get(cls, 0)
        frac = cnt / n_tagged if n_tagged else 0
        lines.append(f"| {cls} | {cnt} | {frac:.3f} |")
    lines.extend([
        "",
        "### Remnant path classification (does any path to remnant avoid the hard set?)",
        "",
        "| Remnant path classification | Count | Fraction |",
        "|-------------------------------|-------|----------|",
    ])
    for rpc in ["no_remnant_path", "remnant_via_hard", "remnant_avoids_hard", "both_types"]:
        cnt = remnant_path_classification_counts.get(rpc, 0)
        frac = cnt / n_tagged if n_tagged else 0
        lines.append(f"| {rpc} | {cnt} | {frac:.3f} |")
    lines.extend([
        "",
        "### Validation note",
        "",
        "Remnant-candidate set = parton-like (quark/gluon) nodes that are proton descendants but not on the struck branch. Proton descendants are now built using a mother-scan fallback when the proton's daughter links are empty (see PYTHIA_REMNANT_DEBUG.md). With this fix, **remnant_candidate_nodes are non-empty in ISRFSR_OFF**; nodes like idx=7 (status -61, beam/remnant-side parton) are included as remnant_candidate.",
        "",
        "## Representative examples",
        "",
    ])

    for cls in ["remnant_only", "hard_only", "both", "neither"]:
        ex = example_by_class.get(cls)
        if not ex:
            lines.append(f"### {cls}")
            lines.append("(No example in this run.)")
            lines.append("")
            continue
        lines.append(f"### {cls}")
        lines.append("")
        lines.append(f"- **Event number:** {ex['event_number']}  |  **Q²:** {ex['q2']:.2f} GeV²  |  **x_bj:** {ex['x_bj']:.4f}")
        tp = ex["tagged_pion"]
        lines.append(f"- **Tagged pion:** idx={tp['idx']}, E_breit={tp['energy_breit']:.3f}, pz_breit={tp['pz_breit']:.3f}")
        lines.append("- **Remnant candidate nodes:**")
        rcn = ex.get("remnant_candidate_nodes", [])
        if not rcn:
            lines.append("  (none)")
        else:
            for n in rcn[:20]:
                lines.append(f"  - idx={n['idx']} pid={n['pid']} status={n['status']} tag={n.get('tag', '')} mothers={n.get('mothers', [])} daughters={n.get('daughters', [])}")
            if len(rcn) > 20:
                lines.append(f"  - ... and {len(rcn)-20} more")
        lines.append("- **Hard-interaction nodes (first 10):**")
        for ns in ex.get("hard_interaction_nodes", [])[:10]:
            lines.append(f"  - idx={ns['idx']} pid={ns['pid']} status={ns['status']} tags={ns.get('tags', [])}")
        lines.append("- **Ancestry trace:** " + str(len(ex.get("ancestry_trace", []))) + " nodes")
        path_rem = ex.get("path_to_first_remnant_hit", [])
        path_hard = ex.get("path_to_first_hard_hit", [])
        if path_rem:
            lines.append("- **Path to first remnant hit:** " + " → ".join(f"idx={s['idx']}(pid={s['pid']})" for s in path_rem))
        else:
            lines.append("- **Path to first remnant hit:** (none)")
        if path_hard:
            lines.append("- **Path to first hard hit:** " + " → ".join(f"idx={s['idx']}(pid={s['pid']})" for s in path_hard))
        else:
            lines.append("- **Path to first hard hit:** (none)")
        lines.append("")

    # Remnant path examples: one where remnant is reached only through hard, one where path avoids hard (if any)
    lines.append("## Remnant path examples (hard vs avoiding hard)")
    lines.append("")
    for rpc in ["remnant_via_hard", "remnant_avoids_hard", "both_types"]:
        ex = example_by_remnant_path.get(rpc)
        if not ex:
            continue
        lines.append(f"### {rpc}")
        lines.append("")
        lines.append(f"- **Event number:** {ex['event_number']}  |  **Q²:** {ex['q2']:.2f}  |  **x_bj:** {ex['x_bj']:.4f}")
        lines.append(f"- **remnant_path_classification:** {ex.get('remnant_path_classification', '')}")
        paths_through = ex.get("paths_to_remnant_hits_through_hard", [])
        paths_avoiding = ex.get("paths_to_remnant_hits_avoiding_hard", [])
        if paths_through:
            lines.append("- **Example path through hard (pion → ... → remnant):**")
            p = paths_through[0]
            lines.append("  " + " → ".join(f"idx={s['idx']}(pid={s['pid']})" for s in p))
        if paths_avoiding:
            lines.append("- **Example path avoiding hard (pion → ... → remnant):**")
            p = paths_avoiding[0]
            lines.append("  " + " → ".join(f"idx={s['idx']}(pid={s['pid']})" for s in p))
        lines.append("")

    with report_path.open("w") as f:
        f.write("\n".join(lines))


def write_lowlevel_summary(
    label: str,
    n_accepted: int,
    n_tagged: int,
    classification_counts: Dict[str, int],
    remnant_path_classification_counts: Dict[str, int],
    example_by_class: Dict[str, Optional[dict]],
    summary_path: Path,
    command: str,
) -> None:
    """Write short low-level summary for ChatGPT."""
    lines = [
        "# Target π⁻ remnant ancestry — low-level summary",
        "",
        "## Command",
        f"```\n{command}\n```",
        "",
        "## Counts",
        f"- Accepted DIS events: {n_accepted}",
        f"- Tagged π⁻ events: {n_tagged}",
        f"- Classification counts: {classification_counts}",
        f"- Remnant path classification counts: {remnant_path_classification_counts}",
        "",
        "## One example per class",
        "",
    ]
    for cls in ["remnant_only", "hard_only", "both", "neither"]:
        ex = example_by_class.get(cls)
        lines.append(f"### {cls}")
        if not ex:
            lines.append("(No example in this run.)")
        else:
            lines.append(f"- event_number={ex['event_number']}, Q²={ex['q2']:.2f}, x_bj={ex['x_bj']:.4f}")
            lines.append(f"- classification={ex['classification']}, reached_remnant={ex['reached_remnant_candidates']}, reached_hard={ex['reached_hard_interaction']}")
            lines.append(f"- remnant_candidate_nodes: {len(ex.get('remnant_candidate_nodes', []))} entries")
            lines.append(f"- path_to_first_remnant_hit: {len(ex.get('path_to_first_remnant_hit', []))} steps")
            lines.append(f"- path_to_first_hard_hit: {len(ex.get('path_to_first_hard_hit', []))} steps")
        lines.append("")
    with summary_path.open("w") as f:
        f.write("\n".join(lines))


def run_label(
    label: str,
    n_events: int,
    max_debug_events: int,
    out_path: Path,
    report_path: Optional[Path],
    max_depth: int,
) -> None:
    pythia = setup_pythia(label)
    ev = pythia.event

    n_generated = 0
    n_accepted = 0
    n_with_target_pi = 0
    n_tagged = 0
    n_reached_hard = 0
    n_cycles = 0
    n_invalid_mother = 0
    n_max_depth = 0
    stop_reason_counts: Dict[str, int] = {}
    stop_node_type_counts: Dict[str, int] = {}
    classification_counts: Dict[str, int] = {}
    remnant_path_classification_counts: Dict[str, int] = {}

    debug_records: List[dict] = []
    example_by_class: Dict[str, Optional[dict]] = {"remnant_only": None, "hard_only": None, "both": None, "neither": None}
    example_by_remnant_path: Dict[str, Optional[dict]] = {
        "no_remnant_path": None,
        "remnant_via_hard": None,
        "remnant_avoids_hard": None,
        "both_types": None,
    }

    with out_path.open("w") as fout:
        while n_accepted < n_events:
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
            n_accepted += 1

            e_in_4 = np.array([ev[e_idx].e(), ev[e_idx].px(), ev[e_idx].py(), ev[e_idx].pz()], dtype=float)
            e_sc_4 = np.array([ev[e_sc_idx].e(), ev[e_sc_idx].px(), ev[e_sc_idx].py(), ev[e_sc_idx].pz()], dtype=float)
            p_in_4 = np.array([ev[p_idx].e(), ev[p_idx].px(), ev[p_idx].py(), ev[p_idx].pz()], dtype=float)
            LT = build_breit_transform(e_in_4, e_sc_4, p_in_4)
            if LT is None:
                continue

            tagged = find_hardest_target_pi_minus(ev, LT)
            if tagged is None:
                continue
            n_with_target_pi += 1
            (tag_idx, p4_lab, p4_breit, E_breit, pz_breit) = tagged
            n_tagged += 1

            hard_nodes, node_summaries, struck_summaries = identify_hard_interaction_nodes(ev)
            struck_list = [s["idx"] for s in struck_summaries]
            remnant_candidate_set, remnant_candidate_nodes = identify_remnant_candidate_set(
                ev, p_idx, hard_nodes, struck_list
            )
            (
                trace,
                parent_map,
                reached_remnant_indices,
                reached_hard_indices,
                first_remnant_hit_step,
                first_hard_hit_step,
                stop_reason,
            ) = trace_backward_bfs(
                ev, tag_idx, hard_nodes, remnant_candidate_set, max_depth=max_depth
            )

            reached_remnant = bool(reached_remnant_indices)
            reached_hard = bool(reached_hard_indices)
            classification = classify_trace_result(reached_remnant, reached_hard)

            path_to_first_remnant_hit: List[Dict] = []
            if first_remnant_hit_step is not None and trace:
                hit_idx = trace[first_remnant_hit_step]["idx"]
                path_to_first_remnant_hit = reconstruct_path(ev, parent_map, tag_idx, hit_idx)
            path_to_first_hard_hit: List[Dict] = []
            if first_hard_hit_step is not None and trace:
                hit_idx = trace[first_hard_hit_step]["idx"]
                path_to_first_hard_hit = reconstruct_path(ev, parent_map, tag_idx, hit_idx)

            # All paths to every reached remnant candidate; classify by whether path avoids hard set
            paths_to_remnant_hits: List[List[Dict]] = [
                reconstruct_path(ev, parent_map, tag_idx, r) for r in reached_remnant_indices
            ]
            (
                has_remnant_path,
                has_remnant_path_avoiding_hard,
                has_remnant_path_through_hard,
                remnant_path_classification,
                paths_to_remnant_hits_avoiding_hard,
                paths_to_remnant_hits_through_hard,
            ) = classify_remnant_paths(paths_to_remnant_hits, hard_nodes)

            stop_reason_counts[stop_reason] = stop_reason_counts.get(stop_reason, 0) + 1
            if reached_hard:
                n_reached_hard += 1
                for i in reached_hard_indices:
                    if i < 0 or i >= ev.size():
                        continue
                    p = ev[i]
                    typ = None
                    if i in struck_list:
                        typ = "struck_quark_candidate"
                    else:
                        for s in struck_list:
                            if i in mothers_of(ev[s]):
                                typ = "mother_of_struck"
                                break
                            ps = ev[s]
                            d1, d2 = int(ps.daughter1()), int(ps.daughter2())
                            if d1 > 0 and d2 >= d1 and d1 <= i <= d2:
                                typ = "daughter_of_struck"
                                break
                        if typ is None:
                            typ = "exchanged_boson_candidate" if abs(p.id()) == 22 else "other_hard_node"
                    stop_node_type_counts[typ] = stop_node_type_counts.get(typ, 0) + 1
            if stop_reason == "cycle_detected":
                n_cycles += 1
            elif stop_reason == "invalid_mother" or stop_reason == "invalid_index":
                n_invalid_mother += 1
            elif stop_reason == "max_depth":
                n_max_depth += 1

            rec = {
                "label": label,
                "event_number": n_tagged - 1,
                "global_event_index": n_generated,
                "q2": float(q2),
                "x_bj": float(xbj),
                "tagged_pion": {
                    "idx": tag_idx,
                    "pid": -211,
                    "status": int(ev[tag_idx].status()),
                    "p4_lab": p4_lab,
                    "p4_breit": p4_breit,
                    "energy_breit": E_breit,
                    "pz_breit": pz_breit,
                },
                "hard_interaction_nodes": node_summaries,
                "struck_candidates": struck_summaries,
                "remnant_candidate_nodes": remnant_candidate_nodes,
                "ancestry_trace": trace,
                "stop_reason": stop_reason,
                "reached_remnant_candidates": reached_remnant,
                "reached_hard_interaction": reached_hard,
                "reached_remnant_node_indices": sorted(reached_remnant_indices),
                "reached_hard_node_indices": sorted(reached_hard_indices),
                "first_remnant_hit_step": first_remnant_hit_step,
                "first_hard_hit_step": first_hard_hit_step,
                "classification": classification,
                "path_to_first_remnant_hit": path_to_first_remnant_hit,
                "path_to_first_hard_hit": path_to_first_hard_hit,
                "has_remnant_path": has_remnant_path,
                "has_remnant_path_avoiding_hard": has_remnant_path_avoiding_hard,
                "has_remnant_path_through_hard": has_remnant_path_through_hard,
                "remnant_path_classification": remnant_path_classification,
                "all_reached_remnant_node_indices": sorted(reached_remnant_indices),
                "paths_to_remnant_hits": paths_to_remnant_hits,
                "paths_to_remnant_hits_avoiding_hard": paths_to_remnant_hits_avoiding_hard,
                "paths_to_remnant_hits_through_hard": paths_to_remnant_hits_through_hard,
                "ambiguity_notes": "",
            }
            classification_counts[classification] = classification_counts.get(classification, 0) + 1
            remnant_path_classification_counts[remnant_path_classification] = remnant_path_classification_counts.get(remnant_path_classification, 0) + 1
            if example_by_class.get(classification) is None:
                example_by_class[classification] = rec
            if example_by_remnant_path.get(remnant_path_classification) is None:
                example_by_remnant_path[remnant_path_classification] = rec
            fout.write(json.dumps(rec) + "\n")

            if len(debug_records) < max_debug_events:
                debug_records.append({
                    "ev": ev,
                    "rec": rec,
                    "label": label,
                    "n_accepted": n_accepted,
                    "tag_idx": tag_idx,
                    "trace": trace,
                    "stop_reason": stop_reason,
                    "hard_nodes": hard_nodes,
                    "node_summaries": node_summaries,
                })

    # Validation and debug output
    print("\n" + "=" * 70)
    print(f"Label: {label}")
    print("=" * 70)
    print("Validation 1 — Breit-frame selection sanity")
    print(f"  Generated events: {n_generated}")
    print(f"  Accepted DIS events: {n_accepted}")
    print(f"  Events with at least one pi- in Breit target (pz_breit>0): (implicit: events that passed to tagging)")
    print(f"  Events with a tagged hardest target-region pi-: {n_tagged}")

    print("\nValidation 2 — Remnant / hard set hits")
    print(f"  Tagged events where ancestry reached the hard-interaction set: {n_reached_hard}")
    print(f"  Classification counts: {classification_counts}")
    if n_tagged > 0:
        print(f"  Fraction reached hard: {n_reached_hard / n_tagged:.3f}")

    print("\nValidation 3 — Trace integrity")
    print(f"  Cycles detected: {n_cycles}")
    print(f"  Invalid mother/index: {n_invalid_mother}")
    print(f"  Max-depth terminations: {n_max_depth}")
    print(f"  Stop reasons: {stop_reason_counts}")

    print("\nValidation 4 — Debug examples")
    for k, d in enumerate(debug_records):
        ev = d["ev"]
        rec = d["rec"]
        print("-" * 70)
        print(f"Debug event {k+1}: event_number={rec['event_number']}, global_index={rec['global_event_index']}")
        print(f"  Q2={rec['q2']:.2f}, x_bj={rec['x_bj']:.4f}")
        tp = rec["tagged_pion"]
        print(f"  Tagged pion idx={tp['idx']}, p4_breit E={tp['energy_breit']:.3f} pz={tp['pz_breit']:.3f}")
        print("  Hard-interaction neighborhood (idx, pid, status, tags):")
        for ns in rec["hard_interaction_nodes"][:15]:
            print(f"    {ns['idx']}: pid={ns['pid']} status={ns['status']} tags={ns['tags']}")
        if len(rec["hard_interaction_nodes"]) > 15:
            print(f"    ... and {len(rec['hard_interaction_nodes'])-15} more")
        print("  Ancestry trace (first and last steps):")
        for i, step in enumerate(rec["ancestry_trace"][:5]):
            print(f"    step {i}: idx={step['idx']} pid={step['pid']} status={step['status']} mothers={step['mothers']}")
        if len(rec["ancestry_trace"]) > 5:
            print("    ...")
            for step in rec["ancestry_trace"][-3:]:
                print(f"    step: idx={step['idx']} pid={step['pid']} status={step['status']} mothers={step['mothers']}")
        print(f"  Stop reason: {rec['stop_reason']}, classification: {rec['classification']}")
        print(f"  Reached remnant: {rec['reached_remnant_node_indices']}, reached hard: {rec['reached_hard_node_indices']}")

    print("\nValidation 5 — Summary of stopping-node types")
    for typ, cnt in sorted(stop_node_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {typ}: {cnt}")

    print("\nJSONL output:", str(out_path))
    if report_path is not None:
        write_remnant_ancestry_report(
            label=label,
            n_accepted=n_accepted,
            n_tagged=n_tagged,
            classification_counts=classification_counts,
            remnant_path_classification_counts=remnant_path_classification_counts,
            example_by_class=example_by_class,
            example_by_remnant_path=example_by_remnant_path,
            report_path=report_path,
        )
        print("Report written:", str(report_path))
        summary_path = report_path.parent / (report_path.stem + "_LOWLEVEL_SUMMARY.md")
        write_lowlevel_summary(
            label=label,
            n_accepted=n_accepted,
            n_tagged=n_tagged,
            classification_counts=classification_counts,
            remnant_path_classification_counts=remnant_path_classification_counts,
            example_by_class=example_by_class,
            summary_path=summary_path,
            command=" ".join(sys.argv),
        )
        print("Low-level summary:", str(summary_path))
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace hardest Breit-target pi- to remnant and hard sets.")
    parser.add_argument("--labels", type=str, default="ISRFSR_OFF", help="Comma-separated labels")
    parser.add_argument("--n-events", type=int, default=500, help="Accepted events per label")
    parser.add_argument("--max-debug-events", type=int, default=3, help="Number of events to print in full")
    parser.add_argument("--out", type=str, default="/tmp/target_pion_remnant_ancestry.jsonl", help="Output JSONL path")
    parser.add_argument("--report", type=str, default="", help="Markdown report path (default: project root TARGET_PION_REMNANT_ANCESTRY_REPORT.md)")
    parser.add_argument("--max-depth", type=int, default=100, help="Max ancestry depth")
    args = parser.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    unknown = [l for l in labels if l not in LABEL_CONFIGS]
    if unknown:
        parser.error(f"Unknown labels: {unknown}")

    out_path = Path(args.out)
    report_path = Path(args.report) if args.report else _PROJECT_ROOT / "TARGET_PION_REMNANT_ANCESTRY_REPORT.md"
    for label in labels:
        run_label(
            label=label,
            n_events=args.n_events,
            max_debug_events=args.max_debug_events,
            out_path=out_path if len(labels) == 1 else out_path.with_name(f"{out_path.stem}_{label}{out_path.suffix}"),
            report_path=report_path if len(labels) == 1 else _PROJECT_ROOT / f"TARGET_PION_REMNANT_ANCESTRY_REPORT_{label}.md",
            max_depth=args.max_depth,
        )


if __name__ == "__main__":
    main()
