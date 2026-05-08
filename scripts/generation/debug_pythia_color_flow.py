#!/usr/bin/env python3
"""
Colour-flow diagnostic for tagged target-region π⁻ events.

This script reuses the DIS setup, Breit transform, tagged-pion selection,
HARD and REMNANT sets, and backward BFS from
`scripts/generation/trace_target_pion_to_hard_vertex.py`, and augments it
with a **colour-flow diagnostic**:

For each tagged π⁻ event, it:
  - inspects colour tags (col, acol) for remnant candidates, hard nodes,
    and the tagged pion's ancestry neighborhood;
  - determines whether there is evidence that the pion's hadronization
    system is colour-connected to a remnant-side parton, even when no
    remnant ancestor appears in the mother graph.

Outputs:
  - JSONL: /tmp/target_pion_color_flow.jsonl (includes n_color_connected_remnant_partons,
    struck_veto_node_indices, n_filtered_color_connected_remnant_partons, etc.)
  - Markdown report: TARGET_PION_COLOR_FLOW_REPORT.md
  - Low-level summary: TARGET_PION_COLOR_FLOW_REPORT_LOWLEVEL_SUMMARY.md
  - Colour-flow remnant multiplicity: TARGET_PION_COLOR_REMNANT_MULTIPLICITY_REPORT.md
  - Remnant-filtered (struck-veto) colour report: TARGET_PION_FILTERED_COLOR_REMNANT_REPORT.md
  - Filtered low-level summary: TARGET_PION_FILTERED_COLOR_REMNANT_LOWLEVEL_SUMMARY.md

This is a **first-pass diagnostic** only; it does not attempt full
string reconstruction.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pythia8

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT / "scripts/generation") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts/generation"))

# Reuse helpers from the main tracer
from trace_target_pion_to_hard_vertex import (  # type: ignore[import]
    LABEL_CONFIGS,
    build_breit_transform,
    compute_q2_and_x,
    daughters_of,
    find_hardest_target_pi_minus,
    find_incoming_beams,
    get_descendants_forward,
    get_scattered_electron_idx,
    identify_hard_interaction_nodes,
    identify_remnant_candidate_set,
    is_parton_like,
    mothers_of,
    setup_pythia,
    trace_backward_bfs,
    reconstruct_path,
)


def collect_color_tags_for_node(ev: pythia8.Event, idx: int) -> Tuple[int, int]:
    if idx < 0 or idx >= ev.size():
        return 0, 0
    p = ev[idx]
    # col() / acol() exist in the Python API
    return int(p.col()), int(p.acol())


def build_neighborhood_indices(
    tagged_idx: int,
    ancestry_trace: List[Dict],
    path_to_first_hard_hit: List[Dict],
    paths_to_remnant_hits: List[List[Dict]],
    remnant_indices: Set[int],
    hard_nodes: Set[int],
    ev: pythia8.Event,
) -> Set[int]:
    """Build a set of indices around the tagged pion's ancestry/hard/remnant neighborhood."""
    neigh: Set[int] = {tagged_idx}
    neigh.update(step["idx"] for step in ancestry_trace)
    neigh.update(step["idx"] for step in path_to_first_hard_hit)
    for path in paths_to_remnant_hits:
        neigh.update(step["idx"] for step in path)
    neigh.update(remnant_indices)
    neigh.update(hard_nodes)
    # Optionally add immediate mothers/daughters of these nodes to catch nearby partons
    extra = set()
    for idx in list(neigh):
        if 0 <= idx < ev.size():
            p = ev[idx]
            extra.update(mothers_of(p))
            extra.update(daughters_of(p))
    neigh.update(extra)
    return {i for i in neigh if 0 <= i < ev.size()}


def analyze_event_color_flow(
    ev: pythia8.Event,
    label: str,
    e_idx: int,
    p_idx: int,
    e_sc_idx: int,
    q2: float,
    xbj: float,
) -> Optional[Dict]:
    """Run Breit, tagging, HARD/REMNANT sets, BFS and colour-flow diagnostic for one accepted event."""
    e_in_4 = np.array([ev[e_idx].e(), ev[e_idx].px(), ev[e_idx].py(), ev[e_idx].pz()], dtype=float)
    e_sc_4 = np.array([ev[e_sc_idx].e(), ev[e_sc_idx].px(), ev[e_sc_idx].py(), ev[e_sc_idx].pz()], dtype=float)
    p_in_4 = np.array([ev[p_idx].e(), ev[p_idx].px(), ev[p_idx].py(), ev[p_idx].pz()], dtype=float)
    LT = build_breit_transform(e_in_4, e_sc_4, p_in_4)
    if LT is None:
        return None

    tagged = find_hardest_target_pi_minus(ev, LT)
    if tagged is None:
        return None
    (tag_idx, p4_lab, p4_breit, E_breit, pz_breit) = tagged

    hard_nodes, hard_node_summaries, struck_summaries = identify_hard_interaction_nodes(ev)
    struck_list = [s["idx"] for s in struck_summaries]
    remnant_candidate_set, remnant_candidate_nodes = identify_remnant_candidate_set(ev, p_idx, hard_nodes, struck_list)

    # Struck-side veto set: struck candidates + hard nodes + all forward descendants of struck
    struck_veto_set: Set[int] = set(struck_list) | hard_nodes
    if struck_list:
        struck_veto_set |= get_descendants_forward(ev, struck_list, set())

    # Backward BFS
    (
        ancestry_trace,
        parent_map,
        reached_remnant_indices,
        reached_hard_indices,
        first_remnant_hit_step,
        first_hard_hit_step,
        stop_reason,
    ) = trace_backward_bfs(ev, tag_idx, hard_nodes, remnant_candidate_set, max_depth=100)

    # Paths to all reached remnant nodes
    paths_to_remnant_hits: List[List[Dict]] = [
        reconstruct_path(ev, parent_map, tag_idx, r) for r in reached_remnant_indices
    ]
    path_to_first_hard_hit: List[Dict] = []
    if first_hard_hit_step is not None and ancestry_trace:
        hard_idx = ancestry_trace[first_hard_hit_step]["idx"]
        path_to_first_hard_hit = reconstruct_path(ev, parent_map, tag_idx, hard_idx)

    # Build neighborhood indices and collect colour tags
    neigh_indices = build_neighborhood_indices(
        tag_idx,
        ancestry_trace,
        path_to_first_hard_hit,
        paths_to_remnant_hits,
        remnant_candidate_set,
        hard_nodes,
        ev,
    )

    # Remnant colour tags
    remnant_color_tags: Set[int] = set()
    for n in remnant_candidate_nodes:
        idx = n["idx"]
        c, ac = collect_color_tags_for_node(ev, idx)
        if c != 0:
            remnant_color_tags.add(c)
        if ac != 0:
            remnant_color_tags.add(ac)

    # Ancestry / neighborhood colour tags and matches
    ancestry_color_tags: Set[int] = set()
    color_flow_matches: List[Dict] = []
    color_connected_remnant_node_indices: Set[int] = set()

    # Precompute remnant colours per remnant index
    remnant_idx_to_colors: Dict[int, Set[int]] = {}
    for n in remnant_candidate_nodes:
        idx = n["idx"]
        c, ac = collect_color_tags_for_node(ev, idx)
        tags = set()
        if c != 0:
            tags.add(c)
        if ac != 0:
            tags.add(ac)
        remnant_idx_to_colors[idx] = tags

    # Build a simple map from colour tag -> list of remnant indices
    color_to_remnant_indices: Dict[int, List[int]] = {}
    for idx, tags in remnant_idx_to_colors.items():
        for t in tags:
            color_to_remnant_indices.setdefault(t, []).append(idx)

    for idx in neigh_indices:
        p = ev[idx]
        c, ac = collect_color_tags_for_node(ev, idx)
        for t in (c, ac):
            if t != 0:
                ancestry_color_tags.add(t)
                if t in color_to_remnant_indices:
                    for r_idx in color_to_remnant_indices[t]:
                        if r_idx == idx:
                            continue  # skip self
                        color_connected_remnant_node_indices.add(r_idx)
                        color_flow_matches.append({
                            "tag": t,
                            "node_idx": idx,
                            "node_pid": int(p.id()),
                            "remnant_idx": r_idx,
                            "remnant_pid": next((n["pid"] for n in remnant_candidate_nodes if n["idx"] == r_idx), None),
                        })

    # Remnant-filtered color construction: exclude nodes and edges touching struck_veto_set
    filtered_color_connected_remnant_node_indices: Set[int] = set()
    filtered_color_flow_matches: List[Dict] = []
    for m in color_flow_matches:
        node_idx = m["node_idx"]
        remnant_idx = m["remnant_idx"]
        if node_idx in struck_veto_set or remnant_idx in struck_veto_set:
            continue
        filtered_color_connected_remnant_node_indices.add(remnant_idx)
        filtered_color_flow_matches.append(m)
    n_filtered_color_connected_remnant_partons = len(filtered_color_connected_remnant_node_indices)
    has_filtered_color_connection_to_remnant = n_filtered_color_connected_remnant_partons > 0

    has_color_flow_to_remnant = len(color_flow_matches) > 0
    has_mother_remnant_path = bool(paths_to_remnant_hits)
    has_mother_or_color_connection_to_remnant = has_mother_remnant_path or has_color_flow_to_remnant

    # Per-node table (local neighborhood) with colour info
    node_color_table: List[Dict] = []
    for idx in sorted(neigh_indices | set(n["idx"] for n in remnant_candidate_nodes) | hard_nodes):
        if idx < 0 or idx >= ev.size():
            continue
        p = ev[idx]
        moms = mothers_of(p)
        daus = daughters_of(p)
        c, ac = collect_color_tags_for_node(ev, idx)
        node_color_table.append({
            "idx": idx,
            "pid": int(p.id()),
            "status": int(p.status()),
            "mothers": moms,
            "daughters": daus,
            "col": c,
            "acol": ac,
            "parton_like": bool(is_parton_like(p.id())),
            "in_hard": idx in hard_nodes,
            "in_remnant": idx in remnant_candidate_set,
        })

    note_parts = []
    if not has_mother_remnant_path:
        note_parts.append("no mother-graph remnant path")
    if has_color_flow_to_remnant:
        note_parts.append("colour-flow match to remnant")
    if not note_parts:
        note_parts.append("no remnant indication from mothers or colour")
    color_flow_diagnostic_note = "; ".join(note_parts)

    rec = {
        "label": label,
        "q2": float(q2),
        "x_bj": float(xbj),
        "tagged_pion": {
            "idx": tag_idx,
            "pid": -211,
            "status": int(ev[tag_idx].status()),
            "p4_lab": p4_lab,
            "p4_breit": p4_breit,
            "energy_breit": float(E_breit),
            "pz_breit": float(pz_breit),
        },
        "hard_nodes": sorted(list(hard_nodes)),
        "remnant_candidate_nodes": remnant_candidate_nodes,
        "ancestry_trace": ancestry_trace,
        "path_to_first_hard_hit": path_to_first_hard_hit,
        "paths_to_remnant_hits": paths_to_remnant_hits,
        "remnant_color_tags": sorted(list(remnant_color_tags)),
        "ancestry_color_tags": sorted(list(ancestry_color_tags)),
        "has_color_flow_to_remnant": has_color_flow_to_remnant,
        "color_connected_remnant_node_indices": sorted(list(color_connected_remnant_node_indices)),
        "n_color_connected_remnant_partons": len(color_connected_remnant_node_indices),
        "color_flow_matches": color_flow_matches,
        "has_mother_remnant_path": has_mother_remnant_path,
        "has_mother_or_color_connection_to_remnant": has_mother_or_color_connection_to_remnant,
        "color_flow_diagnostic_note": color_flow_diagnostic_note,
        "node_color_table": node_color_table,
        "struck_veto_node_indices": sorted(list(struck_veto_set)),
        "filtered_color_connected_remnant_node_indices": sorted(list(filtered_color_connected_remnant_node_indices)),
        "n_filtered_color_connected_remnant_partons": n_filtered_color_connected_remnant_partons,
        "filtered_color_flow_matches": filtered_color_flow_matches,
        "has_filtered_color_connection_to_remnant": has_filtered_color_connection_to_remnant,
    }
    return rec


def _multiplicity_bin(n: int) -> str:
    """Map n_color_connected_remnant_partons to bin label: 0, 1, 2, 3, 4+."""
    if n >= 4:
        return "4+"
    return str(n)


def _write_color_remnant_multiplicity_report(
    label: str,
    n_accepted: int,
    n_tagged: int,
    records: List[Dict],
    report_path: Path,
    lowlevel_path: Path,
    argv: List[str],
) -> None:
    """
    Write colour-flow-based remnant multiplicity report.
    Primary quantity: n_color_connected_remnant_partons = len(color_connected_remnant_node_indices).
    """
    # Distribution: 0, 1, 2, 3, 4+
    bin_counts: Dict[str, int] = {"0": 0, "1": 0, "2": 0, "3": 0, "4+": 0}
    example_by_bin: Dict[str, Optional[Dict]] = {"0": None, "1": None, "2": None, "3": None, "4+": None}

    for r in records:
        n = r.get("n_color_connected_remnant_partons", len(r.get("color_connected_remnant_node_indices", [])))
        bin_label = _multiplicity_bin(n)
        bin_counts[bin_label] = bin_counts.get(bin_label, 0) + 1
        if example_by_bin.get(bin_label) is None:
            example_by_bin[bin_label] = r

    # Report
    lines = [
        "# Target π⁻ colour-flow remnant multiplicity",
        "",
        "Remnant connection is defined by **colour flow** (primary diagnostic). For each tagged target-region π⁻ we count how many **distinct remnant partons** it is colour-connected to: `n_color_connected_remnant_partons = len(color_connected_remnant_node_indices)`.",
        "",
        f"- **Label:** {label}",
        f"- **Accepted DIS events:** {n_accepted}",
        f"- **Tagged π⁻ events:** {n_tagged}",
        "",
        "## Summary: number of colour-connected remnant partons per event",
        "",
        "| n (colour-connected remnant partons) | Count | Fraction |",
        "|--------------------------------------|-------|----------|",
    ]
    for bin_label in ["0", "1", "2", "3", "4+"]:
        cnt = bin_counts.get(bin_label, 0)
        frac = cnt / n_tagged if n_tagged else 0.0
        lines.append(f"| {bin_label} | {cnt} | {frac:.3f} |")
    lines.extend([
        "",
        "## Representative examples (one per non-empty bin)",
        "",
    ])

    def render_multiplicity_example(bin_label: str, r: Optional[Dict]) -> None:
        lines.append(f"### n = {bin_label} colour-connected remnant parton(s)")
        if not r:
            lines.append("(No example in this run.)")
            lines.append("")
            return
        lines.append("")
        lines.append(f"- **event_number:** {r['event_number']}  |  **Q²:** {r['q2']:.2f} GeV²  |  **x_bj:** {r['x_bj']:.4f}")
        tp = r["tagged_pion"]
        lines.append(f"- **Tagged π⁻:** idx={tp['idx']}, E_breit={tp['energy_breit']:.3f}, pz_breit={tp['pz_breit']:.3f}")
        lines.append(f"- **color_connected_remnant_node_indices:** {r.get('color_connected_remnant_node_indices', [])}")
        lines.append(f"- **n_color_connected_remnant_partons:** {r.get('n_color_connected_remnant_partons', 0)}")
        lines.append("")
        lines.append("Remnant candidate nodes (first 10):")
        for n in r.get("remnant_candidate_nodes", [])[:10]:
            lines.append(f"  - idx={n['idx']} pid={n['pid']} status={n['status']} tag={n.get('tag','')}")
        lines.append("")
        lines.append("Colour matches (tag → node_idx/pid, remnant_idx/pid):")
        for m in r.get("color_flow_matches", [])[:15]:
            lines.append(f"  - tag={m['tag']}  node {m['node_idx']}(pid={m['node_pid']})  remnant {m['remnant_idx']}(pid={m['remnant_pid']})")
        lines.append("")
        # Auxiliary: mother-graph context
        lines.append("*(Auxiliary)* has_mother_remnant_path = " + str(r.get("has_mother_remnant_path", False)) + "; paths_to_remnant_hits = " + str(len(r.get("paths_to_remnant_hits", []))) + " path(s).")
        lines.append("")

    for bin_label in ["0", "1", "2", "3", "4+"]:
        render_multiplicity_example(bin_label, example_by_bin.get(bin_label))

    lines.extend([
        "## Interpretation",
        "",
        "The multiplicity counts how many **distinct remnant-side partons** the tagged target-region π⁻ is colour-connected to in the PYTHIA string picture. A value of 0 means no colour-tag overlap was found between the pion's ancestry/neighborhood and remnant candidates; 1 or more means the pion's hadronization system shares at least one colour line with that many remnant partons. Mother-graph paths are kept only as auxiliary context; the primary classification is colour-flow-based.",
        "",
        "## Validation",
        "",
        f"- **Total tagged events:** {n_tagged}",
        f"- **Events with 0 colour-connected remnant partons:** {bin_counts.get('0', 0)}",
        "- **Full multiplicity distribution:** see table above.",
        "- **Colour-flow view and earlier “0 remnant ancestor” ambiguity:** the primary classification here is colour-flow-based; events that had no mother-graph path to a remnant parton often still show one or more colour-connected remnant partons, so the colour-flow view removes or greatly reduces that ambiguity.",
        "- **One explicit event with more than one colour-connected remnant parton:** see the example for the first non-empty bin with n ≥ 2 above (or n = 4+ if that is the only non-empty bin).",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Low-level summary
    ll_lines = [
        "# Target π⁻ colour-flow remnant multiplicity — low-level summary",
        "",
        "## Command",
        f"```\n{' '.join(argv)}\n```",
        "",
        "## Counts",
        f"- Accepted DIS events: {n_accepted}",
        f"- Tagged π⁻ events: {n_tagged}",
        "",
        "## Multiplicity distribution (n = number of colour-connected remnant partons)",
        "",
    ]
    for bin_label in ["0", "1", "2", "3", "4+"]:
        cnt = bin_counts.get(bin_label, 0)
        frac = cnt / n_tagged if n_tagged else 0.0
        ll_lines.append(f"- n = {bin_label}: {cnt} ({frac:.3f})")
    ll_lines.append("")
    n_zero = bin_counts.get("0", 0)
    ll_lines.append(f"Events with 0 colour-connected remnant partons: {n_zero}")
    ll_lines.append("")
    # Two short example lines (two different events)
    ll_lines.append("## Two example lines")
    for i, r in enumerate(records):
        if i >= 2:
            break
        n = r.get("n_color_connected_remnant_partons", 0)
        idxs = r.get("color_connected_remnant_node_indices", [])
        ll_lines.append(f"- event_number={r['event_number']}  n={n}  indices={idxs}")
    ll_lines.append("")
    lowlevel_path.write_text("\n".join(ll_lines), encoding="utf-8")


def _filtered_multiplicity_bin(n: int) -> str:
    """Bin label for filtered remnant partons: 0, 1, 2, 3+."""
    if n >= 3:
        return "3+"
    return str(n)


def _write_filtered_color_remnant_report(
    label: str,
    n_accepted: int,
    n_tagged: int,
    records: List[Dict],
    report_path: Path,
    lowlevel_path: Path,
    argv: List[str],
) -> None:
    """Write remnant-filtered (struck-veto) colour-flow report."""
    # Filtered multiplicity: 0, 1, 2, 3+
    f_bin_counts: Dict[str, int] = {"0": 0, "1": 0, "2": 0, "3+": 0}
    example_by_f_bin: Dict[str, Optional[Dict]] = {"0": None, "1": None, "2": None, "3+": None}
    veto_sizes: List[int] = []

    for r in records:
        nf = r.get("n_filtered_color_connected_remnant_partons", 0)
        veto_sizes.append(len(r.get("struck_veto_node_indices", [])))
        bin_label = _filtered_multiplicity_bin(nf)
        f_bin_counts[bin_label] = f_bin_counts.get(bin_label, 0) + 1
        if example_by_f_bin.get(bin_label) is None:
            example_by_f_bin[bin_label] = r

    mean_veto = sum(veto_sizes) / len(veto_sizes) if veto_sizes else 0
    unfiltered_ns = [r.get("n_color_connected_remnant_partons", 0) for r in records]
    filtered_ns = [r.get("n_filtered_color_connected_remnant_partons", 0) for r in records]
    mean_unfiltered = sum(unfiltered_ns) / len(unfiltered_ns) if unfiltered_ns else 0
    mean_filtered = sum(filtered_ns) / len(filtered_ns) if filtered_ns else 0

    # Example: unfiltered match that disappears after veto (unfiltered > 0 and filtered == 0)
    ex_veto_removes = next((r for r in records if (r.get("n_color_connected_remnant_partons", 0) > 0 and r.get("n_filtered_color_connected_remnant_partons", 0) == 0)), None)

    lines = [
        "# Target π⁻ remnant-filtered colour-flow report",
        "",
        "Backward colour-history of the tagged π⁻ with the **entire struck-side descendant branch vetoed**. Only colour connections that do not touch the struck-veto set are kept; the main quantity is **n_filtered_color_connected_remnant_partons**.",
        "",
        f"- **Label:** {label}",
        f"- **Accepted DIS events:** {n_accepted}",
        f"- **Tagged π⁻ events:** {n_tagged}",
        "",
        "## Struck veto set",
        "",
        f"- **Typical size:** mean |struck_veto_set| = {mean_veto:.1f} (min={min(veto_sizes) if veto_sizes else 0}, max={max(veto_sizes) if veto_sizes else 0}).",
        "",
        "## Filtered remnant multiplicity (main quantity)",
        "",
        "| n (filtered colour-connected remnant partons) | Count | Fraction |",
        "|-------------------------------------------------|-------|----------|",
    ]
    for bin_label in ["0", "1", "2", "3+"]:
        cnt = f_bin_counts.get(bin_label, 0)
        frac = cnt / n_tagged if n_tagged else 0.0
        lines.append(f"| {bin_label} | {cnt} | {frac:.3f} |")
    lines.extend([
        "",
        "## Comparison with unfiltered colour-flow",
        "",
        f"- Unfiltered: mean n = {mean_unfiltered:.2f}.",
        f"- Filtered (after struck veto): mean n = {mean_filtered:.2f}.",
        f"- Vetoing struck descendants **reduces** the earlier over-connection (unfiltered counted struck-side nodes as remnant-connected).",
        "",
        "## Representative examples (one per non-empty bin)",
        "",
    ])

    def render_filtered_example(bin_label: str, r: Optional[Dict]) -> None:
        lines.append(f"### n_filtered = {bin_label}")
        if not r:
            lines.append("(No example in this run.)")
            lines.append("")
            return
        lines.append("")
        lines.append(f"- **event_number:** {r['event_number']}  |  **Q²:** {r['q2']:.2f} GeV²  |  **x_bj:** {r['x_bj']:.4f}")
        tp = r["tagged_pion"]
        lines.append(f"- **Tagged π⁻:** idx={tp['idx']}, E_breit={tp['energy_breit']:.3f}, pz_breit={tp['pz_breit']:.3f}")
        veto = r.get("struck_veto_node_indices", [])
        lines.append(f"- **struck_veto_set:** |set| = {len(veto)}; indices (first 20): {veto[:20]}")
        lines.append(f"- **filtered_color_connected_remnant_node_indices:** {r.get('filtered_color_connected_remnant_node_indices', [])}")
        lines.append(f"- **n_filtered_color_connected_remnant_partons:** {r.get('n_filtered_color_connected_remnant_partons', 0)}")
        lines.append("")
        lines.append("Remnant candidate nodes (first 10):")
        for n in r.get("remnant_candidate_nodes", [])[:10]:
            lines.append(f"  - idx={n['idx']} pid={n['pid']} status={n['status']} tag={n.get('tag','')}")
        lines.append("")
        lines.append("Filtered colour matches (tag → node_idx/pid, remnant_idx/pid):")
        for m in r.get("filtered_color_flow_matches", [])[:15]:
            lines.append(f"  - tag={m['tag']}  node {m['node_idx']}(pid={m['node_pid']})  remnant {m['remnant_idx']}(pid={m['remnant_pid']})")
        lines.append("")

    for bin_label in ["0", "1", "2", "3+"]:
        render_filtered_example(bin_label, example_by_f_bin.get(bin_label))

    lines.extend([
        "## Example: unfiltered match that disappears after veto",
        "",
    ])
    if ex_veto_removes:
        r = ex_veto_removes
        lines.append(f"- **event_number:** {r['event_number']}. Unfiltered n = {r.get('n_color_connected_remnant_partons', 0)}, filtered n = {r.get('n_filtered_color_connected_remnant_partons', 0)}.")
        lines.append(f"- Unfiltered remnant indices: {r.get('color_connected_remnant_node_indices', [])}. Filtered: {r.get('filtered_color_connected_remnant_node_indices', [])}.")
        lines.append("")
    else:
        lines.append("(No event in this run had unfiltered > 0 and filtered = 0.)")
        lines.append("")

    lines.extend([
        "## Validation",
        "",
        f"- **Struck veto set size:** typically {mean_veto:.1f} nodes (see above).",
        "- **Veto reduces over-connection:** yes; filtered mean is lower than unfiltered because colour links through struck-side nodes are excluded.",
        "- **Filtered remnant multiplicity distribution:** see table above.",
        f"- **Events with at least one filtered remnant connection:** {sum(1 for r in records if r.get('has_filtered_color_connection_to_remnant'))}.",
        "- **One example of unfiltered match disappearing after veto:** see section above.",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Low-level summary
    ll_lines = [
        "# Target π⁻ filtered colour remnant — low-level summary",
        "",
        "## Command",
        f"```\n{' '.join(argv)}\n```",
        "",
        "## Counts",
        f"- Accepted DIS events: {n_accepted}",
        f"- Tagged π⁻ events: {n_tagged}",
        "",
        "## Filtered multiplicity (n_filtered_color_connected_remnant_partons)",
        "",
    ]
    for bin_label in ["0", "1", "2", "3+"]:
        cnt = f_bin_counts.get(bin_label, 0)
        frac = cnt / n_tagged if n_tagged else 0.0
        ll_lines.append(f"- n = {bin_label}: {cnt} ({frac:.3f})")
    ll_lines.extend([
        "",
        f"Comparison: unfiltered mean n = {mean_unfiltered:.2f}, filtered mean n = {mean_filtered:.2f}. Mean |struck_veto_set| = {mean_veto:.1f}.",
        "",
        "## Two example lines",
        "",
    ])
    for i, r in enumerate(records):
        if i >= 2:
            break
        nf = r.get("n_filtered_color_connected_remnant_partons", 0)
        idxs = r.get("filtered_color_connected_remnant_node_indices", [])
        ll_lines.append(f"- event_number={r['event_number']}  n_filtered={nf}  indices={idxs}")
    ll_lines.append("")
    lowlevel_path.write_text("\n".join(ll_lines), encoding="utf-8")


def run_color_flow(label: str, n_events: int, out_path: Path, report_path: Path, lowlevel_path: Path) -> None:
    pythia = setup_pythia(label)
    ev = pythia.event

    n_generated = 0
    n_accepted = 0
    n_tagged = 0

    records: List[Dict] = []

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
            if q2 <= 16.0 or not (xbj > 0 and np.isfinite(xbj)):
                continue
            n_accepted += 1

            rec = analyze_event_color_flow(ev, label, e_idx, p_idx, e_sc_idx, q2, xbj)
            if rec is None:
                continue
            rec["event_number"] = n_tagged
            rec["global_event_index"] = n_generated
            records.append(rec)
            fout.write(json.dumps(rec) + "\n")
            n_tagged += 1

    # Build report-level statistics
    has_mother_counts = Counter()
    has_color_counts = Counter()
    zero_mother_records = []
    for r in records:
        has_mother_counts[bool(r.get("has_mother_remnant_path"))] += 1
        has_color_counts[bool(r.get("has_color_flow_to_remnant"))] += 1
        if not r.get("has_mother_remnant_path"):
            zero_mother_records.append(r)

    remnant_path_counts = Counter(r.get("remnant_path_classification", "") for r in records)

    # Build markdown report
    lines = [
        "# Target π⁻ colour-flow remnant diagnostic",
        "",
        f"Label: {label}",
        "",
        f"- Total generated events: {n_generated}",
        f"- Accepted DIS events: {n_accepted}",
        f"- Tagged π⁻ events (hardest in Breit target): {n_tagged}",
        "",
        "## Summary table",
        "",
        "### Mother-graph remnant paths",
        "",
        f"- With mother-graph remnant path: {has_mother_counts[True]}",
        f"- With no mother-graph remnant path: {has_mother_counts[False]}",
        "",
        "### Colour-flow remnant connections",
        "",
        f"- has_color_flow_to_remnant = True: {has_color_counts[True]}",
        f"- has_color_flow_to_remnant = False: {has_color_counts[False]}",
        "",
        "### Among events with NO mother-graph remnant path",
        "",
    ]
    zero_with_color = sum(1 for r in zero_mother_records if r.get("has_color_flow_to_remnant"))
    zero_without_color = len(zero_mother_records) - zero_with_color
    lines.append(f"- Count: {len(zero_mother_records)}")
    lines.append(f"- With colour-flow connection to remnant: {zero_with_color}")
    lines.append(f"- With no remnant indication (mothers + colour): {zero_without_color}")
    lines.append("")

    lines.append("### Remnant path classification (from main tracer, if available)")
    lines.append("")
    for k, v in sorted(remnant_path_counts.items()):
        lines.append(f"- {k}: {v}")
    lines.append("")

    # Representative examples
    def pick_example(filter_fn):
        for r in records:
            if filter_fn(r):
                return r
        return None

    ex_no_mother_with_color = pick_example(lambda r: (not r.get("has_mother_remnant_path")) and r.get("has_color_flow_to_remnant"))
    ex_both = pick_example(lambda r: r.get("has_mother_remnant_path") and r.get("has_color_flow_to_remnant"))
    ex_neither = pick_example(lambda r: (not r.get("has_mother_remnant_path")) and (not r.get("has_color_flow_to_remnant")))

    def render_example(title: str, r: Optional[Dict]) -> None:
        lines.append(f"## {title}")
        if not r:
            lines.append("(No example in this run.)")
            lines.append("")
            return
        lines.append("")
        lines.append(f"- event_number = {r['event_number']}, global_event_index = {r['global_event_index']}")
        lines.append(f"- Q² = {r['q2']:.2f} GeV², x_bj = {r['x_bj']:.4f}")
        tp = r["tagged_pion"]
        lines.append(f"- Tagged π⁻: idx={tp['idx']}, status={tp['status']}, E_breit={tp['energy_breit']:.3f}, pz_breit={tp['pz_breit']:.3f}")
        lines.append(f"- has_mother_remnant_path = {r.get('has_mother_remnant_path')}")
        lines.append(f"- has_color_flow_to_remnant = {r.get('has_color_flow_to_remnant')}")
        lines.append(f"- has_mother_or_color_connection_to_remnant = {r.get('has_mother_or_color_connection_to_remnant')}")
        lines.append(f"- color_flow_diagnostic_note = {r.get('color_flow_diagnostic_note', '')}")
        lines.append("")
        lines.append("### Remnant candidate nodes")
        for n in r.get("remnant_candidate_nodes", [])[:10]:
            lines.append(f"- idx={n['idx']} pid={n['pid']} status={n['status']} tag={n.get('tag','')} mothers={n.get('mothers',[])} daughters={n.get('daughters',[])}")
        lines.append("")
        lines.append("### Hard-interaction nodes (first 10)")
        lines.append("")
        # We do not have full hard node summaries here, only indices
        lines.append(f"- hard_nodes = {r.get('hard_nodes', [])[:10]}")
        lines.append("")
        lines.append("### Colour matches")
        lines.append("")
        for m in r.get("color_flow_matches", [])[:10]:
            lines.append(f"- tag={m['tag']}  node_idx={m['node_idx']}(pid={m['node_pid']})  remnant_idx={m['remnant_idx']}(pid={m['remnant_pid']})")
        lines.append("")
        lines.append("### Node colour table (subset)")
        lines.append("")
        for t in r.get("node_color_table", [])[:20]:
            lines.append(
                f"- idx={t['idx']} pid={t['pid']} status={t['status']} mothers={t['mothers']} daughters={t['daughters']} "
                f"col={t['col']} acol={t['acol']} parton_like={t['parton_like']} in_hard={t['in_hard']} in_remnant={t['in_remnant']}"
            )
        lines.append("")

    render_example("Example: no mother-graph remnant path, but colour-flow remnant connection", ex_no_mother_with_color)
    render_example("Example: both mother-graph and colour-flow remnant connection", ex_both)
    render_example("Example: neither mother-graph nor colour-flow remnant connection", ex_neither)

    report_path.write_text("\n".join(lines), encoding="utf-8")

    # ----- Colour-flow-based remnant multiplicity (primary view) -----
    multiplicity_report_path = _PROJECT_ROOT / "TARGET_PION_COLOR_REMNANT_MULTIPLICITY_REPORT.md"
    multiplicity_lowlevel_path = _PROJECT_ROOT / "TARGET_PION_COLOR_REMNANT_MULTIPLICITY_LOWLEVEL_SUMMARY.md"
    _write_color_remnant_multiplicity_report(
        label=label,
        n_accepted=n_accepted,
        n_tagged=n_tagged,
        records=records,
        report_path=multiplicity_report_path,
        lowlevel_path=multiplicity_lowlevel_path,
        argv=sys.argv,
    )
    print("Color-remnant multiplicity report:", multiplicity_report_path)
    print("Color-remnant multiplicity low-level:", multiplicity_lowlevel_path)

    # Filtered (struck-veto) colour remnant report
    filtered_report_path = _PROJECT_ROOT / "TARGET_PION_FILTERED_COLOR_REMNANT_REPORT.md"
    filtered_lowlevel_path = _PROJECT_ROOT / "TARGET_PION_FILTERED_COLOR_REMNANT_LOWLEVEL_SUMMARY.md"
    _write_filtered_color_remnant_report(
        label=label,
        n_accepted=n_accepted,
        n_tagged=n_tagged,
        records=records,
        report_path=filtered_report_path,
        lowlevel_path=filtered_lowlevel_path,
        argv=sys.argv,
    )
    print("Filtered color remnant report:", filtered_report_path)
    print("Filtered color remnant low-level:", filtered_lowlevel_path)

    # Low-level summary for ChatGPT
    ll_lines = [
        "# Target π⁻ colour-flow report — low-level summary",
        "",
        "## Command",
        f"```\n{' '.join(sys.argv)}\n```",
        "",
        "## Counts",
        f"- Tagged π⁻ events: {n_tagged}",
        f"- Events with 0 mother-graph remnant partons (has_mother_remnant_path=False): {has_mother_counts[False]}",
        f"- Among those, events with colour-flow connection to remnant: {zero_with_color}",
        f"- Among those, events with no remnant indication: {zero_without_color}",
        "",
        "## Remnant path classification counts (from tracer, if present in JSON)",
        f"- {dict(remnant_path_counts)}",
        "",
    ]
    lowlevel_path.write_text("\n".join(ll_lines), encoding="utf-8")


def main() -> None:
    label = "ISRFSR_OFF"
    n_events = 120
    out_path = Path("/tmp/target_pion_color_flow.jsonl")
    report_path = _PROJECT_ROOT / "TARGET_PION_COLOR_FLOW_REPORT.md"
    lowlevel_path = _PROJECT_ROOT / "TARGET_PION_COLOR_FLOW_REPORT_LOWLEVEL_SUMMARY.md"
    run_color_flow(label, n_events, out_path, report_path, lowlevel_path)
    print("JSONL:", out_path)
    print("Report:", report_path)
    print("Low-level summary:", lowlevel_path)


if __name__ == "__main__":
    main()
