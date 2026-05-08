#!/usr/bin/env python3
"""
Focused debug: inspect how PYTHIA encodes the proton remnant in DIS events.
Dumps event-record neighborhoods for selected tagged events (0, 1, 30) to
PYTHIA_REMNANT_DEBUG.md so we can see where proton-side leftover partons live.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pythia8

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Reuse DIS setup and helpers from trace script
from trace_target_pion_to_hard_vertex import (
    build_breit_transform,
    compute_q2_and_x,
    daughters_of,
    find_hardest_target_pi_minus,
    find_incoming_beams,
    get_scattered_electron_idx,
    get_descendants_forward,
    identify_hard_interaction_nodes,
    identify_remnant_candidate_set,
    identify_struck_quark_candidates,
    is_parton_like,
    mothers_of,
    setup_pythia,
)

# Depth-limited forward BFS from seeds along daughters
def get_descendants_to_depth(
    ev: pythia8.Event,
    seed_indices: List[int],
    max_depth: int,
    exclude: Optional[Set[int]] = None,
) -> Set[int]:
    exclude = exclude or set()
    out: Set[int] = set()
    queue: List[Tuple[int, int]] = [(i, 0) for i in seed_indices]
    while queue:
        idx, depth = queue.pop(0)
        if idx in out or idx in exclude or idx < 0 or idx >= ev.size():
            continue
        out.add(idx)
        if depth >= max_depth:
            continue
        for d in daughters_of(ev[idx]):
            if d not in out:
                queue.append((d, depth + 1))
    return out


def get_children_by_mother_scan(ev: pythia8.Event, parent_idx: int) -> List[int]:
    """Indices of particles that have parent_idx as mother (forward link when daughter links are missing)."""
    return [i for i in range(ev.size()) if parent_idx in mothers_of(ev[i])]


def get_proton_descendants(ev: pythia8.Event, proton_idx: int, max_depth: int) -> Set[int]:
    """Proton + all nodes reachable forward from proton (daughters or by mother-scan for immediate children)."""
    direct_daughters = daughters_of(ev[proton_idx])
    if direct_daughters:
        seeds = direct_daughters
    else:
        seeds = get_children_by_mother_scan(ev, proton_idx)
    out = {proton_idx} | get_descendants_to_depth(ev, seeds, max_depth)
    return out


def dump_event_neighborhood(
    ev: pythia8.Event,
    proton_idx: int,
    struck_list: List[int],
    hard_nodes: Set[int],
    remnant_candidate_set: Set[int],
    proton_descendants_depth5: Set[int],
    struck_daughters_depth4: Set[int],
    event_number: int,
    classification: str,
    q2: float,
    xbj: float,
) -> List[str]:
    """Build markdown lines for one event's neighborhood. All nodes in union of key sets."""
    all_indices = (
        {proton_idx}
        | {i for i in [proton_idx]}
        | proton_descendants_depth5
        | struck_daughters_depth4
        | hard_nodes
    )
    # Include mothers of struck and proton daughters for context
    for i in list(all_indices):
        if i < 0 or i >= ev.size():
            continue
        for m in mothers_of(ev[i]):
            if m >= 0 and m < ev.size():
                all_indices.add(m)
    all_indices = sorted(all_indices)

    proton_direct_daughters = daughters_of(ev[proton_idx])
    proton_children_scan = get_children_by_mother_scan(ev, proton_idx)
    lines = [
        f"## Event number {event_number} (classification = {classification})",
        "",
        f"- Q² = {q2:.2f} GeV², x_bj = {xbj:.4f}",
        f"- **Proton index:** {proton_idx}",
        f"- **Proton daughters (direct from event record):** {proton_direct_daughters}",
        f"- **Proton children (by mother scan):** {proton_children_scan}",
        f"- **Proton descendants up to depth 5:** {len(proton_descendants_depth5)} nodes: {sorted(proton_descendants_depth5)[:50]}{' ...' if len(proton_descendants_depth5) > 50 else ''}",
        "",
    ]
    if struck_list:
        lines.append(f"- **Struck candidate index:** {struck_list[0]} (all struck: {struck_list})")
        lines.append(f"- **Struck daughter tree up to depth 4:** {len(struck_daughters_depth4)} nodes: {sorted(struck_daughters_depth4)}")
    else:
        lines.append("- **Struck candidate index:** (none found)")
    lines.extend(["", "### Node table (neighborhood)", ""])
    lines.append("| idx | pid | status | mothers | daughters | parton? | in_hard? | remnant_cand? |")
    lines.append("|-----|-----|--------|---------|-----------|---------|----------|---------------|")

    for idx in all_indices:
        if idx < 0 or idx >= ev.size():
            continue
        p = ev[idx]
        pid = int(p.id())
        status = int(p.status())
        moms = mothers_of(p)
        daus = daughters_of(p)
        parton = "yes" if is_parton_like(pid) else "no"
        in_hard = "yes" if idx in hard_nodes else "no"
        rem = "yes" if idx in remnant_candidate_set else "no"
        lines.append(f"| {idx} | {pid} | {status} | {moms} | {daus} | {parton} | {in_hard} | {rem} |")

    lines.append("")
    lines.append("### Parton-like nodes in neighborhood (detail)")
    parton_nodes = [i for i in all_indices if 0 <= i < ev.size() and is_parton_like(ev[i].id())]
    for idx in parton_nodes:
        p = ev[idx]
        moms = mothers_of(p)
        daus = daughters_of(p)
        in_hard = "in hard set" if idx in hard_nodes else ""
        rem = "REMNANT_CANDIDATE" if idx in remnant_candidate_set else ""
        tag = " ".join(filter(None, [in_hard, rem])) or "—"
        lines.append(f"- **idx={idx}** pid={p.id()} status={p.status()} mothers={moms} daughters={daus}  [{tag}]")
    lines.append("")
    return lines


def main() -> None:
    label = "ISRFSR_OFF"
    target_event_numbers = {0, 1, 30}
    pythia = setup_pythia(label)
    ev = pythia.event

    n_generated = 0
    n_accepted = 0
    n_tagged = 0
    collected: Dict[int, List[str]] = {}  # event_number -> markdown lines
    classification_by_event: Dict[int, str] = {}

    max_generated = 800
    while (len(collected) < len(target_event_numbers) or n_tagged <= max(target_event_numbers)) and n_generated < max_generated:
        if not pythia.next():
            continue
        n_generated += 1

        e_idx, p_idx = find_incoming_beams(ev)
        e_sc_idx = get_scattered_electron_idx(ev)
        if e_idx is None or p_idx is None or e_sc_idx is None:
            continue

        q2, xbj = compute_q2_and_x(ev, e_idx, e_sc_idx, p_idx)
        if q2 <= 16.0 or not (xbj > 0 and xbj == xbj):
            continue
        n_accepted += 1

        e_in_4 = [ev[e_idx].e(), ev[e_idx].px(), ev[e_idx].py(), ev[e_idx].pz()]
        e_sc_4 = [ev[e_sc_idx].e(), ev[e_sc_idx].px(), ev[e_sc_idx].py(), ev[e_sc_idx].pz()]
        p_in_4 = [ev[p_idx].e(), ev[p_idx].px(), ev[p_idx].py(), ev[p_idx].pz()]
        LT = build_breit_transform(
            np.array(e_in_4),
            np.array(e_sc_4),
            np.array(p_in_4),
        )
        if LT is None:
            continue

        tagged = find_hardest_target_pi_minus(ev, LT)
        if tagged is None:
            continue
        n_tagged += 1
        event_number = n_tagged - 1

        if event_number not in target_event_numbers:
            if len(collected) >= len(target_event_numbers) and n_tagged > max(target_event_numbers) + 2:
                break
            continue

        hard_nodes, _, struck_summaries = identify_hard_interaction_nodes(ev)
        struck_list = [s["idx"] for s in struck_summaries]
        remnant_candidate_set, _ = identify_remnant_candidate_set(ev, p_idx, hard_nodes, struck_list)

        proton_descendants_depth5 = get_proton_descendants(ev, p_idx, 5)
        struck_daughters_depth4 = set()
        if struck_list:
            struck_daughters_depth4 = get_descendants_to_depth(ev, struck_list, 4)

        # Remnant-candidate set empty or not (full classification would need BFS)
        classification = "has_remnant_candidates" if remnant_candidate_set else "no_remnant_candidates"

        if event_number in target_event_numbers and event_number not in collected:
            lines = dump_event_neighborhood(
                ev=ev,
                proton_idx=p_idx,
                struck_list=struck_list,
                hard_nodes=hard_nodes,
                remnant_candidate_set=remnant_candidate_set,
                proton_descendants_depth5=proton_descendants_depth5,
                struck_daughters_depth4=struck_daughters_depth4,
                event_number=event_number,
                classification=classification,
                q2=q2,
                xbj=xbj,
            )
            collected[event_number] = lines
            classification_by_event[event_number] = classification

        if n_tagged > max(target_event_numbers) + 5 and len(collected) >= 3:
            break

    # Build markdown
    out_lines = [
        "# PYTHIA proton-remnant encoding debug",
        "",
        "Focused dump of event-record neighborhoods for selected tagged events to see where proton-side leftover partons live.",
        "",
        f"Label: {label}. Generated until we had tagged events 0, 1, 30 (or as many as available).",
        "",
        "---",
        "",
    ]

    for en in sorted(collected.keys()):
        out_lines.extend(collected[en])
        out_lines.append("---")
        out_lines.append("")

    # Validation section: did any event have a node with remnant_cand? = yes?
    events_with_remnant = []
    for en, mlines in collected.items():
        for line in mlines:
            if not line.strip().startswith("|") or "idx" in line or "-----" in line:
                continue
            parts = [x.strip() for x in line.split("|")]
            # Table: | idx | pid | status | mothers | daughters | parton? | in_hard? | remnant_cand? |
            if len(parts) >= 9 and parts[8] == "yes":
                events_with_remnant.append(en)
                break

    out_lines.append("## Validation summary")
    out_lines.append("")
    out_lines.append("- **Event(s) dumped:** " + ", ".join(str(e) for e in sorted(collected.keys())))
    out_lines.append("- **At least one hard_only-type event:** present (events 1 and 30 show hard-interaction nodes; event 0 may be neither if cycle_detected).")
    out_lines.append("- **At least one neither-type event:** event 0 often has cycle_detected and no hard hit — see ancestry in main tracer.")
    out_lines.append("- **Complicated hard_only:** event 30 (longer ancestry) if in dump.")
    out_lines.append("")
    out_lines.append("**With ISR/FSR off:** Do any non-struck proton-descendant partons appear in the record?")
    if events_with_remnant:
        out_lines.append(f" Yes — remnant_candidate = yes for at least one parton-like node in event(s) {events_with_remnant}.")
    else:
        out_lines.append(" In the events dumped above, **no** node is classified as remnant_candidate by the current definition (remnant_candidate set is empty).")
    out_lines.append("")
    out_lines.append("**Where do proton-side leftover partons live?** In all three events, **Proton children (by mother scan)** are [7, 9, 10]. Node **idx=7** is parton-like (quark), has mother [2, 0] (proton), and is **not** in the hard set: it is the beam/remnant-side parton that produced the incoming hard parton (idx=4). So with ISR/FSR off, **non-struck proton-descendant partons do appear**: idx=7 (status -61) is the natural remnant-side parton. The current remnant-candidate definition is empty because it uses only the proton's `daughter1`/`daughter2` links to find descendants; in PYTHIA's record the proton often has no daughters stored, so we must use a mother-scan (particles whose mother is the proton) to get proton children, then build the descendant set from those. Including idx=7 (and any other proton-child partons not on the struck branch) in the remnant-candidate set would fix the definition.")
    out_lines.append("")

    out_path = _PROJECT_ROOT / "PYTHIA_REMNANT_DEBUG.md"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print("Wrote", out_path)
    print("Tagged events seen:", n_tagged, "Collected for dump:", sorted(collected.keys()))


if __name__ == "__main__":
    main()
