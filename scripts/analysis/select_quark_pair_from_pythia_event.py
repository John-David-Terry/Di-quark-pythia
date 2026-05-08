#!/usr/bin/env python3
"""
Proof-of-principle: select struck quark and a partner quark (from q' q̄' pair) from a DIS event.

Event setup: ISR on, FSR off, hadronization off so hard partons are visible.
Selection: incoming quark (status -21), struck outgoing quark (ancestry to incoming), partner quark (different flavour, with matching antiquark).
Outputs: event_dump.json, quark_selection_debug.txt, summary files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pythia8

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

OUTDIR = outputs_dir() / "popf_quark_selection_test"
OUTDIR.mkdir(parents=True, exist_ok=True)

EVENT_DUMP = OUTDIR / "event_dump.json"
QUARK_DEBUG = OUTDIR / "quark_selection_debug.txt"
SUMMARY_HI = OUTDIR / "summary_high_level.txt"
SUMMARY_LO = OUTDIR / "summary_low_level.txt"
SUMMARY_COMB = OUTDIR / "summary_combined.txt"

QUARK_ABS_IDS = {1, 2, 3, 4, 5}
MAX_EVENTS = 10_000


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
    """Set of indices reachable by following mother1/mother2 from start."""
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
    """One path from start backward via mother1 (then mother2 if needed)."""
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


def run_selection() -> Tuple[
    Optional[Dict[str, Any]],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[Dict[str, Any]],
    str,
]:
    """
    Generate until one event has >=2 final quarks and >=1 final antiquark.
    Then run selection logic. Return (event_dump_dict, incoming_idx, struck_idx, partner_idx, debug_info, status).
    """
    pythia = build_pythia()
    ev = pythia.event
    n_tried = 0

    while n_tried < MAX_EVENTS:
        if not pythia.next():
            n_tried += 1
            continue
        n_tried += 1

        # Final-state quarks and antiquarks (abs(id) in 1..5, isFinal())
        final_quarks: List[int] = []
        final_antiquarks: List[int] = []
        for i in range(ev.size()):
            p = ev[i]
            if not p.isFinal():
                continue
            aid = abs(p.id())
            if aid not in QUARK_ABS_IDS:
                continue
            if p.id() > 0:
                final_quarks.append(i)
            else:
                final_antiquarks.append(i)

        if len(final_quarks) < 2 or len(final_antiquarks) < 1:
            continue

        # Step 1: incoming struck quark (status -21, abs(id) in 1..5)
        incoming_quark_index: Optional[int] = None
        for i in range(ev.size()):
            p = ev[i]
            if p.status() != -21:
                continue
            if abs(p.id()) not in QUARK_ABS_IDS:
                continue
            incoming_quark_index = i
            break

        if incoming_quark_index is None:
            continue

        incoming_id = ev[incoming_quark_index].id()

        # Step 2: struck outgoing quark = final quark with same |id| and ancestry to incoming
        struck_quark_index: Optional[int] = None
        ancestors_struck: Dict[int, Set[int]] = {}
        for i in final_quarks:
            if abs(ev[i].id()) != abs(incoming_id):
                continue
            anc = get_ancestors(ev, i)
            ancestors_struck[i] = anc
            if incoming_quark_index in anc:
                struck_quark_index = i
                break

        if struck_quark_index is None:
            continue

        # Step 3: partner quark = first final quark (index != struck) with different flavour
        partner_quark_index: Optional[int] = None
        for i in final_quarks:
            if i == struck_quark_index:
                continue
            if abs(ev[i].id()) == abs(incoming_id):
                continue
            partner_quark_index = i
            break

        if partner_quark_index is None:
            continue

        partner_id = ev[partner_quark_index].id()

        # Step 4: verify q' q̄' pair exists (final antiquark with id == -partner_id)
        has_pair = any(ev[i].id() == -partner_id for i in final_antiquarks)
        if not has_pair:
            continue

        # Build full event dump
        event_records = [particle_record(ev, i) for i in range(ev.size())]
        event_dump = {
            "n_tried": n_tried,
            "n_particles": ev.size(),
            "particles": event_records,
            "incoming_quark_index": incoming_quark_index,
            "struck_quark_index": struck_quark_index,
            "partner_quark_index": partner_quark_index,
            "final_quark_indices": final_quarks,
            "final_antiquark_indices": final_antiquarks,
            "has_qbar_pair": has_pair,
        }

        # Debug info
        def p4(p: Dict[str, Any]) -> str:
            return f"(px={p['px']:.6g}, py={p['py']:.6g}, pz={p['pz']:.6g}, E={p['E']:.6g}, m={p['m']:.6g})"

        rec_in = event_records[incoming_quark_index]
        rec_struck = event_records[struck_quark_index]
        rec_partner = event_records[partner_quark_index]
        chain_struck = ancestry_chain(ev, struck_quark_index)
        chain_partner = ancestry_chain(ev, partner_quark_index)

        debug = {
            "incoming_quark_index": incoming_quark_index,
            "incoming_quark_4vector": p4(rec_in),
            "incoming_pdg_id": rec_in["pdg_id"],
            "struck_quark_index": struck_quark_index,
            "struck_quark_4vector": p4(rec_struck),
            "struck_pdg_id": rec_struck["pdg_id"],
            "partner_quark_index": partner_quark_index,
            "partner_quark_4vector": p4(rec_partner),
            "partner_pdg_id": rec_partner["pdg_id"],
            "final_quark_indices": final_quarks,
            "final_antiquark_indices": final_antiquarks,
            "ancestry_chain_struck": chain_struck,
            "ancestry_chain_partner": chain_partner,
            "has_qbar_pair": has_pair,
        }

        return event_dump, incoming_quark_index, struck_quark_index, partner_quark_index, debug, "PASS"

    return None, None, None, None, None, "FAIL"


def main() -> None:
    event_dump, inc_idx, struck_idx, partner_idx, debug, status = run_selection()

    # Save event dump
    if event_dump is not None:
        with EVENT_DUMP.open("w") as f:
            json.dump(event_dump, f, indent=2)
    else:
        EVENT_DUMP.write_text(json.dumps({"status": "no_valid_event", "message": "No event found within max events."}))

    # Debug file
    lines: List[str] = [
        "Quark selection debug",
        "",
        "incoming quark index: " + (str(inc_idx) if inc_idx is not None else "N/A"),
    ]
    if debug is not None:
        lines.append("incoming quark 4-vector: " + debug["incoming_quark_4vector"])
        lines.append("")
        lines.append("struck quark index: " + str(debug["struck_quark_index"]))
        lines.append("struck quark 4-vector: " + debug["struck_quark_4vector"])
        lines.append("")
        lines.append("partner quark index: " + str(debug["partner_quark_index"]))
        lines.append("partner quark 4-vector: " + debug["partner_quark_4vector"])
        lines.append("")
        lines.append("final-state quarks (indices): " + str(debug["final_quark_indices"]))
        lines.append("final-state antiquarks (indices): " + str(debug["final_antiquark_indices"]))
        lines.append("")
        lines.append("ancestry chain (struck quark, backward): " + str(debug["ancestry_chain_struck"]))
        lines.append("ancestry chain (partner quark, backward): " + str(debug["ancestry_chain_partner"]))
        lines.append("")
        lines.append("has q' q̄' pair (final antiquark id == -partner_id): " + str(debug["has_qbar_pair"]))
    QUARK_DEBUG.write_text("\n".join(lines))

    # Summaries
    if event_dump is not None and debug is not None:
        rec_in = event_dump["particles"][inc_idx]
        rec_struck = event_dump["particles"][struck_idx]
        rec_partner = event_dump["particles"][partner_idx]
        hi = [
            "Quark pair selection — HIGH LEVEL SUMMARY",
            "",
            "1) Was a valid event found? Yes.",
            f"2) Which quarks were selected? Incoming (index {inc_idx}, id={rec_in['pdg_id']}), "
            f"struck (index {struck_idx}, id={rec_struck['pdg_id']}), partner (index {partner_idx}, id={rec_partner['pdg_id']}).",
            "3) Do they share the expected ancestry? Struck quark ancestry includes the incoming quark index. Partner is a different flavour; a matching final-state antiquark exists (q' q̄' pair).",
            "",
        ]
        lo = [
            "Quark pair selection — LOW LEVEL SUMMARY",
            "",
            f"incoming_quark_index: {inc_idx}  pdg_id: {rec_in['pdg_id']}  status: {rec_in['status']}  col/acol: {rec_in['color']}/{rec_in['anticolor']}  mothers: {rec_in['mother1']},{rec_in['mother2']}",
            f"  4-vector: px={rec_in['px']:.6g} py={rec_in['py']:.6g} pz={rec_in['pz']:.6g} E={rec_in['E']:.6g} m={rec_in['m']:.6g}",
            "",
            f"struck_quark_index: {struck_idx}  pdg_id: {rec_struck['pdg_id']}  status: {rec_struck['status']}  col/acol: {rec_struck['color']}/{rec_struck['anticolor']}  mothers: {rec_struck['mother1']},{rec_struck['mother2']}",
            f"  4-vector: px={rec_struck['px']:.6g} py={rec_struck['py']:.6g} pz={rec_struck['pz']:.6g} E={rec_struck['E']:.6g} m={rec_struck['m']:.6g}",
            f"  ancestry_chain: {debug['ancestry_chain_struck']}",
            "",
            f"partner_quark_index: {partner_idx}  pdg_id: {rec_partner['pdg_id']}  status: {rec_partner['status']}  col/acol: {rec_partner['color']}/{rec_partner['anticolor']}  mothers: {rec_partner['mother1']},{rec_partner['mother2']}",
            f"  4-vector: px={rec_partner['px']:.6g} py={rec_partner['py']:.6g} pz={rec_partner['pz']:.6g} E={rec_partner['E']:.6g} m={rec_partner['m']:.6g}",
            f"  ancestry_chain: {debug['ancestry_chain_partner']}",
            "",
            "has_qbar_pair: " + str(debug["has_qbar_pair"]),
            "",
        ]
    else:
        hi = [
            "Quark pair selection — HIGH LEVEL SUMMARY",
            "",
            "1) Was a valid event found? No.",
            "2) Which quarks were selected? N/A",
            "3) Do they share the expected ancestry? N/A",
            "",
        ]
        lo = [
            "Quark pair selection — LOW LEVEL SUMMARY",
            "",
            "No event satisfying the selection criteria was found within the event limit.",
            "",
        ]

    SUMMARY_HI.write_text("\n".join(hi))
    SUMMARY_LO.write_text("\n".join(lo))
    SUMMARY_COMB.write_text(
        "HIGH LEVEL SUMMARY\n\n" + SUMMARY_HI.read_text().strip() + "\n\nLOW LEVEL SUMMARY\n\n" + SUMMARY_LO.read_text().strip() + "\n"
    )

    # Final report to stdout
    print("Status:", status)
    if status == "PASS" and inc_idx is not None and struck_idx is not None and partner_idx is not None:
        print("1) Struck and partner quarks were successfully identified.")
        print(f"2) Indices: incoming={inc_idx}, struck={struck_idx}, partner={partner_idx}")
        print("3) Partner quark appears to originate from a q' q̄' pair: Yes (matching final-state antiquark found).")
    else:
        print("1) Struck and partner quarks were NOT both identified.")
        print("2) Indices: N/A")
        print("3) Partner from q' q̄' pair: N/A")
    print(f"Outputs: {OUTDIR}")


if __name__ == "__main__":
    main()
