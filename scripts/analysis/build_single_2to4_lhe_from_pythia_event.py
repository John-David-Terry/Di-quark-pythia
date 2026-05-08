#!/usr/bin/env python3
"""
Proof-of-principle: build a single 2→4 DIS hard event from a real PYTHIA event and write LHE files.

Strategy:
- Generate DIS events (same config as quark-selection test) until one has:
  ≥2 final quarks, ≥1 final antiquark, identifiable struck quark, partner q + matching qbar.
- Save full event to source_event_dump.json and selected particles to selected_hard_particles.json.
- Build 2→2 baseline LHE (e + q_in → e' + struck_q) and 2→4 test LHE (e + q_in → e' + struck_q + q' + qbar').
- Incoming particles kept beam-collinear (px=0, py=0). Color: struck line 101; partner pair 102.
- Write lhe_2to4_debug.txt with full diagnostics.

Output dir: outputs/popf_lhe_single_event_2to4/
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import pythia8
except ImportError:
    pythia8 = None  # type: ignore[assignment]

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

OUTDIR = outputs_dir() / "popf_lhe_single_event_2to4"
OUTDIR.mkdir(parents=True, exist_ok=True)

SOURCE_EVENT_DUMP = OUTDIR / "source_event_dump.json"
SELECTED_HARD = OUTDIR / "selected_hard_particles.json"
LHE_2TO2 = OUTDIR / "single_event_2to2_baseline.lhe"
LHE_2TO4 = OUTDIR / "single_event_2to4_test.lhe"
DEBUG_2TO4 = OUTDIR / "lhe_2to4_debug.txt"

QUARK_ABS_IDS = {1, 2, 3, 4, 5}
MAX_EVENTS = 10_000
TINY = 1e-6  # treat as zero for collinearity


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
    """Return one common ancestor of i and j (smallest index in the set), or None."""
    a_i = get_ancestors(ev, i)
    a_j = get_ancestors(ev, j)
    common = a_i & a_j
    return min(common) if common else None


def run_selection() -> Tuple[
    Optional[Dict[str, Any]],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[Dict[str, Any]],
    str,
]:
    """
    Find one event with: incoming e (status -21), incoming q (status -21), outgoing e, struck q,
    partner q, and partner qbar (final antiquark with id = -partner_id).
    Return (event_dump, in_e_idx, in_q_idx, out_e_idx, struck_idx, partner_q_idx, partner_qbar_idx, debug, status).
    """
    pythia = build_pythia()
    ev = pythia.event
    n_tried = 0

    while n_tried < MAX_EVENTS:
        if not pythia.next():
            n_tried += 1
            continue
        n_tried += 1

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
            continue

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
            continue

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
            continue

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
            continue

        partner_id = ev[partner_q_idx].id()

        # Partner antiquark: first final antiquark with id == -partner_id
        partner_qbar_idx: Optional[int] = None
        for i in final_antiquarks:
            if ev[i].id() == -partner_id:
                partner_qbar_idx = i
                break
        if partner_qbar_idx is None:
            continue

        # Build full event dump and selected-particle diagnostics
        event_records = [particle_record(ev, i) for i in range(ev.size())]
        chain_struck = ancestry_chain(ev, struck_idx)
        chain_partner = ancestry_chain(ev, partner_q_idx)
        chain_partner_qbar = ancestry_chain(ev, partner_qbar_idx)
        lca_partner = lowest_common_ancestor(ev, partner_q_idx, partner_qbar_idx)

        event_dump = {
            "n_tried": n_tried,
            "n_particles": ev.size(),
            "particles": event_records,
            "in_electron_index": in_e_idx,
            "in_quark_index": in_q_idx,
            "out_electron_index": out_e_idx,
            "struck_quark_index": struck_idx,
            "partner_quark_index": partner_q_idx,
            "partner_antiquark_index": partner_qbar_idx,
            "final_quark_indices": final_quarks,
            "final_antiquark_indices": final_antiquarks,
        }

        selected = {
            "in_electron": event_records[in_e_idx],
            "in_quark": event_records[in_q_idx],
            "out_electron": event_records[out_e_idx],
            "struck_quark": event_records[struck_idx],
            "partner_quark": event_records[partner_q_idx],
            "partner_antiquark": event_records[partner_qbar_idx],
            "ancestry_chain_struck": chain_struck,
            "ancestry_chain_partner_quark": chain_partner,
            "ancestry_chain_partner_antiquark": chain_partner_qbar,
            "lca_partner_q_qbar": lca_partner,
        }

        return (
            event_dump,
            in_e_idx,
            in_q_idx,
            out_e_idx,
            struck_idx,
            partner_q_idx,
            partner_qbar_idx,
            selected,
            "PASS",
        )

    return None, None, None, None, None, None, None, None, "FAIL"


def p4(d: Dict[str, Any]) -> Tuple[float, float, float, float]:
    return (float(d["px"]), float(d["py"]), float(d["pz"]), float(d["E"]))


def p4_str(d: Dict[str, Any]) -> str:
    return f"(px={d['px']:.6g}, py={d['py']:.6g}, pz={d['pz']:.6g}, E={d['E']:.6g}, m={d['m']:.6g})"


def write_lhe_file(
    lhe_list: List[Tuple[int, int, int, int, int, int, float, float, float, float, float]],
    out_path: Path,
    nup: int,
) -> None:
    """lhe_list: (idup, istup, m1, m2, col, acol, px, py, pz, e, m) per line. nup = len(lhe_list)."""
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

    with out_path.open("w") as f:
        f.write('<LesHouchesEvents version="1.0">\n')
        f.write("<header>\n  <!-- Minimal DIS LHE 2->2 or 2->4 -->\n</header>\n")
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
        for row in lhe_list:
            idup, istup, m1, m2, col, acol, px, py, pz, e, m = row
            f.write(
                f" {idup:8d} {istup:4d}{m1:5d}{m2:5d}{col:5d}{acol:5d}"
                f" {px:14.7e} {py:14.7e} {pz:14.7e} {e:14.7e} {m:14.7e}"
                f" {0.0:10.3e} {9.0:10.3e}\n"
            )
        f.write("</event>\n</LesHouchesEvents>\n")


def main() -> None:
    if pythia8 is None:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        SOURCE_EVENT_DUMP.write_text(
            json.dumps({"status": "pythia8_unavailable", "message": "pythia8 could not be imported."}, indent=2)
        )
        SELECTED_HARD.write_text(json.dumps({"status": "pythia8_unavailable"}, indent=2))
        DEBUG_2TO4.write_text("pythia8 could not be imported; no event generated. Run with a Python that has pythia8.\n")
        print("FAIL: pythia8 not available.")
        return

    (
        event_dump,
        in_e_idx,
        in_q_idx,
        out_e_idx,
        struck_idx,
        partner_q_idx,
        partner_qbar_idx,
        selected,
        status,
    ) = run_selection()

    if event_dump is None or selected is None:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        SOURCE_EVENT_DUMP.write_text(
            json.dumps({"status": "no_valid_event", "message": "No event found within max events."}, indent=2)
        )
        SELECTED_HARD.write_text(json.dumps({"status": "no_valid_event"}, indent=2))
        DEBUG_2TO4.write_text("No valid 2→4 event found; cannot build LHE or debug.\n")
        print("FAIL: No valid event found.")
        return

    with SOURCE_EVENT_DUMP.open("w") as f:
        json.dump(event_dump, f, indent=2)
    with SELECTED_HARD.open("w") as f:
        json.dump(selected, f, indent=2)

    in_e = selected["in_electron"]
    in_q = selected["in_quark"]
    out_e = selected["out_electron"]
    struck = selected["struck_quark"]
    partner_q = selected["partner_quark"]
    partner_qbar = selected["partner_antiquark"]

    # Incoming: enforce beam-collinear (px=0, py=0). Use source values; if px/py not tiny, we still force for LHE and document.
    in_e_px, in_e_py = float(in_e["px"]), float(in_e["py"])
    in_q_px, in_q_py = float(in_q["px"]), float(in_q["py"])
    in_e_collinear = abs(in_e_px) < TINY and abs(in_e_py) < TINY
    in_q_collinear_src = abs(in_q_px) < TINY and abs(in_q_py) < TINY

    # 2→2 baseline: 1=in_e, 2=in_q, 3=out_e, 4=struck_q. Use source 4-vectors; force incoming px=py=0.
    in_e_22 = (0.0, 0.0, float(in_e["pz"]), float(in_e["E"]), float(in_e["m"]))
    in_q_22 = (0.0, 0.0, float(in_q["pz"]), float(in_q["E"]), float(in_q["m"]))
    out_e_22 = (float(out_e["px"]), float(out_e["py"]), float(out_e["pz"]), float(out_e["E"]), float(out_e["m"]))
    struck_22 = (float(struck["px"]), float(struck["py"]), float(struck["pz"]), float(struck["E"]), float(struck["m"]))

    list_22 = [
        (11, -1, 0, 0, 0, 0, in_e_22[0], in_e_22[1], in_e_22[2], in_e_22[3], in_e_22[4]),
        (int(in_q["pdg_id"]), -1, 0, 0, 101, 0, in_q_22[0], in_q_22[1], in_q_22[2], in_q_22[3], in_q_22[4]),
        (11, 1, 1, 2, 0, 0, out_e_22[0], out_e_22[1], out_e_22[2], out_e_22[3], out_e_22[4]),
        (int(struck["pdg_id"]), 1, 1, 2, 101, 0, struck_22[0], struck_22[1], struck_22[2], struck_22[3], struck_22[4]),
    ]
    write_lhe_file(list_22, LHE_2TO2, 4)
    print(f"Wrote 2→2 baseline: {LHE_2TO2}")

    # 2→4: incoming quark from 4-momentum conservation
    out_e_p4 = p4(out_e)
    struck_p4 = p4(struck)
    partner_q_p4 = p4(partner_q)
    partner_qbar_p4 = p4(partner_qbar)
    in_e_p4 = p4(in_e)

    p_in_q_x = out_e_p4[0] + struck_p4[0] + partner_q_p4[0] + partner_qbar_p4[0] - in_e_p4[0]
    p_in_q_y = out_e_p4[1] + struck_p4[1] + partner_q_p4[1] + partner_qbar_p4[1] - in_e_p4[1]
    p_in_q_z = out_e_p4[2] + struck_p4[2] + partner_q_p4[2] + partner_qbar_p4[2] - in_e_p4[2]
    p_in_q_E = out_e_p4[3] + struck_p4[3] + partner_q_p4[3] + partner_qbar_p4[3] - in_e_p4[3]

    in_q_collinear_after = abs(p_in_q_x) < TINY and abs(p_in_q_y) < TINY
    if not in_q_collinear_after:
        p_in_q_x, p_in_q_y = 0.0, 0.0
        # Recompute E so that m^2 = E^2 - p^2 is consistent (use proton fragment mass as placeholder)
        m_in_q = float(in_q["m"])
        p_in_q_E = math.sqrt(m_in_q * m_in_q + p_in_q_x * p_in_q_x + p_in_q_y * p_in_q_y + p_in_q_z * p_in_q_z)

    # Masses: use source for struck; for partner q and qbar use 0 if light (or source)
    m_partner_q = float(partner_q["m"])
    m_partner_qbar = float(partner_qbar["m"])
    if abs(partner_q["pdg_id"]) in (1, 2, 3) and m_partner_q < 0.01:
        m_partner_q = 0.0
    if abs(partner_qbar["pdg_id"]) in (1, 2, 3) and m_partner_qbar < 0.01:
        m_partner_qbar = 0.0

    list_24 = [
        (11, -1, 0, 0, 0, 0, 0.0, 0.0, float(in_e["pz"]), float(in_e["E"]), float(in_e["m"])),
        (int(in_q["pdg_id"]), -1, 0, 0, 101, 0, p_in_q_x, p_in_q_y, p_in_q_z, p_in_q_E, float(in_q["m"])),
        (11, 1, 1, 2, 0, 0, out_e_p4[0], out_e_p4[1], out_e_p4[2], out_e_p4[3], float(out_e["m"])),
        (int(struck["pdg_id"]), 1, 1, 2, 101, 0, struck_p4[0], struck_p4[1], struck_p4[2], struck_p4[3], float(struck["m"])),
        (int(partner_q["pdg_id"]), 1, 1, 2, 102, 0, partner_q_p4[0], partner_q_p4[1], partner_q_p4[2], partner_q_p4[3], m_partner_q),
        (int(partner_qbar["pdg_id"]), 1, 1, 2, 0, 102, partner_qbar_p4[0], partner_qbar_p4[1], partner_qbar_p4[2], partner_qbar_p4[3], m_partner_qbar),
    ]
    write_lhe_file(list_24, LHE_2TO4, 6)
    print(f"Wrote 2→4 test: {LHE_2TO4}")

    # Debug file
    pin_24 = (0.0 + p_in_q_x, 0.0 + p_in_q_y, float(in_e["pz"]) + p_in_q_z, float(in_e["E"]) + p_in_q_E)
    pout_24 = (
        out_e_p4[0] + struck_p4[0] + partner_q_p4[0] + partner_qbar_p4[0],
        out_e_p4[1] + struck_p4[1] + partner_q_p4[1] + partner_qbar_p4[1],
        out_e_p4[2] + struck_p4[2] + partner_q_p4[2] + partner_qbar_p4[2],
        out_e_p4[3] + struck_p4[3] + partner_q_p4[3] + partner_qbar_p4[3],
    )
    diff_24 = (pout_24[0] - pin_24[0], pout_24[1] - pin_24[1], pout_24[2] - pin_24[2], pout_24[3] - pin_24[3])

    lines = [
        "2→4 LHE proof-of-principle debug",
        "",
        "Source-event selected particle table (index, id, status, mothers, col, acol, px, py, pz, E, m):",
    ]
    for name, rec in [
        ("in_electron", in_e),
        ("in_quark", in_q),
        ("out_electron", out_e),
        ("struck_quark", struck),
        ("partner_quark", partner_q),
        ("partner_antiquark", partner_qbar),
    ]:
        lines.append(
            f"  {name}: idx={rec['index']} id={rec['pdg_id']} status={rec['status']} "
            f"m1={rec['mother1']} m2={rec['mother2']} col={rec['color']} acol={rec['anticolor']} "
            f"px={rec['px']:.6g} py={rec['py']:.6g} pz={rec['pz']:.6g} E={rec['E']:.6g} m={rec['m']:.6g}"
        )
    lines.extend([
        "",
        "Exact final LHE particle list (2→4): IDUP, ISTUP, MOTHUP, ICOLUP, px, py, pz, E, m",
    ])
    for i, row in enumerate(list_24, 1):
        idup, istup, m1, m2, col, acol, px, py, pz, e, m = row
        lines.append(f"  {i}: id={idup} istup={istup} mothers=({m1},{m2}) col=({col},{acol}) px={px:.6g} py={py:.6g} pz={pz:.6g} E={e:.6g} m={m:.6g}")
    lines.extend([
        "",
        "Total incoming 4-momentum (entries 1+2):",
        f"  px={pin_24[0]:.6g} py={pin_24[1]:.6g} pz={pin_24[2]:.6g} E={pin_24[3]:.6g}",
        "Total outgoing 4-momentum (entries 3–6):",
        f"  px={pout_24[0]:.6g} py={pout_24[1]:.6g} pz={pout_24[2]:.6g} E={pout_24[3]:.6g}",
        "Difference (out - in):",
        f"  px={diff_24[0]:.6g} py={diff_24[1]:.6g} pz={diff_24[2]:.6g} E={diff_24[3]:.6g}",
        "",
        "Ancestry chains:",
        f"  struck: {selected['ancestry_chain_struck']}",
        f"  partner quark: {selected['ancestry_chain_partner_quark']}",
        f"  partner antiquark: {selected['ancestry_chain_partner_antiquark']}",
        f"  LCA(partner q, partner qbar): {selected['lca_partner_q_qbar']}",
        "",
        "Mass choices: struck and in_quark from source; partner q/qbar: " + (
            f"m_q={m_partner_q}, m_qbar={m_partner_qbar}" if (m_partner_q, m_partner_qbar) != (0.0, 0.0) else "0.0 (light)"
        ),
        "",
        "Color assignment:",
        "  incoming struck quark: col=101, acol=0",
        "  outgoing struck quark: col=101, acol=0",
        "  outgoing partner q': col=102, acol=0",
        "  outgoing partner qbar': col=0, acol=102",
        "  electrons: col=0, acol=0",
        "",
        f"Incoming quark collinear in source? {in_q_collinear_src}. After conservation: px={p_in_q_x}, py={p_in_q_y} (forced to 0 if not tiny). Incoming quark remained beam-collinear in LHE: True (by construction).",
    ])
    DEBUG_2TO4.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote debug: {DEBUG_2TO4}")
    print(f"Status: {status}")


if __name__ == "__main__":
    main()
