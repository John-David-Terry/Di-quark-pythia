#!/usr/bin/env python3
"""
Proof-of-principle: build 2→2 baseline and 2→3 test LHE from the same DIS hard-process source.

Reads from the existing hard_process.json (popf_lhe_single_event), saves a copy as
hard_process_source.json in the 2to3 output dir, then:
  - Writes single_event_2to2_baseline.lhe (same minimal 2→2 as working pipeline).
  - Builds 2→3 by splitting the outgoing quark 4-momentum with z=0.8 into struck + spectator.
  - Writes single_event_2to3_test.lhe and lhe_2to3_debug.txt.

No momentum kick; structural test only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

# Source: reuse the same hard process as the working 2→2 pipeline
SOURCE_JSON = outputs_dir() / "popf_lhe_single_event" / "hard_process.json"
OUTDIR = outputs_dir() / "popf_lhe_single_event_2to3"
OUTDIR.mkdir(parents=True, exist_ok=True)
HARD_SOURCE_COPY = OUTDIR / "hard_process_source.json"
LHE_2TO2 = OUTDIR / "single_event_2to2_baseline.lhe"
LHE_2TO3 = OUTDIR / "single_event_2to3_test.lhe"
DEBUG_2TO3 = OUTDIR / "lhe_2to3_debug.txt"

QUARK_IDS = {1, 2, 3, 4, 5, 6}
Z_SPECTATOR = 0.8  # momentum fraction for struck quark; spectator gets (1-z)


def load_hard_process() -> Dict[str, Any]:
    if not SOURCE_JSON.exists():
        raise FileNotFoundError(f"Missing hard-process JSON: {SOURCE_JSON}")
    with SOURCE_JSON.open() as f:
        data = json.load(f)
    # Save copy for 2to3 outputs
    with HARD_SOURCE_COPY.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return data


def build_22_particles(data: Dict[str, Any]) -> Tuple[List[Tuple[int, Dict[str, Any]]], Dict[str, Any]]:
    """Build the same minimal 2→2 LHE particle list as the working pipeline. Returns (lhe_list, event_header)."""
    particles = data.get("particles", [])
    info = data.get("info", {})
    if len(particles) < 6:
        raise RuntimeError("Expected at least 6 process entries.")

    incoming_hard = [p for p in particles if int(p["status"]) == -21][:2]
    outgoing_hard = [p for p in particles if int(p["status"]) == 23][:2]
    if len(incoming_hard) != 2 or len(outgoing_hard) != 2:
        raise RuntimeError("Could not identify 2→2 hard subprocess.")

    lhe_list: List[Tuple[int, Dict[str, Any]]] = [
        (1, dict(incoming_hard[0])),
        (2, dict(incoming_hard[1])),
        (3, dict(outgoing_hard[0])),
        (4, dict(outgoing_hard[1])),
    ]

    xwgtup = float(info.get("weight", 1.0))
    q2 = info.get("Q2")
    scalup = (float(q2) ** 0.5) if q2 and float(q2) > 0 else 10.0
    aqedup = float(info.get("alphaEM", 1.0 / 137.0))
    aqcdup = float(info.get("alphaS", 0.118))
    header = {"xwgtup": xwgtup, "scalup": scalup, "aqedup": aqedup, "aqcdup": aqcdup}
    return lhe_list, header


def write_lhe_file(
    lhe_list: List[Tuple[int, Dict[str, Any]]],
    header: Dict[str, Any],
    out_path: Path,
) -> None:
    """Write minimal LHE file. header: xwgtup, scalup, aqedup, aqcdup."""
    nup = len(lhe_list)
    idprup = 1
    idbmup1, idbmup2 = 11, 2212
    ebmup1, ebmup2 = 18.0, 275.0
    pdfgup1, pdfgup2, pdfsup1, pdfsup2 = 0, 0, 0, 0
    idwtup, nprup = 3, 1
    xsecup, xerrup, xmaxup, lprup = 1.0, 0.0, 1.0, idprup

    with out_path.open("w") as f:
        f.write('<LesHouchesEvents version="1.0">\n')
        f.write("<header>\n  <!-- Minimal DIS LHE -->\n</header>\n")
        f.write("<init>\n")
        f.write(
            f" {idbmup1:8d} {idbmup2:8d} {ebmup1:14.7e} {ebmup2:14.7e} "
            f"{pdfgup1:4d} {pdfgup2:4d} {pdfsup1:4d} {pdfsup2:4d} {idwtup:4d} {nprup:4d}\n"
        )
        f.write(f" {xsecup:14.7e} {xerrup:14.7e} {xmaxup:14.7e} {lprup:4d}\n")
        f.write("</init>\n<event>\n")
        f.write(
            f" {nup:4d} {idprup:4d} {header['xwgtup']:14.7e} {header['scalup']:14.7e} "
            f"{header['aqedup']:14.7e} {header['aqcdup']:14.7e}\n"
        )
        for lhe_idx, p in lhe_list:
            idup = int(p["id"])
            istup = -1 if lhe_idx in (1, 2) else 1
            m1, m2 = (0, 0) if istup == -1 else (1, 2)
            col1, col2 = int(p["col"]), int(p["acol"])
            px, py, pz = float(p["px"]), float(p["py"]), float(p["pz"])
            e, m = float(p["e"]), float(p["m"])
            f.write(
                f" {idup:8d} {istup:4d}{m1:5d}{m2:5d}{col1:5d}{col2:5d}"
                f" {px:14.7e} {py:14.7e} {pz:14.7e} {e:14.7e} {m:14.7e}"
                f" {0.0:10.3e} {9.0:10.3e}\n"
            )
        f.write("</event>\n</LesHouchesEvents>\n")


def build_23_particles(
    list_22: List[Tuple[int, Dict[str, Any]]],
    header: Dict[str, Any],
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Build 2→3 LHE list: same 1–4 as 2→2, then add entry 5 = spectator.
    Split original outgoing quark (entry 4) into struck (z) and spectator (1-z).
    Mass choice: on-shell for split 4-vectors, m_struck = z*m_q, m_spec = (1-z)*m_q.
    Color: spectator as antiquark (id=-2) with acol=101, col=0 to close the incoming-quark color line.
    """
    list_23: List[Tuple[int, Dict[str, Any]]] = []
    for lhe_idx, p in list_22:
        list_23.append((lhe_idx, {k: v for k, v in p.items()}))

    # Original outgoing quark = entry 4 (index 3 in 0-based)
    p_q = list_22[3][1]
    px_q = float(p_q["px"])
    py_q = float(p_q["py"])
    pz_q = float(p_q["pz"])
    e_q = float(p_q["e"])
    m_q = float(p_q["m"])
    z = Z_SPECTATOR

    # Split 4-momentum: p_struck = z * p_q, p_spec = (1-z) * p_q
    px_s = z * px_q
    py_s = z * py_q
    pz_s = z * pz_q
    e_s = z * e_q
    px_sp = (1 - z) * px_q
    py_sp = (1 - z) * py_q
    pz_sp = (1 - z) * pz_q
    e_sp = (1 - z) * e_q

    # On-shell masses for the split: invariant mass of z*p_q is z*m_q, of (1-z)*p_q is (1-z)*m_q
    m_struck = z * m_q
    m_spec = (1 - z) * m_q

    # Replace entry 4 with struck quark (same id, col as original)
    list_23[3] = (
        4,
        {
            "id": int(p_q["id"]),
            "status": 23,
            "mother1": 0,
            "mother2": 0,
            "daughter1": 0,
            "daughter2": 0,
            "col": int(p_q["col"]),
            "acol": int(p_q["acol"]),
            "px": px_s,
            "py": py_s,
            "pz": pz_s,
            "e": e_s,
            "m": m_struck,
        },
    )

    # Entry 5: spectator. Use antiquark id=-2 so acol=101 closes the color line with incoming quark (col=101).
    list_23.append(
        (
            5,
            {
                "id": -2,
                "status": 23,
                "mother1": 0,
                "mother2": 0,
                "daughter1": 0,
                "daughter2": 0,
                "col": 0,
                "acol": 101,
                "px": px_sp,
                "py": py_sp,
                "pz": pz_sp,
                "e": e_sp,
                "m": m_spec,
            },
        )
    )
    return list_23


def main() -> None:
    data = load_hard_process()
    list_22, header = build_22_particles(data)
    write_lhe_file(list_22, header, LHE_2TO2)
    print(f"Wrote 2→2 baseline: {LHE_2TO2}")

    list_23 = build_23_particles(list_22, header)
    write_lhe_file(list_23, header, LHE_2TO3)
    print(f"Wrote 2→3 test: {LHE_2TO3}")

    # Debug file: exact 5-particle list, kinematics, color, assumptions
    p_q = list_22[3][1]
    px_q, py_q, pz_q = float(p_q["px"]), float(p_q["py"]), float(p_q["pz"])
    e_q, m_q = float(p_q["e"]), float(p_q["m"])
    z = Z_SPECTATOR
    lines = [
        "2→3 LHE proof-of-principle debug",
        "",
        "Exact 5-particle LHE list (IDUP, ISTUP, MOTHUP, ICOLUP, px, py, pz, E, m):",
    ]
    for lhe_idx, p in list_23:
        istup = -1 if lhe_idx in (1, 2) else 1
        m1, m2 = (0, 0) if istup == -1 else (1, 2)
        lines.append(
            f"  {lhe_idx}: id={p['id']} istup={istup} mothers=({m1},{m2}) "
            f"col=({p['col']},{p['acol']}) "
            f"px={p['px']:.6g} py={p['py']:.6g} pz={p['pz']:.6g} E={p['e']:.6g} m={p['m']:.6g}"
        )
    pin = [list_22[0][1], list_22[1][1]]
    pout = [list_23[i][1] for i in range(2, 5)]
    tin = (sum(float(p["px"]) for p in pin), sum(float(p["py"]) for p in pin),
           sum(float(p["pz"]) for p in pin), sum(float(p["e"]) for p in pin))
    tout = (sum(float(p["px"]) for p in pout), sum(float(p["py"]) for p in pout),
            sum(float(p["pz"]) for p in pout), sum(float(p["e"]) for p in pout))
    diff = (tout[0] - tin[0], tout[1] - tin[1], tout[2] - tin[2], tout[3] - tin[3])
    lines.extend([
        "",
        "Original 2→2 outgoing quark 4-vector (entry 4):",
        f"  px={px_q:.6g} py={py_q:.6g} pz={pz_q:.6g} E={e_q:.6g} m={m_q:.6g}",
        "",
        f"z (struck fraction) = {z}, spectator fraction = {1-z}",
        "",
        "Derived 4-vectors:",
        f"  struck (entry 4): px={z*px_q:.6g} py={z*py_q:.6g} pz={z*pz_q:.6g} E={z*e_q:.6g} m={z*m_q:.6g}",
        f"  spectator (entry 5): px={(1-z)*px_q:.6g} py={(1-z)*py_q:.6g} pz={(1-z)*pz_q:.6g} E={(1-z)*e_q:.6g} m={(1-z)*m_q:.6g}",
        "",
        "Total incoming 4-momentum (entries 1+2):",
        f"  px={tin[0]:.6g} py={tin[1]:.6g} pz={tin[2]:.6g} E={tin[3]:.6g}",
        "Total outgoing 4-momentum (entries 3+4+5):",
        f"  px={tout[0]:.6g} py={tout[1]:.6g} pz={tout[2]:.6g} E={tout[3]:.6g}",
        "Difference (out - in):",
        f"  px={diff[0]:.6g} py={diff[1]:.6g} pz={diff[2]:.6g} E={diff[3]:.6g}",
        "",
        "Mass handling: struck and spectator use on-shell masses m_struck = z*m_q, m_spec = (1-z)*m_q.",
        "",
        "Color assignment:",
        "  Entry 1 (e): 0,0.  Entry 2 (q): 101,0.  Entry 3 (e): 0,0.",
        "  Entry 4 (struck q): 101,0.  Entry 5 (spectator ubar): 0,101 (closes color 101).",
    ])
    DEBUG_2TO3.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote debug: {DEBUG_2TO3}")


if __name__ == "__main__":
    main()
