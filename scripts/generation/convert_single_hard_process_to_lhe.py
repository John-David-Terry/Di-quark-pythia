#!/usr/bin/env python3
"""
Convert a saved PYTHIA8 hard-process record (JSON) to a minimal single-event LHE file.

Input:
  - outputs/popf_lhe_single_event/hard_process.json

Output:
  - outputs/popf_lhe_single_event/single_event.lhe

We:
  - read the particles from the saved pythia.process dump,
  - map them to the standard LHE fields:
      IDUP, ISTUP, MOTHUP(1:2), ICOLUP(1:2), PUP(1:5), VTIMUP, SPINUP
  - build a minimal <init> and <event> block.

We keep the mapping as transparent as possible; this is a proof-of-principle only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

OUTDIR = outputs_dir() / "popf_lhe_single_event"
IN_JSON = OUTDIR / "hard_process.json"
OUT_LHE_NOMINAL = OUTDIR / "single_event_nominal.lhe"
OUT_LHE_KICKED = OUTDIR / "single_event_kicked.lhe"
KICK_DEBUG_PATH = OUTDIR / "kick_debug.txt"

# Quark PDG IDs (absolute value)
QUARK_IDS = {1, 2, 3, 4, 5, 6}

# Momentum kick: Delta_px = 2 * Lambda_QCD, Lambda_QCD ≈ 0.2 GeV
LAMBDA_QCD_GEV = 0.2
DELTA_PX_GEV = 2.0 * LAMBDA_QCD_GEV  # 0.4 GeV


def load_hard_process() -> Dict[str, Any]:
    if not IN_JSON.exists():
        raise FileNotFoundError(f"Missing hard-process JSON: {IN_JSON}")
    with IN_JSON.open() as f:
        return json.load(f)


def write_single_event_lhe() -> None:
    data = load_hard_process()
    particles: List[Dict[str, Any]] = data.get("particles", [])
    info: Dict[str, Any] = data.get("info", {})

    if not particles:
        raise RuntimeError("No particles in hard_process.json; cannot build LHE.")

    # Event-level quantities with simple fallbacks
    xwgtup = float(info.get("weight", 1.0))
    q2 = info.get("Q2", None)
    if q2 is not None and q2 > 0.0:
        scalup = float(q2) ** 0.5
    else:
        scalup = 10.0  # GeV, arbitrary fallback
    aqedup = float(info.get("alphaEM", 1.0 / 137.0))
    aqcdup = float(info.get("alphaS", 0.118))

    # Process ID: arbitrary but fixed
    idprup = 1

    # <init> block values
    # We use the known beam IDs and energies from the generator config.
    idbmup1, idbmup2 = 11, 2212
    ebmup1, ebmup2 = 18.0, 275.0
    pdfgup1, pdfgup2 = 0, 0   # unknown PDF group
    pdfsup1, pdfsup2 = 0, 0   # unknown PDF set
    idwtup = 3                # unweighted events
    nprup = 1
    xsecup, xerrup, xmaxup, lprup = 1.0, 0.0, 1.0, idprup

    # ---- Build a minimal 2->2 hard-process LHE event from PYTHIA process ----
    #
    # For the DIS test case we observed the following structure in pythia.process:
    #   idx  id   status  mothers  daughters  role
    #     1  11   -12     0  0     3   0      incoming beam lepton
    #     2  2212 -12     0  0     4   0      incoming beam proton
    #     3  11   -21     1  0     5   6      incoming hard lepton
    #     4   2   -21     2  0     5   6      incoming hard quark
    #     5  11    23     3  4     0   0      outgoing hard lepton
    #     6   2    23     3  4     0   0      outgoing hard quark
    #
    # We construct a minimal external 2->2 LHE event using:
    #   - incoming: the two status=-21 partons (3,4)  => ISTUP = -1
    #   - outgoing: the two status=23 partons (5,6)  => ISTUP = +1, mothers=(1,2)
    #
    # If this pattern is not present, we fall back to a conservative heuristic.

    # Index mapping: original process index -> particle dict
    by_idx: Dict[int, Dict[str, Any]] = {int(p["index"]): p for p in particles}

    # Select incoming hard particles (prefer status == -21)
    incoming_hard = [p for p in particles if int(p["status"]) == -21]
    if len(incoming_hard) < 2:
        # fallback: any negative-status non-beam entries
        incoming_hard = [p for p in particles if int(p["status"]) < 0 and int(p["id"]) not in (90,)]
    incoming_hard = incoming_hard[:2]

    # Select outgoing hard particles (prefer status == 23)
    outgoing_hard = [p for p in particles if int(p["status"]) == 23]
    if len(outgoing_hard) < 2:
        outgoing_hard = [p for p in particles if int(p["status"]) > 0]
    outgoing_hard = outgoing_hard[:2]

    if len(incoming_hard) != 2 or len(outgoing_hard) != 2:
        raise RuntimeError(
            "Could not identify a simple 2->2 hard subprocess in pythia.process "
            "(expected two incoming and two outgoing hard particles)."
        )

    # Build LHE particle list in the desired order:
    #   1,2: incoming hard particles (ISTUP=-1)
    #   3,4: outgoing hard particles (ISTUP=+1, MOTHUP=(1,2))
    lhe_particles: List[Tuple[int, Dict[str, Any]]] = []

    # Assign LHE indices explicitly
    #   (lhe_idx, source_particle_dict)
    lhe_particles.append((1, incoming_hard[0]))
    lhe_particles.append((2, incoming_hard[1]))
    lhe_particles.append((3, outgoing_hard[0]))
    lhe_particles.append((4, outgoing_hard[1]))

    # Diagnostics: write mapping for inspection
    mapping_lines: List[str] = []
    mapping_lines.append("Original PYTHIA process particle list:")
    for p in particles:
        mapping_lines.append(
            f" idx={p['index']} id={p['id']} status={p['status']} "
            f"m1={p['mother1']} m2={p['mother2']} d1={p['daughter1']} d2={p['daughter2']} "
            f"col={p['col']} acol={p['acol']} px={p['px']:.6g} py={p['py']:.6g} pz={p['pz']:.6g} E={p['e']:.6g} m={p['m']:.6g}"
        )
    mapping_lines.append("")
    mapping_lines.append("Translated minimal LHE particle list (index, IDUP, ISTUP, MOTHUP(1,2), ICOLUP(1,2), PUP[1:5]):")
    for lhe_idx, p in lhe_particles:
        istup = -1 if lhe_idx in (1, 2) else +1
        m1 = 0
        m2 = 0
        if istup == +1:
            m1, m2 = 1, 2
        mapping_lines.append(
            f" LHE idx={lhe_idx}  IDUP={p['id']}  ISTUP={istup}  "
            f"MOTHUP=({m1},{m2})  ICOLUP=({p['col']},{p['acol']})  "
            f"PUP=({p['px']:.6g},{p['py']:.6g},{p['pz']:.6g},{p['e']:.6g},{p['m']:.6g})"
        )

    debug_path = OUTDIR / "lhe_mapping_debug.txt"
    debug_path.write_text("\n".join(mapping_lines), encoding="utf-8")

    nup = len(lhe_particles)

    def write_lhe(lhe_list: List[Tuple[int, Dict[str, Any]]], out_path: Path) -> None:
        """Write a minimal LHE file from list of (lhe_idx, particle_dict)."""
        with out_path.open("w") as f:
            f.write("<LesHouchesEvents version=\"1.0\">\n")
            f.write("<header>\n")
            f.write("  <!-- Single DIS hard event exported from PYTHIA8. -->\n")
            f.write("</header>\n")
            f.write("<init>\n")
            f.write(
                f" {idbmup1:8d} {idbmup2:8d} "
                f"{ebmup1:14.7e} {ebmup2:14.7e} "
                f"{pdfgup1:4d} {pdfgup2:4d} {pdfsup1:4d} {pdfsup2:4d} "
                f"{idwtup:4d} {nprup:4d}\n"
            )
            f.write(f" {xsecup:14.7e} {xerrup:14.7e} {xmaxup:14.7e} {lprup:4d}\n")
            f.write("</init>\n")
            f.write("<event>\n")
            f.write(
                f" {nup:4d} {idprup:4d} {xwgtup:14.7e} {scalup:14.7e} "
                f"{aqedup:14.7e} {aqcdup:14.7e}\n"
            )
            for lhe_idx, p in lhe_list:
                idup = int(p["id"])
                istup = -1 if lhe_idx in (1, 2) else +1
                m1, m2 = (0, 0) if istup == -1 else (1, 2)
                col1, col2 = int(p["col"]), int(p["acol"])
                px, py, pz = float(p["px"]), float(p["py"]), float(p["pz"])
                e, m = float(p["e"]), float(p["m"])
                f.write(
                    f" {idup:8d} {istup:4d}{m1:5d}{m2:5d}{col1:5d}{col2:5d}"
                    f" {px:14.7e} {py:14.7e} {pz:14.7e} {e:14.7e} {m:14.7e}"
                    f" {0.0:10.3e} {9.0:10.3e}\n"
                )
            f.write("</event>\n")
            f.write("</LesHouchesEvents>\n")

    # ---- 1. Nominal LHE (no kick) ----
    write_lhe(lhe_particles, OUT_LHE_NOMINAL)
    print(f"Wrote nominal LHE to: {OUT_LHE_NOMINAL}")

    # ---- 2. Momentum-kick block ----
    # Identify incoming quark and outgoing struck quark by PDG id (quark = |id| in 1..6).
    # LHE order: 1=incoming A, 2=incoming B, 3=outgoing A, 4=outgoing B. For DIS: 1=e, 2=quark, 3=e, 4=quark.
    incoming_quark_lhe_idx: int | None = None
    outgoing_struck_quark_lhe_idx: int | None = None
    for lhe_idx, p in lhe_particles:
        if abs(int(p["id"])) not in QUARK_IDS:
            continue
        if lhe_idx in (1, 2):
            incoming_quark_lhe_idx = lhe_idx
        else:
            outgoing_struck_quark_lhe_idx = lhe_idx

    if incoming_quark_lhe_idx is None or outgoing_struck_quark_lhe_idx is None:
        raise RuntimeError("Could not identify both incoming and outgoing quark in LHE list.")

    # Copy particle list and apply ±Delta_px to the two quarks; recompute E = sqrt(m^2 + p^2).
    kicked_list: List[Tuple[int, Dict[str, Any]]] = []
    for lhe_idx, p in lhe_particles:
        pc = {k: v for k, v in p.items()}
        px, py, pz = float(p["px"]), float(p["py"]), float(p["pz"])
        e, m = float(p["e"]), float(p["m"])
        if lhe_idx == outgoing_struck_quark_lhe_idx:
            px += DELTA_PX_GEV
            e = math.sqrt(m * m + px * px + py * py + pz * pz)
        elif lhe_idx == incoming_quark_lhe_idx:
            px -= DELTA_PX_GEV
            e = math.sqrt(m * m + px * px + py * py + pz * pz)
        pc["px"], pc["py"], pc["pz"], pc["e"], pc["m"] = px, py, pz, e, m
        kicked_list.append((lhe_idx, pc))

    # ---- 3. Kick debug file ----
    def p4_str(d: Dict[str, Any]) -> str:
        return f"(px={d['px']:.6g}, py={d['py']:.6g}, pz={d['pz']:.6g}, E={d['e']:.6g}, m={d['m']:.6g})"

    orig_in = next(p for i, p in lhe_particles if i == incoming_quark_lhe_idx)
    orig_out = next(p for i, p in lhe_particles if i == outgoing_struck_quark_lhe_idx)
    mod_in = next(p for i, p in kicked_list if i == incoming_quark_lhe_idx)
    mod_out = next(p for i, p in kicked_list if i == outgoing_struck_quark_lhe_idx)

    total_px_orig = sum(float(p["px"]) for _, p in lhe_particles)
    total_px_mod = sum(float(p["px"]) for _, p in kicked_list)
    total_e_orig = sum(float(p["e"]) for _, p in lhe_particles)
    total_e_mod = sum(float(p["e"]) for _, p in kicked_list)

    kick_debug_lines = [
        "Momentum kick debug (single-event DIS LHE)",
        "",
        f"Delta_px used (GeV): {DELTA_PX_GEV}",
        f"Incoming quark: LHE entry {incoming_quark_lhe_idx}  (id={orig_in['id']})",
        f"Outgoing struck quark: LHE entry {outgoing_struck_quark_lhe_idx}  (id={orig_out['id']})",
        "",
        "Original 4-vectors:",
        f"  incoming quark: {p4_str(orig_in)}",
        f"  outgoing struck quark: {p4_str(orig_out)}",
        "",
        "Modified 4-vectors (after ±Delta_px and E recomputed):",
        f"  incoming quark: {p4_str(mod_in)}",
        f"  outgoing struck quark: {p4_str(mod_out)}",
        "",
        f"Original total hard-event px (GeV): {total_px_orig:.6g}",
        f"Modified total hard-event px (GeV): {total_px_mod:.6g}",
        f"Original total hard-event energy (GeV): {total_e_orig:.6g}",
        f"Modified total hard-event energy (GeV): {total_e_mod:.6g}",
        "",
    ]
    KICK_DEBUG_PATH.write_text("\n".join(kick_debug_lines), encoding="utf-8")
    print(f"Wrote kick debug to: {KICK_DEBUG_PATH}")

    # ---- 4. Kicked LHE ----
    write_lhe(kicked_list, OUT_LHE_KICKED)
    print(f"Wrote kicked LHE to: {OUT_LHE_KICKED}")


if __name__ == "__main__":
    write_single_event_lhe()

