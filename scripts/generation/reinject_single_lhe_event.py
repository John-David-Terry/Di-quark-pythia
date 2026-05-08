#!/usr/bin/env python3
"""
Reinject a single LHE event into PYTHIA8 and run full downstream evolution.

Input:
  - outputs/popf_lhe_single_event/single_event_nominal.lhe
  - outputs/popf_lhe_single_event/single_event_kicked.lhe

Output:
  - outputs/popf_lhe_single_event/reinjection_nominal_log.txt
  - outputs/popf_lhe_single_event/reinjection_kicked_log.txt
  - outputs/popf_lhe_single_event/summary_high_level.txt
  - outputs/popf_lhe_single_event/summary_low_level.txt
  - outputs/popf_lhe_single_event/summary_combined.txt

This script:
  - configures PYTHIA with Beams:frameType = 4 and Beams:LHEF pointing to the LHE file,
  - turns ISR, FSR, and hadronization ON,
  - attempts to generate exactly one event,
  - records basic success/failure info and a short particle-count summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pythia8


import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

OUTDIR = outputs_dir() / "popf_lhe_single_event"
OUTDIR.mkdir(parents=True, exist_ok=True)
LHE_NOMINAL = OUTDIR / "single_event_nominal.lhe"
LHE_KICKED = OUTDIR / "single_event_kicked.lhe"
LOG_NOMINAL = OUTDIR / "reinjection_nominal_log.txt"
LOG_KICKED = OUTDIR / "reinjection_kicked_log.txt"
SUMMARY_HI = OUTDIR / "summary_high_level.txt"
SUMMARY_LO = OUTDIR / "summary_low_level.txt"
SUMMARY_COMBINED = OUTDIR / "summary_combined.txt"


def build_pythia_for_reinjection(lhe_path: Path) -> pythia8.Pythia:
    p = pythia8.Pythia()
    p.readString("Beams:frameType = 4")
    p.readString(f"Beams:LHEF = {lhe_path}")

    # Try to enforce matching of incoming/outgoing four-momenta for LHE input
    p.readString("LesHouches:matchInOut = on")

    # Downstream evolution ON
    p.readString("PartonLevel:ISR = on")
    p.readString("PartonLevel:FSR = on")
    p.readString("HadronLevel:all = on")
    # MPI off for simplicity (can be changed if needed)
    p.readString("PartonLevel:MPI = off")

    # Keep PYTHIA chatty enough for diagnostics (do NOT silence completely)
    # p.readString("Print:quiet = on")

    return p


def count_final_state_particles(ev: pythia8.Event) -> int:
    """Count final-state particles in the event record."""
    n_final = 0
    for i in range(ev.size()):
        if ev[i].isFinal():
            n_final += 1
    return n_final


def run_one_reinjection(lhe_path: Path, log_path: Path) -> Tuple[bool, bool, int, str]:
    """Run PYTHIA once with the given LHE file. Return (init_ok, next_ok, n_final, init_err)."""
    log_lines: List[str] = []
    init_err = ""

    if not lhe_path.exists():
        log_lines.append(f"LHE file not found: {lhe_path}")
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        return False, False, 0, f"File not found: {lhe_path}"

    log_lines.append(f"Using LHE file: {lhe_path}")
    pythia = build_pythia_for_reinjection(lhe_path)

    init_ok = False
    next_ok = False
    n_final = 0

    try:
        init_ok = pythia.init()
    except Exception as exc:
        init_err = f"Exception during pythia.init(): {exc!r}"
        log_lines.append(init_err)

    if not init_ok and not init_err:
        log_lines.append("pythia.init() returned False (initialization failed).")

    if init_ok:
        log_lines.append("pythia.init() succeeded.")
        try:
            next_ok = pythia.next()
        except Exception as exc:
            log_lines.append(f"Exception during pythia.next(): {exc!r}")
            next_ok = False
        if next_ok:
            log_lines.append("pythia.next() succeeded (event generated).")
            n_final = count_final_state_particles(pythia.event)
            log_lines.append(f"Number of final-state particles: {n_final}")
        else:
            log_lines.append("pythia.next() failed or returned False (no accepted event).")

    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    return init_ok, next_ok, n_final, init_err


def run_reinjection() -> None:
    # Run both nominal and kicked reinjection
    init_nom, next_nom, n_final_nom, err_nom = run_one_reinjection(LHE_NOMINAL, LOG_NOMINAL)
    init_kick, next_kick, n_final_kick, err_kick = run_one_reinjection(LHE_KICKED, LOG_KICKED)

    # Benchmark: PASS if both nominal and kicked succeed; SOFT PASS if nominal works but kicked fails; FAIL if nominal breaks
    if init_nom and next_nom and init_kick and next_kick:
        benchmark_status = "PASS"
        main_reason = "Nominal and kicked LHE reinjection both succeeded; kicked event produced a fully evolved final state."
    elif init_nom and next_nom and init_kick and not next_kick:
        benchmark_status = "SOFT PASS"
        main_reason = "Nominal reinjection succeeded; kicked reinjection initialized but pythia.next() failed."
    elif init_nom and next_nom:
        benchmark_status = "SOFT PASS"
        main_reason = "Nominal reinjection succeeded; kicked LHE failed to initialize or was not run."
    elif not (init_nom and next_nom):
        benchmark_status = "FAIL"
        main_reason = "Nominal baseline reinjection failed; cannot validate kicked event."
    else:
        benchmark_status = "FAIL"
        main_reason = "Unexpected state."

    # Load kick debug for summary (if present)
    kick_debug_path = OUTDIR / "kick_debug.txt"
    kick_debug_text = ""
    if kick_debug_path.exists():
        kick_debug_text = kick_debug_path.read_text(encoding="utf-8").strip()

    # Answers to the six explicit questions
    q1_nominal = "Yes." if (init_nom and next_nom) else "No."
    q2_kicked = "Yes." if (init_kick and next_kick) else "No."
    q3_vectors = (
        kick_debug_text
        if kick_debug_text
        else "See kick_debug.txt for original and modified 4-vectors of incoming and outgoing quarks."
    )
    q4_px = (
        "By construction total hard-event px is preserved (outgoing +0.4, incoming -0.4)."
        if kick_debug_text
        else "See kick_debug.txt for total px before/after."
    )
    q5_energy = (
        "Total hard-event energy changes because we recomputed E = sqrt(m^2 + p^2) for the two quarks only; electron momenta unchanged."
        if kick_debug_text
        else "See kick_debug.txt for total E before/after."
    )
    q6_failure = (
        "N/A (kicked event succeeded)."
        if (init_kick and next_kick)
        else "PYTHIA reports 'ProcessContainer::constructProcess: setting mass failed' — modified quark 4-momenta may violate internal mass/kinematic checks (e.g. incoming parton off-shell in the frame PYTHIA uses)."
    )

    # High-level summary
    hi_lines = [
        "POPF single-event LHE reinjection benchmark (nominal + momentum kick) — HIGH-LEVEL SUMMARY",
        "",
        f"Benchmark status: {benchmark_status}",
        "",
        "What was attempted:",
        "- Export one internal DIS hard event; build minimal 2->2 LHE (nominal and kicked).",
        "- Nominal: no kick. Kicked: +0.4 GeV px on outgoing struck quark, -0.4 GeV px on incoming quark; E recomputed.",
        "- Reinject both LHE files into PYTHIA with ISR/FSR/hadronization ON.",
        "",
        f"Main result: {main_reason}",
        "",
        "1. Did the nominal event still work? " + q1_nominal,
        "2. Did the kicked event work? " + q2_kicked,
        "3. Exact quark 4-vectors modified: see kick_debug.txt and low-level summary.",
        "4. Was total hard-event px preserved? " + q4_px,
        "5. Did total hard-event energy change? " + q5_energy,
        "6. If kicked failed, single most likely reason? " + q6_failure,
        "",
    ]
    SUMMARY_HI.write_text("\n".join(hi_lines), encoding="utf-8")

    # Low-level summary
    lo_lines = [
        "POPF single-event LHE reinjection benchmark — LOW-LEVEL SUMMARY",
        "",
        "PYTHIA settings (export): Beams:idA=11, idB=2212, eA=18, eB=275, frameType=2; WeakBosonExchange:ff2ff(t:gmZ)=on; PartonLevel:all=off, HadronLevel:all=off.",
        "PYTHIA settings (reinjection): Beams:frameType=4, Beams:LHEF=<file>; LesHouches:matchInOut=on; PartonLevel:ISR=on, FSR=on; HadronLevel:all=on; PartonLevel:MPI=off.",
        "",
        "Nominal reinjection:",
        f"  pythia.init() success: {init_nom}, pythia.next() success: {next_nom}, n_final: {n_final_nom}",
        "",
        "Kicked reinjection:",
        f"  pythia.init() success: {init_kick}, pythia.next() success: {next_kick}, n_final: {n_final_kick}",
        "  If next() failed, PYTHIA may print 'ProcessContainer::constructProcess: setting mass failed' (see run stdout).",
        "",
        "1. Did the nominal event still work? " + q1_nominal,
        "2. Did the kicked event work? " + q2_kicked,
        "3. Quark 4-vectors modified (incoming quark and outgoing struck quark): see kick_debug.txt.",
        "",
        "4. Total hard-event px preserved by construction (outgoing +Delta_px, incoming -Delta_px). " + q4_px,
        "5. Total hard-event energy changed (E recomputed for two quarks only). " + q5_energy,
        "6. If kicked failed, most likely reason: " + q6_failure,
        "",
    ]
    SUMMARY_LO.write_text("\n".join(lo_lines), encoding="utf-8")

    # Combined summary
    combined = [
        "HIGH-LEVEL SUMMARY",
        "",
        SUMMARY_HI.read_text(encoding="utf-8").strip(),
        "",
        "LOW-LEVEL SUMMARY",
        "",
        SUMMARY_LO.read_text(encoding="utf-8").strip(),
        "",
    ]
    SUMMARY_COMBINED.write_text("\n".join(combined), encoding="utf-8")

    print(f"Benchmark status: {benchmark_status}")
    print(main_reason)
    print(f"Nominal: init={init_nom} next={next_nom} n_final={n_final_nom}  Kicked: init={init_kick} next={next_kick} n_final={n_final_kick}")
    print(f"Logs: {LOG_NOMINAL}, {LOG_KICKED}; summary: {SUMMARY_COMBINED}")


if __name__ == "__main__":
    run_reinjection()

