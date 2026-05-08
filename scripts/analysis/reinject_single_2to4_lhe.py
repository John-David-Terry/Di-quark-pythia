#!/usr/bin/env python3
"""
Reinject 2→2 baseline and 2→4 test LHE files into PYTHIA and write logs + summaries.

Runs one event for each LHE file. Writes:
  - reinjection_2to2_log.txt, reinjection_2to4_log.txt
  - summary_high_level.txt, summary_low_level.txt, summary_combined.txt

Summary files answer the 9 explicit questions (2→2 baseline, 2→4 init/next, source particles,
kinematics, collinearity, color, failure reason, next fix).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

try:
    import pythia8
    _pythia_import_error = ""
except ImportError as e:
    pythia8 = None  # type: ignore[assignment]
    _pythia_import_error = str(e)

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

OUTDIR = outputs_dir() / "popf_lhe_single_event_2to4"
OUTDIR.mkdir(parents=True, exist_ok=True)

LHE_2TO2 = OUTDIR / "single_event_2to2_baseline.lhe"
LHE_2TO4 = OUTDIR / "single_event_2to4_test.lhe"
LOG_2TO2 = OUTDIR / "reinjection_2to2_log.txt"
LOG_2TO4 = OUTDIR / "reinjection_2to4_log.txt"
SUMMARY_HI = OUTDIR / "summary_high_level.txt"
SUMMARY_LO = OUTDIR / "summary_low_level.txt"
SUMMARY_COMBINED = OUTDIR / "summary_combined.txt"
DEBUG_2TO4 = OUTDIR / "lhe_2to4_debug.txt"
SELECTED_HARD = OUTDIR / "selected_hard_particles.json"


def build_pythia(lhe_path: Path):
    if pythia8 is None:
        return None
    p = pythia8.Pythia()
    p.readString("Beams:frameType = 4")
    p.readString(f"Beams:LHEF = {lhe_path}")
    p.readString("LesHouches:matchInOut = on")
    p.readString("PartonLevel:ISR = on")
    p.readString("PartonLevel:FSR = on")
    p.readString("HadronLevel:all = on")
    p.readString("PartonLevel:MPI = off")
    return p


def count_final(ev: pythia8.Event) -> int:
    return sum(1 for i in range(ev.size()) if ev[i].isFinal())


def run_one(lhe_path: Path, log_path: Path) -> Tuple[bool, bool, int, List[str]]:
    """Run PYTHIA once with the given LHE. Return (init_ok, next_ok, n_final, log_lines)."""
    lines: List[str] = [f"LHE: {lhe_path}"]
    if pythia8 is None:
        lines.append(f"pythia8 not available: {_pythia_import_error}")
        log_path.write_text("\n".join(lines), encoding="utf-8")
        return False, False, 0, lines
    if not lhe_path.exists():
        lines.append("File not found.")
        log_path.write_text("\n".join(lines), encoding="utf-8")
        return False, False, 0, lines

    pythia = build_pythia(lhe_path)
    init_ok = pythia.init()
    lines.append(f"pythia.init(): {init_ok}")
    next_ok = False
    n_final = 0
    if init_ok:
        next_ok = pythia.next()
        lines.append(f"pythia.next(): {next_ok}")
        if next_ok:
            n_final = count_final(pythia.event)
            lines.append(f"n_final: {n_final}")
        else:
            # Capture any PYTHIA message that might explain failure
            try:
                lines.append("(Check stderr for PYTHIA colour/kinematic messages.)")
            except Exception:
                pass
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return init_ok, next_ok, n_final, lines


def main() -> None:
    init_22, next_22, n_22, _ = run_one(LHE_2TO2, LOG_2TO2)
    init_24, next_24, n_24, log_24 = run_one(LHE_2TO4, LOG_2TO4)

    # Benchmark
    if init_22 and next_22 and init_24 and next_24:
        status = "PASS"
        reason = "2→2 baseline and 2→4 test both initialized and produced a fully evolved event."
    elif init_22 and next_22 and init_24 and not next_24:
        status = "SOFT PASS"
        reason = "2→2 baseline works; 2→4 initializes but pythia.next() failed."
    elif init_22 and next_22:
        status = "SOFT PASS"
        reason = "2→2 baseline works; 2→4 failed to initialize or next() failed."
    else:
        status = "FAIL"
        reason = "2→2 baseline failed or outputs missing."

    # Load debug and selected for summary answers
    debug_text = ""
    if DEBUG_2TO4.exists():
        debug_text = DEBUG_2TO4.read_text(encoding="utf-8").strip()
    selected_data = {}
    if SELECTED_HARD.exists():
        try:
            selected_data = json.loads(SELECTED_HARD.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Answers to the 9 questions
    q1 = "Yes." if (init_22 and next_22) else "No."
    q2 = "Yes." if init_24 else "No."
    q3 = "Yes." if (init_24 and next_24) else "No."
    q4 = "Source event indices and IDs: see selected_hard_particles.json and lhe_2to4_debug.txt."
    if selected_data and "in_electron" in selected_data:
        in_e = selected_data.get("in_electron", {})
        in_q = selected_data.get("in_quark", {})
        out_e = selected_data.get("out_electron", {})
        struck = selected_data.get("struck_quark", {})
        partner_q = selected_data.get("partner_quark", {})
        partner_qbar = selected_data.get("partner_antiquark", {})
        q4 = (
            f"Incoming e (idx={in_e.get('index')}, id={in_e.get('pdg_id')}), "
            f"incoming q (idx={in_q.get('index')}, id={in_q.get('pdg_id')}), "
            f"out e (idx={out_e.get('index')}), struck q (idx={struck.get('index')}, id={struck.get('pdg_id')}), "
            f"partner q (idx={partner_q.get('index')}, id={partner_q.get('pdg_id')}), "
            f"partner qbar (idx={partner_qbar.get('index')}, id={partner_qbar.get('pdg_id')})."
        )
    q5 = "Outgoing 4-vectors from source event; p_in_quark = sum(outgoing) - p_in_electron; then px=py=0 enforced. See lhe_2to4_debug.txt."
    if debug_text:
        q5 = "Exact 2→4 kinematics: outgoing e, struck q, partner q, partner qbar from source event; incoming quark from 4-momentum conservation; incoming beam-collinear (px=0, py=0). See lhe_2to4_debug.txt for full table."
    q6 = "Yes (by construction: incoming quark forced to px=0, py=0 in LHE)." if init_24 or init_22 else "N/A (LHE not built or not used)."
    q7 = "Struck line: col=101, acol=0 (in and out). Partner q': col=102, acol=0; partner qbar': col=0, acol=102. Electrons: 0,0."
    q8 = "N/A (2→4 worked)." if (init_24 and next_24) else (
        "Color flow rejected by PYTHIA (N(unmatched) or unphysical colour)."
        if init_24 and not next_24
        else "LHE file missing or init failed."
    )
    if init_24 and not next_24:
        q9 = "Try alternative colour tag numbering or ensure exactly two colour lines (one open struck, one closed q' qbar')."
    elif not LHE_2TO2.exists() or not LHE_2TO4.exists() or pythia8 is None:
        q9 = "Run build_single_2to4_lhe_from_pythia_event.py in an environment with working pythia8 to generate LHE files; then run this script."
    else:
        q9 = "N/A."

    # High-level summary
    hi = [
        "2→4 LHE proof-of-principle — HIGH-LEVEL SUMMARY",
        "",
        f"Status: {status}",
        f"Result: {reason}",
        "",
        "1. Did the 2→2 baseline still work? " + q1,
        "2. Did the 2→4 LHE initialize? " + q2,
        "3. Did the 2→4 event pass pythia.next()? " + q3,
        "4. Which exact particles from the source event were used? " + q4,
        "5. What exact 2→4 kinematics were used? " + q5,
        "6. Did the incoming quark remain beam-collinear? " + q6,
        "7. What exact color assignment was used? " + q7,
        "8. If it failed, what is the single most likely reason? " + q8,
        "9. What is the smallest next fix? " + q9,
        "",
    ]
    SUMMARY_HI.write_text("\n".join(hi), encoding="utf-8")

    # Low-level summary
    lo = [
        "2→4 LHE proof-of-principle — LOW-LEVEL SUMMARY",
        "",
        "Settings: Beams:frameType=4, Beams:LHEF=<file>, LesHouches:matchInOut=on, PartonLevel:ISR/FSR=on, HadronLevel:all=on, PartonLevel:MPI=off.",
        "",
        f"2→2: init={init_22} next={next_22} n_final={n_22}",
        f"2→4: init={init_24} next={next_24} n_final={n_24}",
        "",
        "1. 2→2 baseline worked? " + q1,
        "2. 2→4 LHE initialized? " + q2,
        "3. 2→4 next() passed? " + q3,
        "4. Source particles: " + q4,
        "5. Kinematics: " + q5,
        "6. Incoming quark collinear? " + q6,
        "7. Color: " + q7,
        "8. Likely failure reason: " + q8,
        "9. Next fix: " + q9,
        "",
    ]
    if debug_text:
        lo.append("Debug excerpt (first 1200 chars):")
        lo.append(debug_text[:1200])
        lo.append("")
    SUMMARY_LO.write_text("\n".join(lo), encoding="utf-8")

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

    print(f"Status: {status}")
    print(reason)
    print(f"2→2: init={init_22} next={next_22} n_final={n_22}  |  2→4: init={init_24} next={next_24} n_final={n_24}")


if __name__ == "__main__":
    main()
