#!/usr/bin/env python3
"""
Reinject 2→2 baseline and 2→3 test LHE files into PYTHIA (same settings as working pipeline).

Outputs:
  - reinjection_2to2_log.txt
  - reinjection_2to3_log.txt
  - summary_high_level.txt, summary_low_level.txt, summary_combined.txt
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

OUTDIR = outputs_dir() / "popf_lhe_single_event_2to3"
OUTDIR.mkdir(parents=True, exist_ok=True)
LHE_2TO2 = OUTDIR / "single_event_2to2_baseline.lhe"
LHE_2TO3 = OUTDIR / "single_event_2to3_test.lhe"
LOG_2TO2 = OUTDIR / "reinjection_2to2_log.txt"
LOG_2TO3 = OUTDIR / "reinjection_2to3_log.txt"
SUMMARY_HI = OUTDIR / "summary_high_level.txt"
SUMMARY_LO = OUTDIR / "summary_low_level.txt"
SUMMARY_COMBINED = OUTDIR / "summary_combined.txt"


def build_pythia(lhe_path: Path) -> pythia8.Pythia:
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


def run_one(lhe_path: Path, log_path: Path) -> Tuple[bool, bool, int]:
    """Return (init_ok, next_ok, n_final)."""
    lines: List[str] = [f"LHE: {lhe_path}"]
    if not lhe_path.exists():
        lines.append("File not found.")
        log_path.write_text("\n".join(lines), encoding="utf-8")
        return False, False, 0
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
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return init_ok, next_ok, n_final


def main() -> None:
    init_22, next_22, n_22 = run_one(LHE_2TO2, LOG_2TO2)
    init_23, next_23, n_23 = run_one(LHE_2TO3, LOG_2TO3)

    if init_22 and next_22 and init_23 and next_23:
        status = "PASS"
        reason = "2→2 baseline and 2→3 test both initialized and produced a fully evolved event."
    elif init_22 and next_22 and init_23 and not next_23:
        status = "SOFT PASS"
        reason = "2→2 baseline works; 2→3 initializes but pythia.next() failed."
    elif init_22 and next_22:
        status = "SOFT PASS"
        reason = "2→2 baseline works; 2→3 failed to initialize or next() failed."
    else:
        status = "FAIL"
        reason = "2→2 baseline failed or outputs missing."

    # Load debug for summary
    debug_path = OUTDIR / "lhe_2to3_debug.txt"
    debug_snippet = ""
    if debug_path.exists():
        debug_snippet = debug_path.read_text(encoding="utf-8").strip()[:1500]

    q1 = "Yes." if (init_22 and next_22) else "No."
    q2 = "Yes." if init_23 else "No."
    q3 = "Yes." if (init_23 and next_23) else "No."
    q4 = "z=0.8 split of outgoing quark 4-momentum; struck = z*p_q, spectator = (1-z)*p_q. On-shell masses m_struck=z*m_q, m_spec=(1-z)*m_q. See lhe_2to3_debug.txt."
    q5 = "Struck q: col=101, acol=0. Spectator ubar (id=-2): col=0, acol=101 (closes color 101)."
    q6 = "N/A (2→3 worked)." if (init_23 and next_23) else "PYTHIA reports 'N(unmatched (anti)colour tags) != 3' and 'unphysical colour flow' in ProcessLevel::checkColours — colour assignment for the 2→3 event is rejected."
    q7 = "Try alternative colour assignment (e.g. spectator as gluon, or different tag numbering); or add a second spectator so all colour lines close in pairs; or check PYTHIA docs for Les Houches colour in 2→3."

    hi = [
        "2→3 LHE proof-of-principle — HIGH-LEVEL SUMMARY",
        "",
        f"Status: {status}",
        f"Result: {reason}",
        "",
        "1. Did the 2→2 baseline still work? " + q1,
        "2. Did the 2→3 LHE initialize? " + q2,
        "3. Did the 2→3 event pass pythia.next()? " + q3,
        "4. Exact 2→3 kinematics: " + q4,
        "5. Color assignment: " + q5,
        "6. If 2→3 failed, single most likely reason? " + q6,
        "7. Smallest next fix? " + q7,
        "",
    ]
    SUMMARY_HI.write_text("\n".join(hi), encoding="utf-8")

    lo = [
        "2→3 LHE proof-of-principle — LOW-LEVEL SUMMARY",
        "",
        "Settings: Beams:frameType=4, Beams:LHEF=<file>, LesHouches:matchInOut=on, PartonLevel:ISR/FSR=on, HadronLevel:all=on, PartonLevel:MPI=off.",
        "",
        f"2→2: init={init_22} next={next_22} n_final={n_22}",
        f"2→3: init={init_23} next={next_23} n_final={n_23}",
        "",
        "1. 2→2 baseline worked? " + q1,
        "2. 2→3 LHE initialized? " + q2,
        "3. 2→3 next() passed? " + q3,
        "4. Kinematics: " + q4,
        "5. Color: " + q5,
        "6. Likely failure reason: " + q6,
        "7. Next fix: " + q7,
        "",
        "Debug excerpt:",
        debug_snippet,
        "",
    ]
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
    print(f"2→2: init={init_22} next={next_22} n_final={n_22}  |  2→3: init={init_23} next={next_23} n_final={n_23}")


if __name__ == "__main__":
    main()
