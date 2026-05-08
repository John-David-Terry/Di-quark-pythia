#!/usr/bin/env python3
"""
Build kicked 2->4 LHE from the working zero-kick 2->4 LHE.

Modifications (only):
  - Outgoing struck quark (LHE entry 4): px += 0.4 GeV, E recomputed on-shell.
  - Outgoing partner quark q' (LHE entry 5): px -= 0.4 GeV, E recomputed on-shell.
All other particles unchanged. Incoming remain beam-collinear.

Reads: outputs/popf_lhe_single_event_2to4/single_event_2to4.lhe
  (falls back to single_event_2to4_test.lhe if the primary file is missing)
Writes to: outputs/popf_lhe_single_event_2to4_kicked/
  - single_event_2to4_kicked.lhe
  - kick_debug.txt
  - kicked_hard_particles.json (optional)

Then runs reinjection on zero-kick and kicked LHE (with python3.11 + pythia8),
writes reinjection_kicked_log.txt and summary_*.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import List, Tuple

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

ZERO_KICK_DIR = outputs_dir() / "popf_lhe_single_event_2to4"
KICKED_DIR = outputs_dir() / "popf_lhe_single_event_2to4_kicked"
ZERO_KICK_LHE_PRIMARY = ZERO_KICK_DIR / "single_event_2to4.lhe"
ZERO_KICK_LHE_FALLBACK = ZERO_KICK_DIR / "single_event_2to4_test.lhe"
KICKED_LHE = KICKED_DIR / "single_event_2to4_kicked.lhe"
KICK_DEBUG = KICKED_DIR / "kick_debug.txt"
KICKED_HARD_JSON = KICKED_DIR / "kicked_hard_particles.json"
REINJECTION_LOG = KICKED_DIR / "reinjection_kicked_log.txt"

DELTA_PX_GEV = 0.4
# LHE particle order: 1=in_e, 2=in_q, 3=out_e, 4=struck_q, 5=partner_q, 6=partner_qbar
IDX_STruck = 4  # 1-based LHE entry
IDX_PARTNER_Q = 5


def zero_kick_lhe_path() -> Path:
    """Prefer single_event_2to4.lhe; use test filename if primary is absent."""
    if ZERO_KICK_LHE_PRIMARY.exists():
        return ZERO_KICK_LHE_PRIMARY
    return ZERO_KICK_LHE_FALLBACK


def parse_lhe_event(lhe_path: Path) -> Tuple[str, str, List[List[float]]]:
    """Return (init_block, event_header_line, list of particle rows). Each row: [idup, istup, m1, m2, col, acol, px, py, pz, e, m] as numbers."""
    text = lhe_path.read_text()
    init_match = re.search(r"<init>\n(.*?)</init>", text, re.DOTALL)
    event_match = re.search(r"<event>\n(.*?)</event>", text, re.DOTALL)
    if not init_match or not event_match:
        raise RuntimeError(f"Could not parse init/event blocks in {lhe_path}")
    init_block = init_match.group(0)
    event_body = event_match.group(1).strip()
    lines = [ln for ln in event_body.split("\n") if ln.strip()]
    if len(lines) < 7:
        raise RuntimeError(f"Event block has too few lines: {len(lines)}")
    event_header = lines[0]
    particles = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) < 11:
            raise RuntimeError(f"Particle line has too few fields: {ln[:80]}")
        idup = int(parts[0])
        istup = int(parts[1])
        m1, m2 = int(parts[2]), int(parts[3])
        col, acol = int(parts[4]), int(parts[5])
        px, py, pz = float(parts[6]), float(parts[7]), float(parts[8])
        e, m = float(parts[9]), float(parts[10])
        vtimup = float(parts[11]) if len(parts) > 11 else 0.0
        spinup = float(parts[12]) if len(parts) > 12 else 9.0
        particles.append([idup, istup, m1, m2, col, acol, px, py, pz, e, m, vtimup, spinup])
    return init_block, event_header, particles


def write_lhe_event(out_path: Path, init_block: str, event_header: str, particles: List[List[float]]) -> None:
    """Write LHE file with same init and event header; particles is list of rows with 13 elements (idup..spinup)."""
    nup = len(particles)
    with out_path.open("w") as f:
        f.write('<LesHouchesEvents version="1.0">\n')
        f.write("<header>\n  <!-- Minimal DIS LHE 2->4 kicked -->\n</header>\n")
        f.write(init_block + "\n")
        f.write("<event>\n")
        f.write(event_header + "\n")
        for row in particles:
            idup, istup, m1, m2, col, acol = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])
            px, py, pz, e, m = row[6], row[7], row[8], row[9], row[10]
            vtimup = row[11] if len(row) > 11 else 0.0
            spinup = row[12] if len(row) > 12 else 9.0
            f.write(
                f" {idup:8d} {istup:4d}{m1:5d}{m2:5d}{col:5d}{acol:5d}"
                f" {px:14.7e} {py:14.7e} {pz:14.7e} {e:14.7e} {m:14.7e}"
                f" {vtimup:10.3e} {spinup:10.3e}\n"
            )
        f.write("</event>\n</LesHouchesEvents>\n")


def build_kicked_lhe() -> Tuple[bool, dict]:
    """Build kicked LHE and debug files. Return (success, debug_info dict)."""
    KICKED_DIR.mkdir(parents=True, exist_ok=True)
    src_lhe = zero_kick_lhe_path()
    if not src_lhe.exists():
        KICK_DEBUG.write_text(f"Zero-kick LHE not found: {ZERO_KICK_LHE_PRIMARY} or {ZERO_KICK_LHE_FALLBACK}\n")
        return False, {"error": "zero_kick_lhe_missing"}

    init_block, event_header, particles = parse_lhe_event(src_lhe)
    if len(particles) != 6:
        KICK_DEBUG.write_text(f"Expected 6 particles, got {len(particles)}\n")
        return False, {"error": "wrong_nup"}

    # Identification (1-based LHE index -> 0-based list index)
    # 1=in_e, 2=in_q, 3=out_e, 4=struck_q, 5=partner_q, 6=partner_qbar
    roles = ["in_electron", "in_quark", "out_electron", "struck_quark", "partner_quark", "partner_antiquark"]
    struck_row = particles[IDX_STruck - 1]
    partner_row = particles[IDX_PARTNER_Q - 1]

    px_s_old = struck_row[6]
    py_s_old = struck_row[7]
    pz_s_old = struck_row[8]
    e_s_old = struck_row[9]
    m_s = struck_row[10]

    px_p_old = partner_row[6]
    py_p_old = partner_row[7]
    pz_p_old = partner_row[8]
    e_p_old = partner_row[9]
    m_p = partner_row[10]

    # Kicks: struck +0.4, partner -0.4
    px_s_new = px_s_old + DELTA_PX_GEV
    py_s_new = py_s_old
    pz_s_new = pz_s_old
    e_s_new = math.sqrt(px_s_new**2 + py_s_new**2 + pz_s_new**2 + m_s**2)

    px_p_new = px_p_old - DELTA_PX_GEV
    py_p_new = py_p_old
    pz_p_new = pz_p_old
    e_p_new = math.sqrt(px_p_new**2 + py_p_new**2 + pz_p_new**2 + m_p**2)

    # Replace in copy
    kicked_particles = [list(p) for p in particles]
    kicked_particles[IDX_STruck - 1][6:10] = [px_s_new, py_s_new, pz_s_new, e_s_new]
    kicked_particles[IDX_PARTNER_Q - 1][6:10] = [px_p_new, py_p_new, pz_p_new, e_p_new]

    # Momentum sums
    def sum4(rows: List[List[float]], idx_off: int = 0) -> Tuple[float, float, float, float]:
        px = sum(r[6] for r in rows)
        py = sum(r[7] for r in rows)
        pz = sum(r[8] for r in rows)
        e = sum(r[9] for r in rows)
        return (px, py, pz, e)

    pin_before = sum4(particles[:2])
    pout_before = sum4(particles[2:])
    pin_after = sum4(kicked_particles[:2])
    pout_after = sum4(kicked_particles[2:])

    debug_info = {
        "zero_kick_source_lhe": str(src_lhe),
        "particle_roles": roles,
        "struck_quark_lhe_index": IDX_STruck,
        "partner_quark_lhe_index": IDX_PARTNER_Q,
        "struck_before": {"px": px_s_old, "py": py_s_old, "pz": pz_s_old, "E": e_s_old, "m": m_s},
        "struck_after": {"px": px_s_new, "py": py_s_new, "pz": pz_s_new, "E": e_s_new, "m": m_s},
        "partner_before": {"px": px_p_old, "py": py_p_old, "pz": pz_p_old, "E": e_p_old, "m": m_p},
        "partner_after": {"px": px_p_new, "py": py_p_new, "pz": pz_p_new, "E": e_p_new, "m": m_p},
        "total_px_before": pout_before[0],
        "total_px_after": pout_after[0],
        "total_py_before": pout_before[1],
        "total_py_after": pout_after[1],
        "total_pz_before": pout_before[2],
        "total_pz_after": pout_after[2],
        "total_E_before": pout_before[3],
        "total_E_after": pout_after[3],
        "incoming_px_before": pin_before[0],
        "incoming_px_after": pin_after[0],
        "delta_px_gev": DELTA_PX_GEV,
    }

    # kick_debug.txt
    lines = [
        "Kicked 2->4 LHE — identification and kinematics",
        "",
        f"Source (zero-kick) file: {src_lhe}",
        "",
        "LHE particle order (1-based):",
        "  1 = incoming electron (id=11)",
        "  2 = incoming struck quark (id=3 in this event)",
        "  3 = outgoing electron (id=11)",
        "  4 = outgoing struck quark (id=3)  <- +0.4 GeV px kick",
        "  5 = outgoing partner quark q' (id=2)  <- -0.4 GeV px kick",
        "  6 = outgoing partner antiquark qbar' (id=-2)",
        "",
        "Particles being kicked: LHE entries 4 (struck) and 5 (partner q').",
        "",
        "Original four-momenta (before kick):",
        f"  Struck quark (entry 4): px={px_s_old:.6g} py={py_s_old:.6g} pz={pz_s_old:.6g} E={e_s_old:.6g} m={m_s:.6g}",
        f"  Partner quark (entry 5): px={px_p_old:.6g} py={py_p_old:.6g} pz={pz_p_old:.6g} E={e_p_old:.6g} m={m_p:.6g}",
        "",
        "Modified four-momenta (after kick, E from on-shell):",
        f"  Struck quark: px={px_s_new:.6g} py={py_s_new:.6g} pz={pz_s_new:.6g} E={e_s_new:.6g} m={m_s:.6g}",
        f"  Partner quark: px={px_p_new:.6g} py={py_p_new:.6g} pz={pz_p_new:.6g} E={e_p_new:.6g} m={m_p:.6g}",
        "",
        "Total hard-event (outgoing) before kick:",
        f"  px={pout_before[0]:.6g} py={pout_before[1]:.6g} pz={pout_before[2]:.6g} E={pout_before[3]:.6g}",
        "Total hard-event (outgoing) after kick:",
        f"  px={pout_after[0]:.6g} py={pout_after[1]:.6g} pz={pout_after[2]:.6g} E={pout_after[3]:.6g}",
        "",
        "Total incoming (unchanged):",
        f"  px={pin_before[0]:.6g} py={pin_before[1]:.6g} pz={pin_before[2]:.6g} E={pin_before[3]:.6g}",
        "",
        "Four-momentum conservation:",
        "  Outgoing px: before and after differ by (struck +0.4, partner -0.4) => net 0. So 3-momentum in x is preserved.",
        "  Outgoing py, pz: unchanged (only px kicks).",
        f"  Outgoing energy: before E_out={pout_before[3]:.6g}, after E_out={pout_after[3]:.6g}. Difference = {pout_after[3] - pout_before[3]:.6g} (on-shell update changes total E slightly).",
        "",
        "Conclusion: 3-momentum is preserved exactly (px net zero from opposite kicks). Total hard-event energy is NOT preserved; the on-shell recomputation of the two quarks increases total outgoing E by a small amount.",
    ]
    KICK_DEBUG.write_text("\n".join(lines))

    write_lhe_event(KICKED_LHE, init_block, event_header, kicked_particles)

    # kicked_hard_particles.json: minimal list of six with 4-vectors (kicked for 4 and 5)
    hard_list = []
    for i, row in enumerate(kicked_particles):
        hard_list.append({
            "lhe_index_1based": i + 1,
            "role": roles[i],
            "id": int(row[0]),
            "istup": int(row[1]),
            "px": row[6], "py": row[7], "pz": row[8], "E": row[9], "m": row[10],
        })
    KICKED_HARD_JSON.write_text(json.dumps({"particles": hard_list, "delta_px_gev": DELTA_PX_GEV}, indent=2))

    return True, debug_info


def run_reinjection() -> Tuple[bool, bool, bool, bool, int, int, int, int, List[str]]:
    """Run PYTHIA on zero-kick and kicked LHE.
    Return (zero_init, zero_next, kick_init, kick_next, n_zero, n_zero_had, n_kick, n_kick_had, log_lines).
    """
    try:
        import pythia8
    except ImportError:
        log = ["pythia8 not available; cannot run reinjection."]
        REINJECTION_LOG.write_text("\n".join(log))
        return False, False, False, False, 0, 0, 0, 0, log

    log_lines: List[str] = []

    def run_one(lhe_path: Path) -> Tuple[bool, bool, int, int]:
        p = pythia8.Pythia()
        p.readString("Beams:frameType = 4")
        p.readString(f"Beams:LHEF = {lhe_path}")
        p.readString("LesHouches:matchInOut = on")
        p.readString("PartonLevel:ISR = on")
        p.readString("PartonLevel:FSR = on")
        p.readString("HadronLevel:all = on")
        p.readString("PartonLevel:MPI = off")
        p.readString("Print:quiet = on")
        init_ok = p.init()
        next_ok = p.next() if init_ok else False
        n_final = 0
        n_had = 0
        if next_ok:
            for i in range(p.event.size()):
                pp = p.event[i]
                if pp.isFinal():
                    n_final += 1
                    if pp.isHadron():
                        n_had += 1
        return init_ok, next_ok, n_final, n_had

    zk = zero_kick_lhe_path()
    log_lines.append("=== Zero-kick 2->4 LHE ===")
    log_lines.append(f"LHE: {zk}")
    zero_init, zero_next, n_zero, n_zero_had = run_one(zk)
    log_lines.append(f"pythia.init(): {zero_init}")
    log_lines.append(f"pythia.next(): {zero_next}")
    log_lines.append(f"n_final: {n_zero}")
    log_lines.append(f"n_final_hadrons: {n_zero_had}")
    log_lines.append("")

    log_lines.append("=== Kicked 2->4 LHE ===")
    log_lines.append(f"LHE: {KICKED_LHE}")
    if not KICKED_LHE.exists():
        log_lines.append("File not found.")
        kick_init, kick_next, n_kick, n_kick_had = False, False, 0, 0
    else:
        kick_init, kick_next, n_kick, n_kick_had = run_one(KICKED_LHE)
        log_lines.append(f"pythia.init(): {kick_init}")
        log_lines.append(f"pythia.next(): {kick_next}")
        log_lines.append(f"n_final: {n_kick}")
        log_lines.append(f"n_final_hadrons: {n_kick_had}")
    REINJECTION_LOG.write_text("\n".join(log_lines))
    return zero_init, zero_next, kick_init, kick_next, n_zero, n_zero_had, n_kick, n_kick_had, log_lines


def write_summaries(
    kicked_written: bool,
    zero_init: bool,
    zero_next: bool,
    kick_init: bool,
    kick_next: bool,
    n_zero: int,
    n_zero_had: int,
    n_kick: int,
    n_kick_had: int,
    debug_info: dict,
) -> None:
    if kick_init and kick_next:
        outcome = "A. Kick accepted"
        fix = "None; kicked test passed."
    elif not kick_init:
        outcome = "B. Init-stage failure"
        fix = "Kicked LHE rejected at init; check LHE format or PYTHIA Les Houches requirements."
    else:
        outcome = "C. Event-generation failure"
        fix = "Init succeeded but next() failed; likely hard-event energy/kinematics or colour/mass handling."

    q1 = "Yes." if kicked_written else "No."
    q2 = "Yes." if (kick_init and kick_next) else "No."
    q3 = "Write stage: ok. Init: " + ("ok" if kick_init else "failed") + ". next(): " + ("ok" if kick_next else "failed") + "."
    q4 = fix

    hi = [
        "Kicked 2->4 LHE test — HIGH-LEVEL SUMMARY",
        "",
        "1. Was the kicked 2->4 LHE successfully written? " + q1,
        "2. Did PYTHIA accept the kicked LHE? " + q2,
        "3. If not, where did it fail (write stage, init, or next())? " + q3,
        "4. What is the single smallest next fix? " + q4,
        "",
        "Outcome classification: " + outcome,
        "",
        "Zero-kick baseline (same 2->4 LHE): init="
        + str(zero_init)
        + " next="
        + str(zero_next)
        + " n_final="
        + str(n_zero)
        + " n_hadrons="
        + str(n_zero_had),
        "Kicked 2->4 LHE: init="
        + str(kick_init)
        + " next="
        + str(kick_next)
        + " n_final="
        + str(n_kick)
        + " n_hadrons="
        + str(n_kick_had),
    ]
    SUMMARY_HI = KICKED_DIR / "summary_high_level.txt"
    SUMMARY_HI.write_text("\n".join(hi))

    # Low-level: exact particles, 4-vectors, momentum sums
    s_before = debug_info.get("struck_before", {})
    s_after = debug_info.get("struck_after", {})
    p_before = debug_info.get("partner_before", {})
    p_after = debug_info.get("partner_after", {})
    lo = [
        "Kicked 2->4 LHE test — LOW-LEVEL SUMMARY",
        "",
        "Particles chosen: LHE entry 4 = outgoing struck quark (id=3), LHE entry 5 = partner quark q' (id=2).",
        "",
        "Original four-vectors:",
        f"  Struck: px={s_before.get('px',0):.6g} py={s_before.get('py',0):.6g} pz={s_before.get('pz',0):.6g} E={s_before.get('E',0):.6g} m={s_before.get('m',0):.6g}",
        f"  Partner q': px={p_before.get('px',0):.6g} py={p_before.get('py',0):.6g} pz={p_before.get('pz',0):.6g} E={p_before.get('E',0):.6g} m={p_before.get('m',0):.6g}",
        "",
        "Modified four-vectors (E = sqrt(px^2+py^2+pz^2+m^2)):",
        f"  Struck: px={s_after.get('px',0):.6g} py={s_after.get('py',0):.6g} pz={s_after.get('pz',0):.6g} E={s_after.get('E',0):.6g}",
        f"  Partner q': px={p_after.get('px',0):.6g} py={p_after.get('py',0):.6g} pz={p_after.get('pz',0):.6g} E={p_after.get('E',0):.6g}",
        "",
        "Total hard-event (outgoing) before/after:",
        f"  px: {debug_info.get('total_px_before',0):.6g} -> {debug_info.get('total_px_after',0):.6g}",
        f"  E:  {debug_info.get('total_E_before',0):.6g} -> {debug_info.get('total_E_after',0):.6g}",
        "",
        "px conservation (entries 4+5 only, py/pz unchanged):",
        f"  px4+px5 old = {s_before.get('px',0) + p_before.get('px',0):.10g}",
        f"  px4+px5 new = {s_after.get('px',0) + p_after.get('px',0):.10g}",
        "",
        "Kicked LHE file written: " + str(kicked_written),
        "Reinjection: zero-kick init="
        + str(zero_init)
        + " next="
        + str(zero_next)
        + " n_hadrons="
        + str(n_zero_had)
        + " | kicked init="
        + str(kick_init)
        + " next="
        + str(kick_next)
        + " n_hadrons="
        + str(n_kick_had),
        "",
        "Root cause (if failure): " + fix,
    ]
    SUMMARY_LO = KICKED_DIR / "summary_low_level.txt"
    SUMMARY_LO.write_text("\n".join(lo))

    SUMMARY_COMBINED = KICKED_DIR / "summary_combined.txt"
    SUMMARY_COMBINED.write_text(
        "HIGH LEVEL SUMMARY\n\n" + SUMMARY_HI.read_text().strip() + "\n\n\nLOW LEVEL SUMMARY\n\n" + SUMMARY_LO.read_text().strip() + "\n"
    )


def main() -> None:
    ok, debug_info = build_kicked_lhe()
    if not ok and "error" in debug_info:
        # Still run reinjection to log zero-kick and missing kicked
        z0, zn, ki, kn, nz, nzh, nk, nkh, _ = run_reinjection()
        write_summaries(False, z0, zn, ki, kn, nz, nzh, nk, nkh, {})
        print("build_kicked_lhe failed:", debug_info.get("error"))
        return
    zero_init, zero_next, kick_init, kick_next, n_zero, n_zero_had, n_kick, n_kick_had, log_lines = run_reinjection()
    # Append reinjection result to kick_debug
    with KICK_DEBUG.open("a") as f:
        f.write("\n\n--- Reinjection ---\n")
        f.write("\n".join(log_lines))
    write_summaries(
        ok, zero_init, zero_next, kick_init, kick_next, n_zero, n_zero_had, n_kick, n_kick_had, debug_info
    )

    sb = debug_info.get("struck_before", {})
    sa = debug_info.get("struck_after", {})
    pb = debug_info.get("partner_before", {})
    pa = debug_info.get("partner_after", {})
    px_sum_old = float(sb.get("px", 0)) + float(pb.get("px", 0))
    px_sum_new = float(sa.get("px", 0)) + float(pa.get("px", 0))

    print("")
    print("=== Single-event 2->4 transverse kick (Δpx = %.1f GeV on 4, − on 5) ===" % DELTA_PX_GEV)
    print("Zero-kick source:", debug_info.get("zero_kick_source_lhe", zero_kick_lhe_path()))
    print("Kicked LHE written:", KICKED_LHE)
    print("")
    print("Particle 4 (outgoing struck quark):")
    print("  px_old=%.10g  px_new=%.10g" % (sb.get("px", 0), sa.get("px", 0)))
    print("  E_old =%.10g  E_new =%.10g" % (sb.get("E", 0), sa.get("E", 0)))
    print("Particle 5 (outgoing partner quark):")
    print("  px_old=%.10g  px_new=%.10g" % (pb.get("px", 0), pa.get("px", 0)))
    print("  E_old =%.10g  E_new =%.10g" % (pb.get("E", 0), pa.get("E", 0)))
    print("")
    print("px conservation (4+5 only):  sum_old=%.10g  sum_new=%.10g  match=%s" % (px_sum_old, px_sum_new, abs(px_sum_old - px_sum_new) < 1e-9))
    print("")
    print("PYTHIA (ISR=on, FSR=on, HadronLevel=on, MPI=off, LesHouches:matchInOut=on)")
    print("  Zero-kick:  init=%s  next=%s  n_final=%d  n_final_hadrons=%d" % (zero_init, zero_next, n_zero, n_zero_had))
    print("  Kicked:     init=%s  next=%s  n_final=%d  n_final_hadrons=%d" % (kick_init, kick_next, n_kick, n_kick_had))
    accepted = bool(kick_init and kick_next)
    print("  Kicked event accepted (init and next both True): %s" % accepted)
    if accepted:
        print("  Success: no init/next failure for this single-event kick test.")
    else:
        print("  See %s and PYTHIA stderr for colour/momentum messages." % REINJECTION_LOG)


if __name__ == "__main__":
    main()
