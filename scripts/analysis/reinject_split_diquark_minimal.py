#!/usr/bin/env python3
"""
Minimal hand-built final-parton reinjection trial (Lund.cc-style).

This intentionally avoids reinjecting the full internal PYTHIA event record.
Instead, for each selected event we:
  - reset PYTHIA event
  - append only a minimal set of final partons with status 23
  - call pythia.next() to run FSR + hadronization

Two test modes are run:
  - 2-parton control: struck q + anti-quark surrogate
  - 3-parton test: struck q + remnant u + remnant d
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    import pythia8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pythia8 import failed: {exc}")

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

OUTDIR = outputs_dir() / "dis_isr_parton_dataset"

IN_SPLIT_EVENT = OUTDIR / "dis_isr_full_event_record_diquark_split.csv"
IN_SPLIT_META = OUTDIR / "dis_isr_diquark_split_metadata.csv"

OUT_LOG = OUTDIR / "reinjection_minimal_trial_log.txt"
OUT_CSV = OUTDIR / "reinjection_minimal_trial_results.csv"

N_TRIAL = 20


def make_hadronizer() -> "pythia8.Pythia":
    p = pythia8.Pythia()
    p.readString("Beams:idA = 2212")
    p.readString("Beams:idB = 2212")
    p.readString("Beams:eCM = 100.0")
    p.readString("ProcessLevel:all = off")
    p.readString("PartonLevel:ISR = off")
    p.readString("PartonLevel:FSR = on")
    p.readString("PartonLevel:MPI = off")
    p.readString("HadronLevel:all = on")
    ok = p.init()
    if not ok:
        raise RuntimeError("pythia.init() failed for minimal hadronizer")
    return p


def count_final_and_hadrons(ev: "pythia8.Event") -> Tuple[int, int]:
    n_final = 0
    n_had = 0
    for i in range(ev.size()):
        pp = ev[i]
        if pp.isFinal():
            n_final += 1
            if pp.isHadron():
                n_had += 1
    return n_final, n_had


def append_parton(p: "pythia8.Pythia", pid: int, col: int, acol: int, px: float, py: float, pz: float, E: float, m: float) -> None:
    # Lund-style append: status=23 final parton.
    p.event.append(int(pid), 23, int(col), int(acol), float(px), float(py), float(pz), float(E), float(m))


def run_one(p: "pythia8.Pythia", particles: List[Tuple[int, int, int, float, float, float, float, float]]) -> Tuple[bool, int, int, str]:
    try:
        p.event.reset()
        for pid, col, acol, px, py, pz, E, m in particles:
            append_parton(p, pid, col, acol, px, py, pz, E, m)
        next_ok = bool(p.next())
        if not next_ok:
            return False, 0, 0, "pythia.next() failed"
        n_final, n_had = count_final_and_hadrons(p.event)
        return True, n_final, n_had, ""
    except Exception as exc:
        return False, 0, 0, str(exc)


def main() -> None:
    ev = pd.read_csv(IN_SPLIT_EVENT).sort_values(["event_id", "particle_index"]).reset_index(drop=True)
    meta = pd.read_csv(IN_SPLIT_META).sort_values("event_id").reset_index(drop=True)
    trial_ids = meta["event_id"].drop_duplicates().head(N_TRIAL).tolist()
    meta_by_id = meta.set_index("event_id")

    p = make_hadronizer()
    rows: List[Dict[str, object]] = []
    log: List[str] = []

    for i, event_id in enumerate(trial_ids, start=1):
        g = ev[ev["event_id"] == event_id].copy()
        m = meta_by_id.loc[event_id]
        struck_idx = int(m["struck_outgoing_index_selected"])
        u_idx = int(m["new_u_index"])
        d_idx = int(m["new_d_index"])

        struck = g[g["particle_index"] == struck_idx].iloc[0]
        u = g[g["particle_index"] == u_idx].iloc[0]
        d = g[g["particle_index"] == d_idx].iloc[0]

        if i <= 5:
            log.append(f"--- event_id={event_id} ---")
            log.append(
                "struck "
                f"(px,py,pz,E)=({float(struck['px'])}, {float(struck['py'])}, {float(struck['pz'])}, {float(struck['E'])})"
            )
            log.append(
                "u "
                f"(px,py,pz,E)=({float(u['px'])}, {float(u['py'])}, {float(u['pz'])}, {float(u['E'])})"
            )
            log.append(
                "d "
                f"(px,py,pz,E)=({float(d['px'])}, {float(d['py'])}, {float(d['pz'])}, {float(d['E'])})"
            )

        # TEST 1: 2-parton control, forced closed string.
        # struck q: col=501, acol=0; anti-q surrogate: col=0, acol=501.
        struck_pid = int(struck["pdg_id"])
        surrogate_pid = -struck_pid if struck_pid != 0 else -2
        parts2 = [
            (struck_pid, 501, 0, float(struck["px"]), float(struck["py"]), float(struck["pz"]), float(struck["E"]), float(struck["m"])),
            (surrogate_pid, 0, 501, float(u["px"]), float(u["py"]), float(u["pz"]), float(u["E"]), float(u["m"])),
        ]
        ok2, nf2, nh2, err2 = run_one(p, parts2)
        rows.append(
            {
                "event_id": int(event_id),
                "test_mode": "2parton_control",
                "build_success": 1,
                "pythia_next_success": int(ok2),
                "hadrons_produced": int(nh2 > 0),
                "n_final_particles": int(nf2),
                "n_final_hadrons": int(nh2),
                "error_message": err2,
                "notes": "struck+antiq surrogate, closed 501 line",
            }
        )
        if i <= 5:
            log.append("test_mode=2parton_control")
            log.append(
                f"append: struck(id={struck_pid},col=501,acol=0), surrogate(id={surrogate_pid},col=0,acol=501)"
            )
            log.append("color_balanced_for_test=True")
            log.append(f"pythia_next={ok2}, n_final={nf2}, n_had={nh2}")

        # TEST 2: 3-parton struck+u+d with intended DIS-style assignment.
        # struck: col=501; u partner: acol=501; d spectator: col=502 open.
        parts3 = [
            (struck_pid, 501, 0, float(struck["px"]), float(struck["py"]), float(struck["pz"]), float(struck["E"]), float(struck["m"])),
            (2, 0, 501, float(u["px"]), float(u["py"]), float(u["pz"]), float(u["E"]), float(u["m"])),
            (1, 502, 0, float(d["px"]), float(d["py"]), float(d["pz"]), float(d["E"]), float(d["m"])),
        ]
        ok3, nf3, nh3, err3 = run_one(p, parts3)
        rows.append(
            {
                "event_id": int(event_id),
                "test_mode": "3parton_struck_u_d",
                "build_success": 1,
                "pythia_next_success": int(ok3),
                "hadrons_produced": int(nh3 > 0),
                "n_final_particles": int(nf3),
                "n_final_hadrons": int(nh3),
                "error_message": err3,
                "notes": "struck-u-d with open spectator color 502",
            }
        )
        if i <= 5:
            log.append("test_mode=3parton_struck_u_d")
            log.append(
                "append: struck(id={},col=501,acol=0), u(id=2,col=0,acol=501), d(id=1,col=502,acol=0)".format(
                    struck_pid
                )
            )
            log.append("color_balanced_for_test=False (expected open baryonic topology)")
            log.append(f"pythia_next={ok3}, n_final={nf3}, n_had={nh3}")
            log.append("")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    summary_lines: List[str] = []
    for mode in ["2parton_control", "3parton_struck_u_d"]:
        d = out_df[out_df["test_mode"] == mode]
        attempted = len(d)
        nxt = int(d["pythia_next_success"].sum())
        had = int(d["hadrons_produced"].sum())
        fail_counts = Counter([x for x in d["error_message"].tolist() if x])
        ex = defaultdict(list)
        for _, r in d.iterrows():
            em = str(r["error_message"])
            if em and len(ex[em]) < 5:
                ex[em].append(int(r["event_id"]))
        summary_lines.append(f"[{mode}] attempted={attempted} pythia_next_success={nxt} hadrons_produced={had}")
        summary_lines.append(f"[{mode}] error_counts={dict(fail_counts)}")
        summary_lines.append(f"[{mode}] error_examples={dict(ex)}")

    OUT_LOG.write_text("\n".join(summary_lines + [""] + log), encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"log={OUT_LOG}")
    print(f"csv={OUT_CSV}")


if __name__ == "__main__":
    main()

