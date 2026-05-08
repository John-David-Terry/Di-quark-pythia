#!/usr/bin/env python3
"""
Minimal direct PYTHIA->PYTHIA reinjection sanity test for q-qbar.

No LHE, no DIS realism, no remnants, no junctions:
  - reset event
  - append q and qbar with matched color line
  - run pythia.next()
  - verify hadron production
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

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
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_LOG = OUTDIR / "minimal_direct_reinjection_q_qbar_log.txt"


def make_pythia() -> "pythia8.Pythia":
    p = pythia8.Pythia()
    p.readString("Beams:idA = 2212")
    p.readString("Beams:idB = 2212")
    p.readString("Beams:eCM = 200.0")
    p.readString("ProcessLevel:all = off")
    p.readString("PartonLevel:ISR = off")
    p.readString("PartonLevel:FSR = on")
    p.readString("PartonLevel:MPI = off")
    p.readString("HadronLevel:all = on")
    ok = p.init()
    if not ok:
        raise RuntimeError("pythia.init() failed")
    return p


def mass_for_pid(pid: int) -> float:
    apid = abs(pid)
    # small constituent-like values are fine for technical test
    if apid in {1, 2, 3}:
        return 0.33
    if apid == 4:
        return 1.5
    if apid == 5:
        return 4.8
    return 0.33


def append_qqbar(
    p: "pythia8.Pythia", pid_q: int, pz_abs: float, status: int
) -> None:
    m = mass_for_pid(pid_q)
    E = math.sqrt(pz_abs * pz_abs + m * m)
    # quark and antiquark, matched single color line
    p.event.append(pid_q, status, 101, 0, 0.0, 0.0, +pz_abs, E, m)
    p.event.append(-pid_q, status, 0, 101, 0.0, 0.0, -pz_abs, E, m)


def event_listing_text(ev: "pythia8.Event", title: str) -> str:
    lines = [f"--- {title} ---"]
    lines.append("idx id status m1 m2 d1 d2 col acol px py pz E m isFinal isHadron")
    for i in range(ev.size()):
        p = ev[i]
        lines.append(
            f"{i:3d} {int(p.id()):4d} {int(p.status()):4d} "
            f"{int(p.mother1()):3d} {int(p.mother2()):3d} "
            f"{int(p.daughter1()):3d} {int(p.daughter2()):3d} "
            f"{int(p.col()):3d} {int(p.acol()):3d} "
            f"{float(p.px()): .6f} {float(p.py()): .6f} {float(p.pz()): .6f} {float(p.e()): .6f} {float(p.m()): .6f} "
            f"{int(p.isFinal())} {int(p.isHadron())}"
        )
    return "\n".join(lines)


def count_hadrons(ev: "pythia8.Event") -> Tuple[int, int]:
    nf, nh = 0, 0
    for i in range(ev.size()):
        pp = ev[i]
        if pp.isFinal():
            nf += 1
            if pp.isHadron():
                nh += 1
    return nf, nh


def run_one(pid_q: int, pz_abs: float, status: int) -> Dict[str, object]:
    p = make_pythia()
    p.event.reset()
    append_qqbar(p, pid_q, pz_abs, status)
    before = event_listing_text(p.event, f"before hadronization pid={pid_q} pz={pz_abs} status={status}")
    ok = bool(p.next())
    after = event_listing_text(p.event, f"after hadronization pid={pid_q} pz={pz_abs} status={status}") if ok else ""
    nf, nh = count_hadrons(p.event) if ok else (0, 0)
    return {
        "pid_q": pid_q,
        "pz_abs": pz_abs,
        "status": status,
        "next_ok": ok,
        "n_final": nf,
        "n_had": nh,
        "before_listing": before,
        "after_listing": after,
    }


def main() -> None:
    # systematic minimal variations only
    statuses = [23, 62]
    flavors = [2, 1]  # u, d
    scales = [5.0, 10.0, 20.0]

    results: List[Dict[str, object]] = []
    log_lines: List[str] = []

    for st in statuses:
        for flv in flavors:
            for pz in scales:
                try:
                    res = run_one(flv, pz, st)
                except Exception as exc:
                    res = {
                        "pid_q": flv,
                        "pz_abs": pz,
                        "status": st,
                        "next_ok": False,
                        "n_final": 0,
                        "n_had": 0,
                        "before_listing": "",
                        "after_listing": "",
                        "error": str(exc),
                    }
                results.append(res)

                hdr = (
                    f"case pid={res['pid_q']} pz={res['pz_abs']} status={res['status']} "
                    f"next_ok={res['next_ok']} n_final={res['n_final']} n_had={res['n_had']}"
                )
                log_lines.append(hdr)
                if "error" in res:
                    log_lines.append(f"error: {res['error']}")
                if res.get("before_listing"):
                    log_lines.append(str(res["before_listing"]))
                if res.get("after_listing"):
                    log_lines.append(str(res["after_listing"]))
                log_lines.append("")

    success_cases = [r for r in results if bool(r["next_ok"]) and int(r["n_had"]) > 0]
    summary = [
        "Minimal q-qbar direct reinjection summary",
        f"total_cases={len(results)}",
        f"next_success_cases={sum(1 for r in results if bool(r['next_ok']))}",
        f"hadronizing_success_cases={len(success_cases)}",
    ]
    OUT_LOG.write_text("\n".join(summary + [""] + log_lines), encoding="utf-8")
    print("\n".join(summary))
    print(f"log={OUT_LOG}")


if __name__ == "__main__":
    main()

