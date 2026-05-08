#!/usr/bin/env python3
"""
Minimal direct PYTHIA->PYTHIA reinjection test for baryonic junction topology.

Scope intentionally minimal:
  - no LHE
  - no hard-process generation
  - hand-built 3q + explicit junction only
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
OUT_LOG = OUTDIR / "minimal_direct_reinjection_junction_qqq_log.txt"


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


def quark_mass(pid: int) -> float:
    apid = abs(pid)
    if apid in {1, 2, 3}:
        return 0.33
    return 0.33


def event_listing_text(ev: "pythia8.Event", title: str) -> str:
    lines = [f"--- {title} ---"]
    lines.append("idx id status m1 m2 d1 d2 col acol px py pz E m isFinal isHadron")
    for i in range(ev.size()):
        pp = ev[i]
        lines.append(
            f"{i:3d} {int(pp.id()):4d} {int(pp.status()):4d} "
            f"{int(pp.mother1()):3d} {int(pp.mother2()):3d} "
            f"{int(pp.daughter1()):3d} {int(pp.daughter2()):3d} "
            f"{int(pp.col()):3d} {int(pp.acol()):3d} "
            f"{float(pp.px()): .6f} {float(pp.py()): .6f} {float(pp.pz()): .6f} "
            f"{float(pp.e()): .6f} {float(pp.m()): .6f} "
            f"{int(pp.isFinal())} {int(pp.isHadron())}"
        )
    return "\n".join(lines)


def junction_listing_text(ev: "pythia8.Event", title: str) -> str:
    lines = [f"--- {title} ---"]
    nj = int(ev.sizeJunction())
    lines.append(f"n_junctions={nj}")
    for j in range(nj):
        kind = int(ev.kindJunction(j))
        c0 = int(ev.colJunction(j, 0))
        c1 = int(ev.colJunction(j, 1))
        c2 = int(ev.colJunction(j, 2))
        s0 = int(ev.statusJunction(j, 0))
        s1 = int(ev.statusJunction(j, 1))
        s2 = int(ev.statusJunction(j, 2))
        lines.append(
            f"junction[{j}] kind={kind} cols=({c0},{c1},{c2}) "
            f"status=({s0},{s1},{s2})"
        )
    return "\n".join(lines)


def count_final_hadrons(ev: "pythia8.Event") -> Tuple[int, int]:
    n_final, n_had = 0, 0
    for i in range(ev.size()):
        pp = ev[i]
        if pp.isFinal():
            n_final += 1
            if pp.isHadron():
                n_had += 1
    return n_final, n_had


def build_three_quarks_with_junction(
    p: "pythia8.Pythia",
    pids: Sequence[int],
    status: int,
    scale: float,
    add_parent_placeholder: bool,
    junction_kind: int,
) -> Dict[str, object]:
    # Symmetric non-collinear 3-body geometry with p_total = 0.
    base = [
        (+3.0, 0.0, +4.0),
        (-3.0, 0.0, +4.0),
        (0.0, 0.0, -8.0),
    ]
    assert len(pids) == 3
    cols = [101, 102, 103]

    p.event.reset()

    parent_idx = -1
    if add_parent_placeholder:
        e_parent = 100.0
        # Optional technical placeholder only.
        parent_idx = int(p.event.append(2212, -21, 0, 0, 0.0, 0.0, 0.0, e_parent, 0.938))

    quark_indices: List[int] = []
    for i, pid in enumerate(pids):
        px, py, pz = base[i]
        px *= scale
        py *= scale
        pz *= scale
        m = quark_mass(pid)
        E = math.sqrt(px * px + py * py + pz * pz + m * m)

        if add_parent_placeholder and parent_idx >= 0:
            idx = int(
                p.event.append(
                    int(pid),
                    int(status),
                    parent_idx,
                    parent_idx,
                    0,
                    0,
                    int(cols[i]),
                    0,
                    float(px),
                    float(py),
                    float(pz),
                    float(E),
                    float(m),
                )
            )
        else:
            idx = int(
                p.event.append(
                    int(pid),
                    int(status),
                    int(cols[i]),
                    0,
                    float(px),
                    float(py),
                    float(pz),
                    float(E),
                    float(m),
                )
            )
        quark_indices.append(idx)

    # From BeamParticle.cc usage: kind=1 for three colour legs (junction),
    # kind=2 for three anticolour legs (anti-junction). Also test kind=3
    # as a possible alternate encoding for colour-side topology.
    jidx = int(p.event.appendJunction(junction_kind, cols[0], cols[1], cols[2]))
    return {
        "quark_indices": quark_indices,
        "junction_index": jidx,
        "junction_kind": int(junction_kind),
        "junction_cols": tuple(cols),
    }


def run_case(
    pids: Sequence[int],
    status: int,
    scale: float,
    add_parent_placeholder: bool,
    junction_kind: int,
) -> Dict[str, object]:
    p = make_pythia()
    topology_info = build_three_quarks_with_junction(
        p, pids, status, scale, add_parent_placeholder, junction_kind
    )

    before_evt = event_listing_text(
        p.event,
        f"before hadronization pids={list(pids)} status={status} scale={scale} parent={add_parent_placeholder} jkind={junction_kind}",
    )
    before_jun = junction_listing_text(
        p.event,
        f"junctions before hadronization pids={list(pids)} status={status} scale={scale} parent={add_parent_placeholder} jkind={junction_kind}",
    )

    next_ok = bool(p.next())
    after_evt = (
        event_listing_text(
            p.event,
            f"after hadronization pids={list(pids)} status={status} scale={scale} parent={add_parent_placeholder} jkind={junction_kind}",
        )
        if next_ok
        else ""
    )
    after_jun = (
        junction_listing_text(
            p.event,
            f"junctions after hadronization pids={list(pids)} status={status} scale={scale} parent={add_parent_placeholder} jkind={junction_kind}",
        )
        if next_ok
        else ""
    )
    n_final, n_had = count_final_hadrons(p.event) if next_ok else (0, 0)

    return {
        "pids": list(pids),
        "status": int(status),
        "scale": float(scale),
        "add_parent_placeholder": bool(add_parent_placeholder),
        "junction_kind": int(junction_kind),
        "next_ok": bool(next_ok),
        "n_final": int(n_final),
        "n_had": int(n_had),
        "before_event_listing": before_evt,
        "before_junction_listing": before_jun,
        "after_event_listing": after_evt,
        "after_junction_listing": after_jun,
        "topology_info": topology_info,
    }


def main() -> None:
    # Minimal systematic scan if first trial fails:
    # vary one axis at a time (status/flavor/scale/placeholder).
    pid_sets = [
        [2, 2, 1],  # uud
        [2, 1, 1],  # udd
    ]
    statuses = [23, 62]
    scales = [1.0, 2.0]  # 1x and 2x of base geometry
    parent_options = [False, True]
    junction_kinds = [1, 3]

    results: List[Dict[str, object]] = []
    log_lines: List[str] = []

    log_lines.append("Junction API pattern used:")
    log_lines.append("- Three quarks are appended as ordinary colored partons.")
    log_lines.append("- One explicit junction is added with event.appendJunction(1,c1,c2,c3).")
    log_lines.append("- kind=1 corresponds to three colour legs (baryonic junction).")
    log_lines.append("- kind=2 would represent an anti-junction (three anticolours).")
    log_lines.append("")

    for pids in pid_sets:
        for status in statuses:
            for scale in scales:
                for use_parent in parent_options:
                    for jkind in junction_kinds:
                        try:
                            res = run_case(pids, status, scale, use_parent, jkind)
                        except Exception as exc:
                            res = {
                                "pids": list(pids),
                                "status": int(status),
                                "scale": float(scale),
                                "add_parent_placeholder": bool(use_parent),
                                "junction_kind": int(jkind),
                                "next_ok": False,
                                "n_final": 0,
                                "n_had": 0,
                                "error": str(exc),
                            }
                        results.append(res)

                        hdr = (
                            f"case pids={res['pids']} status={res['status']} scale={res['scale']} "
                            f"parent={res['add_parent_placeholder']} jkind={res['junction_kind']} "
                            f"next_ok={res['next_ok']} n_final={res['n_final']} n_had={res['n_had']}"
                        )
                        log_lines.append(hdr)
                        if "topology_info" in res:
                            log_lines.append(f"topology_info={res['topology_info']}")
                        if "error" in res:
                            log_lines.append(f"error: {res['error']}")
                        if res.get("before_junction_listing"):
                            log_lines.append(str(res["before_junction_listing"]))
                        if res.get("before_event_listing"):
                            log_lines.append(str(res["before_event_listing"]))
                        if res.get("after_junction_listing"):
                            log_lines.append(str(res["after_junction_listing"]))
                        if res.get("after_event_listing"):
                            log_lines.append(str(res["after_event_listing"]))
                        log_lines.append("")

    n_cases = len(results)
    n_next_ok = sum(1 for r in results if bool(r["next_ok"]))
    n_had_ok = sum(1 for r in results if bool(r["next_ok"]) and int(r["n_had"]) > 0)
    summary = [
        "Minimal qqq+junction direct reinjection summary",
        f"total_cases={n_cases}",
        f"next_success_cases={n_next_ok}",
        f"hadronizing_success_cases={n_had_ok}",
    ]

    OUT_LOG.write_text("\n".join(summary + [""] + log_lines), encoding="utf-8")
    print("\n".join(summary))
    print(f"log={OUT_LOG}")


if __name__ == "__main__":
    main()

