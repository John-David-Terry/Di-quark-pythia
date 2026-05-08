#!/usr/bin/env python3
"""
Export a single internal DIS hard event from PYTHIA8.

This script:
  - sets up e(18 GeV) p(275 GeV) DIS (gm/Z exchange),
  - turns OFF parton-level and hadron-level evolution,
  - generates until it gets one accepted event,
  - reads the full hard process record from `pythia.process`,
  - saves it (plus selected `pythia.info` fields) to JSON:
        outputs/popf_lhe_single_event/hard_process.json
  - prints a human-readable summary of the hard process to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pythia8


import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

OUTDIR = outputs_dir() / "popf_lhe_single_event"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUTDIR / "hard_process.json"


def build_pythia_for_hard_only() -> pythia8.Pythia:
    """Configure PYTHIA8 for DIS hard scattering only (no ISR/FSR/hadronization)."""
    p = pythia8.Pythia()
    # Beams: e- (11) and p (2212) with 18x275 GeV
    p.readString("Beams:idA = 11")
    p.readString("Beams:idB = 2212")
    p.readString("Beams:eA = 18.0")
    p.readString("Beams:eB = 275.0")
    p.readString("Beams:frameType = 2")

    # Neutral-current DIS via gamma/Z exchange
    p.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    p.readString("PDF:lepton = off")

    # Q2 cut
    p.readString("PhaseSpace:Q2Min = 16.0")

    # Turn OFF downstream evolution for this first hard-only dump
    p.readString("PartonLevel:all = off")
    p.readString("HadronLevel:all = off")

    # Keep things deterministic
    p.readString("Random:setSeed = on")
    p.readString("Random:seed = 123456")

    p.init()
    return p


def info_field(info: pythia8.Info, name_candidates: List[str], default: Any = None) -> Any:
    """Try a list of Info methods (no-arg) and return the first that exists; else default."""
    for nm in name_candidates:
        if hasattr(info, nm):
            try:
                return float(getattr(info, nm)())
            except Exception:
                continue
    return default


def export_single_hard_process() -> None:
    pythia = build_pythia_for_hard_only()
    ev = pythia.event
    proc = pythia.process

    n_attempts = 0
    max_attempts = 100000

    print("Generating until one DIS hard event is accepted (hard-only configuration)...")
    while True:
        n_attempts += 1
        if n_attempts > max_attempts:
            raise RuntimeError("Exceeded max_attempts without an accepted event.")
        if not pythia.next():
            continue

        print(f"Accepted event after {n_attempts} attempts. Dumping hard process...")
        break

    # Extract the hard process record from pythia.process
    particles: List[Dict[str, Any]] = []
    for i in range(proc.size()):
        p = proc[i]
        particles.append(
            {
                "index": int(i),
                "id": int(p.id()),
                "status": int(p.status()),
                "mother1": int(p.mother1()),
                "mother2": int(p.mother2()),
                "daughter1": int(p.daughter1()),
                "daughter2": int(p.daughter2()),
                "col": int(p.col()),
                "acol": int(p.acol()),
                "px": float(p.px()),
                "py": float(p.py()),
                "pz": float(p.pz()),
                "e": float(p.e()),
                "m": float(p.m()),
            }
        )

    # Info-level metadata: Python bindings used here do not expose pythia.info,
    # so we leave this as a minimal stub for now.
    info_dict: Dict[str, Any] = {}

    out = {
        "description": "Single DIS hard process exported from PYTHIA8 (pythia.process).",
        "n_attempts": n_attempts,
        "particles": particles,
        "info": info_dict,
    }

    with OUT_JSON.open("w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    # Human-readable table
    print(f"\nHard process written to: {OUT_JSON}\n")
    print("Hard process particle table (pythia.process):")
    header = (
        " idx  id   status  m1  m2  d1  d2   col  acol        px        py        pz         E         m"
    )
    print(header)
    print("-" * len(header))
    for p in particles:
        print(
            f"{p['index']:4d} {p['id']:4d} {p['status']:7d} "
            f"{p['mother1']:3d} {p['mother2']:3d} {p['daughter1']:3d} {p['daughter2']:3d} "
            f"{p['col']:4d} {p['acol']:4d} "
            f"{p['px']:9.3f} {p['py']:9.3f} {p['pz']:9.3f} {p['e']:9.3f} {p['m']:9.3f}"
        )


if __name__ == "__main__":
    export_single_hard_process()

