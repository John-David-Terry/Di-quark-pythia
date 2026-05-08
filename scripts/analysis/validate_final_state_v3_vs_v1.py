#!/usr/bin/env python3
"""
Compare one-step hadronized DIS background (v3 / same as v1 hadronization) vs broken two-step.

Metrics: mean final multiplicity, charged-pion fractions, target-hemisphere π± (same selector
as jet–hadron: pz_breit > 0, pt_breit >= FLIP_Z_PTREL), k_out extraction from pythia.process.

Usage (from repo root)::

  python scripts/analysis/validate_final_state_v3_vs_v1.py --n 2000
  python scripts/analysis/validate_final_state_v3_vs_v1.py --n 500 --seed 2 --broken-two-step
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
_ANAL = Path(__file__).resolve().parent
for _p in (_SRC, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diquark.analyze_events_raw import FLIP_Z_PTREL, flip_z  # noqa: E402

from generate_dis_background_final_state_parquet import (  # noqa: E402
    build_pythia_background,
    collect_final_state_breit,
    hard_subprocess_outgoing_quark_lab_p4_and_index,
    pick_incoming_quark_index,
)
from generate_dis_isr_parton_dataset import try_build_lt_from_event  # noqa: E402


def _target_hemisphere_charged_pion_fraction(rows: list) -> float:
    if not rows:
        return 0.0
    n = 0
    for r in rows:
        ok = False
        for p in r["particles"]:
            if int(p["pdg_id"]) not in (211, -211):
                continue
            if float(p["pz"]) <= 0.0:
                continue
            pt = math.hypot(float(p["px"]), float(p["py"]))
            if pt < FLIP_Z_PTREL:
                continue
            ok = True
            break
        if ok:
            n += 1
    return n / len(rows)


def _any_charged_pion_fraction(rows: list) -> float:
    if not rows:
        return 0.0
    n = 0
    for r in rows:
        if any(int(p["pdg_id"]) in (211, -211) for p in r["particles"]):
            n += 1
    return n / len(rows)


def _mean_multiplicity(rows: list) -> float:
    if not rows:
        return 0.0
    return float(np.mean([len(r["particles"]) for r in rows]))


def run_one_step(n: int, seed: int, *, label: str) -> dict:
    pythia = build_pythia_background(int(seed), hadron_level=True)
    ev = pythia.event
    rows: list = []
    miss = 0
    tries = 0
    while len(rows) < n and tries < n * 80:
        tries += 1
        if not pythia.next():
            continue
        inc_idx = pick_incoming_quark_index(ev)
        if inc_idx is None:
            continue
        if abs(int(ev[inc_idx].id())) != 2:
            continue

        proc = pythia.process
        p4_lab, _oq = hard_subprocess_outgoing_quark_lab_p4_and_index(proc)
        if p4_lab is None:
            miss += 1
            continue

        LT = try_build_lt_from_event(ev)
        if LT is None:
            continue

        _ = LT @ flip_z(p4_lab, FLIP_Z_PTREL)
        eid = len(rows)
        parts = collect_final_state_breit(ev, LT, eid)
        rows.append({"particles": parts})

    return {
        "label": label,
        "accepted": len(rows),
        "tries": tries,
        "hard_subprocess_miss": miss,
        "mean_n_final": _mean_multiplicity(rows),
        "frac_any_pion": _any_charged_pion_fraction(rows),
        "frac_target_pion": _target_hemisphere_charged_pion_fraction(rows),
        "k_out_found_rate": 1.0 - (miss / tries) if tries else 0.0,
    }


def run_broken_two_step(n: int, seed: int, *, label: str) -> dict:
    """Old v2-style path: hadron off, next, read process, forceHadronLevel, next (under-hadronizes)."""
    pythia = build_pythia_background(int(seed), hadron_level=False)
    ev = pythia.event
    rows: list = []
    miss = 0
    tries = 0
    while len(rows) < n and tries < n * 80:
        tries += 1
        if not pythia.next():
            continue
        inc_idx = pick_incoming_quark_index(ev)
        if inc_idx is None:
            continue
        if abs(int(ev[inc_idx].id())) != 2:
            continue

        proc = pythia.process
        p4_lab, _oq = hard_subprocess_outgoing_quark_lab_p4_and_index(proc)
        if p4_lab is None:
            miss += 1
            continue

        pythia.forceHadronLevel(True)
        if not pythia.next():
            pythia.forceHadronLevel(False)
            continue
        pythia.forceHadronLevel(False)

        LT = try_build_lt_from_event(ev)
        if LT is None:
            continue

        eid = len(rows)
        parts = collect_final_state_breit(ev, LT, eid)
        rows.append({"particles": parts})

    return {
        "label": label,
        "accepted": len(rows),
        "tries": tries,
        "hard_subprocess_miss": miss,
        "mean_n_final": _mean_multiplicity(rows),
        "frac_any_pion": _any_charged_pion_fraction(rows),
        "frac_target_pion": _target_hemisphere_charged_pion_fraction(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--broken-two-step",
        action="store_true",
        help="Also run the old v2-style broken path for contrast (fewer final particles).",
    )
    args = ap.parse_args()

    r_ok = run_one_step(args.n, args.seed, label="one_step_hadron_on (v3)")
    print(r_ok)
    if args.broken_two_step:
        r_b = run_broken_two_step(
            min(args.n, 2000),
            args.seed + 10_000,
            label="broken_two_step (old v2)",
        )
        print(r_b)


if __name__ == "__main__":
    main()
