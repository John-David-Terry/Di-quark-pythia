#!/usr/bin/env python3
"""Print P^mu, q^mu, k_out^mu, P_h^mu for a few accepted v3 DIS background events (production conventions)."""

from __future__ import annotations

import argparse
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
)
from generate_dis_isr_parton_dataset import (  # noqa: E402
    extract_beams_from_event,
    hard_subprocess_outgoing_quark_lab_p4_and_index,
    pick_incoming_quark_index,
    try_build_lt_from_event,
)


def _fmt4(v: np.ndarray) -> str:
    e, x, y, z = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
    return f"({e: .6f}, {x: .6f}, {y: .6f}, {z: .6f})"


def _leading_pi_breit(parts: list) -> np.ndarray | None:
    best_e = -1.0
    best = None
    for r in parts:
        if abs(int(r["pdg_id"])) != 211:
            continue
        if float(r["pz"]) <= 0.0:
            continue
        E = float(r["E"])
        if E > best_e:
            best_e = E
            best = np.array(
                [E, float(r["px"]), float(r["py"]), float(r["pz"])],
                dtype=np.float64,
            )
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of accepted events to print")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pythia8

    pythia = build_pythia_background(int(args.seed), hadron_level=True)
    ev = pythia.event
    n_target = int(args.n)
    acc = 0

    print(
        "Conventions (match v3 production):\n"
        "  P^mu   = incoming proton; native lab from extract_beams, then flip_z, then LT -> Breit.\n"
        "  q^mu   = e^-_in - e^-_sc in flip_z frame (same as try_build_lt_from_event), then LT -> Breit.\n"
        "  k_out^mu = outgoing struck quark from pythia.process (lab), flip_z, LT -> Breit.\n"
        "  P_h^mu = leading charged pion (|pdg|=211, pz_Breit>0, max Breit E) from collect_final_state_breit.\n"
        "All four-vectors below are in the DIS Breit-like frame [E, px, py, pz] in GeV.\n"
    )

    while acc < n_target:
        if not pythia.next():
            continue
        inc_idx = pick_incoming_quark_index(ev)
        if inc_idx is None or abs(int(ev[inc_idx].id())) != 2:
            continue
        proc = pythia.process
        p4_lab, oq = hard_subprocess_outgoing_quark_lab_p4_and_index(proc)
        if p4_lab is None or oq < 0:
            continue
        LT = try_build_lt_from_event(ev)
        if LT is None:
            continue
        beams = extract_beams_from_event(ev)
        if beams is None:
            continue
        e_in, e_sc, p_in = beams
        e_in_ev = flip_z(np.asarray(e_in, dtype=np.float64), FLIP_Z_PTREL)
        e_sc_ev = flip_z(np.asarray(e_sc, dtype=np.float64), FLIP_Z_PTREL)
        p_in_ev = flip_z(np.asarray(p_in, dtype=np.float64), FLIP_Z_PTREL)
        qmu = e_in_ev - e_sc_ev

        P_b = LT @ p_in_ev
        q_b = LT @ qmu
        k_b = LT @ flip_z(np.asarray(p4_lab, dtype=np.float64), FLIP_Z_PTREL)

        parts = collect_final_state_breit(ev, LT, acc)
        pih = _leading_pi_breit(parts)

        print(f"--- event_id (accepted index) = {acc} ---")
        print(f"  P^mu   (Breit) = {_fmt4(P_b)}")
        print(f"  q^mu   (Breit) = {_fmt4(q_b)}")
        print(f"  k_out^mu (Breit) = {_fmt4(k_b)}")
        if pih is None:
            print("  P_h^mu (Breit) = (no charged pion with pz_Breit > 0)")
        else:
            print(f"  P_h^mu (Breit) = {_fmt4(pih)}")
        print()
        acc += 1


if __name__ == "__main__":
    main()
