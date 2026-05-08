#!/usr/bin/env python3
"""
Sample N accepted DIS events (same cuts as ``generate_dis_background_final_state_parquet``)
and report transverse momentum of the **struck outgoing quark** in the **Breit frame**
(``flip_z`` + ``try_build_lt_from_event``), using ``identify_struck_outgoing_quark_index``.

Shows whether post-hadronization events still have a resolvable final struck quark with
non-zero |pT| in Breit.
"""

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

from generate_dis_background_final_state_parquet import build_pythia_background  # noqa: E402
from generate_dis_isr_parton_dataset import (  # noqa: E402
    identify_struck_outgoing_quark_index,
    pick_incoming_quark_index,
    try_build_lt_from_event,
)


def build_pythia_background_parton_only(seed: int):
    """Same as background chain but **no hadronization** — final quarks remain in the record."""
    import pythia8

    p = pythia8.Pythia()
    p.readString("Beams:idA = 11")
    p.readString("Beams:idB = 2212")
    p.readString("Beams:eA = 18.0")
    p.readString("Beams:eB = 275.0")
    p.readString("Beams:frameType = 2")
    p.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    p.readString("PhaseSpace:Q2Min = 16.0")
    p.readString("ProcessLevel:all = on")
    p.readString("PDF:lepton = off")
    p.readString("PartonLevel:ISR = on")
    p.readString("PartonLevel:FSR = on")
    p.readString("PartonLevel:MPI = off")
    p.readString("PartonLevel:Remnants = on")
    p.readString("HadronLevel:all = off")
    p.readString(f"Random:seed = {int(seed)}")
    p.readString("Random:setSeed = on")
    p.readString("Print:quiet = on")
    if not p.init():
        raise RuntimeError("PYTHIA init failed (parton-only)")
    return p


def p4_lab_from_pythia_particle(p) -> np.ndarray:
    return np.array([float(p.e()), float(p.px()), float(p.py()), float(p.pz())], dtype=np.float64)


def _run_block(
    label: str,
    p,
    n_target: int,
) -> None:
    ev = p.event
    pt_breit_list: list[float] = []
    unresolved = 0
    gen = 0

    while len(pt_breit_list) < n_target:
        if not p.next():
            continue
        gen += 1
        inc_idx = pick_incoming_quark_index(ev)
        if inc_idx is None:
            continue
        if abs(int(ev[inc_idx].id())) != 2:
            continue
        LT = try_build_lt_from_event(ev)
        if LT is None:
            continue

        k_idx = identify_struck_outgoing_quark_index(ev, inc_idx)
        if k_idx < 0:
            unresolved += 1
            pt_breit_list.append(float("nan"))
            continue
        kp = ev[k_idx]
        p4 = flip_z(p4_lab_from_pythia_particle(kp), FLIP_Z_PTREL)
        p4b = LT @ p4
        pt = float(np.hypot(float(p4b[1]), float(p4b[2])))
        pt_breit_list.append(pt)

    arr = np.asarray(pt_breit_list, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    n_nan = int(np.sum(~np.isfinite(arr)))

    print(f"=== {label} ===")
    print(f"  PYTHIA tries to fill sample={n_target}: {gen}")
    print(f"  identify_struck_outgoing unresolved (-1): {unresolved}/{n_target}")
    if finite.size:
        print(
            f"  |pT|_Breit (GeV), resolved only: min={finite.min():.6g}  max={finite.max():.6g}  "
            f"median={np.median(finite):.6g}  mean={finite.mean():.6g}"
        )
        for eps in (1e-12, 1e-9, 1e-6, 1e-3):
            print(f"    frac |pT| <= {eps:g}: {np.mean(finite <= eps):.4f}")
    else:
        print("  (no resolved struck quarks — no |pT| stats)")
    print(f"  NaN slots in sample list: {n_nan}/{len(arr)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument(
        "--parton-only",
        action="store_true",
        help="Only run HadronLevel:off chain (skip post-hadron block).",
    )
    ap.add_argument(
        "--posthadron-only",
        action="store_true",
        help="Only run full hadronization chain.",
    )
    args = ap.parse_args()

    try:
        import pythia8  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(f"pythia8 required: {exc}") from exc

    n = int(args.n_sample)
    seed = int(args.seed)

    do_post = not args.parton_only
    do_parton = not args.posthadron_only
    if args.parton_only and args.posthadron_only:
        raise SystemExit("choose at most one of --parton-only / --posthadron-only")

    if do_post:
        _run_block("Post-hadron (HadronLevel:on, same as background Parquet)", build_pythia_background(seed), n)
    if do_parton:
        _run_block("Parton-level (HadronLevel:off)", build_pythia_background_parton_only(seed), n)


if __name__ == "__main__":
    main()
