#!/usr/bin/env python3
"""
Benchmark: Breit ↔ lab inverse Lorentz chain for DIS (same LT as analyze_events_raw.build_LT).

- Generates events with PYTHIA (parton-level, no hadronization) matching editable-source settings.
- For each event with a valid lab→Breit matrix LT (Mm1@…@M4), checks:
    inv(LT) @ (LT @ p_flipped) == p_flipped  (round-trip on flipped incoming proton 4-vector).
- Compares numpy inverse to the factorized tail inv(M0)@inv(M1)@…@inv(M4).
- Optionally compares to explicit Mathematica-style factors LT0m…LT4m (inverse factors only; Mm1 separate).

Convention: four-vectors as column arrays [E, px, py, pz] with metric signature (+,-,-,-) in dot4.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
_ANAL = Path(__file__).resolve().parent
for _p in (_SRC, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diquark.analyze_events_raw import FLIP_Z_PTREL, flip_z  # noqa: E402

import generate_dis_isr_parton_dataset as _gen  # noqa: E402
from generate_dis_editable_source_parquet import build_pythia_source  # noqa: E402

try:
    import pythia8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pythia8 import failed: {exc}") from exc


def lt_factors(
    Ee: float, Ep: float, qmu: np.ndarray, x: float, y: float, qT: float, phiq: float, S: float
) -> Optional[Tuple[np.ndarray, ...]]:
    """Return (Mm1, M0, M1, M2, M3, M4) — same as inside build_LT before the final product."""
    Mm1 = np.array(
        [
            [Ee / np.sqrt(S) + np.sqrt(S) / (4.0 * Ee), 0, 0, Ee / np.sqrt(S) - np.sqrt(S) / (4.0 * Ee)],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [Ee / np.sqrt(S) - np.sqrt(S) / (4.0 * Ee), 0, 0, Ee / np.sqrt(S) + np.sqrt(S) / (4.0 * Ee)],
        ]
    )
    M0 = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(phiq), np.sin(phiq), 0],
            [0, -np.sin(phiq), np.cos(phiq), 0],
            [0, 0, 0, 1],
        ]
    )
    den2 = -qT * qT + S * (1 + x) * y
    if den2 <= 0:
        return None
    denom_M1 = 2.0 * y * np.sqrt(S * (-qT * qT + S * (1 + x) * y))
    if denom_M1 == 0:
        return None
    M1 = np.array(
        [
            [(-qT * qT + S * y * (1 + x + y)) / denom_M1, 0, 0, (qT * qT + S * y * (-x + y - 1)) / denom_M1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [(qT * qT + S * y * (-x + y - 1)) / denom_M1, 0, 0, (-qT * qT + S * y * (1 + x + y)) / denom_M1],
        ]
    )
    denom_M2_s1 = np.sqrt(S * (1 + x) * y / den2)
    denom_M2_s2 = np.sqrt(S * (1 + x) * y)
    M2 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1 / denom_M2_s1, 0, qT / denom_M2_s2],
            [0, 0, 1, 0],
            [0, -qT / denom_M2_s2, 0, 1 / denom_M2_s1],
        ]
    )
    num_log = qT + np.sqrt(S * (1 + x) * y)
    den_log = np.sqrt(S * (1 + x) * y) - qT
    if num_log <= 0 or den_log <= 0:
        return None
    eta_m3 = 0.5 * np.log(num_log / den_log)
    denom_M3 = np.sqrt(den2)
    M3 = np.array(
        [
            [np.cosh(eta_m3), -qT / denom_M3, 0, 0],
            [-qT / denom_M3, np.cosh(eta_m3), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    denom_M4 = 2 * np.sqrt(x * (1 + x))
    if denom_M4 == 0:
        return None
    M4 = np.array(
        [
            [(1 + 2 * x) / denom_M4, 0, 0, 1 / denom_M4],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1 / denom_M4, 0, 0, (1 + 2 * x) / denom_M4],
        ]
    )
    return (Mm1, M0, M1, M2, M3, M4)


def explicit_lt0m_lt4m_product(
    x: float, y: float, qT: float, S: float, phi: float
) -> Optional[np.ndarray]:
    """
    User-provided inverse tail LT0m @ LT1m @ LT2m @ LT3m @ LT4m (Breit-related frame → …),
    transcribed from Mathematica. Does NOT include Mm1^-1.
    """
    den2 = -qT * qT + S * (1 + x) * y
    if den2 <= 0:
        return None
    s_root = np.sqrt(S * (1 + x) * y)
    denom_M1 = 2.0 * y * np.sqrt(S * den2)
    if denom_M1 <= 0:
        return None

    LT0m = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ]
    )
    LT1m = np.array(
        [
            [
                (-qT * qT + S * y * (1 + x + y)) / denom_M1,
                0,
                0,
                (-qT * qT + S * (1 + x - y) * y) / denom_M1,
            ],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [
                (-qT * qT + S * (1 + x - y) * y) / denom_M1,
                0,
                0,
                (-qT * qT + S * y * (1 + x + y)) / denom_M1,
            ],
        ]
    )
    r = np.sqrt(S * (1 + x) * y)
    if r <= 0:
        return None
    LT2m = np.array(
        [
            [1, 0, 0, 0],
            [0, 1 / np.sqrt(S * (1 + x) * y / den2), 0, -(qT / r)],
            [0, 0, 1, 0],
            [0, qT / r, 0, 1 / np.sqrt(S * (1 + x) * y / den2)],
        ]
    )
    num_log = qT + s_root
    den_log = s_root - qT
    if num_log <= 0 or den_log <= 0:
        return None
    ch = np.cosh(0.5 * np.log(num_log / den_log))
    den_sq = np.sqrt(-qT * qT + S * (1 + x) * y)
    if den_sq <= 0:
        return None
    dinv = qT * qT / (qT * qT - S * (1 + x) * y) + ch * ch
    if abs(dinv) < 1e-18:
        return None
    LT3m = np.array(
        [
            [ch / dinv, qT / (den_sq * dinv), 0, 0],
            [qT / (den_sq * dinv), ch / dinv, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    den_m4 = 2 * np.sqrt(x * (1 + x))
    if den_m4 <= 0:
        return None
    LT4m = np.array(
        [
            [(1 + 2 * x) / den_m4, 0, 0, -(1 / den_m4)],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-(1 / den_m4), 0, 0, (1 + 2 * x) / den_m4],
        ]
    )
    return LT0m @ LT1m @ LT2m @ LT3m @ LT4m


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark Breit inverse LT vs factorization.")
    ap.add_argument("--n-events", type=int, default=1000, help="Target accepted kinematics with valid LT.")
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--beam-e", type=float, default=5.0)
    ap.add_argument("--beam-p", type=float, default=41.0)
    ap.add_argument("--phase-space-q2-min", type=float, default=4.0)
    args = ap.parse_args()

    pythia = build_pythia_source(
        int(args.seed),
        e_beam=float(args.beam_e),
        p_beam=float(args.beam_p),
        phase_space_q2_min=float(args.phase_space_q2_min),
    )
    ev = pythia.event

    n_target = int(args.n_events)
    round_trip_errs: List[float] = []
    inv_compare_errs: List[float] = []
    tail_compare_errs: List[float] = []
    user_tail_errs: List[float] = []
    ratio_e_over_pz: List[float] = []

    n_gen = 0
    while len(round_trip_errs) < n_target:
        if not pythia.next():
            continue
        n_gen += 1
        LT = _gen.try_build_lt_from_event(ev)
        if LT is None:
            continue
        beams = _gen.extract_beams_from_event(ev)
        if beams is None:
            continue
        _e_in, _e_sc, p_in = beams
        p_in_ev = flip_z(np.asarray(p_in, dtype=np.float64), FLIP_Z_PTREL)
        p_b = LT @ p_in_ev
        inv_LT = np.linalg.inv(LT)
        p_back = inv_LT @ p_b
        round_trip_errs.append(float(np.max(np.abs(p_back - p_in_ev))))

        e_in_evf = flip_z(np.asarray(_e_in), FLIP_Z_PTREL)
        e_sc_evf = flip_z(np.asarray(_e_sc), FLIP_Z_PTREL)
        qmu = e_in_evf - e_sc_evf
        q0, q1, q2, q3 = float(qmu[0]), float(qmu[1]), float(qmu[2]), float(qmu[3])
        Q2 = -(q0 * q0 - q1 * q1 - q2 * q2 - q3 * q3)
        if Q2 <= 0:
            continue
        qT = math.hypot(q1, q2)
        p_dot_q = (
            float(p_in_ev[0]) * q0
            - float(p_in_ev[1]) * q1
            - float(p_in_ev[2]) * q2
            - float(p_in_ev[3]) * q3
        )
        if p_dot_q == 0:
            continue
        x = Q2 / (2.0 * p_dot_q)
        Ee = float(e_in_evf[0])
        Ep = float(p_in_ev[0])
        S = 4.0 * Ee * Ep
        yb = Q2 / (S * x) if S * x > 0 else 0.0
        phiq = math.atan2(q2, q1)
        fac = lt_factors(Ee, Ep, np.array([q0, q1, q2, q3]), x, yb, qT, phiq, S)
        if fac is None:
            continue
        Mm1, M0, M1, M2, M3, M4 = fac
        tail = np.linalg.inv(M0) @ np.linalg.inv(M1) @ np.linalg.inv(M2) @ np.linalg.inv(M3) @ np.linalg.inv(M4)
        inv_full = np.linalg.inv(Mm1) @ tail
        inv_compare_errs.append(float(np.max(np.abs(inv_full - inv_LT))))
        tail_compare_errs.append(float(np.max(np.abs(tail - np.linalg.inv(M4 @ M3 @ M2 @ M1 @ M0)))))

        ut = explicit_lt0m_lt4m_product(x, yb, qT, S, phiq)
        if ut is not None:
            user_tail_errs.append(float(np.max(np.abs(tail - ut))))

        if abs(p_b[3]) > 1e-9:
            ratio_e_over_pz.append(float(p_b[0] / p_b[3]))

    rep = {
        "n_gen": n_gen,
        "n_valid": len(round_trip_errs),
        "beam_e": float(args.beam_e),
        "beam_p": float(args.beam_p),
        "round_trip_max_abs_err": max(round_trip_errs) if round_trip_errs else None,
        "round_trip_mean_abs_err": float(np.mean(round_trip_errs)) if round_trip_errs else None,
        "inv_full_vs_np_inv_LT_max": max(inv_compare_errs) if inv_compare_errs else None,
        "tail_vs_inv_M4M3M2M1M0_max": max(tail_compare_errs) if tail_compare_errs else None,
        "tail_vs_user_LT0_LT4_max": max(user_tail_errs) if user_tail_errs else None,
        "proton_breit_E_over_pz_mean": float(np.mean(ratio_e_over_pz)) if ratio_e_over_pz else None,
        "proton_breit_E_over_pz_std": float(np.std(ratio_e_over_pz)) if ratio_e_over_pz else None,
        "note": "Full Breit→collider for proton uses inv(LT) @ p_b then undo flip_z if storing flip_z(p_lab) in LT path.",
    }
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
