#!/usr/bin/env python3
"""
Compute the four transverse observables from cached shards (ISRFSR_ON only).
Uses the same validated pipeline as analyze_events_raw. No plots.

Run from project root: python scripts/analysis/compute_transverse_observables.py
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from diquark.analyze_events_raw import (
    Qmax_ptrel,
    Qmin_ptrel,
    build_LT,
    dot4,
    flip_z,
    is_hadron,
    p3,
    xmax_ptrel,
    xmin_ptrel,
)
from diquark.cached_shards import iter_events_from_shards

LABEL = "ISRFSR_ON"
FLIP_Z = True


def main():
    S_Rpi_list = []
    D_Rpi_list = []
    S_Jpi_list = []
    D_Jpi_list = []
    n_printed = 0

    for shard_idx, ie, data in iter_events_from_shards(LABEL, flip_z=FLIP_Z):
        e_in_ev = data["event_e_in"]
        p_in_ev = data["event_p_in"]
        e_sc_ev = data["event_e_sc"]
        k_out_ev = data["event_k_out"]
        offsets = data["offsets"]
        pid = data["pid"]
        p4_arr = data["p4"]

        Ep = float(p_in_ev[0])
        Ee = float(e_in_ev[0])
        q0 = e_in_ev[0] - e_sc_ev[0]
        q1 = e_in_ev[1] - e_sc_ev[1]
        q2 = e_in_ev[2] - e_sc_ev[2]
        q3 = e_in_ev[3] - e_sc_ev[3]
        qmu = np.array([q0, q1, q2, q3], dtype=float)
        Q2 = -(q0 * q0 - q1 * q1 - q2 * q2 - q3 * q3)
        if Q2 <= 0:
            continue
        Q = float(np.sqrt(Q2))
        qT = float(np.hypot(q1, q2))
        p_dot_q = p_in_ev[0] * q0 - p_in_ev[1] * q1 - p_in_ev[2] * q2 - p_in_ev[3] * q3
        if p_dot_q == 0:
            continue
        x = Q2 / (2.0 * p_dot_q)
        if not (xmin_ptrel <= x <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel):
            continue
        phiq = float(np.arctan2(q2, q1))
        S = 4.0 * Ee * Ep
        y = Q2 / (S * x)
        LT = build_LT(Ee, Ep, qmu, x, y, qT, phiq, S)
        if LT is None:
            continue
        boost = lambda v: LT @ np.asarray(v, dtype=float)
        P_proton_breit = boost(p_in_ev)
        P_plus = float(P_proton_breit[0] + P_proton_breit[3])
        if P_plus <= 0:
            continue

        start = int(offsets[ie])
        end = int(offsets[ie + 1])
        best_tarE = -1.0
        P_pi_breit = None
        best_tar_pid = None
        for j in range(start, end):
            this_pid = int(pid[j])
            if not is_hadron(this_pid):
                continue
            lab = flip_z(np.asarray(p4_arr[j], dtype=float), FLIP_Z)
            trf = boost(lab)
            E_, px_, py_, pz_ = trf
            if pz_ <= 0:
                continue
            if E_ > best_tarE:
                best_tarE = E_
                P_pi_breit = trf
                best_tar_pid = this_pid
        if P_pi_breit is None or abs(best_tar_pid) != 211:
            continue

        k_in = k_out_ev - qmu
        p_rem_truth = np.asarray(p_in_ev, dtype=float) - k_in
        P_remnant_breit = boost(p_rem_truth)
        if np.linalg.norm(p3(P_remnant_breit)) <= 0:
            continue
        q_breit = boost(qmu)
        den = dot4(P_remnant_breit, q_breit)
        if den <= 0:
            continue
        xL_exact = dot4(P_pi_breit, q_breit) / den
        if xL_exact < 0.01 or xL_exact > 1.0 + 1e-6:
            continue

        P_jet_breit = np.asarray(P_proton_breit, dtype=float) - P_remnant_breit

        # Transverse components (px, py) = indices 1, 2
        pT_rem = np.array([P_remnant_breit[1], P_remnant_breit[2]], dtype=float)
        pT_jet = np.array([P_jet_breit[1], P_jet_breit[2]], dtype=float)
        pT_pi = np.array([P_pi_breit[1], P_pi_breit[2]], dtype=float)

        S_Rpi = float(np.linalg.norm(pT_rem + pT_pi))
        D_Rpi = float(np.linalg.norm(pT_rem - pT_pi))
        S_Jpi = float(np.linalg.norm(pT_jet + pT_pi))
        D_Jpi = float(np.linalg.norm(pT_jet - pT_pi))

        if len(S_Rpi_list) < 50:
            check = pT_rem + pT_jet
            print(f"event {len(S_Rpi_list)}: pT_rem + pT_jet = ({check[0]:.6e}, {check[1]:.6e})")
            n_printed += 1

        S_Rpi_list.append(S_Rpi)
        D_Rpi_list.append(D_Rpi)
        S_Jpi_list.append(S_Jpi)
        D_Jpi_list.append(D_Jpi)

    S_Rpi_arr = np.asarray(S_Rpi_list, dtype=np.float64)
    D_Rpi_arr = np.asarray(D_Rpi_list, dtype=np.float64)
    S_Jpi_arr = np.asarray(S_Jpi_list, dtype=np.float64)
    D_Jpi_arr = np.asarray(D_Jpi_list, dtype=np.float64)

    np.save(_PROJECT_ROOT / "S_Rpi_ISRFSR_ON.npy", S_Rpi_arr)
    np.save(_PROJECT_ROOT / "D_Rpi_ISRFSR_ON.npy", D_Rpi_arr)
    np.save(_PROJECT_ROOT / "S_Jpi_ISRFSR_ON.npy", S_Jpi_arr)
    np.save(_PROJECT_ROOT / "D_Jpi_ISRFSR_ON.npy", D_Jpi_arr)

    n = len(S_Rpi_list)
    print("\n--- Summary ---")
    print(f"Number of processed events: {n}")

    for name, arr in [
        ("S_Rpi", S_Rpi_arr),
        ("D_Rpi", D_Rpi_arr),
        ("S_Jpi", S_Jpi_arr),
        ("D_Jpi", D_Jpi_arr),
    ]:
        print(f"  {name}: mean = {np.mean(arr):.6e}, std = {np.std(arr):.6e}")

    print("\nSaved: S_Rpi_ISRFSR_ON.npy, D_Rpi_ISRFSR_ON.npy, S_Jpi_ISRFSR_ON.npy, D_Jpi_ISRFSR_ON.npy")


if __name__ == "__main__":
    main()
