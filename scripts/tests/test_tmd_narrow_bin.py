#!/usr/bin/env python3
"""
Test TMD-limit behavior in a narrow (x,Q) bin.

Run from project root: python scripts/tests/test_tmd_narrow_bin.py
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None

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

# Narrow kinematic bin
X_LO, X_HI = 0.04, 0.05
Q_LO, Q_HI = 6.0, 8.0


def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / np.maximum(sigma, 1e-9)) ** 2)


def main():
    D_mag_list = []
    D_x_list = []

    for _shard_idx, ie, data in iter_events_from_shards(LABEL, flip_z=FLIP_Z):
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

        # Narrow (x,Q) cut: keep only this bin
        if not (X_LO < x < X_HI) or not (Q_LO < Q < Q_HI):
            continue

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

        # Transverse components (px, py) = indices 1, 2
        px_rem = P_remnant_breit[1]
        py_rem = P_remnant_breit[2]
        px_pi = P_pi_breit[1]
        py_pi = P_pi_breit[2]
        D_vec_x = px_pi - px_rem
        D_vec_y = py_pi - py_rem
        D_mag = float(np.sqrt(D_vec_x ** 2 + D_vec_y ** 2))
        D_x = float(D_vec_x)
        D_mag_list.append(D_mag)
        D_x_list.append(D_x)

    D_mag_arr = np.array(D_mag_list, dtype=float)
    D_x_arr = np.array(D_x_list, dtype=float)
    n = len(D_mag_list)

    # Diagnostics (print first)
    print("--- Diagnostics (narrow bin 0.04 < x < 0.05, 6 < Q < 8) ---")
    print(f"Number of events surviving cut: {n}")
    print(f"Mean(D_mag) = {np.mean(D_mag_arr):.6e}")
    print(f"Std(D_mag)  = {np.std(D_mag_arr):.6e}")
    print(f"Mean(D_x)   = {np.mean(D_x_arr):.6e}")
    print(f"Std(D_x)    = {np.std(D_x_arr):.6e}")

    if n == 0:
        print("No events in bin; nothing to plot.")
        return

    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "stix",
    })
    fontsize = 14

    # (A) Histogram of D_mag
    bins_mag = np.linspace(0.0, 2.0, 50)
    counts_mag, edges_mag = np.histogram(D_mag_arr, bins=bins_mag, density=False)
    centers_mag = 0.5 * (edges_mag[:-1] + edges_mag[1:])

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.step(centers_mag, counts_mag, where="mid", color="steelblue", linewidth=1.5)
    ax1.set_xlabel("Magnitude  $|p_{T,\\pi} - p_{T,{\\rm rem}}|$", fontsize=fontsize)
    ax1.set_ylabel("Counts", fontsize=fontsize)
    ax1.set_title("D_Rpi magnitude (narrow x,Q bin)", fontsize=fontsize)
    ax1.tick_params(direction="in", labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(_PROJECT_ROOT / "test_tmd_D_mag_hist.pdf", format="pdf")
    plt.close(fig1)
    print("Saved: test_tmd_D_mag_hist.pdf")

    # (B) Histogram of D_x and Gaussian fit in |x| < 1.5
    bins_x = np.linspace(-2.0, 2.0, 80)
    counts_x, edges_x = np.histogram(D_x_arr, bins=bins_x, density=False)
    centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])

    fit_mask = (np.abs(centers_x) < 1.5) & (counts_x > 0)
    if np.sum(fit_mask) >= 3 and curve_fit is not None:
        cx = centers_x[fit_mask]
        cy = counts_x[fit_mask].astype(float)
        sigma_y = np.sqrt(np.maximum(cy + 1, 1))
        mu0, sigma0 = float(np.mean(D_x_arr)), float(np.std(D_x_arr))
        A0 = float(np.max(cy))
        try:
            popt, _ = curve_fit(
                gaussian, cx, cy,
                p0=(A0, mu0, max(sigma0, 0.1)),
                sigma=sigma_y,
                absolute_sigma=True,
                maxfev=5000,
                bounds=([0, -5, 0.01], [np.max(cy) * 2, 5, 5]),
            )
            A_fit, mu_fit, sigma_fit = popt[0], popt[1], popt[2]
        except Exception:
            A_fit, mu_fit, sigma_fit = A0, mu0, sigma0
    else:
        A_fit = np.max(counts_x)
        mu_fit = float(np.mean(D_x_arr))
        sigma_fit = float(np.std(D_x_arr)) or 1.0

    x_grid = np.linspace(-2.0, 2.0, 300)
    y_fit = gaussian(x_grid, A_fit, mu_fit, sigma_fit)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.step(centers_x, counts_x, where="mid", label="D_x (counts)", color="coral", linewidth=1.5)
    ax2.plot(x_grid, y_fit, "k--", alpha=0.8, linewidth=2, label="Gaussian fit (|x|<1.5)")
    ax2.set_xlabel("Signed x-component  $(p_{T,\\pi} - p_{T,{\\rm rem}})_x$", fontsize=fontsize)
    ax2.set_ylabel("Counts", fontsize=fontsize)
    ax2.set_title("D_Rpi signed x-component (narrow x,Q bin)", fontsize=fontsize)
    ax2.legend(loc="best", fontsize=fontsize)
    ax2.tick_params(direction="in", labelsize=fontsize)
    ax2.text(0.02, 0.98, f"$\\mu = {mu_fit:.4f}$\n$\\sigma = {sigma_fit:.4f}$",
             transform=ax2.transAxes, fontsize=fontsize, verticalalignment="top")
    plt.tight_layout()
    plt.savefig(_PROJECT_ROOT / "test_tmd_D_x_hist.pdf", format="pdf")
    plt.close(fig2)
    print("Saved: test_tmd_D_x_hist.pdf")


if __name__ == "__main__":
    main()
