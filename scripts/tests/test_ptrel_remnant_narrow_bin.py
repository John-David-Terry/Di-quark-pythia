#!/usr/bin/env python3
"""
Compute pT of hadrons with respect to the remnant transverse axis
in a narrow (x,Q) bin, using cached shards (ISRFSR_ON, FLIP_Z=True).

Run from project root: python scripts/tests/test_ptrel_remnant_narrow_bin.py
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

# Narrow TMD-like bin
X_LO, X_HI = 0.04, 0.05
Q_LO, Q_HI = 6.0, 8.0


def gaussian(x, A, mu, sigma):
    sigma_safe = np.maximum(sigma, 1e-9)
    return A * np.exp(-0.5 * ((x - mu) / sigma_safe) ** 2)


def main():
    total_events = 0
    in_bin_events = 0
    axis_skipped = 0

    pTrel_list = []

    for _shard_idx, ie, data in iter_events_from_shards(LABEL, flip_z=FLIP_Z):
        total_events += 1

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

        # Narrow (x,Q) bin for TMD-like core
        if not (X_LO < x < X_HI) or not (Q_LO < Q < Q_HI):
            continue
        in_bin_events += 1

        # Also require within global pTrel analysis range
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

        # Pion selection in Breit frame (same as validated pipeline)
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

        # Remnant definition (same as validated pTrel pipeline)
        k_in = k_out_ev - qmu
        p_rem_truth = np.asarray(p_in_ev, dtype=float) - k_in
        P_remnant_breit = boost(p_rem_truth)
        r3 = p3(P_remnant_breit)
        if np.linalg.norm(r3) <= 0:
            continue

        # TMD-style remnant transverse axis in Breit
        rT = np.array([P_remnant_breit[1], P_remnant_breit[2]], dtype=float)
        hT = np.array([P_pi_breit[1], P_pi_breit[2]], dtype=float)
        rT_norm = float(np.linalg.norm(rT))
        if rT_norm < 1e-6:
            axis_skipped += 1
            continue
        u_hat = rT / rT_norm
        u_hat_perp = np.array([-u_hat[1], u_hat[0]], dtype=float)

        # Signed perpendicular component of pion pT relative to remnant axis
        pTrel_rem = float(np.dot(hT, u_hat_perp))

        # Additional xL cut (same as validated pTrel)
        q_breit = boost(qmu)
        den = dot4(P_remnant_breit, q_breit)
        if den <= 0:
            continue
        xL_exact = dot4(P_pi_breit, q_breit) / den
        if xL_exact < 0.01 or xL_exact > 1.0 + 1e-6:
            continue

        pTrel_list.append(pTrel_rem)

    pTrel_arr = np.array(pTrel_list, dtype=float)

    # Diagnostics
    print("--- pTrel_rem (wrt remnant axis) in narrow bin ---")
    print(f"Total events iterated          : {total_events}")
    print(f"Events in (0.04<x<0.05, 6<Q<8): {in_bin_events}")
    print(f"Events skipped |rT|<1e-6       : {axis_skipped}")
    print(f"Events used for pTrel_rem      : {pTrel_arr.size}")
    if pTrel_arr.size == 0:
        print("No events survived all cuts; nothing to plot.")
        return
    print(f"Mean(pTrel_rem) = {np.mean(pTrel_arr):.6e}")
    print(f"Std(pTrel_rem)  = {np.std(pTrel_arr):.6e}")

    # Histogram of signed pTrel and Gaussian fit to core
    bins = np.linspace(-2.0, 2.0, 120)
    counts, edges = np.histogram(pTrel_arr, bins=bins, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Fit only core |x|<1.5 with counts>0
    fit_mask = (np.abs(centers) < 1.5) & (counts > 0)
    if np.sum(fit_mask) >= 3 and curve_fit is not None:
        cx = centers[fit_mask]
        cy = counts[fit_mask].astype(float)
        sigma_y = np.sqrt(np.maximum(cy + 1, 1.0))
        mu0 = float(np.mean(pTrel_arr))
        sigma0 = float(np.std(pTrel_arr)) or 0.5
        A0 = float(np.max(cy))
        try:
            popt, _ = curve_fit(
                gaussian,
                cx,
                cy,
                p0=(A0, mu0, max(sigma0, 0.1)),
                sigma=sigma_y,
                absolute_sigma=True,
                maxfev=5000,
                bounds=([0.0, -5.0, 0.01], [np.max(cy) * 2.0, 5.0, 5.0]),
            )
            A_fit, mu_fit, sigma_fit = popt
        except Exception:
            A_fit = A0
            mu_fit = mu0
            sigma_fit = sigma0
    else:
        A_fit = float(np.max(counts))
        mu_fit = float(np.mean(pTrel_arr))
        sigma_fit = float(np.std(pTrel_arr)) or 1.0

    print(f"Fit mu    = {mu_fit:.6e}")
    print(f"Fit sigma = {sigma_fit:.6e}")

    x_grid = np.linspace(-2.0, 2.0, 400)
    y_fit = gaussian(x_grid, A_fit, mu_fit, sigma_fit)

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "stix",
        }
    )
    fontsize = 14

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(centers, counts, where="mid", color="steelblue", linewidth=1.5, label="pTrel_rem counts")
    ax.plot(x_grid, y_fit, "k--", linewidth=2.0, label="Gaussian fit (|x|<1.5)")
    ax.set_xlabel("pT_rel wrt remnant axis [GeV]", fontsize=fontsize)
    ax.set_ylabel("Counts", fontsize=fontsize)
    ax.set_title("pTrel wrt remnant axis (ISR/FSR ON, narrow x,Q bin)", fontsize=fontsize)
    ax.legend(loc="best", fontsize=fontsize)
    ax.tick_params(direction="in", labelsize=fontsize)
    ax.text(
        0.02,
        0.98,
        f"$\\mu = {mu_fit:.4f}$\\n$\\sigma = {sigma_fit:.4f}$",
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
    )
    plt.tight_layout()
    outname_signed = _PROJECT_ROOT / "pTrel_remnant_ISRFSR_ON_narrow_bin.pdf"
    plt.savefig(outname_signed, format="pdf")
    plt.close(fig)
    print(f"Saved: {outname_signed}")

    # Magnitude distribution: pTrel_mag = |pTrel_signed|
    pTrel_mag = np.abs(pTrel_arr)
    print(f"Mean(|pTrel_rem|) = {np.mean(pTrel_mag):.6e}")
    print(f"Std(|pTrel_rem|)  = {np.std(pTrel_mag):.6e}")

    bins_mag = np.linspace(0.0, 2.0, 80)
    counts_mag, edges_mag = np.histogram(pTrel_mag, bins=bins_mag, density=False)
    centers_mag = 0.5 * (edges_mag[:-1] + edges_mag[1:])

    fig_mag, ax_mag = plt.subplots(figsize=(8, 5))
    ax_mag.step(
        centers_mag,
        counts_mag,
        where="mid",
        color="darkgreen",
        linewidth=1.5,
        label="|pTrel_rem| counts",
    )
    ax_mag.set_xlabel("|pT_rel| wrt remnant axis [GeV]", fontsize=fontsize)
    ax_mag.set_ylabel("Counts", fontsize=fontsize)
    ax_mag.set_title("|pTrel| wrt remnant axis (ISR/FSR ON, narrow x,Q bin)", fontsize=fontsize)
    ax_mag.legend(loc="best", fontsize=fontsize)
    ax_mag.tick_params(direction="in", labelsize=fontsize)
    plt.tight_layout()
    outname_mag = _PROJECT_ROOT / "pTrel_remnant_ISRFSR_ON_narrow_bin_mag.pdf"
    plt.savefig(outname_mag, format="pdf")
    plt.close(fig_mag)
    print(f"Saved: {outname_mag}")


if __name__ == "__main__":
    main()

