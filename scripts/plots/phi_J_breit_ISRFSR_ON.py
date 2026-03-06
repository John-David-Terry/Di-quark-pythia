#!/usr/bin/env python3
"""
Compute and plot the distribution of the jet azimuthal angle phi_J
in the Breit frame coordinates (ISRFSR_ON, cached shards).

Definition:
  - Work in the validated Breit-like frame (build_LT).
  - Jet 4-vector in Breit: P_jet_breit = P_proton_breit - P_remnant_breit.
  - Transverse jet momentum: pT_jet = (P_jet_breit.px, P_jet_breit.py).
  - Jet azimuth in Breit frame: phi_J = atan2(pT_jet_y, pT_jet_x) mapped to [0, 2π).

Run from project root: python scripts/plots/phi_J_breit_ISRFSR_ON.py
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

from diquark.analyze_events_raw import (
    Qmax_ptrel,
    Qmin_ptrel,
    build_LT,
    flip_z,
    p3,
    xmax_ptrel,
    xmin_ptrel,
)
from diquark.cached_shards import iter_events_from_shards

LABEL = "ISRFSR_ON"
FLIP_Z = True  # consistent with validated analysis for this label


def main():
    phi_J_vals = []
    total_events = 0
    used_events = 0

    for _shard_idx, ie, data in iter_events_from_shards(LABEL, flip_z=FLIP_Z):
        total_events += 1

        e_in_ev = data["event_e_in"]
        p_in_ev = data["event_p_in"]
        e_sc_ev = data["event_e_sc"]
        k_out_ev = data["event_k_out"]

        Ep = float(p_in_ev[0])
        Ee = float(e_in_ev[0])

        # q in lab (already flipped along z if FLIP_Z is True)
        q0 = e_in_ev[0] - e_sc_ev[0]
        q1 = e_in_ev[1] - e_sc_ev[1]
        q2 = e_in_ev[2] - e_sc_ev[2]
        q3 = e_in_ev[3] - e_sc_ev[3]
        qmu = np.array([q0, q1, q2, q3], dtype=float)

        # Basic DIS kinematics and cuts (same ranges as pTrel analysis)
        Q2 = -(q0 * q0 - q1 * q1 - q2 * q2 - q3 * q3)
        if Q2 <= 0:
            continue
        Q = float(np.sqrt(Q2))
        qT = float(np.hypot(q1, q2))

        # Lorentz-invariant p·q
        p_dot_q = p_in_ev[0] * q0 - p_in_ev[1] * q1 - p_in_ev[2] * q2 - p_in_ev[3] * q3
        if p_dot_q == 0:
            continue
        x = Q2 / (2.0 * p_dot_q)
        if not (xmin_ptrel <= x <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel):
            continue

        phiq = float(np.arctan2(q2, q1))
        S = 4.0 * Ee * Ep
        y = Q2 / (S * x)

        # Breit-like LT and boosts
        LT = build_LT(Ee, Ep, qmu, x, y, qT, phiq, S)
        if LT is None:
            continue
        boost = lambda v: LT @ np.asarray(v, dtype=float)

        # Proton in Breit frame
        p_proton_breit = boost(p_in_ev)
        P_plus = float(p_proton_breit[0] + p_proton_breit[3])
        if P_plus <= 0:
            continue

        # Remnant: k_in = k_out - q, P_rem = P_proton - k_in (lab frame), then boost
        k_in = k_out_ev - qmu
        p_rem_truth = np.asarray(p_in_ev, dtype=float) - k_in
        p_remnant_breit = boost(p_rem_truth)
        if np.linalg.norm(p3(p_remnant_breit)) <= 0:
            continue

        # Jet axis in Breit: P_jet_breit = P_proton_breit - P_remnant_breit
        P_jet_breit = p_proton_breit - p_remnant_breit
        pT_jet = np.array([P_jet_breit[1], P_jet_breit[2]], dtype=float)

        # Skip events with tiny jet pT (ill-defined azimuth)
        norm_jet = float(np.linalg.norm(pT_jet))
        if norm_jet < 1e-8:
            continue

        # Jet azimuth in Breit frame
        phi_J = np.arctan2(pT_jet[1], pT_jet[0])
        if phi_J < 0.0:
            phi_J += 2.0 * np.pi

        phi_J_vals.append(phi_J)
        used_events += 1

    phi_J_vals = np.asarray(phi_J_vals, dtype=float)

    print("--- phi_J in Breit frame (ISRFSR_ON) ---")
    print(f"Total events iterated : {total_events}")
    print(f"Events with valid phi_J: {used_events}")
    if used_events == 0:
        print("No events survived; nothing to plot.")
        return
    print(f"Mean(phi_J) = {np.mean(phi_J_vals):.6f}")
    print(f"Std(phi_J)  = {np.std(phi_J_vals):.6f}")

    # φ-binning in [0, 2π)
    nbins = 24
    hist, edges = np.histogram(phi_J_vals, bins=nbins, range=(0.0, 2.0 * np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    # Normalize to (1 / (N * Δφ)) dN/dφ
    density = hist.astype(float) / (used_events * bin_width)

    # Save arrays for further analysis
    np.save(_PROJECT_ROOT / "phi_J_breit_centers.npy", centers)
    np.save(_PROJECT_ROOT / "phi_J_breit_hist.npy", hist)
    np.save(_PROJECT_ROOT / "phi_J_breit_density.npy", density)

    # Plot normalized distribution vs phi_J
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "stix",
        }
    )
    fontsize = 14
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(
        centers,
        density,
        where="mid",
        color="darkmagenta",
        linewidth=1.5,
        label=r"$\frac{1}{N\,\Delta\phi}\,\frac{dN}{d\phi_J}$",
    )
    ax.set_xlabel(r"$\phi_J$ in Breit frame", fontsize=fontsize)
    ax.set_ylabel(r"Normalized counts", fontsize=fontsize)
    ax.set_title(r"Jet azimuth $\phi_J$ in Breit frame (ISR/FSR ON)", fontsize=fontsize)
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.legend(loc="best", fontsize=fontsize)
    ax.tick_params(direction="in", labelsize=fontsize)
    plt.tight_layout()
    outname = _PROJECT_ROOT / "phi_J_breit_ISRFSR_ON.pdf"
    plt.savefig(outname, format="pdf")
    plt.close(fig)
    print(f"Saved: {outname}")


if __name__ == "__main__":
    main()

