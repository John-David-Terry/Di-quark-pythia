#!/usr/bin/env python3
"""
Compute hadron azimuth relative to the jet axis in the Breit frame,
by rotating each event so that pT_jet lies along +x and then
histogramming the hadron azimuth in that rotated frame.

Uses cached shards for ISRFSR_ON and the validated pTrel pipeline
logic to define the jet axis and selected pion.

Run from project root: python scripts/plots/phi_h_relative_to_jet_ISRFSR_ON.py
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
    dot4,
    flip_z,
    is_hadron,
    p3,
    xmax_ptrel,
    xmin_ptrel,
)
from diquark.cached_shards import iter_events_from_shards


LABEL = "ISRFSR_ON"
FLIP_Z = True  # consistent with pTrel pipeline for this label


def main():
    phi_vals = []
    total_events = 0
    used_events = 0

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
        if not (xmin_ptrel <= x <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel):
            continue

        phiq = float(np.arctan2(q2, q1))
        S = 4.0 * Ee * Ep
        y = Q2 / (S * x)
        LT = build_LT(Ee, Ep, qmu, x, y, qT, phiq, S)
        if LT is None:
            continue
        boost = lambda v: LT @ np.asarray(v, dtype=float)

        p_breit = boost(p_in_ev)
        P_plus = float(p_breit[0] + p_breit[3])
        if P_plus <= 0:
            continue

        # Select leading charged pion in Breit frame (same rule as pTrel)
        start = int(offsets[ie])
        end = int(offsets[ie + 1])
        best_tarE = -1.0
        best_tar_breit = None
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
                best_tar_breit = trf
                best_tar_pid = this_pid
        if best_tar_breit is None or abs(best_tar_pid) != 211:
            continue

        # Remnant and jet axis in Breit frame (validated definitions)
        k_in = k_out_ev - qmu
        p_rem_truth = np.asarray(p_in_ev, dtype=float) - k_in
        p_rem_truth_breit = boost(p_rem_truth)
        if np.linalg.norm(p3(p_rem_truth_breit)) <= 0:
            continue

        q_breit = boost(qmu)
        den = dot4(p_rem_truth_breit, q_breit)
        if den <= 0:
            continue
        xL_exact = dot4(best_tar_breit, q_breit) / den
        if xL_exact < 0.01 or xL_exact > 1.0 + 1e-6:
            continue

        p_proton_breit = np.asarray(p_breit, dtype=float)
        P_jet_breit = p_proton_breit - p_rem_truth_breit

        # Transverse components (px, py) in Breit frame
        pT_jet = np.array([P_jet_breit[1], P_jet_breit[2]], dtype=float)
        pT_h = np.array([best_tar_breit[1], best_tar_breit[2]], dtype=float)

        # Skip events with tiny jet pT (ill-defined azimuth)
        norm_jet = float(np.linalg.norm(pT_jet))
        if norm_jet < 1e-8:
            continue

        # Event-by-event rotation: align pT_jet along +x
        phi_J = np.arctan2(pT_jet[1], pT_jet[0])
        c = np.cos(-phi_J)
        s = np.sin(-phi_J)
        R = np.array([[c, -s], [s, c]], dtype=float)

        pT_jet_rot = R @ pT_jet
        pT_h_rot = R @ pT_h

        # Optional sanity check: pT_jet_rot[1] should be ~ 0
        # (we don't cut on it, just note it if needed)

        # Hadron azimuth in rotated frame
        phi_h = np.arctan2(pT_h_rot[1], pT_h_rot[0])
        if phi_h < 0.0:
            phi_h += 2.0 * np.pi

        phi_vals.append(phi_h)
        used_events += 1

    phi_vals = np.asarray(phi_vals, dtype=float)

    print("--- φ_h relative to jet axis (ISRFSR_ON) ---")
    print(f"Total events iterated : {total_events}")
    print(f"Events with valid φ_h : {used_events}")
    if used_events == 0:
        print("No events survived; nothing to plot.")
        return
    print(f"Mean(phi_h) = {np.mean(phi_vals):.6f}")
    print(f"Std(phi_h)  = {np.std(phi_vals):.6f}")

    # φ-binning
    nbins = 24
    hist, edges = np.histogram(phi_vals, bins=nbins, range=(0.0, 2.0 * np.pi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    # Optionally normalize to (1 / (N * Δφ)) dN/dφ
    density = hist.astype(float) / (used_events * bin_width)

    # Save raw histogram arrays for further analysis if desired
    np.save(_PROJECT_ROOT / "phi_h_rel_jet_centers.npy", centers)
    np.save(_PROJECT_ROOT / "phi_h_rel_jet_hist.npy", hist)
    np.save(_PROJECT_ROOT / "phi_h_rel_jet_density.npy", density)

    # Simple plot of normalized distribution vs φ
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
        color="steelblue",
        linewidth=1.5,
        label=r"$\frac{1}{N\,\Delta\phi}\,\frac{dN}{d\phi}$",
    )
    ax.set_xlabel(r"$\phi_h$ relative to jet axis", fontsize=fontsize)
    ax.set_ylabel(r"Normalized counts", fontsize=fontsize)
    ax.set_title(r"$\phi_h$ distribution (ISR/FSR ON, jet aligned to +x)", fontsize=fontsize)
    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.legend(loc="best", fontsize=fontsize)
    ax.tick_params(direction="in", labelsize=fontsize)
    plt.tight_layout()
    outname = _PROJECT_ROOT / "phi_h_relative_to_jet_ISRFSR_ON.pdf"
    plt.savefig(outname, format="pdf")
    plt.close(fig)
    print(f"Saved: {outname}")


if __name__ == "__main__":
    main()

