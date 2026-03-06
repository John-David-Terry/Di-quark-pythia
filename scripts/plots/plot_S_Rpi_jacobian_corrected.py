#!/usr/bin/env python3
"""
Plot (1 / |ΣpT|) * (dN / d|ΣpT|) for
ΣpT = |pT_pi + pT_rem| using the cached
S_Rpi_ISRFSR_ON.npy array (ISR/FSR ON only).

Run from project root: python scripts/plots/plot_S_Rpi_jacobian_corrected.py
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None


def half_gaussian(r, A, sigma):
    """Half-Gaussian in radius (mu fixed at 0)."""
    sigma_safe = max(sigma, 1e-9)
    return A * np.exp(-0.5 * (r / sigma_safe) ** 2)


def main():
    # 1) Load data
    S_vals = np.load(_PROJECT_ROOT / "S_Rpi_ISRFSR_ON.npy")
    S_vals = S_vals[np.isfinite(S_vals)]
    S_vals = S_vals[S_vals > 0.0]

    if S_vals.size == 0:
        print("No positive finite S_vals; nothing to plot.")
        return

    print(f"N raw S_vals: {S_vals.size}")
    print(f"min(S_vals) = {S_vals.min():.6e}")
    print(f"max(S_vals) = {S_vals.max():.6e}")
    print(f"mean(S_vals) = {S_vals.mean():.6e}")
    print(f"std(S_vals)  = {S_vals.std():.6e}")

    # 2) Histogram in magnitude over core
    xmax = np.quantile(S_vals, 0.995)
    bins = np.linspace(0.0, float(xmax), 100)
    counts, edges = np.histogram(S_vals, bins=bins, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])

    dN_dR = counts / bin_width

    # 3) Divide by R (Jacobian factor)
    eps = 1e-8
    corrected = dN_dR / (centers + eps)

    # Restrict to core for clarity
    core_mask_plot = centers < np.quantile(S_vals, 0.8)
    core_mask_plot &= centers > 1e-3

    centers_core = centers[core_mask_plot]
    corrected_core = corrected[core_mask_plot]
    counts_core = counts[core_mask_plot]

    print(f"N core bins used for plot: {centers_core.size}")

    # 5) Fit Gaussian to corrected distribution (half-Gaussian in R)
    A_fit, sigma_fit = 0.0, 1.0
    if curve_fit is not None and centers_core.size >= 3:
        fit_mask = (centers_core > 0.0) & (centers_core < centers_core.max())
        fit_centers = centers_core[fit_mask]
        fit_corrected = corrected_core[fit_mask]
        fit_counts = counts_core[fit_mask].astype(float)

        if fit_centers.size >= 3:
            sigma_y = np.sqrt(np.maximum(fit_counts + 1.0, 1.0))
            A0 = float(np.max(fit_corrected))
            sigma0 = float(np.std(S_vals)) or 0.5
            try:
                popt, _ = curve_fit(
                    half_gaussian,
                    fit_centers,
                    fit_corrected,
                    p0=(A0, sigma0),
                    sigma=sigma_y,
                    absolute_sigma=True,
                    maxfev=5000,
                    bounds=([0.0, 0.01], [np.inf, 10.0]),
                )
                A_fit, sigma_fit = popt
            except Exception:
                A_fit, sigma_fit = A0, sigma0

    print(f"Fit A     = {A_fit:.6e}")
    print(f"Fit sigma = {sigma_fit:.6e}")

    # 4) Plot corrected distribution (core only) with fit
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
        centers_core,
        corrected_core,
        where="mid",
        color="darkred",
        linewidth=1.5,
        label=r"$(1/R)\, dN/dR$ (core)",
    )

    if A_fit > 0.0 and sigma_fit > 0.0:
        r_grid = np.linspace(0.0, centers_core.max(), 400)
        y_fit = half_gaussian(r_grid, A_fit, sigma_fit)
        ax.plot(
            r_grid,
            y_fit,
            "k--",
            linewidth=2.0,
            label=r"Half-Gaussian fit in $R$",
        )

    ax.set_xlabel(r"$|\Sigma p_T|$ [GeV]", fontsize=fontsize)
    ax.set_ylabel(r"$(1/R)\, dN/dR$ (arb. units)", fontsize=fontsize)
    ax.set_title(
        r"Jacobian-corrected $|\Sigma p_T|$ distribution (ISR/FSR ON)",
        fontsize=fontsize,
    )
    ax.legend(loc="best", fontsize=fontsize)
    ax.tick_params(direction="in", labelsize=fontsize)
    ax.set_xlim(0.0, centers_core.max())
    plt.tight_layout()
    outname = _PROJECT_ROOT / "S_Rpi_ISRFSR_ON_jacobian_corrected.pdf"
    plt.savefig(outname, format="pdf")
    plt.close(fig)
    print(f"Saved: {outname}")


if __name__ == "__main__":
    main()

