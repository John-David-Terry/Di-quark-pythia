#!/usr/bin/env python3
"""
Plot ISR/FSR ON transverse observables with Gaussian fits.

Run from project root: python scripts/plots/plot_observables_isrfsr_on.py
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import hashlib
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib as mpl  # noqa: E402

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Match style from generate_pdf_plots_new / analyze_events_raw
mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
    }
)


def md5_bytes(arr):
    return hashlib.md5(np.asarray(arr).tobytes()).hexdigest()


def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / np.maximum(sigma, 1e-9)) ** 2)


def fit_gaussian_curve_fit(centers, counts, fit_mask):
    """Fit A * exp(-(x-mu)^2/(2*sigma^2)) on masked bins; weight by Poisson sigma."""
    x_fit = centers[fit_mask]
    y_fit = counts[fit_mask].astype(float)
    sigma_y = np.sqrt(np.maximum(y_fit + 1, 1))
    if x_fit.size < 3:
        return 1.0, 0.0, 1.0
    # Initial guess from moments
    w = y_fit
    w_sum = np.sum(w)
    if w_sum <= 0:
        return 1.0, 0.0, 1.0
    mu0 = float(np.sum(w * x_fit) / w_sum)
    var0 = float(np.sum(w * (x_fit - mu0) ** 2) / w_sum)
    sigma0 = float(np.sqrt(max(var0, 1e-6)))
    A0 = float(np.max(y_fit))
    try:
        popt, _ = curve_fit(
            gaussian,
            x_fit,
            y_fit,
            p0=(A0, mu0, sigma0),
            sigma=sigma_y,
            absolute_sigma=True,
            maxfev=5000,
        )
        return float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        return A0, mu0, sigma0


def plot_one(name, xlabel, filename):
    arr_path = _PROJECT_ROOT / f"{name}_ISRFSR_ON.npy"
    vals = np.load(arr_path)
    vals = np.asarray(vals, dtype=np.float64).flatten()

    # --- DIAGNOSE: stats and hashes ---
    N = vals.size
    v_min, v_max = float(np.min(vals)), float(np.max(vals))
    v_mean, v_std = float(np.mean(vals)), float(np.std(vals))
    frac_gt_2 = float(np.sum(vals > 2.0) / N) if N else 0.0
    hash_vals = md5_bytes(vals)
    print(f"[{name}] N={N} min={v_min:.4f} max={v_max:.4f} mean={v_mean:.4f} std={v_std:.4f}")
    print(f"[{name}] md5(vals) = {hash_vals}")
    print(f"[{name}] fraction > 2.0 = {frac_gt_2:.4f}")

    # x-range: extend to ~99.5% quantile so shape is visible
    xmax = float(np.quantile(vals, 0.995))
    bins = np.linspace(0.0, xmax, 80)
    counts, edges = np.histogram(vals, bins=bins, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hash_hist = md5_bytes(counts)
    print(f"[{name}] md5(hist counts) = {hash_hist}")

    # Fit only on core region
    fit_x_max = float(np.quantile(vals, 0.8))
    fit_mask = (centers >= 0) & (centers <= fit_x_max) & (counts > 0)
    A, mu, sigma = fit_gaussian_curve_fit(centers, counts, fit_mask)

    x_grid = np.linspace(0.0, xmax, 400)
    y_fit = gaussian(x_grid, A, mu, sigma)

    fontsize = 20
    mpl.rcParams["font.size"] = fontsize
    mpl.rcParams["axes.labelsize"] = fontsize
    mpl.rcParams["xtick.labelsize"] = fontsize
    mpl.rcParams["ytick.labelsize"] = fontsize
    mpl.rcParams["legend.fontsize"] = fontsize

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(centers, counts, where="mid", label="ISR/FSR On", linewidth=2, alpha=0.8, color="red")
    ax.plot(x_grid, y_fit, "r--", alpha=0.8, linewidth=2, label="Fit")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.tick_params(direction="in", labelsize=fontsize)
    ax.legend(loc="best", frameon=False, fontsize=fontsize)
    ax.set_title("ISR/FSR On", fontsize=fontsize)

    text_str = rf"$\mu = {mu:.3f}$" + "\n" + rf"$\sigma = {sigma:.3f}$"
    ax.text(
        0.97,
        0.95,
        text_str,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment="top",
        horizontalalignment="right",
    )

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.93)
    plt.savefig(_PROJECT_ROOT / filename, format="pdf")
    plt.close(fig)
    print(f"[{name}] fit mu={mu:.4f} sigma={sigma:.4f} (fit range [0, {fit_x_max:.2f}])")
    print(f"Saved: {filename}")


def main():
    if curve_fit is None:
        raise RuntimeError("scipy.optimize.curve_fit required; install scipy")
    configs = [
        ("S_Rpi", r"$S_{R\pi}$", "S_Rpi_ISRFSR_ON_gaussian_fit.pdf"),
        ("D_Rpi", r"$D_{R\pi}$", "D_Rpi_ISRFSR_ON_gaussian_fit.pdf"),
        ("S_Jpi", r"$S_{J\pi}$", "S_Jpi_ISRFSR_ON_gaussian_fit.pdf"),
        ("D_Jpi", r"$D_{J\pi}$", "D_Jpi_ISRFSR_ON_gaussian_fit.pdf"),
    ]
    for name, xlabel, out_pdf in configs:
        plot_one(name, xlabel, out_pdf)


if __name__ == "__main__":
    main()
