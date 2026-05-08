#!/usr/bin/env python3
"""
Validate struck-u / forward-(u,d) 2->2 baseline samples.

Outputs:
  outputs/baseline_2to2_strucku_forward_ud/validation_plots/
    q2_hist.png
    xb_hist.png
    y_hist.png
    ele_E_hist.png
    struck_pT_hist.png
    forward_parton_E_hist.png
    forward_parton_eta_hist.png
    forward_u_vs_forward_d_comparison.png
    validation_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

OUTDIR = outputs_dir() / "baseline_2to2_strucku_forward_ud"
COMBINED = OUTDIR / "combined_baseline.csv"
PLOTS = OUTDIR / "validation_plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def plot_hist_overlay(x_u, x_d, title, xlabel, out, bins=70):
    plt.figure(figsize=(7, 5))
    if len(x_u):
        plt.hist(x_u, bins=bins, density=True, alpha=0.5, label="forward_u")
    if len(x_d):
        plt.hist(x_d, bins=bins, density=True, alpha=0.5, label="forward_d")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main() -> None:
    if not COMBINED.exists():
        raise FileNotFoundError(f"Missing {COMBINED}. Run harvest script first.")
    df = pd.read_csv(COMBINED)
    df_u = df[df["forward_is_u"] == 1].copy()
    df_d = df[df["forward_is_d"] == 1].copy()

    q2_u, q2_d = finite(df_u["Q2"].to_numpy(float)), finite(df_d["Q2"].to_numpy(float))
    xb_u, xb_d = finite(df_u["xB"].to_numpy(float)), finite(df_d["xB"].to_numpy(float))
    y_u, y_d = finite(df_u["y"].to_numpy(float)), finite(df_d["y"].to_numpy(float))
    eleE_u, eleE_d = finite(df_u["ele_E"].to_numpy(float)), finite(df_d["ele_E"].to_numpy(float))
    spT_u, spT_d = finite(df_u["struck_pT"].to_numpy(float)), finite(df_d["struck_pT"].to_numpy(float))
    fE_u, fE_d = finite(df_u["forward_E"].to_numpy(float)), finite(df_d["forward_E"].to_numpy(float))
    feta_u, feta_d = finite(df_u["forward_eta"].to_numpy(float)), finite(df_d["forward_eta"].to_numpy(float))

    plot_hist_overlay(q2_u, q2_d, "Q2 distribution", "Q2 [GeV^2]", PLOTS / "q2_hist.png")
    plot_hist_overlay(xb_u, xb_d, "xB distribution", "xB", PLOTS / "xb_hist.png")
    plot_hist_overlay(y_u, y_d, "y distribution", "y", PLOTS / "y_hist.png")
    plot_hist_overlay(eleE_u, eleE_d, "Outgoing electron energy", "ele_E [GeV]", PLOTS / "ele_E_hist.png")
    plot_hist_overlay(spT_u, spT_d, "Outgoing struck-u pT", "struck_pT [GeV]", PLOTS / "struck_pT_hist.png")
    plot_hist_overlay(fE_u, fE_d, "Hardest forward light parton energy", "forward_E [GeV]", PLOTS / "forward_parton_E_hist.png")
    plot_hist_overlay(feta_u, feta_d, "Hardest forward light parton eta", "forward_eta", PLOTS / "forward_parton_eta_hist.png")

    # Combined comparison panel
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    axes[0].hist(q2_u, bins=60, density=True, alpha=0.5, label="forward_u")
    axes[0].hist(q2_d, bins=60, density=True, alpha=0.5, label="forward_d")
    axes[0].set_title("Q2")
    axes[1].hist(xb_u, bins=60, density=True, alpha=0.5, label="forward_u")
    axes[1].hist(xb_d, bins=60, density=True, alpha=0.5, label="forward_d")
    axes[1].set_title("xB")
    axes[2].hist(spT_u, bins=60, density=True, alpha=0.5, label="forward_u")
    axes[2].hist(spT_d, bins=60, density=True, alpha=0.5, label="forward_d")
    axes[2].set_title("struck_pT")
    axes[3].hist(fE_u, bins=60, density=True, alpha=0.5, label="forward_u")
    axes[3].hist(fE_d, bins=60, density=True, alpha=0.5, label="forward_d")
    axes[3].set_title("forward_E")
    for ax in axes:
        ax.legend()
    fig.suptitle("Forward-u vs Forward-d sample comparison")
    fig.tight_layout()
    fig.savefig(PLOTS / "forward_u_vs_forward_d_comparison.png", dpi=160)
    plt.close(fig)

    summary = {
        "n_forward_u": int(df_u.shape[0]),
        "n_forward_d": int(df_d.shape[0]),
        "n_total": int(df.shape[0]),
        "acceptance_forward_u_over_total_accepted": float(df_u.shape[0] / df.shape[0]) if df.shape[0] else float("nan"),
        "acceptance_forward_d_over_total_accepted": float(df_d.shape[0] / df.shape[0]) if df.shape[0] else float("nan"),
        "q2_mean_forward_u": float(np.nanmean(df_u["Q2"])) if df_u.shape[0] else float("nan"),
        "q2_mean_forward_d": float(np.nanmean(df_d["Q2"])) if df_d.shape[0] else float("nan"),
    }
    (PLOTS / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Validation complete. Plots in {PLOTS}")


if __name__ == "__main__":
    main()

