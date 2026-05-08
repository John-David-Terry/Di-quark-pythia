#!/usr/bin/env python3
"""
Validate the harvested 2->2 DIS baseline dataset.

High-level overview
--------------------
This script reads outputs/baseline_2to2/combined_baseline.csv (produced by
harvest_2to2_baseline.py) and produces:
  - Physically motivated distribution plots for DIS kinematics and selected objects
  - Object-level sanity plots:
      scattered electron, struck quark, and hardest forward d (selected by pplus)
  - A stability comparison:
      compare d selected by pplus (main) vs d selected by pT (secondary)

Low-level execution plan (exact logic)
----------------------------------------
1) Load the combined CSV.
2) Split into struck_u and struck_d by the column `struck_flavor`.
3) For each plot:
   - drop NaNs (e.g. when has_forward_d==0)
   - compute histograms / 2D histograms using numpy/matplotlib
4) Save the requested plots to validation_plots/.
5) Write a compact validation report to validation_plots/validation_summary.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

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

OUTDIR = outputs_dir() / "baseline_2to2"
COMBINED_CSV = OUTDIR / "combined_baseline.csv"
PLOTS_DIR = OUTDIR / "validation_plots"


def _hist_save_1d(ax, x, bins=60, label=None):
    ax.hist(x, bins=bins, density=True, alpha=0.5, label=label)


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_overlay_hist(x_u: np.ndarray, x_d: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 60):
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    _hist_save_1d(ax, x_u, bins=bins, label="struck_u")
    _hist_save_1d(ax, x_d, bins=bins, label="struck_d")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    if len(x_u) > 0 or len(x_d) > 0:
        ax.legend()
    _savefig(out_path)


def plot_2d(x_u: np.ndarray, y_u: np.ndarray, x_d: np.ndarray, y_d: np.ndarray, title: str, xlabel: str, ylabel: str, out_path: Path):
    # Drop NaNs/Infs early to avoid hist2d warnings.
    if len(x_u):
        mask_u = np.isfinite(x_u) & np.isfinite(y_u)
        x_u = x_u[mask_u]
        y_u = y_u[mask_u]
    if len(x_d):
        mask_d = np.isfinite(x_d) & np.isfinite(y_d)
        x_d = x_d[mask_d]
        y_d = y_d[mask_d]
    plt.figure(figsize=(7, 5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Use same bins/range for both
    x_all = np.concatenate([x_u, x_d]) if len(x_u) and len(x_d) else (x_u if len(x_u) else x_d)
    y_all = np.concatenate([y_u, y_d]) if len(y_u) and len(y_d) else (y_u if len(y_u) else y_d)
    if len(x_all) == 0 or len(y_all) == 0:
        _savefig(out_path)
        return
    x_min, x_max = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    bins = 60

    # Plot u
    if len(x_u):
        plt.hist2d(x_u, y_u, bins=bins, range=[[x_min, x_max], [y_min, y_max]], density=True, cmap="Blues", alpha=0.6)
    # Plot d
    if len(x_d):
        plt.hist2d(x_d, y_d, bins=bins, range=[[x_min, x_max], [y_min, y_max]], density=True, cmap="Oranges", alpha=0.6)

    _savefig(out_path)


def main() -> None:
    if not COMBINED_CSV.exists():
        raise FileNotFoundError(f"Missing {COMBINED_CSV}. Run harvest_2to2_baseline.py first.")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(COMBINED_CSV)
    if "struck_flavor" not in df.columns:
        raise RuntimeError("combined_baseline.csv missing struck_flavor column.")

    df_u = df[df["struck_flavor"] == "u"].copy()
    df_d = df[df["struck_flavor"] == "d"].copy()

    def col(arr: pd.Series) -> np.ndarray:
        return arr.to_numpy(dtype=float)

    # DIS kinematics
    q2_u = col(df_u["Q2"])
    q2_d = col(df_d["Q2"])
    xb_u = col(df_u["xB"])
    xb_d = col(df_d["xB"])
    y_u = col(df_u["y"])
    y_d = col(df_d["y"])

    # Object kinematics
    ele_E_u = col(df_u["ele_E"])
    ele_E_d = col(df_d["ele_E"])
    ele_eta_u = col(df_u["ele_eta"])
    ele_eta_d = col(df_d["ele_eta"])

    struck_pT_u = col(df_u["struck_pT"])
    struck_pT_d = col(df_d["struck_pT"])
    struck_eta_u = col(df_u["struck_eta"])
    struck_eta_d = col(df_d["struck_eta"])
    struck_pplus_u = col(df_u["struck_pplus"])
    struck_pplus_d = col(df_d["struck_pplus"])

    # Forward d (pplus selection). Drop NaNs.
    has_fwd_d_u = df_u["has_forward_d"].to_numpy(dtype=int)
    has_fwd_d_d = df_d["has_forward_d"].to_numpy(dtype=int)

    def filter_forward(df_sub: pd.DataFrame, has_arr: np.ndarray) -> pd.DataFrame:
        return df_sub[has_arr == 1].copy()

    df_du = filter_forward(df_u, has_fwd_d_u)
    df_dd = filter_forward(df_d, has_fwd_d_d)

    d_pT_u = col(df_du["d_pplus_pT"]) if len(df_du) else np.array([])
    d_pT_d = col(df_dd["d_pplus_pT"]) if len(df_dd) else np.array([])
    d_eta_u = col(df_du["d_pplus_eta"]) if len(df_du) else np.array([])
    d_eta_d = col(df_dd["d_pplus_eta"]) if len(df_dd) else np.array([])
    d_pplus_u = col(df_du["d_pplus_pplus"]) if len(df_du) else np.array([])
    d_pplus_d = col(df_dd["d_pplus_pplus"]) if len(df_dd) else np.array([])

    # 2D correlations (drop NaNs already)
    d_pplus_vs_struck = None
    if len(df_du):
        x = col(df_du["d_pplus_pplus"])
        y = col(df_du["struck_pplus"])
        d_pplus_vs_struck_u = (x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)])
    else:
        d_pplus_vs_struck_u = (np.array([]), np.array([]))
    if len(df_dd):
        x = col(df_dd["d_pplus_pplus"])
        y = col(df_dd["struck_pplus"])
        d_pplus_vs_struck_d = (x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)])
    else:
        d_pplus_vs_struck_d = (np.array([]), np.array([]))

    # Prepare validation summary
    summary: Dict[str, float] = {}
    summary["n_total_u"] = float(df_u.shape[0])
    summary["n_total_d"] = float(df_d.shape[0])
    summary["frac_forward_d_u"] = float(df_u["has_forward_d"].mean()) if df_u.shape[0] else float("nan")
    summary["frac_forward_d_d"] = float(df_d["has_forward_d"].mean()) if df_d.shape[0] else float("nan")

    # Ensure finite filtering for hist ranges
    def finite(x):
        return x[np.isfinite(x)]

    # Plots list (minimum requested)
    plot_overlay_hist(finite(q2_u), finite(q2_d), "Q2 distribution", "Q2 [GeV^2]", PLOTS_DIR / "q2_hist.png")
    plot_overlay_hist(finite(xb_u), finite(xb_d), "xB distribution", "xB", PLOTS_DIR / "xb_hist.png", bins=60)
    plot_overlay_hist(finite(y_u), finite(y_d), "y distribution", "y", PLOTS_DIR / "y_hist.png", bins=60)
    # xB vs Q2: build masks so x/y arrays align and are finite
    xb_u_all = xb_u
    xb_d_all = xb_d
    q2_u_all = q2_u
    q2_d_all = q2_d
    mask_u = np.isfinite(xb_u_all) & np.isfinite(q2_u_all)
    mask_d = np.isfinite(xb_d_all) & np.isfinite(q2_d_all)
    plot_2d(
        xb_u_all[mask_u],
        q2_u_all[mask_u],
        xb_d_all[mask_d],
        q2_d_all[mask_d],
        "xB vs Q2",
        "xB",
        "Q2",
        PLOTS_DIR / "xb_vs_q2.png",
    )

    plot_overlay_hist(finite(ele_E_u), finite(ele_E_d), "Electron energy", "ele_E [GeV]", PLOTS_DIR / "ele_E_hist.png")
    plot_overlay_hist(finite(ele_eta_u), finite(ele_eta_d), "Electron eta", "ele_eta", PLOTS_DIR / "ele_eta_hist.png", bins=60)

    plot_overlay_hist(finite(struck_pT_u), finite(struck_pT_d), "Struck quark pT", "struck_pT [GeV]", PLOTS_DIR / "struck_pT_hist.png")
    plot_overlay_hist(finite(struck_eta_u), finite(struck_eta_d), "Struck quark eta", "struck_eta", PLOTS_DIR / "struck_eta_hist.png", bins=60)
    plot_overlay_hist(finite(struck_pplus_u), finite(struck_pplus_d), "Struck quark pplus", "struck_pplus [GeV]", PLOTS_DIR / "struck_pplus_hist.png")

    plot_overlay_hist(finite(d_pT_u), finite(d_pT_d), "Forward d pT (selected by pplus)", "d_pplus_pT [GeV]", PLOTS_DIR / "d_pT_hist.png")
    plot_overlay_hist(finite(d_eta_u), finite(d_eta_d), "Forward d eta (selected by pplus)", "d_pplus_eta", PLOTS_DIR / "d_eta_hist.png", bins=60)
    plot_overlay_hist(finite(d_pplus_u), finite(d_pplus_d), "Forward d pplus (selected by pplus)", "d_pplus_pplus", PLOTS_DIR / "d_pplus_hist.png")

    # d_pplus vs struck_pplus
    x_u, y_u = d_pplus_vs_struck_u
    x_d, y_d = d_pplus_vs_struck_d
    plot_2d(x_u, y_u, x_d, y_d, "d_pplus vs struck_pplus", "d_pplus_pplus", "struck_pplus", PLOTS_DIR / "d_pplus_vs_struck_pplus.png")

    # d_pplus vs xB and Q2
    if len(df_du) and len(df_dd):
        xb_du = df_du["xB"].to_numpy(dtype=float)
        q2_du = df_du["Q2"].to_numpy(dtype=float)
        xb_dd = df_dd["xB"].to_numpy(dtype=float)
        q2_dd = df_dd["Q2"].to_numpy(dtype=float)
    else:
        xb_du = np.array([])
        q2_du = np.array([])
        xb_dd = np.array([])
        q2_dd = np.array([])

    x_u = col(df_du["d_pplus_pplus"]) if len(df_du) else np.array([])
    x_d = col(df_dd["d_pplus_pplus"]) if len(df_dd) else np.array([])
    xb_u = col(df_du["xB"]) if len(df_du) else np.array([])
    xb_d = col(df_dd["xB"]) if len(df_dd) else np.array([])
    q2_u_f = col(df_du["Q2"]) if len(df_du) else np.array([])
    q2_d_f = col(df_dd["Q2"]) if len(df_dd) else np.array([])

    plot_2d(x_u, xb_u, x_d, xb_d, "d_pplus vs xB", "d_pplus_pplus", "xB", PLOTS_DIR / "d_pplus_vs_xb.png")
    plot_2d(x_u, q2_u_f, x_d, q2_d_f, "d_pplus vs Q2", "d_pplus_pplus", "Q2", PLOTS_DIR / "d_pplus_vs_q2.png")

    # Comparison: pplus selector vs pt selector (on forward-d events only)
    if len(df_du) and len(df_dd):
        d_pplus_main_u = col(df_du["d_pplus_pplus"])
        d_pplus_alt_u = col(df_du["d_pt_pplus"])
        d_pplus_main_d = col(df_dd["d_pplus_pplus"])
        d_pplus_alt_d = col(df_dd["d_pt_pplus"])
    else:
        d_pplus_main_u = np.array([])
        d_pplus_alt_u = np.array([])
        d_pplus_main_d = np.array([])
        d_pplus_alt_d = np.array([])

    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    if len(d_pplus_main_u):
        ax.hist(d_pplus_main_u, bins=60, density=True, alpha=0.5, label="u main (pplus)", color="C0")
    if len(d_pplus_alt_u):
        ax.hist(d_pplus_alt_u, bins=60, density=True, alpha=0.5, label="u alt (pt)", color="C0", histtype="step")
    if len(d_pplus_main_d):
        ax.hist(d_pplus_main_d, bins=60, density=True, alpha=0.5, label="d main (pplus)", color="C1")
    if len(d_pplus_alt_d):
        ax.hist(d_pplus_alt_d, bins=60, density=True, alpha=0.5, label="d alt (pt)", color="C1", histtype="step")
    ax.set_title("Forward d selector comparison (pplus selector vs pT selector)")
    ax.set_xlabel("selected d pplus")
    ax.set_ylabel("density")
    ax.legend()
    _savefig(PLOTS_DIR / "comparison_d_selector_pplus_vs_pt.png")

    summary_path = PLOTS_DIR / "validation_summary.json"
    summary.update(
        {
            "n_forward_d_u": float((df_u["has_forward_d"] == 1).sum()),
            "n_forward_d_d": float((df_d["has_forward_d"] == 1).sum()),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Validation plots written to:", PLOTS_DIR)


if __name__ == "__main__":
    main()

