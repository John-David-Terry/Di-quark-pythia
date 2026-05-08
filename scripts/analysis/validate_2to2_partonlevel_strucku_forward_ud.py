#!/usr/bin/env python3
"""Validate parton-level struck-u forward-(u,d) baseline dataset."""

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

OUTDIR = outputs_dir() / "baseline_2to2_partonlevel_strucku_forward_ud"
COMBINED = OUTDIR / "combined_baseline.csv"
PLOTS = OUTDIR / "validation_plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def finite(arr):
    a = np.asarray(arr, dtype=float)
    return a[np.isfinite(a)]


def hist_overlay(u, d, title, xlabel, out, bins=70):
    plt.figure(figsize=(7, 5))
    if len(u):
        plt.hist(u, bins=bins, density=True, alpha=0.5, label="forward_u")
    if len(d):
        plt.hist(d, bins=bins, density=True, alpha=0.5, label="forward_d")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main() -> None:
    if not COMBINED.exists():
        raise FileNotFoundError(f"Missing {COMBINED}")
    df = pd.read_csv(COMBINED)
    u = df[df["forward_is_u"] == 1].copy()
    d = df[df["forward_is_d"] == 1].copy()

    hist_overlay(finite(u["Q2"]), finite(d["Q2"]), "Q2", "Q2 [GeV^2]", PLOTS / "q2_hist.png")
    hist_overlay(finite(u["xB"]), finite(d["xB"]), "xB", "xB", PLOTS / "xb_hist.png")
    hist_overlay(finite(u["y"]), finite(d["y"]), "y", "y", PLOTS / "y_hist.png")
    hist_overlay(finite(u["ele_E"]), finite(d["ele_E"]), "Electron energy", "ele_E [GeV]", PLOTS / "ele_E_hist.png")
    hist_overlay(finite(u["ele_eta"]), finite(d["ele_eta"]), "Electron eta", "ele_eta", PLOTS / "ele_eta_hist.png")
    hist_overlay(finite(u["struck_pT"]), finite(d["struck_pT"]), "Struck-u pT", "struck_pT [GeV]", PLOTS / "struck_pT_hist.png")
    hist_overlay(finite(u["struck_eta"]), finite(d["struck_eta"]), "Struck-u eta", "struck_eta", PLOTS / "struck_eta_hist.png")
    hist_overlay(finite(u["forward_E"]), finite(d["forward_E"]), "Forward parton E (maxE selector)", "forward_E [GeV]", PLOTS / "forward_parton_E_hist.png")
    hist_overlay(finite(u["forward_eta"]), finite(d["forward_eta"]), "Forward parton eta", "forward_eta", PLOTS / "forward_parton_eta_hist.png")

    # PID counts
    counts = df["forward_pid"].value_counts().to_dict()
    plt.figure(figsize=(6, 4))
    labels = [str(k) for k in sorted(counts.keys())]
    vals = [counts[int(k)] for k in sorted(counts.keys())]
    plt.bar(labels, vals)
    plt.title("Forward parton PID counts")
    plt.xlabel("forward_pid")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(PLOTS / "forward_parton_pid_counts.png", dpi=160)
    plt.close()

    # Combined comparison panel
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    axes[0].hist(finite(u["Q2"]), bins=60, density=True, alpha=0.5, label="forward_u")
    axes[0].hist(finite(d["Q2"]), bins=60, density=True, alpha=0.5, label="forward_d")
    axes[0].set_title("Q2")
    axes[1].hist(finite(u["xB"]), bins=60, density=True, alpha=0.5, label="forward_u")
    axes[1].hist(finite(d["xB"]), bins=60, density=True, alpha=0.5, label="forward_d")
    axes[1].set_title("xB")
    axes[2].hist(finite(u["forward_E"]), bins=60, density=True, alpha=0.5, label="forward_u")
    axes[2].hist(finite(d["forward_E"]), bins=60, density=True, alpha=0.5, label="forward_d")
    axes[2].set_title("forward_E")
    axes[3].hist(finite(u["forward_eta"]), bins=60, density=True, alpha=0.5, label="forward_u")
    axes[3].hist(finite(d["forward_eta"]), bins=60, density=True, alpha=0.5, label="forward_d")
    axes[3].set_title("forward_eta")
    for ax in axes:
        ax.legend()
    fig.suptitle("Forward-u vs Forward-d comparison (parton-level)")
    fig.tight_layout()
    fig.savefig(PLOTS / "forward_u_vs_forward_d_comparison.png", dpi=160)
    plt.close(fig)

    summary = {
        "n_forward_u": int(u.shape[0]),
        "n_forward_d": int(d.shape[0]),
        "n_total": int(df.shape[0]),
        "fraction_forward_u": float(u.shape[0] / df.shape[0]) if df.shape[0] else float("nan"),
        "fraction_forward_d": float(d.shape[0] / df.shape[0]) if df.shape[0] else float("nan"),
        "q2_mean_forward_u": float(np.nanmean(u["Q2"])) if u.shape[0] else float("nan"),
        "q2_mean_forward_d": float(np.nanmean(d["Q2"])) if d.shape[0] else float("nan"),
    }
    (PLOTS / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Validation complete. Plots written to {PLOTS}")


if __name__ == "__main__":
    main()

