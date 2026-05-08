#!/usr/bin/env python3
"""
Step histograms for split π+/π− jet–hadron observables.

Uses the same binning and (1/σ) dσ/dx normalization style as
analyze_jet_hadron_transverse_observables.py. No PYTHIA dependency.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_jh():
    path = PROJECT_ROOT / "scripts" / "analysis" / "analyze_jet_hadron_transverse_observables.py"
    spec = importlib.util.spec_from_file_location("_jh_obs_mod_fig", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


JH = _load_jh()

# Fixed y-range for per-bin R_{pi-/pi+} ratio strips (below density panels).
PI_PM_RATIO_YLIM: Tuple[float, float] = (0.9, 1.1)
# Wider range for azimuthal decorrelation (phi_hJ).
PI_PM_RATIO_YLIM_AZIMUTH: Tuple[float, float] = (0.8, 1.2)


def _ratio_ylim_for_obs_key(obs_key: str) -> Tuple[float, float]:
    return PI_PM_RATIO_YLIM_AZIMUTH if obs_key == "phi_hJ" else PI_PM_RATIO_YLIM


def dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """CSV → list[dict] for plotting; normalize ok column and NaN."""
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        d: Dict[str, Any] = {}
        for k, v in row.items():
            if pd.isna(v):
                d[k] = None
            elif k == "ok" and isinstance(v, str):
                d[k] = v.strip().lower() in ("true", "1", "yes")
            else:
                d[k] = v
        records.append(d)
    return records


def _vals_for(
    records: List[Dict[str, Any]],
    sample: str,
    pion: str,
    key: str,
    *,
    finite_only: bool,
) -> np.ndarray:
    raw: List[float] = []
    for r in records:
        if r.get("sample") != sample or r.get("pion") != pion:
            continue
        if not r.get("ok"):
            continue
        if key not in r or r[key] is None:
            continue
        try:
            raw.append(float(r[key]))
        except (TypeError, ValueError):
            continue
    arr = np.asarray(raw, dtype=np.float64)
    if finite_only:
        arr = arr[np.isfinite(arr)]
    return arr


def _vals_for_combined(
    records: List[Dict[str, Any]],
    pion: str,
    key: str,
    *,
    finite_only: bool,
) -> np.ndarray:
    """All rows with ``ok`` and matching ``pion``, ignoring ``sample`` (pooled subsamples)."""
    raw: List[float] = []
    for r in records:
        if r.get("pion") != pion:
            continue
        if not r.get("ok"):
            continue
        if key not in r or r[key] is None:
            continue
        try:
            raw.append(float(r[key]))
        except (TypeError, ValueError):
            continue
    arr = np.asarray(raw, dtype=np.float64)
    if finite_only:
        arr = arr[np.isfinite(arr)]
    return arr


def _add_step_curve(
    ax: Any,
    values: np.ndarray,
    bins: np.ndarray,
    binw: float,
    label: str,
    color: str,
) -> None:
    if len(values) == 0:
        return
    centers = 0.5 * (bins[:-1] + bins[1:])
    hist, _ = np.histogram(values, bins=bins)
    n = len(values)
    density = hist / (n * binw) if n > 0 else hist.astype(float)
    ax.step(centers, density, where="mid", label=label, color=color, linewidth=1.5)


def _add_pi_minus_over_pi_plus_ratio_panel(
    ax: Any,
    vp: np.ndarray,
    vm: np.ndarray,
    bins: np.ndarray,
    *,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot ((1/sigma^- ) d sigma^- / dx) / ((1/sigma^+) d sigma^+ / dx) per bin.

    With histogram counts this is:
      (hm / n_m) / (hp / n_p)
    since the common bin-width factor cancels.
    """
    n_p, n_m = int(len(vp)), int(len(vm))
    hp, _ = np.histogram(vp, bins=bins)
    hm, _ = np.histogram(vm, bins=bins)
    hp = hp.astype(np.float64)
    hm = hm.astype(np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ratio = np.full_like(hp, np.nan, dtype=np.float64)
    err = np.full_like(hp, np.nan, dtype=np.float64)
    if n_p > 0 and n_m > 0:
        for i in range(len(hp)):
            Np, Nm = float(hp[i]), float(hm[i])
            if Np > 0 and Nm > 0:
                ratio[i] = (Nm / float(n_m)) / (Np / float(n_p))
                # Poisson-style uncertainty from bin counts (n_p, n_m treated as fixed sample sizes).
                err[i] = ratio[i] * float(np.sqrt(1.0 / Nm + 1.0 / Np))
    valid = np.isfinite(ratio) & (hp > 0) & (hm > 0)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.8)
    if np.any(valid):
        ax.errorbar(
            centers[valid],
            ratio[valid],
            yerr=err[valid],
            fmt="o",
            markersize=3,
            color="black",
            elinewidth=0.8,
            capsize=2,
            label=r"$R_{\pi^-/\pi^+}$",
        )
        ax.legend(loc="best", frameon=False, fontsize=8)
    ax.set_ylabel(r"$R_{\pi^-/\pi^+}$")
    lo, hi = ylim if ylim is not None else PI_PM_RATIO_YLIM
    ax.set_ylim(lo, hi)
    ax.tick_params(direction="in")


def _write_one_pi_pm_figure_two_rows(
    *,
    records: List[Dict[str, Any]],
    samples: List[Tuple[str, str]],
    figure_dir: Path,
    file_key: str,
    bins: np.ndarray,
    binw: float,
    obs_key: str,
    finite_only: bool,
    ylabel_dens: str,
    xlabel: str,
) -> Path:
    """
    Two stacked **panels** (altered / unchanged). Each panel = density on top + π⁻/π⁺ ratio
    strip below (``make_axes_locatable``), so the figure reads as two rows, not four siblings.
    """
    fig, ax_mains = plt.subplots(2, 1, figsize=(6.0, 9.0), sharex=True)
    for ix, (ax_m, (samp, stitle)) in enumerate(zip(ax_mains, samples)):
        vp = _vals_for(records, samp, "piplus", obs_key, finite_only=finite_only)
        vm = _vals_for(records, samp, "piminus", obs_key, finite_only=finite_only)
        _add_step_curve(ax_m, vp, bins, binw, r"$\pi^+$", "blue")
        _add_step_curve(ax_m, vm, bins, binw, r"$\pi^-$", "red")
        ax_m.set_ylabel(ylabel_dens)
        ax_m.set_title(stitle)
        h, lab = ax_m.get_legend_handles_labels()
        if lab:
            ax_m.legend(loc="best", frameon=False)
        ax_m.tick_params(direction="in")
        div = make_axes_locatable(ax_m)
        ax_r = div.append_axes("bottom", size="32%", pad=0.07, sharex=ax_m)
        _add_pi_minus_over_pi_plus_ratio_panel(
            ax_r, vp, vm, bins, ylim=_ratio_ylim_for_obs_key(obs_key)
        )
        ax_m.tick_params(labelbottom=False)
        if ix < len(ax_mains) - 1:
            ax_r.tick_params(labelbottom=False)
        else:
            ax_r.set_xlabel(xlabel)
    plt.tight_layout()
    out = (
        figure_dir
        / f"{JH.OUTPUT_PREFIX}_{file_key}_{JH.HADRON_TAG}_{JH.FRAME_TAG}_split_pi_pm_comparison.pdf"
    )
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def _write_one_pi_pm_figure_combined(
    *,
    records: List[Dict[str, Any]],
    figure_dir: Path,
    file_key: str,
    bins: np.ndarray,
    binw: float,
    obs_key: str,
    finite_only: bool,
    ylabel_dens: str,
    xlabel: str,
    title: str,
    filename_suffix: str = "",
) -> Path:
    """
    One observable: top = π⁺ and π⁻ (pooled over all ``sample`` values), bottom = R_{π⁻/π⁺}.
    """
    fig, ax_m = plt.subplots(1, 1, figsize=(6.0, 4.2))
    vp = _vals_for_combined(records, "piplus", obs_key, finite_only=finite_only)
    vm = _vals_for_combined(records, "piminus", obs_key, finite_only=finite_only)
    _add_step_curve(ax_m, vp, bins, binw, r"$\pi^+$", "blue")
    _add_step_curve(ax_m, vm, bins, binw, r"$\pi^-$", "red")
    ax_m.set_ylabel(ylabel_dens)
    ax_m.set_title(title)
    h, lab = ax_m.get_legend_handles_labels()
    if lab:
        ax_m.legend(loc="best", frameon=False)
    ax_m.tick_params(direction="in")
    div = make_axes_locatable(ax_m)
    ax_r = div.append_axes("bottom", size="32%", pad=0.07, sharex=ax_m)
    _add_pi_minus_over_pi_plus_ratio_panel(
        ax_r, vp, vm, bins, ylim=_ratio_ylim_for_obs_key(obs_key)
    )
    ax_m.tick_params(labelbottom=False)
    ax_r.set_xlabel(xlabel)
    plt.tight_layout()
    suf = filename_suffix if filename_suffix else ""
    out = (
        figure_dir
        / f"{JH.OUTPUT_PREFIX}_{file_key}_{JH.HADRON_TAG}_{JH.FRAME_TAG}_combined_pi_pm{suf}.pdf"
    )
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


def write_combined_pi_pm_comparison_pdfs(
    records: List[Dict[str, Any]],
    figure_dir: Path,
    *,
    title: str = r"$\pi^\pm$ (pooled over subsamples)",
    filename_suffix: str = "",
) -> List[Path]:
    """
    Four PDFs (angle, sum, diff, P̄_t). Each file: one π⁺/π⁻ density panel + R_{π⁻/π⁺} below,
    pooling all CSV rows that pass the ``ok`` filter (any ``sample``).
    """
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    bins_a = np.linspace(JH.ANGLE_RANGE[0], JH.ANGLE_RANGE[1], JH.ANGLE_BINS + 1)
    binw_a = (JH.ANGLE_RANGE[1] - JH.ANGLE_RANGE[0]) / JH.ANGLE_BINS
    written.append(
        _write_one_pi_pm_figure_combined(
            records=records,
            figure_dir=figure_dir,
            file_key="angle",
            bins=bins_a,
            binw=binw_a,
            obs_key="phi_hJ",
            finite_only=True,
            ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\phi_{hJ}}$",
            xlabel=r"$\phi_{hJ}$ [rad]",
            title=title,
            filename_suffix=filename_suffix,
        )
    )

    bins_sd = np.linspace(JH.SUM_DIFF_RANGE[0], JH.SUM_DIFF_RANGE[1], JH.SUM_DIFF_BINS + 1)
    binw_sd = (JH.SUM_DIFF_RANGE[1] - JH.SUM_DIFF_RANGE[0]) / JH.SUM_DIFF_BINS

    written.append(
        _write_one_pi_pm_figure_combined(
            records=records,
            figure_dir=figure_dir,
            file_key="sum_mag",
            bins=bins_sd,
            binw=binw_sd,
            obs_key="sum_mag_GeV",
            finite_only=False,
            ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\bar{P}_{hJ}}$",
            xlabel=r"$\bar{P}_{hJ}$ [GeV] ($|p_{T,\mathrm{jet}}+p_{T,h}|$)",
            title=title,
            filename_suffix=filename_suffix,
        )
    )
    written.append(
        _write_one_pi_pm_figure_combined(
            records=records,
            figure_dir=figure_dir,
            file_key="diff_mag",
            bins=bins_sd,
            binw=binw_sd,
            obs_key="delta_P_hJ_GeV",
            finite_only=False,
            ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\Delta P_{hJ}}$",
            xlabel=r"$\Delta P_{hJ}$ [GeV]",
            title=title,
            filename_suffix=filename_suffix,
        )
    )
    written.append(
        _write_one_pi_pm_figure_combined(
            records=records,
            figure_dir=figure_dir,
            file_key="Pbar_t",
            bins=bins_sd,
            binw=binw_sd,
            obs_key="Pbar_t_GeV",
            finite_only=False,
            ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}|\bar P_t|}$",
            xlabel=r"$|\bar P_t|$ [GeV] ($\frac{1}{2}|p_{T,\mathrm{jet}}+p_{T,h}|$)",
            title=title,
            filename_suffix=filename_suffix,
        )
    )

    return written


def write_split_pi_pm_comparison_pdfs(
    records: List[Dict[str, Any]],
    figure_dir: Path,
) -> List[Path]:
    """
    Write comparison PDFs: φ_hJ, |pT_jet+pT_h|, |pT_jet−pT_h|, and |P̄_t| (half-sum magnitude).
    **Two** stacked panels (altered / unchanged). Each panel shows π⁺/π⁻ densities with a
    **N(π⁻)/N(π⁺)** ratio strip directly beneath it (not a separate third/fourth subplot row).

    Each event contributes at most one row; ``pion`` tags the leading charged pion’s sign.
    """
    figure_dir.mkdir(parents=True, exist_ok=True)
    samples: List[Tuple[str, str]] = [
        ("altered", "Altered (split)"),
        ("unchanged", "Unchanged"),
    ]
    written: List[Path] = []

    bins_a = np.linspace(JH.ANGLE_RANGE[0], JH.ANGLE_RANGE[1], JH.ANGLE_BINS + 1)
    binw_a = (JH.ANGLE_RANGE[1] - JH.ANGLE_RANGE[0]) / JH.ANGLE_BINS
    written.append(
        _write_one_pi_pm_figure_two_rows(
            records=records,
            samples=samples,
            figure_dir=figure_dir,
            file_key="angle",
            bins=bins_a,
            binw=binw_a,
            obs_key="phi_hJ",
            finite_only=True,
            ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\phi_{hJ}}$",
            xlabel=r"$\phi_{hJ}$ [rad]",
        )
    )

    bins_sd = np.linspace(JH.SUM_DIFF_RANGE[0], JH.SUM_DIFF_RANGE[1], JH.SUM_DIFF_BINS + 1)
    binw_sd = (JH.SUM_DIFF_RANGE[1] - JH.SUM_DIFF_RANGE[0]) / JH.SUM_DIFF_BINS

    written.append(
        _write_one_pi_pm_figure_two_rows(
            records=records,
            samples=samples,
            figure_dir=figure_dir,
            file_key="sum_mag",
            bins=bins_sd,
            binw=binw_sd,
            obs_key="sum_mag_GeV",
            finite_only=False,
            ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\bar{P}_{hJ}}$",
            xlabel=r"$\bar{P}_{hJ}$ [GeV] ($|p_{T,\mathrm{jet}}+p_{T,h}|$)",
        )
    )
    written.append(
        _write_one_pi_pm_figure_two_rows(
            records=records,
            samples=samples,
            figure_dir=figure_dir,
            file_key="diff_mag",
            bins=bins_sd,
            binw=binw_sd,
            obs_key="delta_P_hJ_GeV",
            finite_only=False,
            ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\Delta P_{hJ}}$",
            xlabel=r"$\Delta P_{hJ}$ [GeV]",
        )
    )
    written.append(
        _write_one_pi_pm_figure_two_rows(
            records=records,
            samples=samples,
            figure_dir=figure_dir,
            file_key="Pbar_t",
            bins=bins_sd,
            binw=binw_sd,
            obs_key="Pbar_t_GeV",
            finite_only=False,
            ylabel_dens=r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}|\bar P_t|}$",
            xlabel=r"$|\bar P_t|$ [GeV] ($\frac{1}{2}|p_{T,\mathrm{jet}}+p_{T,h}|$)",
        )
    )

    return written
