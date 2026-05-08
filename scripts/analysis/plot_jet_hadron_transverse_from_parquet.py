#!/usr/bin/env python3
"""
Plot jet–hadron transverse observables from
``produce_dis_final_state_jet_hadron_transverse.py`` output (Parquet).

Uses rows with ``ok == True``. Use ``--arm background`` for the ~900k DIS background
only (no altered reinject). Default ``--arm all`` pools every arm.

For each observable:
  * **Top:** normalized π⁺ vs π⁻ densities (same binning as
    ``analyze_jet_hadron_transverse_observables.py``).
  * **Bottom:** **π⁻/π⁺** ratio per bin = ``N₋(bin) / N₊(bin)`` with Poisson-derived
    uncertainty (delta method: ``R √(1/N₋ + 1/N₊)`` when both counts > 0).

Optional ``--by-arm`` restores legacy PDFs comparing background vs altered (π⁺/π⁻ panels).

Example:
  python scripts/analysis/plot_jet_hadron_transverse_from_parquet.py --arm background
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from diquark.paths import analysis_outputs_dir

ANGLE_BINS = 40
ANGLE_RANGE = (0.0, np.pi)
SUM_DIFF_BINS = 50
SUM_DIFF_MAX_GEV = 5.0
SUM_DIFF_RANGE = (0.0, SUM_DIFF_MAX_GEV)

OUTPUT_PREFIX = "jet_hadron_transverse"
TAG_SUFFIX_BASE = "final_state_parquet"
HADRON_TAG = "target_leading_pion"
FRAME_TAG = "Breit"

_PI_SPECIES: Tuple[Tuple[int, str], ...] = (
    (211, r"$\pi^+$"),
    (-211, r"$\pi^-$"),
)


def _default_parquet_path() -> Path:
    return (
        Path.home()
        / "Data"
        / "dis_jet_hadron_from_final_state_v1"
        / "jet_hadron_transverse_v1"
        / "rows.parquet"
    )


def _plot_pair(
    ax: plt.Axes,
    a: np.ndarray,
    b: np.ndarray,
    bins: np.ndarray,
    xrange: tuple[float, float],
    n_bins: int,
    xlabel: str,
    ylabel: str,
    label_a: str,
    label_b: str,
    color_a: str,
    color_b: str,
) -> None:
    binw = (xrange[1] - xrange[0]) / n_bins
    centers = 0.5 * (bins[:-1] + bins[1:])
    na, nb = len(a), len(b)
    ha, _ = np.histogram(a, bins=bins)
    hb, _ = np.histogram(b, bins=bins)
    da = ha / (na * binw) if na > 0 else ha.astype(float)
    db = hb / (nb * binw) if nb > 0 else hb.astype(float)
    ax.step(centers, da, where="mid", label=label_a, color=color_a, linewidth=1.5)
    ax.step(centers, db, where="mid", label=label_b, color=color_b, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=False)
    ax.tick_params(direction="in")


def _arrays_for_col(df: pd.DataFrame, col: str, angle: bool) -> np.ndarray:
    v = df[col].to_numpy(dtype=np.float64)
    if angle:
        v = v[np.isfinite(v)]
    return v


def _hist_counts(arr: np.ndarray, bins: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(arr, bins=bins)
    return h.astype(np.float64)


def _plot_combined_pi_density_and_ratio(
    out_path: Path,
    df_ok: pd.DataFrame,
    col: str,
    bins: np.ndarray,
    xrange: tuple[float, float],
    n_bins: int,
    xlabel: str,
    ylabel_dens: str,
    title: str,
    angle: bool,
) -> None:
    pip = df_ok[df_ok["pion_pdg"] == 211]
    pim = df_ok[df_ok["pion_pdg"] == -211]
    v_p = _arrays_for_col(pip, col, angle)
    v_m = _arrays_for_col(pim, col, angle)
    n_p, n_m = len(v_p), len(v_m)

    hp = _hist_counts(v_p, bins)
    hm = _hist_counts(v_m, bins)
    binw = (xrange[1] - xrange[0]) / n_bins
    centers = 0.5 * (bins[:-1] + bins[1:])

    dens_p = hp / (n_p * binw) if n_p > 0 else hp
    dens_m = hm / (n_m * binw) if n_m > 0 else hm

    ratio = np.full_like(hp, np.nan, dtype=np.float64)
    err = np.full_like(hp, np.nan, dtype=np.float64)
    for i in range(len(hp)):
        Np, Nm = float(hp[i]), float(hm[i])
        if Np > 0:
            ratio[i] = Nm / Np
            if Nm > 0:
                err[i] = ratio[i] * float(np.sqrt(1.0 / Nm + 1.0 / Np))
            else:
                err[i] = 0.0

    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(6.2, 7.0), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]}
    )
    ax0.step(centers, dens_p, where="mid", label=rf"$\pi^+$ ($N={n_p}$)", color="crimson", linewidth=1.5)
    ax0.step(centers, dens_m, where="mid", label=rf"$\pi^-$ ($N={n_m}$)", color="darkblue", linewidth=1.5)
    ax0.set_ylabel(ylabel_dens)
    ax0.legend(loc="best", frameon=False)
    ax0.tick_params(direction="in")
    ax0.set_title(title, fontsize=11)
    if n_p > 0 and n_m > 0:
        ax0.text(
            0.02,
            0.98,
            rf"global $N_{{\pi^-}}/N_{{\pi^+}} = {n_m/n_p:.4f}$",
            transform=ax0.transAxes,
            va="top",
            fontsize=9,
        )

    valid = np.isfinite(ratio) & (hp > 0)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.8)
    ax1.errorbar(
        centers[valid],
        ratio[valid],
        yerr=err[valid],
        fmt="o",
        markersize=3,
        color="black",
        elinewidth=0.8,
        capsize=2,
        label=r"$N_{\pi^-}/N_{\pi^+}$ per bin",
    )
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(r"$N_{\pi^-} / N_{\pi^+}$")
    ax1.set_ylim(*( (0.8, 1.2) if angle else (0.9, 1.1) ))
    ax1.legend(loc="best", frameon=False, fontsize=8)
    ax1.tick_params(direction="in")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_split_species_by_arm(
    out_path: Path,
    bg: pd.DataFrame,
    alt: pd.DataFrame,
    col: str,
    bins: np.ndarray,
    xrange: tuple[float, float],
    n_bins: int,
    xlabel: str,
    ylabel: str,
    suptitle: str,
    angle: bool,
) -> None:
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.0, 7.0), sharex=True)
    for ax, (pdg, title) in zip((ax_top, ax_bot), _PI_SPECIES):
        bg_s = bg[bg["pion_pdg"] == pdg]
        alt_s = alt[alt["pion_pdg"] == pdg]
        a = _arrays_for_col(bg_s, col, angle)
        b = _arrays_for_col(alt_s, col, angle)
        _plot_pair(
            ax,
            a,
            b,
            bins,
            xrange,
            n_bins,
            xlabel,
            ylabel,
            "background (LO jet proxy)",
            "altered reinject",
            "blue",
            "red",
        )
        ax.set_title(f"{title}   (N_bg={len(a)}, N_alt={len(b)})", fontsize=10)
    fig.suptitle(suptitle, fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot jet–hadron transverse observables from Parquet.")
    ap.add_argument("--parquet", type=Path, default=None, help="Path to rows.parquet")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: analysis_outputs_dir())")
    ap.add_argument(
        "--arm",
        choices=("all", "background", "altered_reinject"),
        default="all",
        help="Restrict to one dataset arm (default: pool all ok rows).",
    )
    ap.add_argument(
        "--by-arm",
        action="store_true",
        help="Also write legacy PDFs (background vs altered, π⁺/π⁻ panels).",
    )
    ap.add_argument(
        "--tag-suffix",
        type=str,
        default=None,
        help=(
            "Override PDF filename segment (default: final_state_parquet_{arm}_pi_ratio). "
            "Example: final_state_parquet_background_pi_ratio"
        ),
    )
    ap.add_argument(
        "--plot-note",
        type=str,
        default="",
        help="Optional note appended to the angle-plot title.",
    )
    args = ap.parse_args()

    pq = (args.parquet or _default_parquet_path()).expanduser().resolve()
    if not pq.is_file():
        raise SystemExit(f"Parquet not found: {pq}")

    out_dir = (args.out_dir or analysis_outputs_dir()).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(pq)
    if "ok" not in df.columns or "pion_pdg" not in df.columns:
        raise SystemExit("Parquet missing required columns (ok, pion_pdg)")
    df_ok_all = df[df["ok"] == True].copy()  # noqa: E712
    if df_ok_all.empty:
        raise SystemExit("No rows with ok==True")

    df_ok = df_ok_all
    if args.arm != "all":
        if "arm" not in df_ok.columns:
            raise SystemExit("Parquet missing 'arm' column for --arm filter")
        df_ok = df_ok[df_ok["arm"] == args.arm].copy()
        if df_ok.empty:
            raise SystemExit(f"No ok rows with arm=={args.arm!r}")

    arm_tag = args.arm if args.arm != "all" else "combined"
    tag_suffix = (
        args.tag_suffix
        if args.tag_suffix
        else f"{TAG_SUFFIX_BASE}_{arm_tag}_pi_ratio"
    )

    bins_angle = np.linspace(ANGLE_RANGE[0], ANGLE_RANGE[1], ANGLE_BINS + 1)
    bins_sd = np.linspace(SUM_DIFF_RANGE[0], SUM_DIFF_RANGE[1], SUM_DIFF_BINS + 1)

    stem = f"{OUTPUT_PREFIX}_{tag_suffix}_{HADRON_TAG}_{FRAME_TAG}"
    saved: list[Path] = []

    title_suffix = {
        "all": r"all arms ($\pi^+$ vs $\pi^-$)",
        "background": r"background only ($\sim$900k), $\pi^+$ vs $\pi^-$",
        "altered_reinject": r"altered reinject only, $\pi^+$ vs $\pi^-$",
    }[args.arm]
    if args.plot_note:
        title_suffix = f"{title_suffix} — {args.plot_note}"
    sum_title = (
        r"$|\mathbf{p}_{T,\mathrm{jet}} + \mathbf{p}_{T,h}|$ — "
        + ("combined arms" if args.arm == "all" else args.arm.replace("_", r"\,"))
    )
    diff_title = (
        r"$|\mathbf{p}_{T,\mathrm{jet}} - \mathbf{p}_{T,h}|$ — "
        + ("combined arms" if args.arm == "all" else args.arm.replace("_", r"\,"))
    )

    _plot_combined_pi_density_and_ratio(
        out_dir / f"{stem}_angle.pdf",
        df_ok,
        "obs_angle_rad",
        bins_angle,
        ANGLE_RANGE,
        ANGLE_BINS,
        r"$\phi_{hJ}$ [rad]",
        r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\phi_{hJ}}$",
        rf"$\phi_{{hJ}}$ — {title_suffix}",
        angle=True,
    )
    saved.append(out_dir / f"{stem}_angle.pdf")

    _plot_combined_pi_density_and_ratio(
        out_dir / f"{stem}_sum_mag.pdf",
        df_ok,
        "obs_sum_mag_GeV",
        bins_sd,
        SUM_DIFF_RANGE,
        SUM_DIFF_BINS,
        r"$| \mathbf{p}_{T,\mathrm{jet}} + \mathbf{p}_{T,h} |$ [GeV]",
        r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}|\mathbf{p}_{T,\mathrm{jet}}+\mathbf{p}_{T,h}|}$",
        sum_title,
        angle=False,
    )
    saved.append(out_dir / f"{stem}_sum_mag.pdf")

    _plot_combined_pi_density_and_ratio(
        out_dir / f"{stem}_diff_mag.pdf",
        df_ok,
        "obs_diff_mag_GeV",
        bins_sd,
        SUM_DIFF_RANGE,
        SUM_DIFF_BINS,
        r"$| \mathbf{p}_{T,\mathrm{jet}} - \mathbf{p}_{T,h} |$ [GeV]",
        r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}|\mathbf{p}_{T,\mathrm{jet}}-\mathbf{p}_{T,h}|}$",
        diff_title,
        angle=False,
    )
    saved.append(out_dir / f"{stem}_diff_mag.pdf")

    if args.by_arm:
        bg = df_ok_all[df_ok_all["arm"] == "background"]
        alt = df_ok_all[df_ok_all["arm"] == "altered_reinject"]
        if bg.empty or alt.empty:
            print("Warning: --by-arm skipped (missing background or altered rows).")
        else:
            leg = f"{OUTPUT_PREFIX}_final_state_parquet_bg_vs_altered_{HADRON_TAG}_{FRAME_TAG}"
            _plot_split_species_by_arm(
                out_dir / f"{leg}_angle_pi_pm.pdf",
                bg,
                alt,
                "obs_angle_rad",
                bins_angle,
                ANGLE_RANGE,
                ANGLE_BINS,
                r"$\phi_{hJ}$ [rad]",
                r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}\phi_{hJ}}$",
                r"Legacy: $\phi_{hJ}$ by arm",
                angle=True,
            )
            saved.append(out_dir / f"{leg}_angle_pi_pm.pdf")
            _plot_split_species_by_arm(
                out_dir / f"{leg}_sum_mag_pi_pm.pdf",
                bg,
                alt,
                "obs_sum_mag_GeV",
                bins_sd,
                SUM_DIFF_RANGE,
                SUM_DIFF_BINS,
                r"$| \mathbf{p}_{T,\mathrm{jet}} + \mathbf{p}_{T,h} |$ [GeV]",
                r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}|\mathbf{p}_{T,\mathrm{jet}}+\mathbf{p}_{T,h}|}$",
                r"Legacy: sum mag by arm",
                angle=False,
            )
            saved.append(out_dir / f"{leg}_sum_mag_pi_pm.pdf")
            _plot_split_species_by_arm(
                out_dir / f"{leg}_diff_mag_pi_pm.pdf",
                bg,
                alt,
                "obs_diff_mag_GeV",
                bins_sd,
                SUM_DIFF_RANGE,
                SUM_DIFF_BINS,
                r"$| \mathbf{p}_{T,\mathrm{jet}} - \mathbf{p}_{T,h} |$ [GeV]",
                r"$\dfrac{1}{\sigma}\dfrac{\mathrm{d}\sigma}{\mathrm{d}|\mathbf{p}_{T,\mathrm{jet}}-\mathbf{p}_{T,h}|}$",
                r"Legacy: diff mag by arm",
                angle=False,
            )
            saved.append(out_dir / f"{leg}_diff_mag_pi_pm.pdf")

    n_tot = len(df_ok)
    n_p = int((df_ok["pion_pdg"] == 211).sum())
    n_m = int((df_ok["pion_pdg"] == -211).sum())
    print(
        f"arm={args.arm}  ok events: {n_tot}  "
        f"(π+={n_p}, π-={n_m}, global π⁻/π⁺={n_m/n_p if n_p else float('nan'):.5f})"
    )
    for p in saved:
        print(f"Saved {p}")


if __name__ == "__main__":
    main()
