#!/usr/bin/env python3
"""
Overlay a single (x,Q) panel from the eta_hadron_*_xQ_regions_3x2 grid for two beams.

The original figure is 3 rows (Q) × 2 columns (sea | valence). The top-right panel is:
  Valence (x > 0.05) and 2 < Q < 5 GeV.

Reads ETA_XQ_* shards (same as analyze_events_raw.make_eta_hadron_xQ_grid_for_beam).

Example:
  python scripts/analysis/plot_eta_xq_valence_lowQ_overlay_two_beams.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
import numpy as np

# Body text (ticks, axes): UNIFORM/--font-pt. Regions: LEGEND_FONT_SCALE × pt; legend + kinematics: × LEGEND_KIN_SHRINK more.
UNIFORM_FONT_PT = 36.0
# Region titles + baseline “secondary” size vs body text (≈30% smaller).
LEGEND_FONT_SCALE = 0.7
# Legend entries + kinematics cuts: extra 10% smaller than that baseline.
LEGEND_KIN_SHRINK = 0.9


def _base_style_rc(pt: float) -> dict:
    """RcParams merged under rc_context so imports cannot leave mixed sizes."""
    return {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": pt,
        "axes.labelsize": pt,
        "axes.titlesize": pt,
        "xtick.labelsize": pt,
        "ytick.labelsize": pt,
        "legend.fontsize": pt,
        "legend.title_fontsize": pt,
        "figure.titlesize": pt,
        "figure.labelsize": pt,
    }


def _enforce_uniform_fontsize(fig: mpl.figure.Figure, pt: float) -> None:
    """Set fontsize on every Text artist (ticks, axis labels with math, legend, annotations)."""
    fig.canvas.draw()
    for txt in fig.findobj(Text):
        txt.set_fontsize(pt)


def _apply_secondary_fontsizes(
    ax: mpl.axes.Axes,
    region_pt: float,
    legend_kin_pt: float,
    region_texts: list[Text],
    kin_text: Optional[Text],
) -> None:
    """After _enforce_uniform_fontsize: restore region / legend / kinematics sizes."""
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(legend_kin_pt)
    for t in region_texts:
        t.set_fontsize(region_pt)
    if kin_text is not None:
        kin_text.set_fontsize(legend_kin_pt)


mpl.rcParams.update(_base_style_rc(UNIFORM_FONT_PT))

from diquark.analyze_events_raw import (  # noqa: E402
    EIC_REGIONS,
    ETA_BINS,
    ETA_RANGE,
    ETA_XQ_LABEL_BY_BEAM,
    classify_Q_bin_index,
    classify_x_bin_index,
    compute_x_Q_eta_one_event,
    flip_z,
    list_shards,
    load_shard,
)
from diquark.paths import analysis_outputs_dir  # noqa: E402

# Top-right of 3×2 grid: first Q row, valence column
IQ_PANEL = 0  # 2 < Q < 5 GeV
IX_PANEL = 1  # x > 0.05 (Valence)
FLIP_Z_ETA = False


def _verify_shard_beams(shard_dir: Path, Ee_nom: float, Ep_nom: float, tol: float = 0.02) -> None:
    mp = shard_dir / "meta.json"
    if not mp.exists():
        return
    meta = json.loads(mp.read_text())
    Ee = float(meta.get("E_e", -1))
    Ep = float(meta.get("E_p", -1))
    if abs(Ee - Ee_nom) > tol or abs(Ep - Ep_nom) > tol:
        raise RuntimeError(
            f"Shard {shard_dir} meta beams E_e={Ee}, E_p={Ep} do not match nominal "
            f"E_e={Ee_nom}, E_p={Ep_nom} (tol={tol}). Wrong label directory?"
        )


def collect_panel_counts(
    Ee_nom: float,
    Ep_nom: float,
    shard_label: str,
    max_events: int | None,
) -> tuple[np.ndarray, int, int]:
    """Histogram counts in ETA_BINS for the selected (x,Q) cell; plus n_ev and n_eta_in_range."""
    shards = list_shards(shard_label)
    if not shards:
        raise FileNotFoundError(
            f"No shards for {shard_label}. Generate with:\n"
            f"  python scripts/generation/generate_events_raw.py --labels {shard_label}"
        )
    _verify_shard_beams(shards[0], Ee_nom, Ep_nom)

    edges = np.linspace(ETA_RANGE[0], ETA_RANGE[1], ETA_BINS + 1)
    counts = np.zeros(ETA_BINS, dtype=np.int64)
    n_events_cell = 0
    n_eta_in_window = 0
    processed = 0

    for shard_path in shards:
        if max_events is not None and processed >= max_events:
            break
        data = load_shard(shard_path)
        e_in = data["event_e_in"]
        p_in = data["event_p_in"]
        e_sc = data["event_e_sc"]
        offsets = data["offsets"]
        pid = data["pid"]
        p4_arr = data["p4"]
        Ne = e_in.shape[0]

        for ie in range(Ne):
            if max_events is not None and processed >= max_events:
                break
            res = compute_x_Q_eta_one_event(
                e_in[ie], p_in[ie], e_sc[ie], offsets, pid, p4_arr, ie, FLIP_Z_ETA
            )
            if res is None:
                continue
            x, Q, eta = res
            ix = classify_x_bin_index(x)
            iq = classify_Q_bin_index(Q)
            if ix != IX_PANEL or iq != IQ_PANEL:
                continue

            e_in_ev = flip_z(np.asarray(e_in[ie], dtype=float), FLIP_Z_ETA)
            p_in_ev = flip_z(np.asarray(p_in[ie], dtype=float), FLIP_Z_ETA)
            if abs(float(e_in_ev[0]) - Ee_nom) > 0.05 or abs(float(p_in_ev[0]) - Ep_nom) > 0.05:
                raise RuntimeError(
                    f"Beam mismatch for {shard_label}: "
                    f"Ee={e_in_ev[0]}, Ep={p_in_ev[0]} vs nominal ({Ee_nom}, {Ep_nom})"
                )

            n_events_cell += 1
            processed += 1
            if ETA_RANGE[0] <= eta < ETA_RANGE[1]:
                n_eta_in_window += 1
                bin_idx = np.searchsorted(edges, eta, side="right") - 1
                if 0 <= bin_idx < ETA_BINS:
                    counts[bin_idx] += 1

    return counts, n_events_cell, n_eta_in_window


def density_from_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(ETA_RANGE[0], ETA_RANGE[1], ETA_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binw = np.diff(edges)
    tc = float(counts.sum())
    if tc <= 0:
        return centers, np.zeros_like(counts, dtype=float), binw
    dens = counts.astype(float) / (tc * binw)
    return centers, dens, binw


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PDF (default: analysis_outputs_dir / eta_hadron_valence_2to5Q_5x41_vs_9x275_overlay.pdf)",
    )
    p.add_argument("--max-events", type=int, default=None, help="Cap classified events per beam")
    p.add_argument(
        "--font-pt",
        type=float,
        default=UNIFORM_FONT_PT,
        metavar="PT",
        help=f"Single fontsize (points) for all text; default {UNIFORM_FONT_PT:g}",
    )
    args = p.parse_args()

    out = args.out
    if out is None:
        out = analysis_outputs_dir() / "eta_hadron_valence_2to5Q_5x41_vs_9x275_overlay.pdf"

    beams = [
        ((5, 41), ETA_XQ_LABEL_BY_BEAM[(5, 41)], "C0", r"$5\times 41$"),
        ((9, 275), ETA_XQ_LABEL_BY_BEAM[(9, 275)], "C3", r"$9\times 275$"),
    ]

    pt = float(args.font_pt)

    series = []
    ymax = 0.0
    for (Ee, Ep), label, color, leg in beams:
        counts, n_ev, n_eta = collect_panel_counts(Ee, Ep, label, args.max_events)
        centers, dens, _binw = density_from_counts(counts)
        series.append((leg, label, color, counts, n_ev, n_eta, centers, dens))
        if dens.size and dens.sum() > 0:
            ymax = max(ymax, float(np.max(dens)))
    if ymax <= 0:
        ymax = 1.0

    # Left-to-right EIC η bands: match EIC_REGIONS in analyze_events_raw
    region_band_styles = list(
        zip(
            EIC_REGIONS,
            ("tab:blue", "tab:green", "tab:red"),
            ("Central", "B0", "Forward"),
        )
    )

    # Figure inches scale lightly with font so layout stays balanced (vector PDF: pt is absolute).
    fs = max(pt / 12.0, 1.0)
    figsize = (9.0 * fs**0.35, 6.0 * fs**0.35)

    region_pt = pt * LEGEND_FONT_SCALE
    legend_kin_pt = region_pt * LEGEND_KIN_SHRINK
    font_prop_region = FontProperties(family="serif", size=region_pt)
    font_prop_legend_kin = FontProperties(family="serif", size=legend_kin_pt)

    with mpl.rc_context(_base_style_rc(pt)):
        fig, ax = plt.subplots(figsize=figsize)
        for leg, _label, color, counts, _n_ev, _n_eta, centers, dens in series:
            tc = int(counts.sum())
            if tc > 0:
                ax.step(
                    centers,
                    dens,
                    where="mid",
                    color=color,
                    linewidth=2.0 * (pt / 24.0),
                    label=leg,
                )
            else:
                ax.plot([], [], color=color, label=leg)

        for (eta_lo, eta_hi), band_color, name in region_band_styles:
            ax.axvspan(eta_lo, eta_hi, alpha=0.2, color=band_color, zorder=0)

        ax.set_xlim(ETA_RANGE[0], ETA_RANGE[1])
        ax.set_ylim(0.0, ymax * 1.08)
        ax.set_xlabel(r"$\eta_h$")
        ax.set_ylabel(r"$\dfrac{1}{\sigma} \dfrac{d\sigma}{d\eta_h}$")
        ax.grid(False)
        ax.tick_params(direction="in", axis="both", which="both", labelsize=pt)

        region_texts: list[Text] = []

        # Region labels (slightly larger than legend + kinematics after LEGEND_KIN_SHRINK)
        y_top = ymax * 1.08 * 0.97
        for (eta_lo, eta_hi), band_color, name in region_band_styles:
            xc = 0.5 * (eta_lo + eta_hi)
            rt = ax.text(
                xc,
                y_top,
                name,
                ha="center",
                va="top",
                color=band_color,
                fontproperties=font_prop_region,
                fontweight="medium",
                zorder=4,
            )
            region_texts.append(rt)

        # Horizontal center of the Forward η band → legend sits just under "Forward"
        _flo, _fhi = EIC_REGIONS[2]
        forward_eta_c = 0.5 * (_flo + _fhi)
        forward_x_axes = (forward_eta_c - ETA_RANGE[0]) / (ETA_RANGE[1] - ETA_RANGE[0])

        handles = list(ax.get_legend_handles_labels()[0])
        leg = ax.legend(
            handles=handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(forward_x_axes, 0.875),
            bbox_transform=ax.transAxes,
            borderaxespad=0.1,
            labelspacing=0.32,
            handletextpad=0.4,
            handlelength=1.75,
            columnspacing=0.8,
            prop=font_prop_legend_kin,
        )

        # Layout + font pass so legend bbox is correct; then stack kinematics under the legend, same left edge.
        plt.tight_layout(pad=0.4)
        _enforce_uniform_fontsize(fig, pt)
        _apply_secondary_fontsizes(ax, region_pt, legend_kin_pt, region_texts, None)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        gap_axes = 0.012
        try:
            tb = leg.get_tightbbox(renderer)
        except (AttributeError, TypeError):
            tb = None
        if tb is not None:
            bb_ax = tb.transformed(ax.transAxes.inverted())
            kin_x = float(bb_ax.x0)
            kin_y = float(bb_ax.y0) - gap_axes
        else:
            kin_x = max(forward_x_axes - 0.12, 0.02)
            kin_y = 0.75

        kin = ax.text(
            kin_x,
            max(kin_y, 0.04),
            r"$x > 0.05$" + "\n" + r"$5~\mathrm{GeV} > Q > 2~\mathrm{GeV}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="0.28",
            fontproperties=font_prop_legend_kin,
            zorder=5,
            clip_on=True,
        )

        _enforce_uniform_fontsize(fig, pt)
        _apply_secondary_fontsizes(ax, region_pt, legend_kin_pt, region_texts, kin)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
