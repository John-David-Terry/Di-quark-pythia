#!/usr/bin/env python3
"""
Two complementary analyses for ETA_XQ beams (5×41 and 9×275):

  --mode all-hadrons (default)
      Every stored hadron (|PDG|≥100) in every event → detector η bands.
      (“Hadron-weighted”; dominated by soft multiplicity.)

  --mode panel-events
      Same subsample as the overlay PDF top-right panel and η(x,Q) grids:
      events with **valence** x and **2<Q<5 GeV**, using the **single leading
      forward hadron η** per event (`compute_x_Q_eta_one_event`). Fractions are
      **percent of those panel events** whose η falls in each band — comparable
      to reading integrated probability mass from the η histogram / detector
      summaries like `Detectors.pdf`, **not** comparable to all-hadrons mode.

Run from project root, e.g.:
  python scripts/analysis/quantify_hadron_fractions_eic_regions.py
  python scripts/analysis/quantify_hadron_fractions_eic_regions.py --mode panel-events
  python scripts/analysis/quantify_hadron_fractions_eic_regions.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.analyze_events_raw import (  # noqa: E402
    EIC_REGIONS,
    ETA_XQ_LABEL_BY_BEAM,
    classify_Q_bin_index,
    classify_x_bin_index,
    compute_x_Q_eta_one_event,
    flip_z,
    is_hadron,
    list_shards,
    load_shard,
)
from diquark.paths import analysis_outputs_dir  # noqa: E402

# Match overlay plot for ETA_XQ samples
FLIP_Z_ETA = False

BEAMS: List[Tuple[float, float]] = [(5, 41), (9, 275)]
REGION_LABELS = ["Central", "B0", "Forward", "other"]

# Same cell as overlay `eta_hadron_*_xQ_regions_3x2.pdf` top-right panel
_PANEL_IX = 1  # Valence x > 0.05
_PANEL_IQ = 0  # 2 GeV < Q < 5 GeV


def _lab_eta_h(p4_row: np.ndarray) -> Optional[float]:
    """Lab-frame η_h from four-momentum (same as leading-hadron η in analyze_events_raw)."""
    v = flip_z(np.asarray(p4_row, dtype=float), FLIP_Z_ETA)
    px, py, pz = float(v[1]), float(v[2]), float(v[3])
    p_mag = float(np.sqrt(px * px + py * py + pz * pz))
    if p_mag <= 0:
        return None
    den = max(p_mag - pz, 1e-12)
    num = max(p_mag + pz, 1e-12)
    return float(0.5 * np.log(num / den))


def _classify_region(eta: float) -> str:
    """Assign η to one of Central / B0 / Forward / other (half-open [lo, hi))."""
    for name, (lo, hi) in zip(REGION_LABELS[:3], EIC_REGIONS):
        if lo <= eta < hi:
            return name
    return "other"


def _accumulate_label(shard_label: str, max_events: Optional[int]) -> Dict[str, int]:
    shards = list_shards(shard_label)
    counts = {k: 0 for k in REGION_LABELS}
    processed_events = 0
    for shard_path in shards:
        data = load_shard(shard_path)
        offsets = data["offsets"]
        pid = data["pid"]
        p4 = data["p4"]
        n_ev = data["event_e_in"].shape[0]
        for ie in range(n_ev):
            if max_events is not None and processed_events >= max_events:
                return counts
            start = int(offsets[ie])
            end = int(offsets[ie + 1])
            for j in range(start, end):
                if not is_hadron(int(pid[j])):
                    continue
                eta = _lab_eta_h(p4[j])
                if eta is None:
                    counts["other"] += 1
                    continue
                counts[_classify_region(eta)] += 1
            processed_events += 1
    return counts


def _accumulate_panel_events(
    shard_label: str,
    Ee_nom: float,
    Ep_nom: float,
    max_events: Optional[int],
) -> Tuple[Dict[str, int], int]:
    """
    Events passing valence + (2,5) GeV Q bin; one η per event (leading forward hadron).
    Returns (region_counts, n_events_in_panel).
    """
    shards = list_shards(shard_label)
    counts = {k: 0 for k in REGION_LABELS}
    n_in_panel = 0
    processed = 0
    for shard_path in shards:
        data = load_shard(shard_path)
        e_in = data["event_e_in"]
        p_in = data["event_p_in"]
        e_sc = data["event_e_sc"]
        offsets = data["offsets"]
        pid = data["pid"]
        p4_arr = data["p4"]
        n_ev = e_in.shape[0]
        for ie in range(n_ev):
            if max_events is not None and processed >= max_events:
                return counts, n_in_panel
            res = compute_x_Q_eta_one_event(
                e_in[ie], p_in[ie], e_sc[ie], offsets, pid, p4_arr, ie, FLIP_Z_ETA
            )
            processed += 1
            if res is None:
                continue
            x, Q, eta = res
            ix = classify_x_bin_index(x)
            iq = classify_Q_bin_index(Q)
            if ix != _PANEL_IX or iq != _PANEL_IQ:
                continue
            e_in_ev = flip_z(np.asarray(e_in[ie], dtype=float), FLIP_Z_ETA)
            p_in_ev = flip_z(np.asarray(p_in[ie], dtype=float), FLIP_Z_ETA)
            if abs(float(e_in_ev[0]) - Ee_nom) > 0.05 or abs(float(p_in_ev[0]) - Ep_nom) > 0.05:
                continue
            n_in_panel += 1
            counts[_classify_region(eta)] += 1
    return counts, n_in_panel


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("all-hadrons", "panel-events"),
        default="all-hadrons",
        help="all-hadrons: count every hadron; panel-events: leading η in valence, 2<Q<5 panel",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Stop after this many events per beam (full sample if omitted)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON only",
    )
    args = p.parse_args()

    results: Dict[str, Dict] = {}
    for Ee, Ep in BEAMS:
        key = f"{int(Ee)}x{int(Ep)}"
        lab = ETA_XQ_LABEL_BY_BEAM[(Ee, Ep)]
        if args.mode == "all-hadrons":
            cts = _accumulate_label(lab, args.max_events)
            n_tot = sum(cts.values())
            pct = {k: (100.0 * cts[k] / n_tot if n_tot else 0.0) for k in REGION_LABELS}
            results[key] = {
                "mode": "all-hadrons",
                "beam_GeV": [Ee, Ep],
                "shard_label": lab,
                "hadron_counts": cts,
                "hadron_fraction_percent": {k: round(pct[k], 4) for k in REGION_LABELS},
                "n_hadrons_total": n_tot,
                "eta_bands_eta_units": {
                    "Central": {"eta_min": EIC_REGIONS[0][0], "eta_max": EIC_REGIONS[0][1]},
                    "B0": {"eta_min": EIC_REGIONS[1][0], "eta_max": EIC_REGIONS[1][1]},
                    "Forward": {"eta_min": EIC_REGIONS[2][0], "eta_max": EIC_REGIONS[2][1]},
                },
                "note": (
                    "Fractions over all stored hadrons per event in shards (no PYTHIA status in numpy)."
                ),
            }
        else:
            cts, n_panel = _accumulate_panel_events(lab, float(Ee), float(Ep), args.max_events)
            n_tot = n_panel
            pct = {k: (100.0 * cts[k] / n_tot if n_tot else 0.0) for k in REGION_LABELS}
            results[key] = {
                "mode": "panel-events",
                "beam_GeV": [Ee, Ep],
                "shard_label": lab,
                "selection": {
                    "x_bin": "Valence (x > 0.05)",
                    "Q_bin_GeV": "2 < Q < 5",
                    "eta_definition": "leading forward hadron lab η (same as eta_hadron_*_xQ_regions PDF)",
                },
                "event_counts_in_panel": cts,
                "event_fraction_percent": {k: round(pct[k], 4) for k in REGION_LABELS},
                "n_events_in_panel": n_tot,
                "eta_bands_eta_units": {
                    "Central": {"eta_min": EIC_REGIONS[0][0], "eta_max": EIC_REGIONS[0][1]},
                    "B0": {"eta_min": EIC_REGIONS[1][0], "eta_max": EIC_REGIONS[1][1]},
                    "Forward": {"eta_min": EIC_REGIONS[2][0], "eta_max": EIC_REGIONS[2][1]},
                },
                "note": (
                    "Denominator = events in this (x,Q) cell only; numerator = where that event’s "
                    "leading-hadron η falls."
                ),
            }

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("EIC η bands (half-open [η_min, η_max)):")
    for name, (lo, hi) in zip(REGION_LABELS[:3], EIC_REGIONS):
        print(f"  {name}: {lo} ≤ η < {hi}")
    print(f"  other: η outside the three bands above\n")

    if args.mode == "all-hadrons":
        print("Mode: **all hadrons** (lab η per particle)\n")
        for Ee, Ep in BEAMS:
            key = f"{int(Ee)}x{int(Ep)}"
            r = results[key]
            print(f"--- {key} GeV ({r['shard_label']}) — N_hadrons = {r['n_hadrons_total']:,} ---")
            for k in REGION_LABELS:
                n = r["hadron_counts"][k]
                pc = r["hadron_fraction_percent"][k]
                print(f"  {k:8s}  {n:12,}  ({pc:.4f}%)")
            print()
        out_json = analysis_outputs_dir() / "hadron_fractions_eic_eta_bands_5x41_9x275.json"
    else:
        print(
            "Mode: **panel events** — Valence, 2<Q<5 GeV; one leading-hadron η per event "
            "(matches overlay top-right panel)\n"
        )
        for Ee, Ep in BEAMS:
            key = f"{int(Ee)}x{int(Ep)}"
            r = results[key]
            print(
                f"--- {key} GeV ({r['shard_label']}) — "
                f"N_events in panel = {r['n_events_in_panel']:,} ---"
            )
            for k in REGION_LABELS:
                n = r["event_counts_in_panel"][k]
                pc = r["event_fraction_percent"][k]
                print(f"  {k:8s}  {n:12,}  ({pc:.4f}%)")
            print()
        out_json = analysis_outputs_dir() / "event_fractions_eic_panel_valence_q2to5_5x41_9x275.json"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_json.resolve()}")


if __name__ == "__main__":
    main()
