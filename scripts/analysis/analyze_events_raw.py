#!/usr/bin/env python3
"""
Analyze raw PYTHIA shards (no PYTHIA). Reproduces:
  1. eta_hadron_EIC_hardware_QCD_regions.pdf
  2. pTrel_target_pion_wrt_remnant_axis_2D_fit_gaussian_comparison.pdf

Run from project root: python scripts/analysis/analyze_events_raw.py
"""
import sys
from pathlib import Path

# Ensure src is on path for diquark package (works without pip install)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.analyze_events_raw import run_eta_analysis_and_plot, run_ptrel_comparison_and_plot

if __name__ == "__main__":
    print("Analyzing raw shards -> eta_hadron_EIC_hardware_QCD_regions.pdf")
    run_eta_analysis_and_plot()
    print("Analyzing raw shards -> pTrel_target_pion_wrt_remnant_axis_2D_fit_gaussian_comparison.pdf")
    run_ptrel_comparison_and_plot()
    print("Done.")
