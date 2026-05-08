"""
unchanged_direct analysis artifact — v1 contract for jet–hadron transverse observables.

This schema is the minimal per-event record needed to compute the three transverse
observables defined in ``analyze_jet_hadron_transverse_observables.py`` (angle between
``pT_jet`` and ``pT_hadron``, ``|pT_jet + pT_h|``, ``|pT_jet - pT_h|``) **without**
calling PYTHIA again, **given** that ``k_out`` and the leading target pion are already
in the **same DIS Breit frame** used by the reinjection pipeline.

A. Proposed columns (v1, one row per event)
--------------------------------------------
  event_id                  int64
  arm                       string  # "unchanged" | "validation_reinject" | …
  source_lineage            string  # e.g. "full_event_record_breit" | "reinject_parton_csv"
  ok                        bool
  failure_reason            string  # "" if ok

  xB, Q2, Q                 float64 # DIS kinematics (same conventions as split_kinematics_extract)

  k_out_breit_E             float64 # struck outgoing quark, Breit frame, GeV
  k_out_breit_px            float64
  k_out_breit_py            float64
  k_out_breit_pz            float64

  pion_pdg                  int64 # ±211
  pion_breit_E              float64
  pion_breit_px             float64
  pion_breit_py             float64
  pion_breit_pz             float64

  obs_angle_rad             float64 # [0, pi] or NaN if below PT_MIN_ANGLE
  obs_sum_mag_GeV           float64 # |pT_jet + pT_h|
  obs_diff_mag_GeV          float64 # |pT_jet - pT_h|

  n_final_hadrons_used      int64   # count of final hadrons seen when selecting π (diagnostic)

B. Why each field exists
------------------------
  event_id — join key for validation vs direct, and for bookkeeping.
  arm — distinguishes bulk unchanged_direct from validation_reinject rows in combined studies.
  source_lineage — audit trail: proves whether row came from native full-event record or post-reinject scan.
  ok / failure_reason — same semantics as kinematics extract (beam/DIS/pion failures).
  xB, Q2, Q — DIS context for cuts and plotting; required to match existing analysis windows.
  k_out_breit_* — defines the jet axis for pT_jet = (px, py) in the transverse observables docstring.
  pion_breit_* — leading |pdg|==211 in target hemisphere (pz_breit>0), same rule as split_kinematics_extract.
  obs_* — the three published observables; stored so downstream only reads Parquet/CSV, not PYTHIA.
  n_final_hadrons_used — lightweight diagnostic when pion missing or samples differ.

C. Frame
--------
  All four-momenta with suffix ``_breit`` are in the **DIS Breit frame** used by
  ``generate_dis_isr_parton_dataset.py`` when ``use_breit`` is enabled (i.e. after the
  same lab→Breit map as the split / reinject pipeline with ``--csv-momenta-frame breit``).

D. Units and conventions
------------------------
  Momenta and energy: **GeV** (same as PYTHIA CSV exports).
  Q2: **GeV^2**; Q: **GeV**; xB: **dimensionless** Bjorken x.
  obs_angle_rad: radians in **[0, π]** (aligned → 0, back-to-back → π), NaN if either
  transverse vector has |pT| < PT_MIN_ANGLE (same threshold as analyze_jet_hadron_transverse_observables).

E. Layout
---------
  **One row per event** (flat, wide schema). Recommended on-disk format: **Parquet** with
  ``pyarrow`` (few columns, many rows — avoid millions of tiny files; shard later if needed).

F. Smallest set to run observable analysis without PYTHIA
---------------------------------------------------------
  Strict minimum to recompute the three observables from stored numbers only:
  ``k_out_breit_px``, ``k_out_breit_py``, ``pion_breit_px``, ``pion_breit_py``
  (plus PT_MIN_ANGLE rule). The full four-vectors and DIS variables are retained so the
  artifact stays aligned with the existing kinematics row and future cuts (e.g. x_L) can
  be added without changing the jet/pion definition.
"""

from __future__ import annotations

# Column order for Parquet/CSV writers (single source of truth)
UNCHANGED_DIRECT_JET_HADRON_V1_COLUMNS = (
    "event_id",
    "arm",
    "source_lineage",
    "ok",
    "failure_reason",
    "xB",
    "Q2",
    "Q",
    "k_out_breit_E",
    "k_out_breit_px",
    "k_out_breit_py",
    "k_out_breit_pz",
    "pion_pdg",
    "pion_breit_E",
    "pion_breit_px",
    "pion_breit_py",
    "pion_breit_pz",
    "obs_angle_rad",
    "obs_sum_mag_GeV",
    "obs_diff_mag_GeV",
    "n_final_hadrons_used",
)

__all__ = ["UNCHANGED_DIRECT_JET_HADRON_V1_COLUMNS"]
