"""Helpers for DIS final-state Parquet (background) with optional per-event struck-quark k_out."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

EVENTS_K_OUT_BREIT_COLUMNS: tuple[str, ...] = (
    "k_out_breit_px",
    "k_out_breit_py",
    "k_out_breit_pz",
    "k_out_breit_E",
)


def events_table_has_k_out_breit(events_df: pd.DataFrame) -> bool:
    """True if ``events_df`` has all Breit-frame k_out columns (may be NaN in rows)."""
    cols = events_df.columns
    return all(c in cols for c in EVENTS_K_OUT_BREIT_COLUMNS)


def k_out_breit_four_vector_from_events_row(row: pd.Series) -> np.ndarray:
    """
    Return k_out as ``[E, px, py, pz]`` (same ordering as parton / jet-hadron producers).
    Non-finite components yield NaNs in the array.
    """
    return np.array(
        [
            float(row["k_out_breit_E"]),
            float(row["k_out_breit_px"]),
            float(row["k_out_breit_py"]),
            float(row["k_out_breit_pz"]),
        ],
        dtype=np.float64,
    )


def k_out_breit_is_valid(row: pd.Series, cols: Iterable[str] = EVENTS_K_OUT_BREIT_COLUMNS) -> bool:
    """True if all k_out columns are finite."""
    try:
        return all(np.isfinite(float(row[c])) for c in cols)
    except (KeyError, TypeError, ValueError):
        return False
