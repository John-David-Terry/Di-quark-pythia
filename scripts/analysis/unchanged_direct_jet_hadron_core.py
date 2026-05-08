#!/usr/bin/env python3
"""
Build jet–hadron transverse observable rows from either:
  (1) full DIS event record (Breit, post-hadronization) — unchanged_direct, no PYTHIA
  (2) parton-level split CSV + reinject — validation_reinject / current pipeline
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_ANAL = Path(__file__).resolve().parent
for _p in (_ROOT / "src", _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diquark.analyze_events_raw import is_hadron

from jet_hadron_observables_split_pi_pm import (  # noqa: E402
    extract_beams_arrays,
    Qmax_ptrel,
    Qmin_ptrel,
    resolve_struck_index,
    xmax_ptrel,
    xmin_ptrel,
)

# ---------------------------------------------------------------------------
# Jet–hadron helpers (same binning conventions as analyze_jet_hadron_transverse_observables)
# ---------------------------------------------------------------------------
PT_MIN_ANGLE = 1e-6


def _load_jet_hadron_transverse_mod():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "analysis" / "analyze_jet_hadron_transverse_observables.py"
    spec = importlib.util.spec_from_file_location("_jh_tr", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_JH = _load_jet_hadron_transverse_mod()


def transverse_three_from_k_out_and_pion_breit(
    k_out_breit: np.ndarray,
    pion_breit: np.ndarray,
) -> Tuple[float, float, float]:
    """Return (obs_angle_rad or nan, obs_sum_mag_GeV, obs_diff_mag_GeV)."""
    pT_j = _JH._pT_vec_breit(k_out_breit)
    pT_h = _JH._pT_vec_breit(pion_breit)
    ang = _JH._angle_between_2d(pT_j, pT_h)
    if ang is None:
        ang_f = float("nan")
    else:
        ang_f = float(ang)
    sum_mag = float(np.linalg.norm(pT_j + pT_h))
    diff_mag = float(np.linalg.norm(pT_j - pT_h))
    return ang_f, sum_mag, diff_mag


def _final_hadrons_tuples_from_full_event_df(df: pd.DataFrame) -> List[Tuple[int, np.ndarray]]:
    """Final hadrons from Breit-frame full-event rows (isFinal==1)."""
    out: List[Tuple[int, np.ndarray]] = []
    for _, r in df.iterrows():
        if int(r["isFinal"]) != 1:
            continue
        pid = int(r["pdg_id"])
        if not is_hadron(pid):
            continue
        p4 = np.array(
            [float(r["E"]), float(r["px"]), float(r["py"]), float(r["pz"])],
            dtype=np.float64,
        )
        out.append((pid, p4))
    return out


def _import_split_kinematics():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "analysis" / "split_kinematics_extract.py"
    spec = importlib.util.spec_from_file_location("_ske", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_ske = _import_split_kinematics()


def row_from_full_event_breit(
    df_event: pd.DataFrame,
    event_id: int,
    md_map: Optional[pd.DataFrame],
    arm: str,
    source_lineage: str,
) -> Dict[str, Any]:
    """
    df_event: all rows for one event_id from dis_isr_full_event_record (Breit momenta).
    """
    row_base: Dict[str, Any] = {
        "event_id": int(event_id),
        "arm": arm,
        "source_lineage": source_lineage,
        "ok": False,
        "failure_reason": "",
        "xB": float("nan"),
        "Q2": float("nan"),
        "Q": float("nan"),
        "k_out_breit_E": float("nan"),
        "k_out_breit_px": float("nan"),
        "k_out_breit_py": float("nan"),
        "k_out_breit_pz": float("nan"),
        "pion_pdg": 0,
        "pion_breit_E": float("nan"),
        "pion_breit_px": float("nan"),
        "pion_breit_py": float("nan"),
        "pion_breit_pz": float("nan"),
        "obs_angle_rad": float("nan"),
        "obs_sum_mag_GeV": float("nan"),
        "obs_diff_mag_GeV": float("nan"),
        "n_final_hadrons_used": 0,
    }

    pdg = df_event["pdg_id"].to_numpy()
    status = df_event["status"].to_numpy()
    is_final = df_event["isFinal"].to_numpy()
    E = df_event["E"].to_numpy(dtype=np.float64)
    px = df_event["px"].to_numpy(dtype=np.float64)
    py = df_event["py"].to_numpy(dtype=np.float64)
    pz = df_event["pz"].to_numpy(dtype=np.float64)
    pidx = df_event["particle_index"].to_numpy()

    e_in, e_sc, p_in, beam_msg = extract_beams_arrays(
        pdg, status, is_final, E, px, py, pz
    )
    if e_in is None:
        row_base["failure_reason"] = beam_msg
        return row_base

    meta_path = None
    idx, _ = resolve_struck_index(df_event, int(event_id), meta_path, md_map)
    if idx is None:
        row_base["failure_reason"] = "struck_unresolved"
        return row_base

    k_hits = np.flatnonzero(pidx == idx)
    if k_hits.size != 1:
        row_base["failure_reason"] = "k_out_row"
        return row_base
    ik = int(k_hits[0])
    k_out = np.array([E[ik], px[ik], py[ik], pz[ik]], dtype=np.float64)

    e_in_ev, e_sc_ev, p_in_ev = e_in, e_sc, p_in
    k_out_ev = k_out

    qmu = e_in_ev - e_sc_ev
    Q2 = -(qmu[0] * qmu[0] - qmu[1] * qmu[1] - qmu[2] * qmu[2] - qmu[3] * qmu[3])
    if Q2 <= 0:
        row_base["failure_reason"] = "Q2"
        return row_base
    Q = float(np.sqrt(Q2))
    p_dot_q = float(
        p_in_ev[0] * qmu[0]
        - p_in_ev[1] * qmu[1]
        - p_in_ev[2] * qmu[2]
        - p_in_ev[3] * qmu[3]
    )
    if p_dot_q == 0:
        row_base["failure_reason"] = "pdotq"
        return row_base
    xB = Q2 / (2.0 * p_dot_q)
    if not (xmin_ptrel <= xB <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel):
        row_base["failure_reason"] = "xQ_window"
        return row_base

    hadrons = _final_hadrons_tuples_from_full_event_df(df_event)
    row_base["n_final_hadrons_used"] = len(hadrons)
    pi_pid, pi_p4 = _ske.leading_target_pion_breit(hadrons)
    if pi_pid is None or pi_p4 is None:
        row_base["failure_reason"] = "no_pion_candidate"
        row_base["xB"] = float(xB)
        row_base["Q2"] = float(Q2)
        row_base["Q"] = float(Q)
        return row_base

    k_out_breit = k_out_ev
    ang, sm, dm = transverse_three_from_k_out_and_pion_breit(k_out_breit, pi_p4)

    row_base.update(
        {
            "ok": True,
            "failure_reason": "",
            "xB": float(xB),
            "Q2": float(Q2),
            "Q": float(Q),
            "k_out_breit_E": float(k_out_breit[0]),
            "k_out_breit_px": float(k_out_breit[1]),
            "k_out_breit_py": float(k_out_breit[2]),
            "k_out_breit_pz": float(k_out_breit[3]),
            "pion_pdg": int(pi_pid),
            "pion_breit_E": float(pi_p4[0]),
            "pion_breit_px": float(pi_p4[1]),
            "pion_breit_py": float(pi_p4[2]),
            "pion_breit_pz": float(pi_p4[3]),
            "obs_angle_rad": float(ang),
            "obs_sum_mag_GeV": float(sm),
            "obs_diff_mag_GeV": float(dm),
        }
    )
    return row_base


def row_from_parton_csv_reinject_fixed(
    df_parton: pd.DataFrame,
    event_id: int,
    md_map: Optional[pd.DataFrame],
    meta_path: Optional[Path],
    sample: str,
    pythia,
    arm: str,
    source_lineage: str,
) -> Dict[str, Any]:
    idx, _ = resolve_struck_index(df_parton, int(event_id), meta_path, md_map)
    row_base: Dict[str, Any] = {
        "event_id": int(event_id),
        "arm": arm,
        "source_lineage": source_lineage,
        "ok": False,
        "failure_reason": "",
        "xB": float("nan"),
        "Q2": float("nan"),
        "Q": float("nan"),
        "k_out_breit_E": float("nan"),
        "k_out_breit_px": float("nan"),
        "k_out_breit_py": float("nan"),
        "k_out_breit_pz": float("nan"),
        "pion_pdg": 0,
        "pion_breit_E": float("nan"),
        "pion_breit_px": float("nan"),
        "pion_breit_py": float("nan"),
        "pion_breit_pz": float("nan"),
        "obs_angle_rad": float("nan"),
        "obs_sum_mag_GeV": float("nan"),
        "obs_diff_mag_GeV": float("nan"),
        "n_final_hadrons_used": 0,
    }
    if idx is None:
        row_base["failure_reason"] = "struck_unresolved"
        return row_base

    rec = _ske.extract_one_event(
        df_parton,
        int(idx),
        sample,
        0,
        pythia,
        True,
        timings=None,
    )
    if not rec.get("ok", False):
        row_base["failure_reason"] = str(rec.get("failure_reason") or rec.get("reason") or "")
        return row_base

    k_out_breit = np.array(
        [
            rec["k_out_breit_E"],
            rec["k_out_breit_px"],
            rec["k_out_breit_py"],
            rec["k_out_breit_pz"],
        ],
        dtype=np.float64,
    )
    pi_p4 = np.array(
        [
            rec["pion_breit_E"],
            rec["pion_breit_px"],
            rec["pion_breit_py"],
            rec["pion_breit_pz"],
        ],
        dtype=np.float64,
    )
    ang, sm, dm = transverse_three_from_k_out_and_pion_breit(k_out_breit, pi_p4)
    row_base.update(
        {
            "ok": True,
            "failure_reason": "",
            "xB": float(rec["xB"]),
            "Q2": float(rec["Q2"]),
            "Q": float(rec["Q"]),
            "k_out_breit_E": float(k_out_breit[0]),
            "k_out_breit_px": float(k_out_breit[1]),
            "k_out_breit_py": float(k_out_breit[2]),
            "k_out_breit_pz": float(k_out_breit[3]),
            "pion_pdg": int(rec.get("pion_pdg", 0)),
            "pion_breit_E": float(pi_p4[0]),
            "pion_breit_px": float(pi_p4[1]),
            "pion_breit_py": float(pi_p4[2]),
            "pion_breit_pz": float(pi_p4[3]),
            "obs_angle_rad": float(ang),
            "obs_sum_mag_GeV": float(sm),
            "obs_diff_mag_GeV": float(dm),
            "n_final_hadrons_used": int(rec.get("n_final_hadrons", 0)),
        }
    )
    return row_base
