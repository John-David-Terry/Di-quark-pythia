#!/usr/bin/env python3
"""
Per-event jet–hadron transverse observables from DIS **final-state-only** Parquet.

Computes the same three Breit-frame quantities as ``unchanged_direct_jet_hadron_core`` /
``analyze_jet_hadron_transverse_observables``:

  * ``obs_angle_rad`` — azimuthal angle between ``pT_jet`` and ``pT_hadron`` in ``[0, pi]``
  * ``obs_sum_mag_GeV`` — ``|pT_jet + pT_hadron|``
  * ``obs_diff_mag_GeV`` — ``|pT_jet - pT_hadron|``

Hadron: leading ``|pdg|==211`` in the target hemisphere ``pz_Breit > 0`` (same as
``split_kinematics_extract.leading_target_pion_breit``).

Jet (struck outgoing quark proxy):

  * **altered_reinject** — exact ``k_out`` from editable parton Parquet (Breit), joined on
    ``event_id`` to ``final_state_v1`` hadrons after reinjection.
  * **background** — legacy Parquet has **no** per-event struck quark. With
    ``--background-jet lo_collinear_qt0``, we build an **LO collinear** proxy jet
    (``q_T=0`` in the flipped-lab frame). With ``--background-jet events_k_out`` and
    ``final_state_v3`` (or legacy ``final_state_v2``) events shards that contain ``k_out_breit_*``,
    the jet uses the **true**
    outgoing struck-quark four-vector stored at generation time (symmetric with altered).

Example (defaults match typical ``~/Data`` layout)::

  python scripts/analysis/produce_dis_final_state_jet_hadron_transverse.py \\
    --out-dir ~/Data/dis_jet_hadron_from_final_state_v1 \\
    --mode both --pythia-seed 12345

Requires: numpy, pandas, pyarrow, tqdm, pythia8 (pythia8 only for beam sampling in
background proxy mode).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tqdm is required; install from requirements.txt") from exc

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
_ANAL = Path(__file__).resolve().parent
for _p in (_SRC, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diquark.analyze_events_raw import (  # noqa: E402
    FLIP_Z_PTREL,
    Qmax_ptrel,
    Qmin_ptrel,
    build_LT,
    flip_z,
    is_hadron,
    xmax_ptrel,
    xmin_ptrel,
)
from diquark.final_state_parquet import (  # noqa: E402
    events_table_has_k_out_breit,
    k_out_breit_four_vector_from_events_row,
    k_out_breit_is_valid,
)
from diquark.paths import write_run_manifest  # noqa: E402
from generate_dis_background_final_state_parquet import build_pythia_background  # noqa: E402
from generate_dis_isr_parton_dataset import extract_beams_from_event  # noqa: E402
from jet_hadron_observables_split_pi_pm import find_outgoing_struck_quark_noisr  # noqa: E402
from unchanged_direct_jet_hadron_core import (  # noqa: E402
    _import_split_kinematics,
    transverse_three_from_k_out_and_pion_breit,
)
from unchanged_direct_schema import UNCHANGED_DIRECT_JET_HADRON_V1_COLUMNS  # noqa: E402

_ske = _import_split_kinematics()
leading_target_pion_breit = _ske.leading_target_pion_breit

DATASET_SUBDIR = "jet_hadron_transverse_v1"


def _empty_row(
    event_id: int,
    arm: str,
    source_lineage: str,
    failure_reason: str,
) -> Dict[str, Any]:
    return {
        "event_id": int(event_id),
        "arm": arm,
        "source_lineage": source_lineage,
        "ok": False,
        "failure_reason": failure_reason,
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


def _hadrons_from_final_particles(g: pd.DataFrame) -> List[Tuple[int, np.ndarray]]:
    out: List[Tuple[int, np.ndarray]] = []
    for _, r in g.iterrows():
        pid = int(r["pdg_id"])
        if not is_hadron(pid):
            continue
        p4 = np.array(
            [float(r["E"]), float(r["px"]), float(r["py"]), float(r["pz"])],
            dtype=np.float64,
        )
        out.append((pid, p4))
    return out


def _sample_flipped_beams(seed: int, max_tries: int = 50_000) -> Tuple[np.ndarray, np.ndarray, float, float]:
    try:
        import pythia8
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pythia8 is required for background jet proxy beam sampling") from exc

    p = build_pythia_background(int(seed))
    ev = p.event
    for _ in range(max_tries):
        if not p.next():
            continue
        beams = extract_beams_from_event(ev)
        if beams is None:
            continue
        e_in, _e_sc, p_in = beams
        e_in_ev = flip_z(e_in, FLIP_Z_PTREL)
        p_in_ev = flip_z(p_in, FLIP_Z_PTREL)
        Ee = float(e_in_ev[0])
        Ep = float(p_in_ev[0])
        if Ee <= 0 or Ep <= 0:
            continue
        return e_in_ev, p_in_ev, Ee, Ep
    raise RuntimeError(f"could not extract beams after {max_tries} tries (seed={seed})")


def _solve_q_collinear_flip(
    p_in_ev: np.ndarray,
    Q2: np.ndarray,
    xB: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each event, find (nu, qz) with q=(nu,0,0,qz) in the flipped-lab frame such that
    q^2 = -Q2 and 2 P·q = Q2/x (metric +---).
    Returns (nu, qz, ok) as float arrays; ok is 1 where a real root was chosen.
    """
    Ep = float(p_in_ev[0])
    pz = float(p_in_ev[3])
    rhs = Q2 / (2.0 * xB)
    nu = np.full_like(Q2, np.nan, dtype=np.float64)
    qz = np.full_like(Q2, np.nan, dtype=np.float64)
    ok = np.zeros_like(Q2, dtype=np.int8)

    m = np.isfinite(Q2) & np.isfinite(xB) & (Q2 > 0) & (xB > 0) & np.isfinite(rhs)
    idxs = np.flatnonzero(m)
    inv_Ep = 1.0 / Ep
    inv_Ep2 = inv_Ep * inv_Ep
    a = (pz * pz) * inv_Ep2 - 1.0

    for i in idxs:
        rhs_i = float(rhs[i])
        Q2_i = float(Q2[i])
        b = 2.0 * pz * rhs_i * inv_Ep2
        c = (rhs_i * rhs_i) * inv_Ep2 + Q2_i
        if abs(a) < 1e-22:
            if abs(b) < 1e-22:
                continue
            qz_i = -c / b
            nu_i = (rhs_i + pz * qz_i) * inv_Ep
        else:
            disc = b * b - 4.0 * a * c
            if disc < 0:
                continue
            s = math.sqrt(disc)
            qz_a = (-b - s) / (2.0 * a)
            qz_b = (-b + s) / (2.0 * a)
            nu_a = (rhs_i + pz * qz_a) * inv_Ep
            nu_b = (rhs_i + pz * qz_b) * inv_Ep
            # Pick the root with larger |nu| (heuristic; both can be kinematically valid)
            if abs(nu_b) > abs(nu_a):
                qz_i, nu_i = qz_b, nu_b
            else:
                qz_i, nu_i = qz_a, nu_a
        chk = nu_i * nu_i - qz_i * qz_i
        if not np.isfinite(chk) or abs(chk + Q2_i) > 1e-2 * max(1.0, Q2_i):
            continue
        nu[i] = nu_i
        qz[i] = qz_i
        ok[i] = 1
    return nu, qz, ok


def _emit_jh_row(rows: List[Dict[str, Any]], progress: Any, row: Dict[str, Any]) -> None:
    rows.append(row)
    if progress is not None:
        progress.update(1)


def _process_background_shard(
    particles_df: pd.DataFrame,
    events_df: pd.DataFrame,
    e_in_ev: np.ndarray,
    p_in_ev: np.ndarray,
    Ee: float,
    Ep: float,
    S: float,
    source_lineage: str,
    apply_xQ_window: bool,
    *,
    progress: Any = None,
    max_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    parts_map = {int(k): v for k, v in particles_df.groupby("event_id", sort=False)}

    ev_sorted = events_df.sort_values("event_id")
    eids = ev_sorted["event_id"].to_numpy(dtype=np.int64)
    Q2a = ev_sorted["Q2"].to_numpy(dtype=np.float64)
    xBa = ev_sorted["xB"].to_numpy(dtype=np.float64)

    y_a = Q2a / (S * xBa)
    nu_a, qz_a, q_ok = _solve_q_collinear_flip(p_in_ev, Q2a, xBa)

    for i in range(len(eids)):
        if max_rows is not None and len(rows) >= max_rows:
            break
        eid = int(eids[i])
        Q2 = float(Q2a[i])
        xB = float(xBa[i])
        base = _empty_row(eid, "background", source_lineage, "")

        g = parts_map.get(eid)
        if g is None or g.empty:
            base["failure_reason"] = "missing_particles"
            _emit_jh_row(rows, progress, base)
            continue

        hadrons = _hadrons_from_final_particles(g)
        base["n_final_hadrons_used"] = len(hadrons)
        pi_pid, pi_p4 = leading_target_pion_breit(hadrons)
        if pi_pid is None or pi_p4 is None:
            base["failure_reason"] = "no_pion_candidate"
            base["Q2"] = Q2
            base["xB"] = xB
            base["Q"] = float(np.sqrt(Q2)) if Q2 > 0 else float("nan")
            _emit_jh_row(rows, progress, base)
            continue

        if not (np.isfinite(Q2) and np.isfinite(xB) and Q2 > 0 and xB > 0):
            base["failure_reason"] = "bad_Q2_xB"
            _emit_jh_row(rows, progress, base)
            continue

        Q = float(np.sqrt(Q2))
        if apply_xQ_window and (
            not (xmin_ptrel <= xB <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel)
        ):
            base["failure_reason"] = "xQ_window"
            base["xB"] = xB
            base["Q2"] = Q2
            base["Q"] = Q
            _emit_jh_row(rows, progress, base)
            continue

        if int(q_ok[i]) != 1:
            base["failure_reason"] = "q_collinear_solve_fail"
            base["xB"] = xB
            base["Q2"] = Q2
            base["Q"] = Q
            _emit_jh_row(rows, progress, base)
            continue

        nu = float(nu_a[i])
        qz = float(qz_a[i])
        qmu = np.array([nu, 0.0, 0.0, qz], dtype=np.float64)
        y = float(y_a[i])
        if not (np.isfinite(y) and y > 0):
            base["failure_reason"] = "bad_y"
            base["xB"] = xB
            base["Q2"] = Q2
            base["Q"] = Q
            _emit_jh_row(rows, progress, base)
            continue

        LT = build_LT(Ee, Ep, qmu, float(xB), y, 0.0, 0.0, S)
        if LT is None:
            base["failure_reason"] = "build_LT_fail"
            base["xB"] = xB
            base["Q2"] = Q2
            base["Q"] = Q
            _emit_jh_row(rows, progress, base)
            continue

        k_ev = xB * p_in_ev + qmu
        k_out_breit = LT @ k_ev

        ang, sm, dm = transverse_three_from_k_out_and_pion_breit(k_out_breit, pi_p4)
        _emit_jh_row(
            rows,
            progress,
            {
                "event_id": eid,
                "arm": "background",
                "source_lineage": source_lineage,
                "ok": True,
                "failure_reason": "",
                "xB": xB,
                "Q2": Q2,
                "Q": Q,
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
                "n_final_hadrons_used": len(hadrons),
            },
        )
    return rows


def _process_background_shard_events_k_out(
    particles_df: pd.DataFrame,
    events_df: pd.DataFrame,
    source_lineage: str,
    apply_xQ_window: bool,
    *,
    progress: Any = None,
    max_rows: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Background jet from ``k_out_breit_*`` on the events row (``final_state_v3`` / v2 schema)."""
    rows: List[Dict[str, Any]] = []
    parts_map = {int(k): v for k, v in particles_df.groupby("event_id", sort=False)}

    ev_sorted = events_df.sort_values("event_id")
    eids = ev_sorted["event_id"].to_numpy(dtype=np.int64)

    for i in range(len(eids)):
        if max_rows is not None and len(rows) >= max_rows:
            break
        eid = int(eids[i])
        er = ev_sorted.iloc[i]
        base = _empty_row(eid, "background", source_lineage, "")

        g = parts_map.get(eid)
        if g is None or g.empty:
            base["failure_reason"] = "missing_particles"
            _emit_jh_row(rows, progress, base)
            continue

        hadrons = _hadrons_from_final_particles(g)
        base["n_final_hadrons_used"] = len(hadrons)
        pi_pid, pi_p4 = leading_target_pion_breit(hadrons)
        if pi_pid is None or pi_p4 is None:
            base["failure_reason"] = "no_pion_candidate"
            base["Q2"] = float(er.get("Q2", float("nan")))
            base["xB"] = float(er.get("xB", float("nan")))
            Q2 = base["Q2"]
            base["Q"] = float(np.sqrt(Q2)) if np.isfinite(Q2) and Q2 > 0 else float("nan")
            _emit_jh_row(rows, progress, base)
            continue

        Q2 = float(er.get("Q2", float("nan")))
        xB = float(er.get("xB", float("nan")))
        if not (np.isfinite(Q2) and np.isfinite(xB) and Q2 > 0 and xB > 0):
            base["failure_reason"] = "bad_Q2_xB"
            _emit_jh_row(rows, progress, base)
            continue

        Q = float(np.sqrt(Q2))
        if apply_xQ_window and (
            not (xmin_ptrel <= xB <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel)
        ):
            base["failure_reason"] = "xQ_window"
            base["xB"] = xB
            base["Q2"] = Q2
            base["Q"] = Q
            _emit_jh_row(rows, progress, base)
            continue

        if not k_out_breit_is_valid(er):
            base["failure_reason"] = "missing_k_out_breit"
            base["xB"] = xB
            base["Q2"] = Q2
            base["Q"] = Q
            _emit_jh_row(rows, progress, base)
            continue

        k_out_breit = k_out_breit_four_vector_from_events_row(er)
        ang, sm, dm = transverse_three_from_k_out_and_pion_breit(k_out_breit, pi_p4)
        _emit_jh_row(
            rows,
            progress,
            {
                "event_id": eid,
                "arm": "background",
                "source_lineage": source_lineage,
                "ok": True,
                "failure_reason": "",
                "xB": xB,
                "Q2": Q2,
                "Q": Q,
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
                "n_final_hadrons_used": len(hadrons),
            },
        )
    return rows


def _struck_parton_row(df_ev: pd.DataFrame, ev_meta: pd.Series) -> Tuple[Optional[pd.Series], str]:
    so = int(ev_meta["struck_outgoing_index"])
    if so > 0:
        hit = df_ev[df_ev["particle_index"] == so]
        if hit.empty:
            return None, "struck_outgoing_row_missing"
        return hit.iloc[0], "struck_outgoing_index"
    si = int(ev_meta["struck_incoming_index"])
    reco = find_outgoing_struck_quark_noisr(df_ev.sort_values("particle_index"), si)
    if not reco.get("success"):
        return None, str(reco.get("failure_reason") or "struck_reco_fail")
    sel = reco.get("selected_row")
    if sel is None:
        return None, "struck_reco_none"
    return sel, str(reco.get("selection_mode") or "reco")


def _load_success_event_ids(altered_root: Path) -> set:
    mdir = altered_root / "altered_metadata"
    out: set = set()
    if not mdir.is_dir():
        return out
    for pq in sorted(mdir.glob("shard_*.parquet")):
        df = pd.read_parquet(pq, columns=["event_id", "alteration_succeeded"])
        sub = df[df["alteration_succeeded"] == 1]
        if sub.empty:
            continue
        out.update(int(x) for x in sub["event_id"].tolist())
    return out


def _build_event_index(parent: Path, sub: str) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """Load all particle groups and concat event tables from ``<parent>/<sub>/``."""
    base = parent / sub
    pdir = base / "particles"
    edir = base / "events"
    parts: Dict[int, pd.DataFrame] = {}
    ev_frames: List[pd.DataFrame] = []
    for pq in sorted(pdir.glob("shard_*.parquet")):
        pdf = pd.read_parquet(pq)
        for eid, g in pdf.groupby("event_id", sort=False):
            parts[int(eid)] = g
    for eq in sorted(edir.glob("shard_*.parquet")):
        ev_frames.append(pd.read_parquet(eq))
    ev_all = pd.concat(ev_frames, ignore_index=True) if ev_frames else pd.DataFrame()
    return parts, ev_all


def _process_altered(
    altered_particles: Dict[int, pd.DataFrame],
    altered_events: pd.DataFrame,
    final_particles: Dict[int, pd.DataFrame],
    final_events: pd.DataFrame,
    success_ids: set,
    source_lineage: str,
    max_events: int,
    apply_xQ_window: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    fe = final_events.set_index("event_id")
    ae = altered_events.set_index("event_id")

    ids = [int(i) for i in fe.index.tolist() if int(i) in success_ids]
    ids.sort()
    if max_events > 0:
        ids = ids[:max_events]

    for eid in tqdm(ids, desc="altered_reinject", unit="ev", mininterval=0.3):
        base = _empty_row(eid, "altered_reinject", source_lineage, "")
        if eid not in ae.index:
            base["failure_reason"] = "missing_altered_event_row"
            rows.append(base)
            continue
        if eid not in final_particles:
            base["failure_reason"] = "missing_final_state_particles"
            rows.append(base)
            continue
        if eid not in altered_particles:
            base["failure_reason"] = "missing_altered_partons"
            rows.append(base)
            continue

        ev_f = fe.loc[eid]
        if isinstance(ev_f, pd.DataFrame):
            ev_f = ev_f.iloc[0]
        Q2 = float(ev_f.get("Q2", float("nan")))
        xB = float(ev_f.get("xB", float("nan")))

        g_fin = final_particles[eid]
        hadrons = _hadrons_from_final_particles(g_fin)
        base["n_final_hadrons_used"] = len(hadrons)
        pi_pid, pi_p4 = leading_target_pion_breit(hadrons)
        if pi_pid is None or pi_p4 is None:
            base["failure_reason"] = "no_pion_candidate"
            base["Q2"] = Q2
            base["xB"] = xB
            base["Q"] = float(np.sqrt(Q2)) if Q2 > 0 else float("nan")
            rows.append(base)
            continue

        df_alt = altered_particles[eid]
        meta = ae.loc[eid]
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]
        prow, _mode = _struck_parton_row(df_alt, meta)
        if prow is None:
            base["failure_reason"] = "struck_unresolved"
            base["Q2"] = Q2
            base["xB"] = xB
            base["Q"] = float(np.sqrt(Q2)) if Q2 > 0 else float("nan")
            rows.append(base)
            continue

        k_out_breit = np.array(
            [
                float(prow["E"]),
                float(prow["px"]),
                float(prow["py"]),
                float(prow["pz"]),
            ],
            dtype=np.float64,
        )

        if not (np.isfinite(Q2) and np.isfinite(xB) and Q2 > 0 and xB > 0):
            base["failure_reason"] = "bad_Q2_xB"
            rows.append(base)
            continue
        Q = float(np.sqrt(Q2))
        if apply_xQ_window and (
            not (xmin_ptrel <= xB <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel)
        ):
            base["failure_reason"] = "xQ_window"
            base["xB"] = xB
            base["Q2"] = Q2
            base["Q"] = Q
            rows.append(base)
            continue

        ang, sm, dm = transverse_three_from_k_out_and_pion_breit(k_out_breit, pi_p4)
        rows.append(
            {
                "event_id": eid,
                "arm": "altered_reinject",
                "source_lineage": source_lineage,
                "ok": True,
                "failure_reason": "",
                "xB": xB,
                "Q2": Q2,
                "Q": Q,
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
                "n_final_hadrons_used": len(hadrons),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Jet–hadron transverse rows from final-state Parquet.")
    ap.add_argument(
        "--mode",
        choices=("background", "altered", "both"),
        default="both",
        help="Which population to process.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help=f"Parent directory; writes {DATASET_SUBDIR}/rows.parquet and run_summary.json",
    )
    ap.add_argument(
        "--background-root",
        type=Path,
        default=Path.home() / "Data" / "dis_isr_background_final_state",
        help="Parent directory containing the final-state dataset folder.",
    )
    ap.add_argument(
        "--background-final-state-subdir",
        type=str,
        default="final_state_v1",
        help="Dataset folder under --background-root (e.g. final_state_v3 for k_out).",
    )
    ap.add_argument(
        "--altered-input-parent",
        type=Path,
        default=Path.home() / "Data" / "dis_isr_editable_altered_100k",
        help="Parent containing altered_100k_parquet/ (partons + metadata).",
    )
    ap.add_argument(
        "--altered-reinject-parent",
        type=Path,
        default=Path.home() / "Data" / "dis_isr_altered_reinject_100k",
        help="Parent of final_state_v1 for altered reinject hadrons.",
    )
    ap.add_argument(
        "--background-jet",
        choices=("lo_collinear_qt0", "events_k_out", "skip"),
        default="lo_collinear_qt0",
        help="Jet definition for background: LO proxy, events-table k_out (v3 Parquet), or skip.",
    )
    ap.add_argument(
        "--pythia-seed",
        type=int,
        default=12345,
        help="Seed for beam sampling (should match background generation for best consistency).",
    )
    ap.add_argument(
        "--skip-xQ-window",
        action="store_true",
        help="Do not apply the pTrel analysis Q/x window; fill observables whenever pion+jets resolve.",
    )
    ap.add_argument(
        "--max-background-events",
        type=int,
        default=0,
        help="If >0, stop after this many background rows (debug).",
    )
    ap.add_argument(
        "--max-altered-events",
        type=int,
        default=0,
        help="If >0, cap altered_reinject events processed (debug).",
    )
    args = ap.parse_args()

    out_root = args.out_dir.expanduser().resolve()
    out_ds = out_root / DATASET_SUBDIR
    out_ds.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    apply_xQ = not bool(args.skip_xQ_window)
    summary: Dict[str, Any] = {
        "mode": args.mode,
        "background_jet": args.background_jet,
        "apply_xQ_window": apply_xQ,
        "out_dir": str(out_root),
        "n_background_rows": 0,
        "n_altered_rows": 0,
        "background_ok": 0,
        "altered_ok": 0,
    }

    if args.mode in ("background", "both") and args.background_jet != "skip":
        fs = args.background_root.expanduser().resolve() / str(args.background_final_state_subdir)
        if not fs.is_dir():
            raise SystemExit(f"missing background dataset: {fs}")

        manifest_path = fs / "manifest.parquet"
        if not manifest_path.is_file():
            raise SystemExit(f"missing {manifest_path}")

        man = pd.read_parquet(manifest_path)
        ev_shards = sorted(
            fs / str(r["shard_path"])
            for _, r in man.iterrows()
            if str(r["dataset"]) == "events"
        )
        p_shards = sorted(
            fs / str(r["shard_path"])
            for _, r in man.iterrows()
            if str(r["dataset"]) == "particles"
        )
        if len(ev_shards) != len(p_shards):
            raise SystemExit("manifest events/particles shard count mismatch")

        use_events_k_out = args.background_jet == "events_k_out"
        if use_events_k_out:
            probe = pd.read_parquet(ev_shards[0])
            if not events_table_has_k_out_breit(probe):
                raise SystemExit(
                    f"--background-jet events_k_out requires k_out columns in events Parquet; "
                    f"missing in {ev_shards[0]}"
                )
            bg_lineage = "background_final_state_v3_events_table_k_out_breit"
        else:
            bg_lineage = "background_final_state_v1_lo_collinear_qt0_jet_proxy"
            e_in_ev, p_in_ev, Ee, Ep = _sample_flipped_beams(int(args.pythia_seed))
            S = 4.0 * Ee * Ep

        n_bg_cap = int(args.max_background_events)
        n_bg_done = 0
        stop_bg = False
        bg_total_events = int(man.loc[man["dataset"] == "events", "n_events"].sum())
        pbar_total = min(n_bg_cap, bg_total_events) if n_bg_cap > 0 else bg_total_events
        event_pbar = tqdm(
            total=max(1, pbar_total),
            desc="Building jet–hadron rows",
            unit="evt",
            mininterval=0.3,
            smoothing=0.05,
        )
        try:
            for esp, psp in zip(ev_shards, p_shards):
                if stop_bg:
                    break
                if not esp.is_file() or not psp.is_file():
                    raise FileNotFoundError(f"missing shard pair {esp} / {psp}")
                remain: Optional[int] = None
                if n_bg_cap > 0:
                    remain = n_bg_cap - n_bg_done
                    if remain <= 0:
                        stop_bg = True
                        break
                edf = pd.read_parquet(esp)
                pdf = pd.read_parquet(psp)
                if use_events_k_out:
                    shard_rows = _process_background_shard_events_k_out(
                        pdf, edf, bg_lineage, apply_xQ, progress=event_pbar, max_rows=remain
                    )
                else:
                    shard_rows = _process_background_shard(
                        pdf,
                        edf,
                        e_in_ev,
                        p_in_ev,
                        Ee,
                        Ep,
                        S,
                        bg_lineage,
                        apply_xQ,
                        progress=event_pbar,
                        max_rows=remain,
                    )
                all_rows.extend(shard_rows)
                n_bg_done += len(shard_rows)
                if n_bg_cap > 0 and n_bg_done >= n_bg_cap:
                    stop_bg = True
        finally:
            event_pbar.close()

        summary["n_background_rows"] = sum(1 for r in all_rows if r["arm"] == "background")
        summary["background_ok"] = sum(1 for r in all_rows if r["arm"] == "background" and r["ok"])
    elif args.mode in ("background", "both") and args.background_jet == "skip":
        summary["n_background_rows"] = 0

    if args.mode in ("altered", "both"):
        alt_root = args.altered_input_parent.expanduser().resolve() / "altered_100k_parquet"
        rein_root = args.altered_reinject_parent.expanduser().resolve() / "final_state_v1"
        if not alt_root.is_dir():
            raise SystemExit(f"missing altered parton root: {alt_root}")
        if not rein_root.is_dir():
            raise SystemExit(f"missing altered reinject root: {rein_root}")

        success_ids = _load_success_event_ids(alt_root)
        if not success_ids:
            raise SystemExit(f"no alteration_succeeded events under {alt_root / 'altered_metadata'}")

        tqdm.write("Loading altered parton Parquet into memory (particle groups + events)…")
        alt_parts, alt_ev = _build_event_index(alt_root.parent, "altered_100k_parquet")
        tqdm.write("Loading altered reinject final-state Parquet…")
        fin_parts, fin_ev = _build_event_index(rein_root.parent, "final_state_v1")

        alt_lineage = "altered_reinject_join_editable_parton_k_out_breit"
        alt_rows = _process_altered(
            alt_parts,
            alt_ev,
            fin_parts,
            fin_ev,
            success_ids,
            alt_lineage,
            int(args.max_altered_events),
            apply_xQ,
        )
        all_rows.extend(alt_rows)
        summary["n_altered_rows"] = len(alt_rows)
        summary["altered_ok"] = sum(1 for r in alt_rows if r["ok"])

    if not all_rows:
        raise SystemExit("No rows produced (check --mode and inputs).")

    out_df = pd.DataFrame(all_rows)
    out_df = out_df[[c for c in UNCHANGED_DIRECT_JET_HADRON_V1_COLUMNS if c in out_df.columns]]
    out_path = out_ds / "rows.parquet"
    out_df.to_parquet(out_path, index=False, compression="snappy")

    summary["n_total_rows"] = int(len(out_df))
    summary["n_ok_total"] = int(out_df["ok"].sum()) if "ok" in out_df.columns else 0
    if "failure_reason" in out_df.columns:
        bad = out_df[~out_df["ok"]]
        summary["failure_reason_counts"] = (
            bad["failure_reason"].value_counts().head(50).to_dict() if len(bad) else {}
        )

    summ_path = out_ds / "run_summary.json"
    summ_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {len(out_df)} rows -> {out_path}")

    write_run_manifest(
        run_label="dis_final_state_jet_hadron_transverse",
        script_name="produce_dis_final_state_jet_hadron_transverse.py",
        top_level_dirs_written=[str(out_ds)],
        approximate_files_created=2,
        extra={"summary_path": str(summ_path), "rows_path": str(out_path)},
    )


if __name__ == "__main__":
    main()
