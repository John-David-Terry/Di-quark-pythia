#!/usr/bin/env python3
"""
Diagnostic: compare pion selection (pz_Breit>0) vs same-sign-as-transformed-proton Breit pz.

Uses the same PYTHIA setup, acceptance, LT, flip_z, and final-state collection as v3 production.
Does not change physics settings.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
_ANAL = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_ANAL) not in sys.path:
    sys.path.insert(0, str(_ANAL))

from diquark.analyze_events_raw import FLIP_Z_PTREL, flip_z  # noqa: E402

from generate_dis_background_final_state_parquet import (  # noqa: E402
    build_pythia_background,
    collect_final_state_breit,
)
from generate_dis_isr_parton_dataset import (  # noqa: E402
    extract_beams_from_event,
    hard_subprocess_outgoing_quark_lab_p4_and_index,
    pick_incoming_quark_index,
    try_build_lt_from_event,
)


def _theta_transverse_rad(
    j_px: float, j_py: float, h_px: float, h_py: float
) -> Tuple[float, float]:
    """Opening angle in transverse plane: theta = arccos(pT_J·pT_h / (|pT_J||pT_h|)). Returns (theta_rad, cos_theta)."""
    jt = math.hypot(j_px, j_py)
    ht = math.hypot(h_px, h_py)
    if jt <= 0.0 or ht <= 0.0:
        return float("nan"), float("nan")
    c = (j_px * h_px + j_py * h_py) / (jt * ht)
    c = max(-1.0, min(1.0, c))
    return float(math.acos(c)), float(c)


def _pick_pion_a(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Tuple[float, Dict[str, Any]]] = None
    for r in rows:
        if abs(int(r["pdg_id"])) != 211:
            continue
        pz = float(r["pz"])
        if pz <= 0.0:
            continue
        E = float(r["E"])
        if best is None or E > best[0]:
            best = (E, r)
    return best[1] if best else None


def _pick_pion_b(rows: List[Dict[str, Any]], p_in_breit_pz: float) -> Optional[Dict[str, Any]]:
    if p_in_breit_pz == 0.0:
        return None
    best: Optional[Tuple[float, Dict[str, Any]]] = None
    for r in rows:
        if abs(int(r["pdg_id"])) != 211:
            continue
        pz = float(r["pz"])
        if pz * p_in_breit_pz <= 0.0:
            continue
        E = float(r["E"])
        if best is None or E > best[0]:
            best = (E, r)
    return best[1] if best else None


def _pion_csv_fields(prefix: str, r: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if r is None:
        return {
            f"{prefix}_pdg": float("nan"),
            f"{prefix}_breit_E": float("nan"),
            f"{prefix}_breit_px": float("nan"),
            f"{prefix}_breit_py": float("nan"),
            f"{prefix}_breit_pz": float("nan"),
        }
    return {
        f"{prefix}_pdg": int(r["pdg_id"]),
        f"{prefix}_breit_E": float(r["E"]),
        f"{prefix}_breit_px": float(r["px"]),
        f"{prefix}_breit_py": float(r["py"]),
        f"{prefix}_breit_pz": float(r["pz"]),
    }


def _sign_str(x: float) -> str:
    if not math.isfinite(x):
        return "nan"
    if x > 0:
        return "+"
    if x < 0:
        return "-"
    return "0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-accept", type=int, default=100, help="Target number of accepted events")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--first-print", type=int, default=20, help="Print compact table for first N accepted")
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=_PROJECT_ROOT / "outputs" / "debug_breit_hemisphere_check" / "debug_100_events.csv",
    )
    args = ap.parse_args()

    n_target = int(args.n_accept)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    import pythia8

    pythia = build_pythia_background(int(args.seed), hadron_level=True)
    ev = pythia.event

    rows_out: List[Dict[str, Any]] = []
    accepted = 0
    total_tried = 0

    while accepted < n_target:
        if not pythia.next():
            continue
        total_tried += 1

        inc_idx = pick_incoming_quark_index(ev)
        if inc_idx is None:
            continue
        if abs(int(ev[inc_idx].id())) != 2:
            continue

        proc = pythia.process
        p4_lab, oq = hard_subprocess_outgoing_quark_lab_p4_and_index(proc)
        if p4_lab is None or oq < 0:
            continue

        LT = try_build_lt_from_event(ev)
        if LT is None:
            continue

        beams = extract_beams_from_event(ev)
        if beams is None:
            continue
        _e_in, _e_sc, p_in_lab = beams
        p4_p_breit = LT @ flip_z(np.asarray(p_in_lab, dtype=np.float64), FLIP_Z_PTREL)
        p_in_breit_pz = float(p4_p_breit[3])

        p4_j_breit = LT @ flip_z(np.asarray(p4_lab, dtype=np.float64), FLIP_Z_PTREL)
        k_out_breit_E = float(p4_j_breit[0])
        k_out_breit_px = float(p4_j_breit[1])
        k_out_breit_py = float(p4_j_breit[2])
        k_out_breit_pz = float(p4_j_breit[3])

        parts = collect_final_state_breit(ev, LT, accepted)
        pion_a = _pick_pion_a(parts)
        pion_b = _pick_pion_b(parts, p_in_breit_pz)

        rec: Dict[str, Any] = {
            "event_id": int(accepted),
            "p_in_breit_pz": p_in_breit_pz,
            "k_out_breit_pz": k_out_breit_pz,
            "k_out_breit_px": k_out_breit_px,
            "k_out_breit_py": k_out_breit_py,
            "k_out_breit_E": k_out_breit_E,
        }

        rec.update(_pion_csv_fields("pionA", pion_a))
        rec.update(_pion_csv_fields("pionB", pion_b))

        if pion_a is not None:
            th_a, c_a = _theta_transverse_rad(
                k_out_breit_px, k_out_breit_py, float(pion_a["px"]), float(pion_a["py"])
            )
        else:
            th_a, c_a = float("nan"), float("nan")
        if pion_b is not None:
            th_b, c_b = _theta_transverse_rad(
                k_out_breit_px, k_out_breit_py, float(pion_b["px"]), float(pion_b["py"])
            )
        else:
            th_b, c_b = float("nan"), float("nan")

        rec["theta_A"] = th_a
        rec["theta_B"] = th_b
        rec["cos_theta_A"] = c_a
        rec["cos_theta_B"] = c_b

        same = False
        if pion_a is not None and pion_b is not None:
            if int(pion_a["pdg_id"]) == int(pion_b["pdg_id"]):
                va = np.array(
                    [float(pion_a["E"]), float(pion_a["px"]), float(pion_a["py"]), float(pion_a["pz"])],
                    dtype=np.float64,
                )
                vb = np.array(
                    [float(pion_b["E"]), float(pion_b["px"]), float(pion_b["py"]), float(pion_b["pz"])],
                    dtype=np.float64,
                )
                same = bool(np.allclose(va, vb, rtol=1e-9, atol=1e-9))
        rec["same_pion_AB"] = int(same)

        rows_out.append(rec)
        accepted += 1

    df = pd.DataFrame(rows_out)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows, {total_tried} generator attempts)")

    n = len(df)
    fp = min(int(args.first_print), n)
    print("\n=== First {} accepted events (compact) ===".format(fp))
    print(
        "ev_id | sgn(p_pz) | sgn(k_pz) | sgn(piA_pz) | sgn(piB_pz) | theta_A | theta_B"
    )
    for i in range(fp):
        r = df.iloc[i]
        print(
            f"{int(r['event_id']):5d} | {_sign_str(float(r['p_in_breit_pz'])):^9} | "
            f"{_sign_str(float(r['k_out_breit_pz'])):^9} | {_sign_str(float(r['pionA_breit_pz'])):^11} | "
            f"{_sign_str(float(r['pionB_breit_pz'])):^11} | "
            f"{r['theta_A']:7.4f} | {r['theta_B']:7.4f}"
        )

    # --- Summary ---
    f_pp = float(np.mean(df["p_in_breit_pz"].to_numpy(dtype=np.float64) > 0)) if n else float("nan")
    f_kn = float(np.mean(df["k_out_breit_pz"].to_numpy(dtype=np.float64) < 0)) if n else float("nan")

    n_a = int(np.isfinite(df["pionA_breit_E"].to_numpy(dtype=np.float64)).sum())
    n_b = int(np.isfinite(df["pionB_breit_E"].to_numpy(dtype=np.float64)).sum())
    f_same = float(np.mean(df["same_pion_AB"].to_numpy() == 1)) if n else float("nan")

    def stats(col: str) -> Tuple[float, float]:
        x = df[col].to_numpy(dtype=np.float64)
        m = np.isfinite(x)
        if not m.any():
            return float("nan"), float("nan")
        return float(np.mean(x[m])), float(np.median(x[m]))

    mean_a, med_a = stats("theta_A")
    mean_b, med_b = stats("theta_B")
    mean_ca, med_ca = stats("cos_theta_A")
    mean_cb, med_cb = stats("cos_theta_B")

    cA = df["cos_theta_A"].to_numpy(dtype=np.float64)
    cB = df["cos_theta_B"].to_numpy(dtype=np.float64)
    mA = np.isfinite(cA)
    mB = np.isfinite(cB)

    print("\n=== Summary ===")
    print(f"1. Fraction with p_in_breit_pz > 0:     {f_pp:.4f}")
    print(f"2. Fraction with k_out_breit_pz < 0:    {f_kn:.4f}")
    print(f"3. Events with Selection A pion found:  {n_a} / {n}")
    print(f"4. Events with Selection B pion found:  {n_b} / {n}")
    print(f"5. Fraction A and B same pion (4-mom):  {f_same:.4f}")
    print("6. Mean | median:")
    print(f"   theta_A:   {mean_a:.6f} | {med_a:.6f}")
    print(f"   theta_B:   {mean_b:.6f} | {med_b:.6f}")
    print(f"   cos_th_A:  {mean_ca:.6f} | {med_ca:.6f}")
    print(f"   cos_th_B:  {mean_cb:.6f} | {med_cb:.6f}")
    print("7. cos_theta counts (finite only):")
    print(f"   cos_theta_A > 0: {int((cA > 0)[mA].sum())}   cos_theta_A < 0: {int((cA < 0)[mA].sum())}")
    print(f"   cos_theta_B > 0: {int((cB > 0)[mB].sum())}   cos_theta_B < 0: {int((cB < 0)[mB].sum())}")
    print(f"\nGenerator attempts until {n} accepted: {total_tried}")


if __name__ == "__main__":
    main()
