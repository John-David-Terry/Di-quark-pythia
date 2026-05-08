#!/usr/bin/env python3
"""
Validate one altered event CSV (+ sibling .meta.json) from split_90_10/altered/.
Prints per-check lines and overall PASS/FAIL.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

DEFAULT_ALTERED_DIR = outputs_dir() / "dis_isr_parton_dataset" / "split_90_10" / "altered"

TOL_FRAC = 0.02
TOL_PX = 1e-5
TOL_ONSHELL = 1e-3


def _mag(px: float, py: float, pz: float) -> float:
    return math.sqrt(px * px + py * py + pz * pz)


def _onshell(E: float, px: float, py: float, pz: float, m: float) -> float:
    return E * E - px * px - py * py - pz * pz - m * m


def inspect_one(csv_path: Path, meta: Dict[str, Any]) -> Tuple[bool, List[str]]:
    lines: List[str] = []
    ok_all = True
    df = pd.read_csv(csv_path).sort_values("particle_index")

    orig_dq_idx = int(meta["original_diquark_index"])
    ch = str(meta["split_channel"])
    exp_strip = int(meta["new_particle_pdgs"]["stripped"])
    exp_aq = int(meta["new_particle_pdgs"]["antiquark"])
    exp_dau = int(meta["new_particle_pdgs"]["daughter_diquark"])
    i_s = int(meta["stripped_quark_index"])
    i_aq = int(meta["created_antiquark_index"])
    i_d = int(meta["daughter_diquark_index"])
    i_st = int(meta["struck_quark_index"])
    delta = float(meta["kick_delta_px_gev"])
    tag_a = int(meta["color_tag_A"])
    A = tag_a

    # 1) Original diquark row removed (index must not appear)
    if orig_dq_idx in set(df["particle_index"].astype(int).values):
        lines.append("FAIL: original diquark particle_index still present in CSV")
        ok_all = False
    else:
        lines.append("PASS: original diquark row removed (index absent)")

    # 2) Three replacement particles
    row_s = df[df["particle_index"] == i_s]
    row_aq = df[df["particle_index"] == i_aq]
    row_d = df[df["particle_index"] == i_d]
    if len(row_s) != 1 or len(row_aq) != 1 or len(row_d) != 1:
        lines.append("FAIL: could not find stripped / antiquark / daughter rows by index")
        return False, lines

    ps = int(row_s.iloc[0]["pdg_id"])
    pa = int(row_aq.iloc[0]["pdg_id"])
    pdg_d = int(row_d.iloc[0]["pdg_id"])
    if ps == exp_strip and pa == exp_aq and pdg_d == exp_dau:
        lines.append(f"PASS: three-body PDGs match channel {ch} ({exp_strip}, {exp_dau}, {exp_aq})")
    else:
        lines.append(f"FAIL: PDGs got ({ps}, {pdg_d}, {pa}), expected ({exp_strip}, {exp_dau}, {exp_aq})")
        ok_all = False

    rs = row_s.iloc[0]
    ra = row_aq.iloc[0]
    rd = row_d.iloc[0]

    # 3) Momentum fraction magnitudes
    ms = _mag(float(rs["px"]), float(rs["py"]), float(rs["pz"]))
    ma = _mag(float(ra["px"]), float(ra["py"]), float(ra["pz"]))
    md = _mag(float(rd["px"]), float(rd["py"]), float(rd["pz"]))
    if "original_diquark_px" not in meta:
        lines.append("SKIP: meta missing original_diquark_px/py/pz (re-run split script for this check)")
    else:
        odx = float(meta["original_diquark_px"])
        ody = float(meta["original_diquark_py"])
        odz = float(meta["original_diquark_pz"])
        p_dq = _mag(odx, ody, odz)
        if p_dq < 1e-9:
            lines.append("FAIL: |p_dq| ~ 0")
            ok_all = False
        else:
            r0, r1, r2 = ms / p_dq, ma / p_dq, md / p_dq
            ok3 = (
                abs(r0 - 0.5) < TOL_FRAC
                and abs(r1 - 0.05) < TOL_FRAC
                and abs(r2 - 0.45) < TOL_FRAC
            )
            if ok3:
                lines.append(f"PASS: |p| fractions ~ 0.5/0.05/0.45 (got {r0:.4f}, {r1:.4f}, {r2:.4f})")
            else:
                lines.append(f"FAIL: |p| fractions {r0:.4f}, {r1:.4f}, {r2:.4f} vs 0.5, 0.05, 0.45")
                ok_all = False

    # 4) Kick on px
    px_b_s = float(meta.get("stripped_px_before_kick", float("nan")))
    px_b_st = float(meta.get("struck_px_before_kick", float("nan")))
    if math.isnan(px_b_s) or math.isnan(px_b_st):
        lines.append("SKIP: no stripped_px_before_kick / struck_px_before_kick in meta (regenerate split)")
    else:
        px_s = float(rs["px"])
        row_st = df[df["particle_index"] == i_st].iloc[0]
        px_st = float(row_st["px"])
        ds = px_s - px_b_s
        dst = px_st - px_b_st
        if abs(ds - delta) < TOL_PX and abs(dst + delta) < TOL_PX:
            lines.append(f"PASS: kick px (stripped +{ds:.6g}, struck {dst:.6g}) ~= ±{delta}")
        else:
            lines.append(f"FAIL: kick expected ±{delta}, got stripped Δpx={ds}, struck Δpx={dst}")
            ok_all = False

    # 5) Color tag A
    nc_a = sum(int(r["col"]) == A for _, r in df.iterrows())
    na_a = sum(int(r["acol"]) == A for _, r in df.iterrows())
    if nc_a == 1 and na_a == 1:
        lines.append(f"PASS: color A={A} appears once as col and once as acol")
    else:
        lines.append(f"FAIL: color A={A} n_col={nc_a} n_acol={na_a}")
        ok_all = False

    # 6) On-shell
    bad = []
    for lab, row in ("stripped", rs), ("antiquark", ra), ("daughter", rd), ("struck", df[df["particle_index"] == i_st].iloc[0]):
        E = float(row["E"])
        px, py, pz, m = float(row["px"]), float(row["py"]), float(row["pz"]), float(row["m"])
        inv = _onshell(E, px, py, pz, m)
        if abs(inv) > TOL_ONSHELL:
            bad.append(f"{lab} inv2={inv}")
    if not bad:
        lines.append("PASS: on-shell (|E^2-p^2-m^2|<=tol) for stripped, antiquark, daughter, struck")
    else:
        lines.append("FAIL: on-shell " + "; ".join(bad))
        ok_all = False

    return ok_all, lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect one altered event CSV.")
    ap.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=None,
        help="Path to event_XXXXXX.csv (default: first altered/*.csv in --altered-dir)",
    )
    ap.add_argument(
        "--altered-dir",
        type=Path,
        default=DEFAULT_ALTERED_DIR,
        help="Directory containing altered event_*.csv",
    )
    ap.add_argument("--all", action="store_true", help="Run on every event_*.csv in altered-dir")
    args = ap.parse_args()

    altered_dir: Path = args.altered_dir
    if not altered_dir.is_dir():
        raise SystemExit(f"altered dir not found: {altered_dir}")

    if args.all:
        paths = sorted(altered_dir.glob("event_*.csv"))
    elif args.csv is not None:
        paths = [args.csv]
    else:
        paths = sorted(altered_dir.glob("event_*.csv"))[:1]
        if not paths:
            raise SystemExit("No event_*.csv in altered dir; pass a path explicitly.")

    n_pass = 0
    for csv_path in paths:
        meta_path = csv_path.with_suffix(".meta.json")
        if not meta_path.exists():
            print(f"{csv_path.name}: SKIP (no {meta_path.name})")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ok, lines = inspect_one(csv_path, meta)
        print(f"\n--- {csv_path.name} (event_id={meta.get('original_event_id')}) ---")
        for ln in lines:
            print(f"  {ln}")
        print(f"  >>> {'PASS' if ok else 'FAIL'}")
        if ok:
            n_pass += 1

    if len(paths) > 1:
        print(f"\n=== summary: {n_pass}/{len(paths)} PASS ===")


if __name__ == "__main__":
    main()
