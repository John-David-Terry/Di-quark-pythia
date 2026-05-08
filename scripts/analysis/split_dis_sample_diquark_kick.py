#!/usr/bin/env python3
"""
Split DIS parton-level CSV events into 90% unchanged vs 10% altered.

Altered events: replace final (ud) diquark (PDG 2101) with a 3-body split
``(ud) -> [d] + (us) + sbar`` (PYTHIA ``su_0`` diquark PDG 3201),
then apply balanced px kicks to the stripped [d] and the outgoing struck quark only.

We want event-by-event comparisons between kicked and unkicked events after PYTHIA
reinjects and hadronizes them; this script prepares paired hard-event CSVs (unchanged
vs altered) from the same source sample.

No junctions, no LHE, no full history replay.

Companion tools: print_split_90_10_summary.py, inspect_altered_event.py, reinject_altered_events.py

**Parquet (preferred for large samples):** ``split_dis_sample_diquark_kick_parquet.py`` reads
``editable_source_v1`` shards and writes ``split_90_10_parquet/`` (see ``split_output_parquet.py``).
This CSV driver remains for legacy workflows.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import count_files_under, outputs_dir, write_run_manifest

OUTDIR = outputs_dir() / "dis_isr_parton_dataset"
FULL_EVENT_CSV = OUTDIR / "dis_isr_full_event_record.csv"
METADATA_CSV = OUTDIR / "dis_isr_event_metadata.csv"

# Output root: two subdirs unchanged/ and altered/
DEFAULT_SPLIT_ROOT = OUTDIR / "split_90_10"

M_LIGHT = 0.33
DIQUARK_UD0 = 2101
# (u,s) diquark in PYTHIA naming order (``su_0``); same flavor content as "(us)".
DIQUARK_SU0 = 3201
TOL_3M = 1e-6
TOL_ONSHELL = 1e-3


def _load_dis_helpers():
    path = Path(__file__).resolve().parent / "modify_dis_isr_parton_dataset.py"
    spec = importlib.util.spec_from_file_location("dis_modify", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_DIS = _load_dis_helpers()
find_outgoing_struck_quark_noisr = _DIS.find_outgoing_struck_quark_noisr


_pythia_pd = None


def _pythia_mass0(pid: int) -> float:
    global _pythia_pd
    try:
        import pythia8
    except ImportError:
        ap = abs(pid)
        return {2101: 0.57933, 3201: 0.80473, 1103: 0.57933}.get(ap, M_LIGHT)
    if _pythia_pd is None:
        p = pythia8.Pythia()
        p.readString("Print:quiet = on")
        p.init()
        _pythia_pd = p.particleData
    return float(_pythia_pd.m0(abs(pid)))


def mass_for_pid(pid: int) -> float:
    ap = abs(pid)
    if ap in (1, 2, 3):
        return M_LIGHT
    if 1000 <= ap < 10000:
        return _pythia_mass0(pid)
    return M_LIGHT


def e_on_shell(px: float, py: float, pz: float, m: float) -> float:
    return math.sqrt(max(0.0, px * px + py * py + pz * pz + m * m))


def kinematic_extras(px: float, py: float, pz: float, E: float) -> Tuple[float, float, float, float]:
    pT = math.hypot(px, py)
    if pT < 1e-14:
        eta = math.copysign(1e6, pz) if pz != 0 else 0.0
    else:
        eta = math.asinh(pz / pT)
    phi = math.atan2(py, px)
    m = math.sqrt(max(0.0, E * E - px * px - py * py - pz * pz))
    return pT, eta, phi, m


def find_final_diquark_kick_partner_row(df_event: pd.DataFrame) -> Optional[pd.Series]:
    """Prefer final ud0 (2101); else highest-E final diquark-like (1000 <= |pdg| < 10000)."""
    r0 = find_final_ud0_diquark(df_event)
    if r0 is not None:
        return r0
    cand = df_event[(df_event["isFinal"] == 1) & (df_event["pdg_id"].abs() >= 1000) & (df_event["pdg_id"].abs() < 10000)]
    if cand.empty:
        return None
    return cand.loc[cand["E"].idxmax()]


def find_final_ud0_diquark(df_event: pd.DataFrame) -> Optional[pd.Series]:
    cand = df_event[(df_event["isFinal"] == 1) & (df_event["pdg_id"] == DIQUARK_UD0)]
    if cand.empty:
        return None
    if len(cand) == 1:
        return cand.iloc[0]
    idxs = set(int(x) for x in cand["particle_index"].values)
    if 32 in idxs:
        return cand[cand["particle_index"] == 32].iloc[0]
    return cand.loc[cand["E"].idxmax()]


def next_free_color_tag(df_event: pd.DataFrame) -> int:
    used = set()
    for _, r in df_event.iterrows():
        c, a = int(r["col"]), int(r["acol"])
        if c > 0:
            used.add(c)
        if a > 0:
            used.add(a)
    return (max(used) + 1) if used else 501


def validate_surgery_color_tags(df: pd.DataFrame, tag_a: int, tag_b: int) -> Tuple[bool, str]:
    """
    Full DIS CSVs often have multiple lines with the same col tag (e.g. several col=101).
    We require the *new* string A to appear exactly once as col and once as acol.
    For B we only check that at least one particle still carries acol=B (the created
    antiquark), matching the moved diquark endpoint.
    """
    nc_a = sum(int(r["col"]) == tag_a for _, r in df.iterrows())
    na_a = sum(int(r["acol"]) == tag_a for _, r in df.iterrows())
    if nc_a != 1 or na_a != 1:
        return False, f"tag A ({tag_a}): n_col={nc_a} n_acol={na_a}"
    na_b = sum(int(r["acol"]) == tag_b for _, r in df.iterrows())
    if na_b < 1:
        return False, f"tag B ({tag_b}): missing acol endpoint"
    return True, "ok"


def channel_ab(ch: str) -> Tuple[int, int, int]:
    """(stripped_id, daughter_diquark_id, antiquark_id)."""
    if ch == "C":
        return 1, DIQUARK_SU0, -3  # (ud) -> d + (us-like su_0) + sbar
    raise ValueError(f"unknown split channel {ch!r} (expected 'C')")


def build_three_body_from_diquark(
    dq_row: pd.Series, channel: str
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    px, py, pz = float(dq_row["px"]), float(dq_row["py"]), float(dq_row["pz"])
    f0, f1, f2 = 0.5, 0.05, 0.45
    ps = (f0 * px, f0 * py, f0 * pz)
    pa = (f1 * px, f1 * py, f1 * pz)
    pd = (f2 * px, f2 * py, f2 * pz)
    sid, did, aid = channel_ab(channel)
    ms, mdq, ma = mass_for_pid(sid), mass_for_pid(did), mass_for_pid(aid)
    Es = e_on_shell(ps[0], ps[1], ps[2], ms)
    Ea = e_on_shell(pa[0], pa[1], pa[2], ma)
    Ed = e_on_shell(pd[0], pd[1], pd[2], mdq)
    return (
        _row_kin(sid, ps[0], ps[1], ps[2], Es, ms),
        _row_kin(aid, pa[0], pa[1], pa[2], Ea, ma),
        _row_kin(did, pd[0], pd[1], pd[2], Ed, mdq),
    )


def _row_kin(pdg: int, px: float, py: float, pz: float, E: float, m: float) -> Dict[str, float]:
    pT, eta, phi, mcalc = kinematic_extras(px, py, pz, E)
    return {"px": px, "py": py, "pz": pz, "E": E, "m": mcalc, "pT": pT, "eta": eta, "phi": phi}


def replace_diquark_three_body(
    g: pd.DataFrame,
    dq_row: pd.Series,
    channel: str,
    tag_a: int,
    tag_b: int,
) -> Tuple[pd.DataFrame, int, int, int, Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Returns (df, idx_stripped, idx_antiq, idx_daughter, kin_s, kin_a, kin_d)."""
    dq_idx = int(dq_row["particle_index"])
    g2 = g[g["particle_index"] != dq_idx].copy()
    max_idx = int(g2["particle_index"].max()) if len(g2) else 0
    i1, i2, i3 = max_idx + 1, max_idx + 2, max_idx + 3
    kin_s, kin_a, kin_d = build_three_body_from_diquark(dq_row, channel)
    sid, did, aid = channel_ab(channel)

    m1 = int(dq_row["mother1"])
    m2 = int(dq_row["mother2"])
    st = int(dq_row["status"])

    common = {
        "event_id": int(dq_row["event_id"]),
        "status": st,
        "mother1": m1,
        "mother2": m2,
        "daughter1": 0,
        "daughter2": 0,
        "isFinal": 1,
    }
    stripped = {
        **common,
        "particle_index": i1,
        "pdg_id": sid,
        "col": tag_a,
        "acol": 0,
        **kin_s,
    }
    antiq = {
        **common,
        "particle_index": i2,
        "pdg_id": aid,
        "col": 0,
        "acol": tag_b,
        **kin_a,
    }
    daughter = {
        **common,
        "particle_index": i3,
        "pdg_id": did,
        "col": 0,
        "acol": tag_a,
        **kin_d,
    }
    out = pd.concat([g2, pd.DataFrame([stripped, antiq, daughter])], ignore_index=True)
    out = out.sort_values("particle_index").reset_index(drop=True)
    return out, i1, i2, i3, kin_s, kin_a, kin_d


def apply_px_kick_pair(df: pd.DataFrame, idx_a: int, idx_b: int, delta: float) -> pd.DataFrame:
    out = df.copy()
    for idx, sign in ((idx_a, 1.0), (idx_b, -1.0)):
        m = out["particle_index"] == idx
        if not m.any():
            raise RuntimeError(f"kick target index {idx} missing")
        row = out.loc[m].iloc[0]
        px = float(row["px"]) + sign * delta
        py, pz = float(row["py"]), float(row["pz"])
        m0 = float(row["m"])
        E = e_on_shell(px, py, pz, m0)
        pT, eta, phi, mcalc = kinematic_extras(px, py, pz, E)
        out.loc[m, ["px", "py", "pz", "E", "m", "pT", "eta", "phi"]] = [
            px,
            py,
            pz,
            E,
            mcalc,
            pT,
            eta,
            phi,
        ]
    return out


def write_event_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("particle_index").to_csv(path, index=False)


def process_one_event_split(
    g: pd.DataFrame,
    event_id: int,
    *,
    event_in_alter_pool: bool,
    md_row: Optional[pd.Series],
    mode: str,
    delta_px: float,
    rng: random.Random,
    forced_split_channel: Optional[Literal["C"]] = None,
) -> Tuple[pd.DataFrame, Literal["unchanged", "altered"], Optional[Dict[str, Any]], Dict[str, int]]:
    """
    Core 90/10 alteration physics (same as the CSV driver's per-event loop).

    Returns:
        particles_out — sorted by ``particle_index``, ready to persist.
        branch — ``"unchanged"`` or ``"altered"``.
        altered_meta — surgery dict (old ``.meta.json`` payload) if ``branch=="altered"``, else ``None``.
        stats_delta — single non-zero counter increment for aggregation (e.g. ``written_unchanged``).
    """
    g = g.sort_values("particle_index").reset_index(drop=True)
    stats_delta: Dict[str, int] = {}

    struck_in = int(md_row["struck_incoming_index"]) if md_row is not None else -1

    if not event_in_alter_pool:
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    reco = find_outgoing_struck_quark_noisr(g, struck_in)
    if not bool(reco.get("success", False)) or reco.get("selected_row") is None:
        stats_delta["fallback_unchanged_struck_fail"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    struck_row = reco["selected_row"]
    struck_idx = int(struck_row["particle_index"])

    if mode == "breit_px_kick_only":
        dq_row = find_final_diquark_kick_partner_row(g)
        if dq_row is None:
            stats_delta["fallback_unchanged_no_ud_diquark"] = 1
            stats_delta["written_unchanged"] = 1
            return g, "unchanged", None, stats_delta
        dq_idx = int(dq_row["particle_index"])
        if dq_idx == struck_idx:
            stats_delta["fallback_unchanged_validation_fail"] = 1
            stats_delta["written_unchanged"] = 1
            return g, "unchanged", None, stats_delta
        try:
            g_kick = apply_px_kick_pair(g.copy(), dq_idx, struck_idx, delta_px)
        except RuntimeError:
            stats_delta["fallback_unchanged_validation_fail"] = 1
            stats_delta["written_unchanged"] = 1
            return g, "unchanged", None, stats_delta
        meta = {
            "original_event_id": int(event_id),
            "mode": "breit_px_kick_only",
            "struck_quark_index": struck_idx,
            "kick_partner_particle_index": dq_idx,
            "kick_delta_px_gev": delta_px,
        }
        stats_delta["written_altered"] = 1
        return g_kick.sort_values("particle_index").reset_index(drop=True), "altered", meta, stats_delta

    dq_row = find_final_ud0_diquark(g)
    if dq_row is None:
        stats_delta["fallback_unchanged_no_ud_diquark"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    channel = "C" if forced_split_channel is None else forced_split_channel
    if channel != "C":
        raise ValueError("only split channel C is supported")
    stats_delta["altered_channel_C"] = 1

    B = int(dq_row["acol"])
    if B <= 0:
        stats_delta["fallback_unchanged_validation_fail"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    g_pre = g[g["particle_index"] != int(dq_row["particle_index"])].copy()
    tag_a = next_free_color_tag(g_pre)

    g_alt, i_strip, i_aq, i_dau, ks, ka, kd = replace_diquark_three_body(
        g, dq_row, channel, tag_a, B
    )

    px_dq, py_dq, pz_dq = float(dq_row["px"]), float(dq_row["py"]), float(dq_row["pz"])
    sx = ks["px"] + ka["px"] + kd["px"] - px_dq
    sy = ks["py"] + ka["py"] + kd["py"] - py_dq
    sz = ks["pz"] + ka["pz"] + kd["pz"] - pz_dq
    if max(abs(sx), abs(sy), abs(sz)) > TOL_3M:
        stats_delta["fallback_unchanged_validation_fail"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    px_before = float(g_alt[g_alt["particle_index"] == i_strip]["px"].iloc[0]) + float(
        g_alt[g_alt["particle_index"] == struck_idx]["px"].iloc[0]
    )

    try:
        g_kick = apply_px_kick_pair(g_alt, i_strip, struck_idx, delta_px)
    except RuntimeError:
        stats_delta["fallback_unchanged_validation_fail"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    px_after = float(g_kick[g_kick["particle_index"] == i_strip]["px"].iloc[0]) + float(
        g_kick[g_kick["particle_index"] == struck_idx]["px"].iloc[0]
    )
    if abs(px_after - px_before) > TOL_3M:
        stats_delta["fallback_unchanged_validation_fail"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    ok_c, msg_c = validate_surgery_color_tags(g_kick, tag_a, B)
    if not ok_c:
        stats_delta["fallback_unchanged_validation_fail"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    onshell_ok = True
    for idx in (i_strip, i_aq, i_dau, struck_idx):
        r = g_kick[g_kick["particle_index"] == idx].iloc[0]
        px, py, pz, E = float(r["px"]), float(r["py"]), float(r["pz"]), float(r["E"])
        m = float(r["m"])
        if abs(E * E - px * px - py * py - pz * pz - m * m) > TOL_ONSHELL:
            onshell_ok = False
            break
    if not onshell_ok:
        stats_delta["fallback_unchanged_validation_fail"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    row_s_pre = g_alt[g_alt["particle_index"] == i_strip].iloc[0]
    row_st_pre = g_alt[g_alt["particle_index"] == struck_idx].iloc[0]
    stripped_px_before_kick = float(row_s_pre["px"])
    struck_px_before_kick = float(row_st_pre["px"])

    sid, did, aid = channel_ab(channel)
    meta = {
        "original_event_id": int(event_id),
        "split_channel": channel,
        "channel_description": "(ud)->[d]+(us)+sbar",
        "original_diquark_index": int(dq_row["particle_index"]),
        "original_diquark_pdg": int(dq_row["pdg_id"]),
        "original_diquark_px": float(dq_row["px"]),
        "original_diquark_py": float(dq_row["py"]),
        "original_diquark_pz": float(dq_row["pz"]),
        "struck_quark_index": struck_idx,
        "kick_delta_px_gev": delta_px,
        "stripped_px_before_kick": stripped_px_before_kick,
        "struck_px_before_kick": struck_px_before_kick,
        "stripped_quark_index": i_strip,
        "created_antiquark_index": i_aq,
        "daughter_diquark_index": i_dau,
        "new_particle_pdgs": {"stripped": sid, "antiquark": aid, "daughter_diquark": did},
        "color_tag_A": tag_a,
        "color_tag_B": B,
    }
    stats_delta["written_altered"] = 1
    return g_kick.sort_values("particle_index").reset_index(drop=True), "altered", meta, stats_delta


def main() -> None:
    ap = argparse.ArgumentParser(description="90/10 DIS split: unchanged vs diquark-split+kick altered.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for 90/10 assignment.")
    ap.add_argument("--delta-px", type=float, default=0.4, help="Balanced px kick (GeV) on stripped d and struck quark.")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_SPLIT_ROOT, help="Output root (contains unchanged/ altered/).")
    ap.add_argument("--max-events", type=int, default=0, help="If >0, only process first N event_ids (for testing).")
    ap.add_argument(
        "--full-event-csv",
        type=Path,
        default=FULL_EVENT_CSV,
        help="Source full event record CSV.",
    )
    ap.add_argument(
        "--metadata-csv",
        type=Path,
        default=METADATA_CSV,
        help="Source event metadata CSV.",
    )
    ap.add_argument(
        "--mode",
        choices=("diquark_kick", "breit_px_kick_only"),
        default="diquark_kick",
        help=(
            "diquark_kick: replace (ud)0 diquark by 3-body then ±px kick (legacy). "
            "breit_px_kick_only: no topology change; ±px on struck quark + diquark partner "
            "in the CSV frame (use with Breit-frame generator output)."
        ),
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_root: Path = args.out_root
    unchanged_dir = out_root / "unchanged"
    altered_dir = out_root / "altered"
    unchanged_dir.mkdir(parents=True, exist_ok=True)
    altered_dir.mkdir(parents=True, exist_ok=True)

    ev = pd.read_csv(args.full_event_csv).sort_values(["event_id", "particle_index"]).reset_index(drop=True)
    md = pd.read_csv(args.metadata_csv).sort_values("event_id").reset_index(drop=True)
    md_map = md.set_index("event_id")

    all_ids = sorted(ev["event_id"].unique().tolist())
    if args.max_events > 0:
        all_ids = all_ids[: args.max_events]

    ids_shuffled = list(all_ids)
    rng.shuffle(ids_shuffled)
    n_alter = int(round(0.1 * len(ids_shuffled)))
    alter_set = set(ids_shuffled[:n_alter])
    unchanged_set = set(ids_shuffled[n_alter:])

    stats = {
        "mode": args.mode,
        "seed": args.seed,
        "delta_px": args.delta_px,
        "total_ids": len(all_ids),
        "planned_altered": n_alter,
        "planned_unchanged": len(unchanged_set),
        "written_unchanged": 0,
        "written_altered": 0,
        "altered_channel_C": 0,
        "fallback_unchanged_no_ud_diquark": 0,
        "fallback_unchanged_struck_fail": 0,
        "fallback_unchanged_validation_fail": 0,
    }

    for event_id in all_ids:
        g = ev[ev["event_id"] == event_id].copy()
        md_row = md_map.loc[event_id] if event_id in md_map.index else None
        particles_out, branch, altered_meta, delta = process_one_event_split(
            g,
            int(event_id),
            event_in_alter_pool=(event_id in alter_set),
            md_row=md_row,
            mode=str(args.mode),
            delta_px=float(args.delta_px),
            rng=rng,
        )
        for k, v in delta.items():
            stats[k] = stats.get(k, 0) + v
        if branch == "unchanged":
            write_event_csv(particles_out, unchanged_dir / f"event_{int(event_id):06d}.csv")
        else:
            alt_path = altered_dir / f"event_{int(event_id):06d}.csv"
            meta_path = altered_dir / f"event_{int(event_id):06d}.meta.json"
            write_event_csv(particles_out, alt_path)
            if altered_meta is not None:
                meta_path.write_text(json.dumps(altered_meta, indent=2), encoding="utf-8")

    stats["total_input_events"] = int(stats["total_ids"])
    stats["attempted_altered"] = int(stats["planned_altered"])
    stats["failed_altered"] = int(stats["attempted_altered"] - stats["written_altered"])

    summary_path = out_root / "split_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    n_files, capped = count_files_under(out_root)
    manifest_path = write_run_manifest(
        run_label="split_dis_sample_diquark_kick",
        script_name="scripts/analysis/split_dis_sample_diquark_kick.py",
        top_level_dirs_written=[str(out_root)],
        approximate_files_created=n_files,
        approximate_files_capped=capped,
        extra={"split_summary_json": str(summary_path)},
    )

    print("=== split_dis_sample_diquark_kick ===")
    print(f"out_root={out_root}")
    print(f"total event ids processed: {stats['total_ids']}")
    print(f"planned altered (10%): {stats['planned_altered']}; written altered: {stats['written_altered']}")
    print(f"planned unchanged (90%): {stats['planned_unchanged']}; written unchanged rows: {stats['written_unchanged']}")
    print(f"altered channel C (ud->d+su0+sbar): {stats.get('altered_channel_C', 0)}")
    print(
        f"fallback to unchanged (no ud diquark): {stats['fallback_unchanged_no_ud_diquark']}, "
        f"(struck fail): {stats['fallback_unchanged_struck_fail']}, "
        f"(validation): {stats['fallback_unchanged_validation_fail']}"
    )
    print(f"summary_json={summary_path}")
    print(f"run_manifest={manifest_path}")


if __name__ == "__main__":
    main()
