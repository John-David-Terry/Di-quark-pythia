#!/usr/bin/env python3
"""
Controlled PYTHIA settings scan: same v3 acceptance + jet–hadron transverse angle observables,
varying only readString overrides. 1000 accepted events per configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
_ANAL = Path(__file__).resolve().parent
for _p in (_SRC, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diquark.analyze_events_raw import FLIP_Z_PTREL, flip_z  # noqa: E402

from generate_dis_background_final_state_parquet import (  # noqa: E402
    _try_q2_xb,
    collect_final_state_breit,
)
from generate_dis_isr_parton_dataset import (  # noqa: E402
    extract_beams_from_event,
    hard_subprocess_outgoing_quark_lab_p4_and_index,
    pick_incoming_quark_index,
    try_build_lt_from_event,
)

from debug_jet_hadron_angle_biases import (  # noqa: E402
    _empty_obj,
    build_obj_from_pion_breit,
    build_obj_from_sum_breit,
    cut_layer_masks,
    is_physical_hadron,
    summarize_mask,
)

import pythia8  # noqa: E402


def baseline_readstrings(seed: int, *, hadron_level: bool) -> List[str]:
    """Mirror generate_dis_background_final_state_parquet.build_pythia_background (before init)."""
    hl = "on" if hadron_level else "off"
    return [
        "Beams:idA = 11",
        "Beams:idB = 2212",
        "Beams:eA = 18.0",
        "Beams:eB = 275.0",
        "Beams:frameType = 2",
        "WeakBosonExchange:ff2ff(t:gmZ) = on",
        "PhaseSpace:Q2Min = 16.0",
        "ProcessLevel:all = on",
        "PDF:lepton = off",
        "PartonLevel:ISR = on",
        "PartonLevel:FSR = on",
        "PartonLevel:MPI = off",
        "PartonLevel:Remnants = on",
        f"HadronLevel:all = {hl}",
        f"Random:seed = {int(seed)}",
        "Random:setSeed = on",
        "Print:quiet = on",
    ]


def build_pythia_background_with_overrides(
    seed: int,
    overrides: List[str],
    *,
    hadron_level: bool = True,
) -> pythia8.Pythia:
    """Baseline DIS background + extra readString lines, then init()."""
    p = pythia8.Pythia()
    for s in baseline_readstrings(seed, hadron_level=hadron_level):
        p.readString(s)
    for s in overrides:
        t = s.strip()
        if not t or t.startswith("#"):
            continue
        p.readString(t)
    if not p.init():
        raise RuntimeError("PYTHIA init failed (baseline + overrides)")
    return p


def read_pythia_tune_defaults() -> Dict[str, float]:
    """Defaults after baseline init (for documentation in summaries)."""
    p = build_pythia_background_with_overrides(0, [], hadron_level=True)
    out: Dict[str, float] = {}
    for k in ("StringPT:sigma", "StringZ:aLund", "StringZ:bLund", "ColourReconnection:mode"):
        try:
            out[k] = float(p.settings.parm(k))
        except Exception:
            out[k] = float("nan")
    return out


def _empty_row_dict(event_id: int) -> Dict[str, Any]:
    keys = (
        "lead_pi_E_ok",
        "lead_pi_E_pdg",
        "lead_pi_E_E",
        "lead_pi_E_px",
        "lead_pi_E_py",
        "lead_pi_E_pz",
        "lead_pi_E_pht",
        "lead_pi_E_zlc",
        "lead_pi_E_theta",
        "lead_pi_E_cos_theta",
        "lead_pi_pT_ok",
        "lead_pi_pT_pdg",
        "lead_pi_pT_E",
        "lead_pi_pT_px",
        "lead_pi_pT_py",
        "lead_pi_pT_pz",
        "lead_pi_pT_pht",
        "lead_pi_pT_zlc",
        "lead_pi_pT_theta",
        "lead_pi_pT_cos_theta",
        "PT_sum_proton_side_ok",
        "PT_sum_proton_side_px",
        "PT_sum_proton_side_py",
        "PT_sum_proton_side_pht",
        "PT_sum_proton_side_zlc",
        "PT_sum_proton_side_theta",
        "PT_sum_proton_side_cos_theta",
    )
    r = {"event_id": int(event_id)}
    for k in keys:
        r[k] = float("nan")
    for k in ("lead_pi_E_ok", "lead_pi_pT_ok", "PT_sum_proton_side_ok"):
        r[k] = 0
    for k in ("lead_pi_E_pdg", "lead_pi_pT_pdg"):
        r[k] = float("nan")
    return r


def collect_one_config_events(
    pythia: pythia8.Pythia,
    n_target: int,
    *,
    max_total_next_attempts: int = 5_000_000,
) -> Tuple[List[Dict[str, Any]], int, str]:
    """Return (rows, n_successful_next_calls, status_note). Same acceptance as v3."""
    ev = pythia.event
    rows: List[Dict[str, Any]] = []
    accepted = 0
    total_gen = 0
    note = ""
    tries = 0
    while accepted < n_target:
        tries += 1
        if tries > max_total_next_attempts:
            note = (
                f"STOPPED_EARLY: exceeded max_total_next_attempts={max_total_next_attempts} "
                f"with only {accepted} accepted events."
            )
            break
        if not pythia.next():
            continue
        total_gen += 1
        inc_idx = pick_incoming_quark_index(ev)
        if inc_idx is None or abs(int(ev[inc_idx].id())) != 2:
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
        p_in_lab = np.asarray(p_in_lab, dtype=np.float64)

        p4_j_b = LT @ flip_z(np.asarray(p4_lab, dtype=np.float64), FLIP_Z_PTREL)
        kpx, kpy, kpz = float(p4_j_b[1]), float(p4_j_b[2]), float(p4_j_b[3])
        pjt = float(math.hypot(kpx, kpy))

        q2, xb = _try_q2_xb(pythia)
        parts = collect_final_state_breit(ev, LT, accepted)

        pool: List[Tuple[int, np.ndarray]] = []
        for r in parts:
            if abs(int(r["pdg_id"])) != 211:
                continue
            if float(r["pz"]) <= 0.0:
                continue
            p4b = np.array(
                [float(r["E"]), float(r["px"]), float(r["py"]), float(r["pz"])],
                dtype=np.float64,
            )
            pool.append((int(r["pdg_id"]), p4b))

        lead_e = _empty_obj()
        if pool:
            pid, p4b = max(pool, key=lambda t: float(t[1][0]))
            lead_e = build_obj_from_pion_breit(p4b, pid, LT, kpx, kpy, p_in_lab)
        lead_pt = _empty_obj()
        if pool:
            pid, p4b = max(pool, key=lambda t: math.hypot(float(t[1][1]), float(t[1][2])))
            lead_pt = build_obj_from_pion_breit(p4b, pid, LT, kpx, kpy, p_in_lab)

        sx, sy = 0.0, 0.0
        for r in parts:
            if not is_physical_hadron(int(r["pdg_id"])):
                continue
            if float(r["pz"]) <= 0.0:
                continue
            sx += float(r["px"])
            sy += float(r["py"])
        sum_had = build_obj_from_sum_breit(sx, sy, kpx, kpy)
        if abs(sx) + abs(sy) <= 0.0:
            sum_had = _empty_obj()

        row = _empty_row_dict(accepted)
        row["Q2"] = float(q2)
        row["xB"] = float(xb)
        row["k_out_breit_px"] = kpx
        row["k_out_breit_py"] = kpy
        row["k_out_breit_pz"] = kpz
        row["pJT"] = pjt

        def fill(prefix: str, o: Any) -> None:
            row[f"{prefix}_ok"] = int(o.ok)
            row[f"{prefix}_pdg"] = o.pdg
            row[f"{prefix}_E"] = o.E
            row[f"{prefix}_px"] = o.px
            row[f"{prefix}_py"] = o.py
            row[f"{prefix}_pz"] = o.pz
            row[f"{prefix}_pht"] = o.pht
            row[f"{prefix}_zlc"] = o.zlc
            row[f"{prefix}_theta"] = o.theta
            row[f"{prefix}_cos_theta"] = o.cos_theta

        fill("lead_pi_E", lead_e)
        fill("lead_pi_pT", lead_pt)
        row["PT_sum_proton_side_ok"] = int(sum_had.ok)
        row["PT_sum_proton_side_px"] = sum_had.px
        row["PT_sum_proton_side_py"] = sum_had.py
        row["PT_sum_proton_side_pht"] = sum_had.pht
        row["PT_sum_proton_side_zlc"] = sum_had.zlc
        row["PT_sum_proton_side_theta"] = sum_had.theta
        row["PT_sum_proton_side_cos_theta"] = sum_had.cos_theta

        rows.append(row)
        accepted += 1
    return rows, total_gen, note


def write_config_summary(
    df: pd.DataFrame,
    path: Path,
    config_name: str,
    overrides: List[str],
    n_gen: int,
    defaults: Dict[str, float],
    notes: str = "",
    early_note: str = "",
) -> Dict[str, Any]:
    """Write summary.txt and return compact metrics for cross-plot."""
    pjt = df["pJT"].to_numpy(dtype=np.float64)
    lines: List[str] = []
    lines.append(f"configuration = {config_name}")
    lines.append(f"n_accepted = {len(df)}")
    lines.append(f"n_generator_next_calls = {n_gen}")
    lines.append("overrides:")
    for o in overrides:
        lines.append(f"  {o}")
    lines.append(f"PYTHIA_tune_defaults_probed = {json.dumps(defaults)}")
    if notes:
        lines.append(f"notes = {notes}")
    if early_note:
        lines.append(f"WARNING = {early_note}")
    lines.append("")

    metrics: Dict[str, Any] = {"name": config_name, "n": len(df), "n_gen": n_gen}

    def block(obj: str, layers: Tuple[str, ...]) -> None:
        lines.append(f"=== {obj} ===")
        theta = df[f"{obj}_theta"].to_numpy(dtype=np.float64)
        cos_t = df[f"{obj}_cos_theta"].to_numpy(dtype=np.float64)
        pht = df[f"{obj}_pht"].to_numpy(dtype=np.float64)
        zlc = df[f"{obj}_zlc"].to_numpy(dtype=np.float64)
        ok = df[f"{obj}_ok"].to_numpy(dtype=np.int64) == 1
        for ln in layers:
            masks = cut_layer_masks(obj, pjt, pht, zlc, ok)
            st = summarize_mask(theta, cos_t, masks[ln])
            lines.append(
                f"  {ln}: n={st['n']} mean_theta={st['mean_theta']:.6f} "
                f"median_theta={st['median_theta']:.6f} mean_cos={st['mean_cos']:.6f} "
                f"frac_cos_pos={st['frac_cos_pos']:.5f} frac_cos_neg={st['frac_cos_neg']:.5f}"
            )
            metrics[f"{obj}_{ln}_mean_cos"] = st["mean_cos"]
            metrics[f"{obj}_{ln}_n"] = st["n"]
        lines.append("")

    block("lead_pi_E", ("layer0", "layer2", "layer3"))
    block("lead_pi_pT", ("layer0", "layer2", "layer3"))
    block("PT_sum_proton_side", ("layer0", "layer2"))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-accept", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJECT_ROOT / "outputs" / "debug_jet_hadron_pythia_settings_scan",
    )
    ap.add_argument(
        "--include-remnants-off",
        action="store_true",
        help=(
            "Also run PartonLevel:Remnants = off (WARNING: in PYTHIA 8.312 this DIS setup can make "
            "next() extremely slow or appear to hang; default is to skip and document.)"
        ),
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    defaults = read_pythia_tune_defaults()
    n_target = int(args.n_accept)
    base_seed = int(args.seed)

    # (name, overrides, human note)
    configs: List[Tuple[str, List[str], str]] = [
        ("baseline", [], "Production-style baseline."),
        ("fsr_off", ["PartonLevel:FSR = off"], "FSR disabled."),
        ("isr_off", ["PartonLevel:ISR = off"], "ISR disabled."),
        (
            "stringpt_sigma_0p18",
            ["StringPT:sigma = 0.18"],
            f"Smaller fragmentation pT kick (baseline StringPT:sigma={defaults['StringPT:sigma']}).",
        ),
        (
            "stringpt_sigma_0p52",
            ["StringPT:sigma = 0.52"],
            f"Larger fragmentation pT kick (baseline StringPT:sigma={defaults['StringPT:sigma']}).",
        ),
        (
            "lund_hard",
            ["StringZ:aLund = 0.52", "StringZ:bLund = 1.15"],
            f"Harder Lund: baseline aLund={defaults['StringZ:aLund']}, bLund={defaults['StringZ:bLund']}.",
        ),
        (
            "lund_soft",
            ["StringZ:aLund = 0.88", "StringZ:bLund = 0.78"],
            f"Softer Lund: baseline aLund={defaults['StringZ:aLund']}, bLund={defaults['StringZ:bLund']}.",
        ),
        (
            "colour_reconnect_mode1",
            ["ColourReconnection:mode = 1"],
            f"Colour reconnection mode 1 (baseline mode={defaults['ColourReconnection:mode']}).",
        ),
    ]
    remnants_note = (
        "Default scan **skips** `PartonLevel:Remnants = off`: a smoke test showed `pythia.next()` "
        "can run extremely slowly or hang for this e+p DIS configuration in PYTHIA 8.312. "
        "Use --include-remnants-off to attempt it anyway (not recommended for batch runs)."
    )
    if args.include_remnants_off:
        configs.insert(
            3,
            (
                "remnants_off",
                ["PartonLevel:Remnants = off"],
                "Beam remnants disabled (may be pathological; see scan_meta).",
            ),
        )
    (out_dir / "REMNANTS_OFF_POLICY.txt").write_text(remnants_note + "\n", encoding="utf-8")

    all_metrics: List[Dict[str, Any]] = []
    all_dfs: Dict[str, pd.DataFrame] = {}
    failures: List[str] = []

    for idx, (name, ovr, note) in enumerate(configs):
        seed = base_seed + idx * 10_000
        csv_path = out_dir / f"{name}_events.csv"
        sum_path = out_dir / f"{name}_summary.txt"
        try:
            p = build_pythia_background_with_overrides(seed, ovr, hadron_level=True)
        except RuntimeError as exc:
            failures.append(f"{name}: INIT FAILED: {exc}")
            sum_path.write_text(
                "\n".join(
                    [
                        f"configuration = {name}",
                        f"FAILED: {exc}",
                        "overrides:",
                        *[f"  {x}" for x in ovr],
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            continue
        rows, n_gen, early_note = collect_one_config_events(p, n_target)
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        all_dfs[name] = df
        m = write_config_summary(
            df, sum_path, name, ovr, n_gen, defaults, notes=note, early_note=early_note
        )
        if early_note:
            m["early_stop"] = early_note
        m["overrides"] = ovr
        all_metrics.append(m)

    # --- comparison plots (only successful configs) ---
    ok_names = [m["name"] for m in all_metrics]

    def thetas_for(name: str, obj: str, layer: str) -> np.ndarray:
        df = all_dfs[name]
        pjt = df["pJT"].to_numpy(dtype=np.float64)
        pht = df[f"{obj}_pht"].to_numpy(dtype=np.float64)
        zlc = df[f"{obj}_zlc"].to_numpy(dtype=np.float64)
        ok = df[f"{obj}_ok"].to_numpy(dtype=np.int64) == 1
        m = cut_layer_masks(obj, pjt, pht, zlc, ok)[layer]
        return df[f"{obj}_theta"].to_numpy(dtype=np.float64)[m]

    def plot_overlay(
        filename: str,
        obj: str,
        layer: str,
        title: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bins = np.linspace(0.0, math.pi, 31)
        for nm in ok_names:
            t = thetas_for(nm, obj, layer)
            t = t[np.isfinite(t)]
            if t.size == 0:
                continue
            ax.hist(t, bins=bins, density=True, histtype="step", linewidth=1.2, label=f"{nm} (N={t.size})")
        ax.set_xlabel("theta (rad)")
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.legend(fontsize=6, loc="upper right")
        ax.set_xlim(0.0, math.pi)
        fig.tight_layout()
        fig.savefig(out_dir / filename)
        plt.close(fig)

    if ok_names:
        plot_overlay(
            "compare_lead_pi_E_layer0.pdf",
            "lead_pi_E",
            "layer0",
            "lead_pi_E theta, Layer 0 (all configs)",
        )
        plot_overlay(
            "compare_lead_pi_E_layer3.pdf",
            "lead_pi_E",
            "layer3",
            "lead_pi_E theta, Layer 3 (pJT,p_hT,z_LC>0.2)",
        )
        plot_overlay(
            "compare_lead_pi_pT_layer3.pdf",
            "lead_pi_pT",
            "layer3",
            "lead_pi_pT theta, Layer 3",
        )
        plot_overlay(
            "compare_PT_sum_proton_side_layer2.pdf",
            "PT_sum_proton_side",
            "layer2",
            "PT_sum_proton_side theta, Layer 2",
        )

        # mean cos bar summary
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(ok_names))
        w = 0.25
        y1 = [float(m.get("lead_pi_E_layer3_mean_cos", float("nan"))) for m in all_metrics if m["name"] in ok_names]
        y2 = [float(m.get("lead_pi_pT_layer3_mean_cos", float("nan"))) for m in all_metrics]
        y3 = [float(m.get("PT_sum_proton_side_layer2_mean_cos", float("nan"))) for m in all_metrics]
        ax.bar(x - w, y1, width=w, label="lead_pi_E L3")
        ax.bar(x, y2, width=w, label="lead_pi_pT L3")
        ax.bar(x + w, y3, width=w, label="PT_sum L2")
        ax.set_xticks(x)
        ax.set_xticklabels(ok_names, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("mean cos(theta)")
        ax.axhline(0.0, color="k", linewidth=0.5)
        ax.legend()
        ax.set_title("Mean cos(theta) by configuration")
        fig.tight_layout()
        fig.savefig(out_dir / "compare_mean_cos_summary.pdf")
        plt.close(fig)

    scan_meta = {
        "n_accept_per_config": n_target,
        "base_seed": base_seed,
        "seed_rule": "seed = base_seed + config_index * 10000",
        "pythia_tune_defaults": defaults,
        "stringpt_scan_values": {
            "baseline": defaults["StringPT:sigma"],
            "small": 0.18,
            "large": 0.52,
        },
        "lund_scan_values": {
            "baseline_a": defaults["StringZ:aLund"],
            "baseline_b": defaults["StringZ:bLund"],
            "hard": {"aLund": 0.52, "bLund": 1.15},
            "soft": {"aLund": 0.88, "bLund": 0.78},
        },
        "failures": failures,
        "remnants_off_policy": remnants_note,
        "include_remnants_off_flag": bool(args.include_remnants_off),
    }
    (out_dir / "scan_meta.json").write_text(json.dumps(scan_meta, indent=2), encoding="utf-8")

    # --- terminal summary ---
    print("PYTHIA defaults (probed on baseline):", json.dumps(defaults))
    print()
    print(remnants_note)
    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(" ", f)
        print()

    def grab(m_list: List[Dict[str, Any]], name: str, key: str) -> str:
        for m in m_list:
            if m["name"] == name:
                v = m.get(key, float("nan"))
                return f"{v:.5f}" if isinstance(v, (int, float)) and np.isfinite(v) else str(v)
        return "n/a"

    print("=== mean cos(theta) quick comparison ===")
    for label, key in (
        ("lead_pi_E L3", "lead_pi_E_layer3_mean_cos"),
        ("lead_pi_pT L3", "lead_pi_pT_layer3_mean_cos"),
        ("PT_sum_proton_side L2", "PT_sum_proton_side_layer2_mean_cos"),
    ):
        print(f"\n{label}:")
        for m in all_metrics:
            v = m.get(key, float("nan"))
            vs = f"{float(v):.5f}" if isinstance(v, (int, float)) and np.isfinite(float(v)) else str(v)
            print(f"  {m['name']}: {vs}")

    # bullets
    baseline = next((m for m in all_metrics if m["name"] == "baseline"), None)
    if baseline:
        def d(name: str, key: str) -> float:
            m = next((x for x in all_metrics if x["name"] == name), None)
            if not m:
                return float("nan")
            return float(m.get(key, float("nan"))) - float(baseline.get(key, float("nan")))

        print("\n=== Interpretation (heuristic) ===")
        fsr = d("fsr_off", "lead_pi_E_layer3_mean_cos")
        isr = d("isr_off", "lead_pi_E_layer3_mean_cos")
        rem = d("remnants_off", "lead_pi_E_layer3_mean_cos") if any(
            m["name"] == "remnants_off" for m in all_metrics
        ) else float("nan")
        spt_s = d("stringpt_sigma_0p18", "lead_pi_E_layer3_mean_cos")
        spt_l = d("stringpt_sigma_0p52", "lead_pi_E_layer3_mean_cos")
        if np.isfinite(fsr) and abs(fsr) > 0.03:
            print("- Turning **FSR** off moves mean cosθ at L3 for lead_pi_E by ~{:.3f} (material).".format(fsr))
        else:
            print("- Turning **FSR** off has a **small** effect on lead_pi_E L3 mean cosθ in this scan.")
        if np.isfinite(isr) and abs(isr) > 0.03:
            print("- Turning **ISR** off shifts lead_pi_E L3 mean cosθ by ~{:.3f}.".format(isr))
        else:
            print("- Turning **ISR** off has a **small** effect on lead_pi_E L3 mean cosθ here.")
        if np.isfinite(rem) and abs(rem) > 0.03:
            print("- **Remnants off** shifts lead_pi_E L3 mean cosθ by ~{:.3f}.".format(rem))
        elif any(m["name"] == "remnants_off" for m in all_metrics):
            print("- **Remnants off** leaves lead_pi_E L3 mean cosθ nearly unchanged in this sample.")
        else:
            print("- **Remnants off** was not run (see REMNANTS_OFF_POLICY.txt).")
        if np.isfinite(spt_s) and np.isfinite(spt_l) and (abs(spt_s) + abs(spt_l)) > 0.04:
            print(
                "- **StringPT:sigma** variations move lead_pi_E L3 mean cosθ "
                f"(Δ small sigma ≈ {spt_s:+.3f}, Δ large sigma ≈ {spt_l:+.3f}) — fragmentation pT kick matters."
            )
        else:
            print("- **StringPT:sigma** end-points move lead_pi_E L3 mean cosθ only modestly here.")
        sumstab = []
        for nm in ("baseline", "stringpt_sigma_0p18", "fsr_off"):
            m = next((x for x in all_metrics if x["name"] == nm), None)
            if not m:
                continue
            pe = m.get("lead_pi_E_layer3_mean_cos", float("nan"))
            ps = m.get("PT_sum_proton_side_layer2_mean_cos", float("nan"))
            if np.isfinite(float(pe)) and np.isfinite(float(ps)):
                sumstab.append(abs(float(pe) - float(ps)))
        if sumstab:
            print(
                "- |mean cosθ(lead_pi_E L3) − mean cosθ(PT_sum L2)| is typically ~{:.3f}–{:.3f}: "
                "single-pion vs summed proton-side vector can disagree.".format(
                    min(sumstab), max(sumstab)
                )
            )

    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
