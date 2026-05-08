#!/usr/bin/env python3
"""
1000-accepted-event diagnostic: jet vs proton-side transverse opening angle under several
hadron definitions and cut layers. Same v3 acceptance and Breit conventions as production.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    build_pythia_background,
    collect_final_state_breit,
)
from generate_dis_isr_parton_dataset import (  # noqa: E402
    extract_beams_from_event,
    hard_subprocess_outgoing_quark_lab_p4_and_index,
    pick_incoming_quark_index,
    try_build_lt_from_event,
)

OBJECT_KEYS = (
    "lead_pi_E",
    "lead_pi_zlc",
    "lead_pi_pT",
    "PT_sum_proton_side",
    "PT_sum_pi_proton_side",
)

SINGLE_PI_OBJECTS = ("lead_pi_E", "lead_pi_zlc", "lead_pi_pT")
SUM_OBJECTS = ("PT_sum_proton_side", "PT_sum_pi_proton_side")


def is_physical_hadron(pid: int) -> bool:
    """Final-state strongly-interacting candidates (includes light mesons/baryons like π, K, p)."""
    ap = abs(int(pid))
    if ap in (11, 12, 13, 14, 15, 16, 22):
        return False
    if 1 <= ap <= 6 or ap == 21:
        return False
    return True


def _theta_cos_from_2d(
    jx: float, jy: float, hx: float, hy: float
) -> Tuple[float, float]:
    jt = math.hypot(jx, jy)
    ht = math.hypot(hx, hy)
    if jt <= 0.0 or ht <= 0.0:
        return float("nan"), float("nan")
    c = (jx * hx + jy * hy) / (jt * ht)
    c = max(-1.0, min(1.0, c))
    return float(math.acos(c)), float(c)


def breit_four_to_lab_native(p4_breit: np.ndarray, LT: np.ndarray) -> np.ndarray:
    """Inverse of production path: p4_breit = LT @ flip_z(p4_lab_native)."""
    LT_inv = np.linalg.inv(np.asarray(LT, dtype=np.float64))
    p4_flip = LT_inv @ np.asarray(p4_breit, dtype=np.float64).reshape(4)
    return flip_z(p4_flip, FLIP_Z_PTREL)


def z_lc_lab_frame(
    p4_h_lab: np.ndarray,
    p_in_lab: np.ndarray,
) -> float:
    """Same z_LC as compute_zlc_replay_v3_background (lab proton direction n = p_p/|p_p|)."""
    Ep = float(p_in_lab[0])
    pp = np.asarray(p_in_lab[1:4], dtype=np.float64)
    pnorm = float(np.linalg.norm(pp))
    if pnorm <= 1e-18:
        return float("nan")
    n_hat = pp / pnorm
    P_plus = Ep + float(np.dot(pp, n_hat))
    if not math.isfinite(P_plus) or P_plus <= 1e-12:
        return float("nan")
    Eh = float(p4_h_lab[0])
    ph = np.asarray(p4_h_lab[1:4], dtype=np.float64)
    h_plus = Eh + float(np.dot(ph, n_hat))
    return float(h_plus / P_plus)


@dataclass
class ObjKinematics:
    """Per-object kinematics for one event (single hadron or summed 2-vector)."""

    ok: bool
    pdg: float  # nan for sums
    E: float
    px: float
    py: float
    pz: float
    pht: float
    zlc: float
    theta: float
    cos_theta: float


def _empty_obj() -> ObjKinematics:
    return ObjKinematics(
        ok=False,
        pdg=float("nan"),
        E=float("nan"),
        px=float("nan"),
        py=float("nan"),
        pz=float("nan"),
        pht=float("nan"),
        zlc=float("nan"),
        theta=float("nan"),
        cos_theta=float("nan"),
    )


def build_obj_from_pion_breit(
    p4_b: np.ndarray,
    pdg: int,
    LT: np.ndarray,
    pjx: float,
    pjy: float,
    p_in_lab: np.ndarray,
) -> ObjKinematics:
    p4_lab = breit_four_to_lab_native(p4_b, LT)
    zlc = z_lc_lab_frame(p4_lab, p_in_lab)
    E, px, py, pz = (float(p4_b[0]), float(p4_b[1]), float(p4_b[2]), float(p4_b[3]))
    pht = float(math.hypot(px, py))
    th, cth = _theta_cos_from_2d(pjx, pjy, px, py)
    return ObjKinematics(
        ok=math.isfinite(th) and math.isfinite(cth),
        pdg=float(pdg),
        E=E,
        px=px,
        py=py,
        pz=pz,
        pht=pht,
        zlc=zlc,
        theta=th,
        cos_theta=cth,
    )


def build_obj_from_sum_breit(
    sum_px: float,
    sum_py: float,
    pjx: float,
    pjy: float,
) -> ObjKinematics:
    pht = float(math.hypot(sum_px, sum_py))
    th, cth = _theta_cos_from_2d(pjx, pjy, sum_px, sum_py)
    return ObjKinematics(
        ok=math.isfinite(th) and math.isfinite(cth),
        pdg=float("nan"),
        E=float("nan"),
        px=sum_px,
        py=sum_py,
        pz=float("nan"),
        pht=pht,
        zlc=float("nan"),
        theta=th,
        cos_theta=cth,
    )


def summarize_mask(theta: np.ndarray, cos_t: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    m = mask & np.isfinite(theta) & np.isfinite(cos_t)
    n = int(np.sum(m))
    if n == 0:
        return {
            "n": 0,
            "mean_theta": float("nan"),
            "median_theta": float("nan"),
            "mean_cos": float("nan"),
            "frac_cos_pos": float("nan"),
            "frac_cos_neg": float("nan"),
        }
    th = theta[m]
    ct = cos_t[m]
    return {
        "n": n,
        "mean_theta": float(np.mean(th)),
        "median_theta": float(np.median(th)),
        "mean_cos": float(np.mean(ct)),
        "frac_cos_pos": float(np.mean(ct > 0.0)),
        "frac_cos_neg": float(np.mean(ct < 0.0)),
    }


def cut_layer_masks(
    obj_key: str,
    pjt: np.ndarray,
    pht: np.ndarray,
    zlc: np.ndarray,
    ok: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return boolean masks for each cut layer (all require object ok / valid kinematics)."""
    base = ok & np.isfinite(pjt) & np.isfinite(pht)
    m0 = base.copy()
    m1 = base & (pjt > 0.2)
    m2 = base & (pjt > 0.2) & (pht > 0.2)
    if obj_key in SINGLE_PI_OBJECTS:
        m3 = m2 & np.isfinite(zlc) & (zlc > 0.2)
        mt = base & (pjt > 0.4) & (pht > 0.4) & np.isfinite(zlc) & (zlc > 0.2)
    else:
        # Summed vectors: no single-hadron z_LC; strongest comparable cut is Layer 2.
        m3 = np.zeros_like(m0, dtype=bool)
        mt = np.zeros_like(m0, dtype=bool)
    return {"layer0": m0, "layer1": m1, "layer2": m2, "layer3": m3, "layer_tight": mt}


def plot_hist_theta(ax, theta_list: List[np.ndarray], labels: List[str], title: str) -> None:
    bins = np.linspace(0.0, math.pi, 31)
    for th, lab in zip(theta_list, labels):
        t = th[np.isfinite(th)]
        if t.size == 0:
            continue
        ax.hist(
            t,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=f"{lab} (N={t.size})",
        )
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0.0, math.pi)


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
        default=_PROJECT_ROOT / "outputs" / "debug_jet_hadron_angle_biases",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "debug_1000_events.csv"
    summary_path = out_dir / "summary.txt"

    import pythia8

    n_target = int(args.n_accept)
    pythia = build_pythia_background(int(args.seed), hadron_level=True)
    ev = pythia.event

    event_rows: List[Dict[str, Any]] = []

    accepted = 0
    total_gen = 0

    while accepted < n_target:
        if not pythia.next():
            continue
        total_gen += 1

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
        p_in_lab = np.asarray(p_in_lab, dtype=np.float64)

        p4_j_b = LT @ flip_z(np.asarray(p4_lab, dtype=np.float64), FLIP_Z_PTREL)
        kpx, kpy, kpz = float(p4_j_b[1]), float(p4_j_b[2]), float(p4_j_b[3])
        pjt = float(math.hypot(kpx, kpy))

        p4_p_b = LT @ flip_z(p_in_lab.copy(), FLIP_Z_PTREL)
        p_in_breit_pz = float(p4_p_b[3])

        q2, xb = _try_q2_xb(pythia)

        parts = collect_final_state_breit(ev, LT, accepted)

        pool: List[Tuple[int, np.ndarray]] = []
        for r in parts:
            pid = int(r["pdg_id"])
            if abs(pid) != 211:
                continue
            pz = float(r["pz"])
            if pz <= 0.0:
                continue
            p4b = np.array(
                [float(r["E"]), float(r["px"]), float(r["py"]), float(r["pz"])],
                dtype=np.float64,
            )
            pool.append((pid, p4b))

        # Leading by E
        lead_e = _empty_obj()
        if pool:
            pid, p4b = max(pool, key=lambda t: float(t[1][0]))
            lead_e = build_obj_from_pion_breit(p4b, pid, LT, kpx, kpy, p_in_lab)

        # Leading by z_LC
        lead_z = _empty_obj()
        if pool:
            best = None
            best_z = -1e300
            for pid, p4b in pool:
                p4_lab = breit_four_to_lab_native(p4b, LT)
                zv = z_lc_lab_frame(p4_lab, p_in_lab)
                if np.isfinite(zv) and zv > best_z:
                    best_z = float(zv)
                    best = (pid, p4b)
            if best is not None:
                lead_z = build_obj_from_pion_breit(best[1], best[0], LT, kpx, kpy, p_in_lab)

        # Leading by pT (Breit)
        lead_pt = _empty_obj()
        if pool:
            pid, p4b = max(pool, key=lambda t: math.hypot(float(t[1][1]), float(t[1][2])))
            lead_pt = build_obj_from_pion_breit(p4b, pid, LT, kpx, kpy, p_in_lab)

        # Full proton-side hadronic sum
        sx, sy = 0.0, 0.0
        for r in parts:
            if not is_physical_hadron(int(r["pdg_id"])):
                continue
            if float(r["pz"]) <= 0.0:
                continue
            sx += float(r["px"])
            sy += float(r["py"])
        sum_had = build_obj_from_sum_breit(sx, sy, kpx, kpy)
        if not (abs(sx) + abs(sy) > 0.0):
            sum_had = _empty_obj()

        # Proton-side charged pion sum
        pix, piy = 0.0, 0.0
        for pid, p4b in pool:
            pix += float(p4b[1])
            piy += float(p4b[2])
        sum_pi = build_obj_from_sum_breit(pix, piy, kpx, kpy)
        if not pool:
            sum_pi = _empty_obj()

        row: Dict[str, Any] = {
            "event_id": int(accepted),
            "Q2": float(q2),
            "xB": float(xb),
            "k_out_breit_px": kpx,
            "k_out_breit_py": kpy,
            "k_out_breit_pz": kpz,
            "p_in_breit_pz": p_in_breit_pz,
            "pJT": pjt,
        }

        objs = {
            "lead_pi_E": lead_e,
            "lead_pi_zlc": lead_z,
            "lead_pi_pT": lead_pt,
            "PT_sum_proton_side": sum_had,
            "PT_sum_pi_proton_side": sum_pi,
        }
        for k, o in objs.items():
            prefix = k
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

        event_rows.append(row)
        accepted += 1

    df = pd.DataFrame(event_rows)
    df.to_csv(csv_path, index=False)

    # --- Build numpy arrays for summaries ---
    n = len(df)
    pjt = df["pJT"].to_numpy(dtype=np.float64)

    summary_lines: List[str] = []
    summary_lines.append(f"n_accepted={n}  generator_tries={total_gen}")
    summary_lines.append("")
    summary_lines.append(
        "Layer definitions: L0=no extra cut (object must be constructible with finite θ); "
        "L1=pJT>0.2; L2=L1+p_hT>0.2; L3=L2+z_LC>0.2 (single-pion objects only); "
        "L_tight=pJT>0.4, p_hT>0.4, z_LC>0.2 (single-pion only). "
        "Summed-vector objects omit z_LC cuts; L3/L_tight masks are empty for them."
    )
    summary_lines.append("")

    layer_names = ["layer0", "layer1", "layer2", "layer3", "layer_tight"]
    stats_table: Dict[Tuple[str, str], Dict[str, float]] = {}

    for ok in OBJECT_KEYS:
        prefix = ok
        theta = df[f"{prefix}_theta"].to_numpy(dtype=np.float64)
        cos_t = df[f"{prefix}_cos_theta"].to_numpy(dtype=np.float64)
        pht_o = df[f"{prefix}_pht"].to_numpy(dtype=np.float64)
        zlc_o = df[f"{prefix}_zlc"].to_numpy(dtype=np.float64)
        obj_ok = df[f"{prefix}_ok"].to_numpy(dtype=np.int64) == 1

        masks = cut_layer_masks(ok, pjt, pht_o, zlc_o, obj_ok)
        for ln in layer_names:
            st = summarize_mask(theta, cos_t, masks[ln])
            stats_table[(ok, ln)] = st
            summary_lines.append(
                f"{ok}  {ln}:  n={st['n']}  mean_theta={st['mean_theta']:.5f}  "
                f"median_theta={st['median_theta']:.5f}  mean_cos={st['mean_cos']:.5f}  "
                f"frac(cos>0)={st['frac_cos_pos']:.4f}  frac(cos<0)={st['frac_cos_neg']:.4f}"
            )
        summary_lines.append("")

    # --- Tertiles for lead_pi_E: zLC, pht, pJT ---
    summary_lines.append("=== Tertiles (lead_pi_E), mean cosθ and mean θ per bin ===")
    mref = (df["lead_pi_E_ok"] == 1) & np.isfinite(df["lead_pi_E_cos_theta"].to_numpy())
    dref = df.loc[mref].copy()
    for col, label in (
        ("lead_pi_E_zlc", "z_LC (leading-E pion)"),
        ("lead_pi_E_pht", "p_hT Breit (leading-E pion)"),
        ("pJT", "p_JT Breit"),
    ):
        try:
            if dref[col].nunique() < 3:
                summary_lines.append(f"{label}: insufficient unique values for tertiles")
                continue
            dref["_bin"] = pd.qcut(dref[col], 3, duplicates="drop")
            g = dref.groupby("_bin", observed=True)
            summary_lines.append(label + ":")
            for name, sub in g:
                summary_lines.append(
                    f"  bin={name}  n={len(sub)}  mean_cosθ={float(sub['lead_pi_E_cos_theta'].mean()):.5f}  "
                    f"mean_θ={float(sub['lead_pi_E_theta'].mean()):.5f}"
                )
        except Exception as exc:  # pragma: no cover
            summary_lines.append(f"{label}: tertile split failed ({exc})")
    summary_lines.append("")

    # --- Terminal one-liners ---
    def mc(layer: str, key: str) -> float:
        return float(stats_table[(key, layer)]["mean_cos"])

    print(f"Wrote {csv_path}  (n={n}, tries={total_gen})")
    print("\n--- Mean cos theta (quick) ---")
    for key in OBJECT_KEYS:
        l3n = int(stats_table[(key, "layer3")]["n"])
        ltn = int(stats_table[(key, "layer_tight")]["n"])
        l3s = f"{mc('layer3', key):.5f}" if l3n > 0 else "n/a (no L3 for sums)"
        lts = f"{mc('layer_tight', key):.5f}" if ltn > 0 else "n/a (no L_tight for sums)"
        print(
            f"  {key}:  L0 mean cos = {mc('layer0', key):.5f}   "
            f"L3 mean cos = {l3s}   L_tight mean cos = {lts}"
        )

    # --- Conclusion heuristic ---
    dcos = abs(mc("layer0", "lead_pi_E") - mc("layer3", "lead_pi_E"))
    mc_e0 = mc("layer0", "lead_pi_E")
    mc_z0 = mc("layer0", "lead_pi_zlc")
    mc_pt0 = mc("layer0", "lead_pi_pT")
    mc_e3 = mc("layer3", "lead_pi_E")
    mc_z3 = mc("layer3", "lead_pi_zlc")
    mc_pt3 = mc("layer3", "lead_pi_pT")
    dz0 = abs(mc_e0 - mc_z0) if all(math.isfinite(x) for x in (mc_e0, mc_z0)) else float("nan")
    de0 = abs(mc_e0 - mc_pt0) if all(math.isfinite(x) for x in (mc_e0, mc_pt0)) else float("nan")
    dz3 = abs(mc_e3 - mc_z3) if all(math.isfinite(x) for x in (mc_e3, mc_z3)) else float("nan")
    de3 = abs(mc_e3 - mc_pt3) if all(math.isfinite(x) for x in (mc_e3, mc_pt3)) else float("nan")
    # Fair same-cut comparison: Layer 2 (pJT,p_hT) for pion vs sums (no z_LC required).
    csum_h = mc("layer2", "PT_sum_proton_side")
    csum_pi = mc("layer2", "PT_sum_pi_proton_side")
    cpi_l2 = mc("layer2", "lead_pi_E")

    conclusion: List[str] = []
    if dcos > 0.08:
        conclusion.append(
            "The jet–hadron cosθ distribution shifts materially between no cut and the triple cut "
            "(cut-induced bias is plausible for the near-θ→0 enhancement)."
        )
    else:
        conclusion.append(
            "Mean cosθ changes only modestly from no cut to the triple cut (cuts alone may not "
            "explain a sharp near-zero peak)."
        )
    if (
        math.isfinite(de3)
        and de3 > 0.08
        and int(stats_table[("lead_pi_E", "layer3")]["n"]) > 0
    ):
        conclusion.append(
            "Under the triple cut (L3), leading the proton-side π by p_T vs by E shifts mean cosθ "
            "materially — cut+selection interplay (not just hemisphere) can reshape the angle."
        )
    elif (
        math.isfinite(dz0)
        and math.isfinite(de0)
        and dz0 < 0.04
        and de0 < 0.04
        and math.isfinite(dz3)
        and math.isfinite(de3)
        and dz3 < 0.04
        and de3 < 0.04
    ):
        conclusion.append(
            "Leading π by E vs z_LC vs p_T give similar mean cosθ at Layer 0 and Layer 3 in this "
            "sample (little sensitivity to which leading-pion rule is used)."
        )
    else:
        conclusion.append(
            "Leading-pion rule sensitivity is mixed: compare Layer 0 vs Layer 3 lines in summary.txt "
            "for E vs z_LC vs p_T."
        )
    if (
        math.isfinite(csum_h)
        and math.isfinite(cpi_l2)
        and (csum_h - cpi_l2) < -0.08
    ):
        conclusion.append(
            "At Layer 2 (pJT,p_hT>0.2), the full proton-side hadronic p_T sum is more back-to-back "
            "with the jet (smaller cosθ) than the single leading-E pion — a single pion is a weak "
            "proxy for the proton-side recoil plane."
        )
    elif (
        math.isfinite(csum_h)
        and math.isfinite(cpi_l2)
        and (csum_h - cpi_l2) > 0.08
    ):
        conclusion.append(
            "At Layer 2, the summed proton-side hadronic vector is less back-to-back than the "
            "leading-E pion (unexpected vs a simple dijet picture; inspect further)."
        )
    else:
        conclusion.append(
            "At Layer 2, leading-E pion vs full proton-side p_T sum show similar mean cosθ "
            "(no strong proxy-vs-sum separation in this sample)."
        )
    if (
        math.isfinite(csum_pi)
        and math.isfinite(cpi_l2)
        and (csum_pi - cpi_l2) < -0.08
    ):
        conclusion.append(
            "The all-pion proton-side p_T sum is more back-to-back than the single leading-E pion "
            "at Layer 2 (multi-pion recoil on the proton side matters)."
        )


    summary_lines.append("=== Automated conclusion (heuristic) ===")
    for line in conclusion:
        summary_lines.append(line)

    print("\n=== Conclusion (heuristic) ===")
    for line in conclusion:
        print(line)

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {summary_path}")

    # --- Histograms ---
    th_E = df["lead_pi_E_theta"].to_numpy(dtype=np.float64)
    m0_E = cut_layer_masks("lead_pi_E", pjt, df["lead_pi_E_pht"].to_numpy(), df["lead_pi_E_zlc"].to_numpy(), df["lead_pi_E_ok"].to_numpy() == 1)["layer0"]
    m3_E = cut_layer_masks("lead_pi_E", pjt, df["lead_pi_E_pht"].to_numpy(), df["lead_pi_E_zlc"].to_numpy(), df["lead_pi_E_ok"].to_numpy() == 1)["layer3"]

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_hist_theta(ax, [th_E[m0_E]], ["lead_pi_E"], "theta, lead_pi_E (Layer 0)")
    fig.tight_layout()
    fig.savefig(out_dir / "theta_lead_pi_E_nocut.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_hist_theta(
        ax,
        [th_E[m3_E]],
        ["lead_pi_E"],
        "theta, lead_pi_E (Layer 3: pJT,p_hT,z_LC > 0.2)",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "theta_lead_pi_E_cutlayer3.pdf")
    plt.close(fig)

    thetas_nocut = []
    labels_nc = []
    for key in OBJECT_KEYS:
        t = df[f"{key}_theta"].to_numpy(dtype=np.float64)
        m0 = cut_layer_masks(
            key,
            pjt,
            df[f"{key}_pht"].to_numpy(dtype=np.float64),
            df[f"{key}_zlc"].to_numpy(dtype=np.float64),
            df[f"{key}_ok"].to_numpy(dtype=np.int64) == 1,
        )["layer0"]
        thetas_nocut.append(t[m0])
        labels_nc.append(key)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_hist_theta(ax, thetas_nocut, labels_nc, "theta compare objects (Layer 0)")
    fig.tight_layout()
    fig.savefig(out_dir / "theta_compare_objects_nocut.pdf")
    plt.close(fig)

    thetas_c3 = []
    labels_c3 = []
    for key in OBJECT_KEYS:
        t = df[f"{key}_theta"].to_numpy(dtype=np.float64)
        masks = cut_layer_masks(
            key,
            pjt,
            df[f"{key}_pht"].to_numpy(dtype=np.float64),
            df[f"{key}_zlc"].to_numpy(dtype=np.float64),
            df[f"{key}_ok"].to_numpy(dtype=np.int64) == 1,
        )
        # For sums use layer2 as "strong cut" analogue
        mcut = masks["layer3"] if key in SINGLE_PI_OBJECTS else masks["layer2"]
        thetas_c3.append(t[mcut])
        labels_c3.append(key if key in SINGLE_PI_OBJECTS else f"{key} (L2, no z_LC)")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_hist_theta(
        ax,
        thetas_c3,
        labels_c3,
        "theta compare objects (pions: Layer 3; sums: Layer 2, no z_LC)",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "theta_compare_objects_cutlayer3.pdf")
    plt.close(fig)

    print(f"Plots saved under {out_dir}")


if __name__ == "__main__":
    main()
