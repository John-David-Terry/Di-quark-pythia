#!/usr/bin/env python3
"""
Harvest clean parton-level 2->2 DIS baseline with:
  - struck quark fixed to u
  - hardest forward light parton (u or d) selected by max energy E.

Generator setup (clean parton-level):
  ISR off, FSR off, MPI off, HadronLevel off, ColourReconnection off.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import pythia8

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

OUTDIR = outputs_dir() / "baseline_2to2_partonlevel_strucku_forward_ud"
OUTDIR.mkdir(parents=True, exist_ok=True)


def p4_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3])


def pT(px: float, py: float) -> float:
    return math.hypot(px, py)


def eta(E: float, pz: float, pt: float) -> Optional[float]:
    if pt < 1e-12:
        return None
    num = E + pz
    den = E - pz
    if num <= 0 or den <= 0:
        return None
    return 0.5 * math.log(num / den)


def phi(px: float, py: float) -> float:
    return math.atan2(py, px)


def pplus_pminus(E: float, pz: float) -> Tuple[float, float]:
    return E + pz, E - pz


def dis_kin(e_in: Sequence[float], e_out: Sequence[float], p_in: Sequence[float]) -> Tuple[float, float, float, float]:
    q = (e_in[0] - e_out[0], e_in[1] - e_out[1], e_in[2] - e_out[2], e_in[3] - e_out[3])
    q2 = p4_dot(q, q)
    Q2 = max(0.0, -q2)
    p_dot_q = p4_dot(p_in, q)
    p_dot_k = p4_dot(p_in, e_in)
    if abs(p_dot_q) < 1e-12 or abs(p_dot_k) < 1e-12:
        return Q2, float("nan"), float("nan"), float("nan")
    xB = Q2 / (2.0 * p_dot_q)
    y = p_dot_q / p_dot_k
    pq = (p_in[0] + q[0], p_in[1] + q[1], p_in[2] + q[2], p_in[3] + q[3])
    W2 = p4_dot(pq, pq)
    return float(Q2), float(xB), float(y), float(W2)


def get_scattered_electron_idx(ev: pythia8.Event) -> Optional[int]:
    i44 = None
    best = None
    best_e = -1.0
    for i in range(ev.size()):
        p = ev[i]
        if p.id() != 11 or p.status() <= 0:
            continue
        if p.status() == 44:
            i44 = i
            break
        if p.e() > best_e:
            best = i
            best_e = float(p.e())
    return i44 if i44 is not None else best


def incoming_beams(ev: pythia8.Event) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[Tuple[float, float, float, float]]]:
    e_in = None
    p_in = None
    for i in range(ev.size()):
        p = ev[i]
        if p.id() == 11 and p.status() == -12:
            e_in = (float(p.e()), float(p.px()), float(p.py()), float(p.pz()))
        if p.id() == 2212 and p.status() < 0:
            p_in = (float(p.e()), float(p.px()), float(p.py()), float(p.pz()))
    return e_in, p_in


def get_ancestors(ev: pythia8.Event, start: int, max_depth: int = 60) -> Set[int]:
    seen: Set[int] = set()
    q = [start]
    for _ in range(max_depth):
        if not q:
            break
        idx = q.pop(0)
        if idx in seen or idx <= 0 or idx >= ev.size():
            continue
        seen.add(idx)
        m1, m2 = ev[idx].mother1(), ev[idx].mother2()
        if m1 > 0:
            q.append(m1)
        if m2 > 0 and m2 != m1:
            q.append(m2)
    return seen


def incoming_struck_idx(ev: pythia8.Event) -> Tuple[Optional[int], bool]:
    cand = []
    for i in range(ev.size()):
        p = ev[i]
        if p.status() == -21 and abs(p.id()) in {1, 2, 3, 4, 5}:
            cand.append(i)
    if not cand:
        return None, False
    if len(cand) > 1:
        return max(cand, key=lambda j: ev[j].e()), True
    return cand[0], False


def outgoing_struck_u_idx(ev: pythia8.Event, incoming_idx: int) -> Optional[int]:
    best23 = None
    best23_pp = -1e300
    bestsh = None
    bestsh_pp = -1e300
    bestfb = None
    bestfb_e = -1e300
    for i in range(ev.size()):
        if i == incoming_idx:
            continue
        p = ev[i]
        if abs(p.id()) != 2:
            continue
        if incoming_idx not in get_ancestors(ev, i):
            continue
        pp = float(p.e() + p.pz())
        abss = abs(p.status())
        if abss == 23:
            if pp > best23_pp:
                best23_pp = pp
                best23 = i
        elif 63 <= abss <= 68:
            if pp > bestsh_pp:
                bestsh_pp = pp
                bestsh = i
        else:
            if p.e() > bestfb_e:
                bestfb_e = float(p.e())
                bestfb = i
    return best23 if best23 is not None else (bestsh if bestsh is not None else bestfb)


def forward_light_idxs(ev: pythia8.Event, eta_cut: float) -> List[int]:
    out = []
    for i in range(ev.size()):
        p = ev[i]
        if p.status() <= 0:
            continue
        if p.id() not in (1, 2):
            continue
        px, py, pz, E = float(p.px()), float(p.py()), float(p.pz()), float(p.e())
        e = eta(E, pz, pT(px, py))
        if e is None or e <= eta_cut:
            continue
        out.append(i)
    return out


def obj_dict(px: float, py: float, pz: float, E: float, m: float) -> Dict[str, float]:
    pt = pT(px, py)
    et = eta(E, pz, pt)
    ph = phi(px, py)
    pp, pm = pplus_pminus(E, pz)
    return {
        "px": float(px), "py": float(py), "pz": float(pz), "E": float(E), "m": float(m),
        "pT": float(pt), "eta": float("nan") if et is None else float(et),
        "phi": float(ph), "pplus": float(pp), "pminus": float(pm),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows, columns=columns)
    mode = "a" if path.exists() else "w"
    header = not path.exists()
    df.to_csv(path, mode=mode, header=header, index=False, float_format="%.10g")


def build_pythia() -> pythia8.Pythia:
    p = pythia8.Pythia()
    p.readString("Beams:idA = 11")
    p.readString("Beams:idB = 2212")
    p.readString("Beams:eA = 18.0")
    p.readString("Beams:eB = 275.0")
    p.readString("Beams:frameType = 2")
    p.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    p.readString("HardQCD:all = off")
    p.readString("PDF:lepton = off")
    p.readString("PhaseSpace:Q2Min = 16.0")
    p.readString("PartonLevel:ISR = off")
    p.readString("PartonLevel:FSR = off")
    p.readString("PartonLevel:MPI = off")
    p.readString("HadronLevel:all = off")
    p.readString("ColourReconnection:reconnect = off")
    p.readString("Random:setSeed = on")
    p.readString("Random:seed = 123456")
    p.init()
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest struck-u, forward-(u,d) clean parton-level baseline.")
    parser.add_argument("--target-forward-u", type=int, default=100000)
    parser.add_argument("--target-forward-d", type=int, default=100000)
    parser.add_argument("--eta-forward-cut", type=float, default=0.0)
    parser.add_argument("--chunk-accepted", type=int, default=20000)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--source-config", type=str, default="PARTONLEVEL_ISRFSR_OFF")
    args = parser.parse_args()

    forward_u_csv = OUTDIR / "forward_u_baseline.csv"
    forward_d_csv = OUTDIR / "forward_d_baseline.csv"
    combined_csv = OUTDIR / "combined_baseline.csv"
    summary_json = OUTDIR / "summary.json"
    readme_md = OUTDIR / "README_summary.md"
    for p in (forward_u_csv, forward_d_csv, combined_csv):
        if p.exists():
            p.unlink()

    columns = [
        "event_id", "struck_flavor", "forward_pid", "forward_is_u", "forward_is_d",
        "source_config", "selection_metric", "has_forward_light_quark",
        "Q2", "xB", "y", "W2",
        "ele_px", "ele_py", "ele_pz", "ele_E", "ele_m", "ele_pT", "ele_eta", "ele_phi", "ele_pplus", "ele_pminus",
        "struck_px", "struck_py", "struck_pz", "struck_E", "struck_m", "struck_pT", "struck_eta", "struck_phi", "struck_pplus", "struck_pminus",
        "forward_px", "forward_py", "forward_pz", "forward_E", "forward_m", "forward_pT", "forward_eta", "forward_phi", "forward_pplus", "forward_pminus",
        "forward_selector_energy_pid", "forward_selector_pplus_pid",
    ]

    counters: Dict[str, Any] = {
        "target_forward_u": args.target_forward_u,
        "target_forward_d": args.target_forward_d,
        "accepted_forward_u": 0,
        "accepted_forward_d": 0,
        "accepted_total": 0,
        "generated_total": 0,
        "events_no_electron": 0,
        "events_no_incoming_struck": 0,
        "ambiguous_struck_assignment": 0,
        "events_incoming_not_u": 0,
        "events_no_outgoing_struck_u": 0,
        "events_no_forward_light_quark": 0,
        "events_forward_pid_other": 0,
        "forward_selector_disagree_count": 0,
        "eta_forward_cut": args.eta_forward_cut,
    }

    chunk_u: List[Dict[str, Any]] = []
    chunk_d: List[Dict[str, Any]] = []
    accepted_since_flush = 0
    event_id = 0
    next_u = args.progress_every
    next_d = args.progress_every
    t0 = time.time()

    pythia = build_pythia()
    ev = pythia.event

    while counters["accepted_forward_u"] < args.target_forward_u or counters["accepted_forward_d"] < args.target_forward_d:
        if not pythia.next():
            continue
        counters["generated_total"] += 1

        ei = get_scattered_electron_idx(ev)
        if ei is None:
            counters["events_no_electron"] += 1
            continue

        e_in, p_in = incoming_beams(ev)
        if e_in is None or p_in is None:
            continue

        in_idx, amb = incoming_struck_idx(ev)
        if in_idx is None:
            counters["events_no_incoming_struck"] += 1
            continue
        if amb:
            counters["ambiguous_struck_assignment"] += 1
        if abs(ev[in_idx].id()) != 2:
            counters["events_incoming_not_u"] += 1
            continue

        so_idx = outgoing_struck_u_idx(ev, in_idx)
        if so_idx is None:
            counters["events_no_outgoing_struck_u"] += 1
            continue

        cand = forward_light_idxs(ev, args.eta_forward_cut)
        if not cand:
            counters["events_no_forward_light_quark"] += 1
            continue
        iE = max(cand, key=lambda j: float(ev[j].e()))
        iPP = max(cand, key=lambda j: float(ev[j].e() + ev[j].pz()))
        if iE != iPP:
            counters["forward_selector_disagree_count"] += 1
        fpid = int(ev[iE].id())
        if fpid not in (1, 2):
            counters["events_forward_pid_other"] += 1
            continue

        if fpid == 2 and counters["accepted_forward_u"] >= args.target_forward_u:
            continue
        if fpid == 1 and counters["accepted_forward_d"] >= args.target_forward_d:
            continue

        event_id += 1
        Q2, xB, y, W2 = dis_kin(
            e_in=e_in,
            e_out=(float(ev[ei].e()), float(ev[ei].px()), float(ev[ei].py()), float(ev[ei].pz())),
            p_in=p_in,
        )
        eobj = obj_dict(float(ev[ei].px()), float(ev[ei].py()), float(ev[ei].pz()), float(ev[ei].e()), float(ev[ei].m()))
        sobj = obj_dict(float(ev[so_idx].px()), float(ev[so_idx].py()), float(ev[so_idx].pz()), float(ev[so_idx].e()), float(ev[so_idx].m()))
        fobj = obj_dict(float(ev[iE].px()), float(ev[iE].py()), float(ev[iE].pz()), float(ev[iE].e()), float(ev[iE].m()))

        row = {
            "event_id": event_id,
            "struck_flavor": "u",
            "forward_pid": fpid,
            "forward_is_u": 1 if fpid == 2 else 0,
            "forward_is_d": 1 if fpid == 1 else 0,
            "source_config": args.source_config,
            "selection_metric": "maxE",
            "has_forward_light_quark": 1,
            "Q2": Q2, "xB": xB, "y": y, "W2": W2,
            "ele_px": eobj["px"], "ele_py": eobj["py"], "ele_pz": eobj["pz"], "ele_E": eobj["E"], "ele_m": eobj["m"],
            "ele_pT": eobj["pT"], "ele_eta": eobj["eta"], "ele_phi": eobj["phi"], "ele_pplus": eobj["pplus"], "ele_pminus": eobj["pminus"],
            "struck_px": sobj["px"], "struck_py": sobj["py"], "struck_pz": sobj["pz"], "struck_E": sobj["E"], "struck_m": sobj["m"],
            "struck_pT": sobj["pT"], "struck_eta": sobj["eta"], "struck_phi": sobj["phi"], "struck_pplus": sobj["pplus"], "struck_pminus": sobj["pminus"],
            "forward_px": fobj["px"], "forward_py": fobj["py"], "forward_pz": fobj["pz"], "forward_E": fobj["E"], "forward_m": fobj["m"],
            "forward_pT": fobj["pT"], "forward_eta": fobj["eta"], "forward_phi": fobj["phi"], "forward_pplus": fobj["pplus"], "forward_pminus": fobj["pminus"],
            "forward_selector_energy_pid": int(ev[iE].id()),
            "forward_selector_pplus_pid": int(ev[iPP].id()),
        }

        if fpid == 2:
            chunk_u.append(row)
            counters["accepted_forward_u"] += 1
        else:
            chunk_d.append(row)
            counters["accepted_forward_d"] += 1
        counters["accepted_total"] += 1
        accepted_since_flush += 1

        if accepted_since_flush >= args.chunk_accepted:
            write_csv(forward_u_csv, chunk_u, columns)
            write_csv(forward_d_csv, chunk_d, columns)
            chunk_u.clear()
            chunk_d.clear()
            accepted_since_flush = 0

        show = False
        if counters["accepted_forward_u"] >= next_u:
            show = True
            while counters["accepted_forward_u"] >= next_u:
                next_u += args.progress_every
        if counters["accepted_forward_d"] >= next_d:
            show = True
            while counters["accepted_forward_d"] >= next_d:
                next_d += args.progress_every
        if show:
            elapsed = time.time() - t0
            au = counters["accepted_forward_u"]
            ad = counters["accepted_forward_d"]
            ru = max(0, args.target_forward_u - au)
            rd = max(0, args.target_forward_d - ad)
            rate_u = au / elapsed if elapsed > 0 else 0.0
            rate_d = ad / elapsed if elapsed > 0 else 0.0
            eta_s = max((ru / rate_u) if rate_u > 0 else float("inf"), (rd / rate_d) if rate_d > 0 else float("inf"))
            print(
                f"[progress] gen={counters['generated_total']} acc_forward_u={au}/{args.target_forward_u} "
                f"acc_forward_d={ad}/{args.target_forward_d} elapsed_s={elapsed:.1f} eta_remaining_s={eta_s:.1f}",
                flush=True,
            )

    write_csv(forward_u_csv, chunk_u, columns)
    write_csv(forward_d_csv, chunk_d, columns)

    dfu = pd.read_csv(forward_u_csv)
    dfd = pd.read_csv(forward_d_csv)
    dfc = pd.concat([dfu, dfd], ignore_index=True)
    dfc.to_csv(combined_csv, index=False, float_format="%.10g")

    counters["n_forward_u_rows"] = int(dfu.shape[0])
    counters["n_forward_d_rows"] = int(dfd.shape[0])
    counters["n_combined_rows"] = int(dfc.shape[0])
    counters["acceptance_forward_u_over_generated"] = counters["accepted_forward_u"] / counters["generated_total"] if counters["generated_total"] else float("nan")
    counters["acceptance_forward_d_over_generated"] = counters["accepted_forward_d"] / counters["generated_total"] if counters["generated_total"] else float("nan")
    counters["elapsed_seconds"] = time.time() - t0
    summary_json.write_text(json.dumps(counters, indent=2), encoding="utf-8")

    readme_md.write_text(
        "\n".join(
            [
                "# Parton-level 2->2 baseline (struck-u, forward u/d)",
                "",
                "Generator settings:",
                "- DIS: WeakBosonExchange on, HardQCD off, Q2Min=16",
                "- Beams: e(18 GeV) on p(275 GeV), frameType=2",
                "- ISR off, FSR off, MPI off, HadronLevel off",
                "- ColourReconnection off",
                "",
                "Selection:",
                "- require incoming struck flavor = u",
                "- identify outgoing struck u via ancestry to incoming struck u",
                "- forward light candidates: final-state pid in {1,2}, eta>0",
                "- hardest forward selected by max energy E",
                "",
                "Outputs:",
                f"- {forward_u_csv}",
                f"- {forward_d_csv}",
                f"- {combined_csv}",
                f"- {summary_json}",
            ]
        ),
        encoding="utf-8",
    )

    print("Harvest complete.", flush=True)
    print(json.dumps({
        "accepted_forward_u": counters["accepted_forward_u"],
        "accepted_forward_d": counters["accepted_forward_d"],
        "generated_total": counters["generated_total"],
    }, indent=2))


if __name__ == "__main__":
    main()

