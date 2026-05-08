#!/usr/bin/env python3
"""
Harvest 2->2 DIS baseline with corrected definition:
  - struck quark is always u
  - forward tag is the most energetic forward light quark (u or d), eta > 0

High-level overview
-------------------
This script generates DIS events with the same kinematics/settings as prior shard/baseline runs
and builds two accepted samples:
  1) forward-u sample: struck=u and hardest forward light quark is u (pid=2)
  2) forward-d sample: struck=u and hardest forward light quark is d (pid=1)

Targets:
  - 100000 accepted forward-u
  - 100000 accepted forward-d

For each accepted event, store:
  - outgoing electron 4-vector (+ derived vars)
  - outgoing struck-u 4-vector (+ derived vars)
  - hardest forward light quark 4-vector (+ derived vars), selected by largest E
  - event metadata and DIS kinematics (Q2, xB, y, W2)

Low-level identification logic
------------------------------
1) Scattered electron:
   - pick e- with status>0, prefer status==44; else highest-energy e-.
2) Incoming struck quark tag:
   - status==-21 and abs(id) in {1..5}; if multiple, choose highest-E and flag ambiguity.
3) Require incoming struck flavor == u (abs(id)==2), else skip.
4) Outgoing struck-u:
   - quark with abs(id)==2 whose ancestry contains incoming struck-quark index.
   - priority: abs(status)==23 (max pplus), then 63<=abs(status)<=68 (max pplus), else max E.
5) Forward light-quark candidates:
   - id in {1,2}, status>0, eta>eta_forward_cut (default 0)
   - choose primary candidate by largest energy E
   - also record side diagnostic candidate chosen by largest pplus
6) Accept only if a forward light candidate exists and its pid is 1 or 2:
   - pid==2 -> forward-u sample
   - pid==1 -> forward-d sample

Outputs:
  outputs/baseline_2to2_strucku_forward_ud/
    forward_u_baseline.csv
    forward_d_baseline.csv
    combined_baseline.csv
    summary.json
    README_summary.md
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

OUTDIR = outputs_dir() / "baseline_2to2_strucku_forward_ud"
OUTDIR.mkdir(parents=True, exist_ok=True)


def p4_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3])


def pT_from(px: float, py: float) -> float:
    return math.hypot(px, py)


def eta_from(E: float, pz: float, pt: float, eps: float = 1e-12) -> Optional[float]:
    if pt < eps:
        return None
    num = E + pz
    den = E - pz
    if num <= 0 or den <= 0:
        return None
    return 0.5 * math.log(num / den)


def phi_from(px: float, py: float) -> float:
    return math.atan2(py, px)


def pplus_pminus(E: float, pz: float) -> Tuple[float, float]:
    return (E + pz, E - pz)


def compute_dis_kinematics(e_in: Sequence[float], e_out: Sequence[float], p_in: Sequence[float]) -> Tuple[float, float, float, float]:
    q = (e_in[0] - e_out[0], e_in[1] - e_out[1], e_in[2] - e_out[2], e_in[3] - e_out[3])
    q2 = p4_dot(q, q)
    Q2 = -q2
    if Q2 < 0:
        Q2 = 0.0
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
    best_44 = None
    best_i = None
    best_e = -1.0
    for i in range(ev.size()):
        p = ev[i]
        if p.id() != 11 or p.status() <= 0:
            continue
        if p.status() == 44:
            best_44 = i
            break
        if p.e() > best_e:
            best_e = float(p.e())
            best_i = i
    return best_44 if best_44 is not None else best_i


def find_incoming_beams(ev: pythia8.Event) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[Tuple[float, float, float, float]]]:
    e_in = None
    p_in = None
    for i in range(ev.size()):
        p = ev[i]
        if p.id() == 11 and p.status() == -12:
            e_in = (float(p.e()), float(p.px()), float(p.py()), float(p.pz()))
        if p.id() == 2212 and p.status() < 0:
            p_in = (float(p.e()), float(p.px()), float(p.py()), float(p.pz()))
    return e_in, p_in


def get_ancestors(ev: pythia8.Event, start: int, max_depth: int = 80) -> Set[int]:
    seen: Set[int] = set()
    queue = [start]
    for _ in range(max_depth):
        if not queue:
            break
        idx = queue.pop(0)
        if idx in seen or idx <= 0 or idx >= ev.size():
            continue
        seen.add(idx)
        m1, m2 = ev[idx].mother1(), ev[idx].mother2()
        if m1 > 0:
            queue.append(m1)
        if m2 > 0 and m2 != m1:
            queue.append(m2)
    return seen


def find_incoming_struck_idx(ev: pythia8.Event) -> Tuple[Optional[int], bool]:
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


def find_outgoing_struck_u_idx(ev: pythia8.Event, incoming_u_idx: int) -> Optional[int]:
    best23 = None
    best23_pp = -1e300
    bestsh = None
    bestsh_pp = -1e300
    bestfb = None
    bestfb_e = -1e300
    for i in range(ev.size()):
        if i == incoming_u_idx:
            continue
        p = ev[i]
        if abs(p.id()) != 2:
            continue
        if incoming_u_idx not in get_ancestors(ev, i):
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


def forward_light_candidates(ev: pythia8.Event, eta_cut: float) -> List[int]:
    out: List[int] = []
    for i in range(ev.size()):
        p = ev[i]
        if p.status() <= 0:
            continue
        if p.id() not in (1, 2):
            continue
        px, py, pz, E = float(p.px()), float(p.py()), float(p.pz()), float(p.e())
        eta = eta_from(E, pz, pT_from(px, py))
        if eta is None or eta <= eta_cut:
            continue
        out.append(i)
    return out


def obj_fields(px: float, py: float, pz: float, E: float, m: float) -> Dict[str, float]:
    pt = pT_from(px, py)
    eta = eta_from(E, pz, pt)
    phi = phi_from(px, py)
    pp, pm = pplus_pminus(E, pz)
    return {
        "px": float(px), "py": float(py), "pz": float(pz), "E": float(E), "m": float(m),
        "pT": float(pt), "eta": float("nan") if eta is None else float(eta), "phi": float(phi),
        "pplus": float(pp), "pminus": float(pm),
    }


def write_rows_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
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
    p.readString("PartonLevel:ISR = on")
    p.readString("PartonLevel:FSR = on")
    p.readString("PartonLevel:MPI = off")
    p.readString("HadronLevel:all = on")
    p.readString("ColourReconnection:reconnect = off")
    p.readString("Random:setSeed = on")
    p.readString("Random:seed = 123456")
    p.init()
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest struck-u / forward-u,d 2->2 baseline.")
    parser.add_argument("--target-forward-u", type=int, default=100000)
    parser.add_argument("--target-forward-d", type=int, default=100000)
    parser.add_argument("--eta-forward-cut", type=float, default=0.0)
    parser.add_argument("--chunk-accepted", type=int, default=20000)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--source-config", type=str, default="ISRFSR_ON")
    args = parser.parse_args()

    f_u_csv = OUTDIR / "forward_u_baseline.csv"
    f_d_csv = OUTDIR / "forward_d_baseline.csv"
    comb_csv = OUTDIR / "combined_baseline.csv"
    summary_path = OUTDIR / "summary.json"
    readme_path = OUTDIR / "README_summary.md"

    for p in (f_u_csv, f_d_csv, comb_csv):
        if p.exists():
            p.unlink()

    cols = [
        "event_id", "source_config", "struck_flavor", "forward_pid", "forward_is_u", "forward_is_d",
        "has_forward_light_quark", "Q2", "xB", "y", "W2",
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
        "generated_total": 0,
        "accepted_total": 0,
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
    t0 = time.time()
    next_u = args.progress_every
    next_d = args.progress_every

    pythia = build_pythia()
    ev = pythia.event

    while counters["accepted_forward_u"] < args.target_forward_u or counters["accepted_forward_d"] < args.target_forward_d:
        if not pythia.next():
            continue
        counters["generated_total"] += 1

        ele_idx = get_scattered_electron_idx(ev)
        if ele_idx is None:
            counters["events_no_electron"] += 1
            continue

        e_in, p_in = find_incoming_beams(ev)
        if e_in is None or p_in is None:
            continue

        in_idx, ambiguous = find_incoming_struck_idx(ev)
        if in_idx is None:
            counters["events_no_incoming_struck"] += 1
            continue
        if ambiguous:
            counters["ambiguous_struck_assignment"] += 1

        if abs(ev[in_idx].id()) != 2:
            counters["events_incoming_not_u"] += 1
            continue

        struck_idx = find_outgoing_struck_u_idx(ev, in_idx)
        if struck_idx is None:
            counters["events_no_outgoing_struck_u"] += 1
            continue

        cand = forward_light_candidates(ev, args.eta_forward_cut)
        if not cand:
            counters["events_no_forward_light_quark"] += 1
            continue

        forward_energy_idx = max(cand, key=lambda j: float(ev[j].e()))
        forward_pplus_idx = max(cand, key=lambda j: float(ev[j].e() + ev[j].pz()))
        if forward_energy_idx != forward_pplus_idx:
            counters["forward_selector_disagree_count"] += 1

        forward_pid = int(ev[forward_energy_idx].id())
        if forward_pid not in (1, 2):
            counters["events_forward_pid_other"] += 1
            continue

        if forward_pid == 2 and counters["accepted_forward_u"] >= args.target_forward_u:
            continue
        if forward_pid == 1 and counters["accepted_forward_d"] >= args.target_forward_d:
            continue

        # Build row
        event_id += 1
        Q2, xB, y, W2 = compute_dis_kinematics(
            e_in=e_in,
            e_out=(float(ev[ele_idx].e()), float(ev[ele_idx].px()), float(ev[ele_idx].py()), float(ev[ele_idx].pz())),
            p_in=p_in,
        )

        e_row = obj_fields(float(ev[ele_idx].px()), float(ev[ele_idx].py()), float(ev[ele_idx].pz()), float(ev[ele_idx].e()), float(ev[ele_idx].m()))
        s_row = obj_fields(float(ev[struck_idx].px()), float(ev[struck_idx].py()), float(ev[struck_idx].pz()), float(ev[struck_idx].e()), float(ev[struck_idx].m()))
        f_row = obj_fields(float(ev[forward_energy_idx].px()), float(ev[forward_energy_idx].py()), float(ev[forward_energy_idx].pz()), float(ev[forward_energy_idx].e()), float(ev[forward_energy_idx].m()))

        row = {
            "event_id": event_id,
            "source_config": args.source_config,
            "struck_flavor": "u",
            "forward_pid": forward_pid,
            "forward_is_u": 1 if forward_pid == 2 else 0,
            "forward_is_d": 1 if forward_pid == 1 else 0,
            "has_forward_light_quark": 1,
            "Q2": Q2, "xB": xB, "y": y, "W2": W2,
            "ele_px": e_row["px"], "ele_py": e_row["py"], "ele_pz": e_row["pz"], "ele_E": e_row["E"], "ele_m": e_row["m"],
            "ele_pT": e_row["pT"], "ele_eta": e_row["eta"], "ele_phi": e_row["phi"], "ele_pplus": e_row["pplus"], "ele_pminus": e_row["pminus"],
            "struck_px": s_row["px"], "struck_py": s_row["py"], "struck_pz": s_row["pz"], "struck_E": s_row["E"], "struck_m": s_row["m"],
            "struck_pT": s_row["pT"], "struck_eta": s_row["eta"], "struck_phi": s_row["phi"], "struck_pplus": s_row["pplus"], "struck_pminus": s_row["pminus"],
            "forward_px": f_row["px"], "forward_py": f_row["py"], "forward_pz": f_row["pz"], "forward_E": f_row["E"], "forward_m": f_row["m"],
            "forward_pT": f_row["pT"], "forward_eta": f_row["eta"], "forward_phi": f_row["phi"], "forward_pplus": f_row["pplus"], "forward_pminus": f_row["pminus"],
            "forward_selector_energy_pid": int(ev[forward_energy_idx].id()),
            "forward_selector_pplus_pid": int(ev[forward_pplus_idx].id()),
        }

        if forward_pid == 2:
            chunk_u.append(row)
            counters["accepted_forward_u"] += 1
        else:
            chunk_d.append(row)
            counters["accepted_forward_d"] += 1
        counters["accepted_total"] += 1
        accepted_since_flush += 1

        if accepted_since_flush >= args.chunk_accepted:
            write_rows_csv(f_u_csv, chunk_u, cols)
            write_rows_csv(f_d_csv, chunk_d, cols)
            chunk_u.clear()
            chunk_d.clear()
            accepted_since_flush = 0

        do_progress = False
        if counters["accepted_forward_u"] >= next_u:
            do_progress = True
            while counters["accepted_forward_u"] >= next_u:
                next_u += args.progress_every
        if counters["accepted_forward_d"] >= next_d:
            do_progress = True
            while counters["accepted_forward_d"] >= next_d:
                next_d += args.progress_every
        if do_progress:
            elapsed = time.time() - t0
            au = counters["accepted_forward_u"]
            ad = counters["accepted_forward_d"]
            ru = max(0, args.target_forward_u - au)
            rd = max(0, args.target_forward_d - ad)
            rate_u = au / elapsed if elapsed > 0 else 0.0
            rate_d = ad / elapsed if elapsed > 0 else 0.0
            eta = max((ru / rate_u) if rate_u > 0 else float("inf"), (rd / rate_d) if rate_d > 0 else float("inf"))
            print(
                f"[progress] gen={counters['generated_total']} "
                f"acc_forward_u={au}/{args.target_forward_u} "
                f"acc_forward_d={ad}/{args.target_forward_d} "
                f"elapsed_s={elapsed:.1f} eta_remaining_s={eta:.1f}",
                flush=True,
            )

    # flush remainder
    write_rows_csv(f_u_csv, chunk_u, cols)
    write_rows_csv(f_d_csv, chunk_d, cols)

    # combined
    df_u = pd.read_csv(f_u_csv)
    df_d = pd.read_csv(f_d_csv)
    df_c = pd.concat([df_u, df_d], ignore_index=True)
    df_c.to_csv(comb_csv, index=False, float_format="%.10g")

    counters["n_forward_u_rows"] = int(df_u.shape[0])
    counters["n_forward_d_rows"] = int(df_d.shape[0])
    counters["n_combined_rows"] = int(df_c.shape[0])
    counters["acceptance_forward_u_over_generated"] = counters["accepted_forward_u"] / counters["generated_total"] if counters["generated_total"] else float("nan")
    counters["acceptance_forward_d_over_generated"] = counters["accepted_forward_d"] / counters["generated_total"] if counters["generated_total"] else float("nan")
    counters["elapsed_seconds"] = time.time() - t0
    summary_path.write_text(json.dumps(counters, indent=2), encoding="utf-8")

    readme_path.write_text(
        "\n".join(
            [
                "# 2->2 baseline: struck-u with forward light-quark tagging",
                "",
                "Definition used:",
                "- keep events where struck quark is u",
                "- forward light candidates: final-state quarks with pid in {1,2}, eta>0",
                "- choose hardest forward light by largest E (primary selector)",
                "- classify accepted event as forward-u (pid=2) or forward-d (pid=1)",
                "",
                "Stored objects per accepted event:",
                "- outgoing electron",
                "- outgoing struck u",
                "- hardest forward light quark",
                "",
                "Outputs:",
                f"- {f_u_csv}",
                f"- {f_d_csv}",
                f"- {comb_csv}",
                f"- {summary_path}",
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

