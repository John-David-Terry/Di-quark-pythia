#!/usr/bin/env python3
"""
Replay the v3 DIS background acceptance (same as generate_dis_background_final_state_parquet v3)
and compute light-cone momentum fraction of the **leading target-hemisphere charged pion**
(same Breit selection as jet–hadron: max Breit E among π± with pz_Breit > 0).

Definition (PYTHIA native lab, incoming proton direction **n** = **p**_p/|**p**_p|):

  z_LC = (E_h + **p**_h·**n**) / (E_p + **p**_p·**n**) = (E_h + **p**_h·**n**) / (E_p + |**p**_p|)

so the denominator is the proton light-cone ``+'' component along the actual proton
three-momentum (not necessarily +z). Using ``E + p_z'' alone is wrong when the proton
beam is not along +z.

Requires: pythia8, numpy, tqdm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
_ANAL = Path(__file__).resolve().parent
for _p in (_SRC, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diquark.analyze_events_raw import FLIP_Z_PTREL, flip_z  # noqa: E402

from generate_dis_background_final_state_parquet import build_pythia_background  # noqa: E402
from generate_dis_isr_parton_dataset import (  # noqa: E402
    extract_beams_from_event,
    hard_subprocess_outgoing_quark_lab_p4_and_index,
    pick_incoming_quark_index,
    try_build_lt_from_event,
)


def _leading_target_pion_lab_index(ev, LT: np.ndarray) -> int | None:
    """Breit-frame leading π± in current hemisphere; return PYTHIA lab index of winner."""
    best_e = -1.0
    best_i: int | None = None
    for i in range(ev.size()):
        p = ev[i]
        if not p.isFinal():
            continue
        if abs(int(p.id())) != 211:
            continue
        p4_lab = np.array([p.e(), p.px(), p.py(), p.pz()], dtype=np.float64)
        p4_b = LT @ flip_z(p4_lab, FLIP_Z_PTREL)
        if float(p4_b[3]) <= 0.0:
            continue
        eb = float(p4_b[0])
        if eb > best_e:
            best_e = eb
            best_i = int(i)
    return best_i


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-accepted", type=int, default=900_000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument(
        "--out-npy",
        type=Path,
        default=None,
        help="Optional path to save z_LC float32 array (finite values only).",
    )
    ap.add_argument(
        "--triple-cut",
        type=float,
        nargs=3,
        metavar=("Z_MIN", "PHT_MIN", "PJT_MIN"),
        default=None,
        help="Count accepted events with z_LC>Z_MIN, P_hT>PHT_MIN, P_jT>PJT_MIN (Breit GeV). Example: --triple-cut 0.2 0.2 0.2",
    )
    ap.add_argument(
        "--write-triple-cut-event-ids",
        type=Path,
        default=None,
        metavar="PATH.npy",
        help="With --triple-cut, save int64 array of event_id (accepted index) for each passing event.",
    )
    args = ap.parse_args()
    if args.write_triple_cut_event_ids is not None and args.triple_cut is None:
        raise SystemExit("--write-triple-cut-event-ids requires --triple-cut")

    try:
        from tqdm import tqdm
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("tqdm required") from exc

    n_target = int(args.n_accepted)
    pythia = build_pythia_background(int(args.seed), hadron_level=True)
    ev = pythia.event

    z_list: list[float] = []
    n_no_pion = 0
    n_bad_Pplus = 0
    n_triple_cut = 0
    triple_cut_event_ids: list[int] = []
    cut_z: float | None = None
    cut_pht: float | None = None
    cut_pjt: float | None = None
    if args.triple_cut is not None:
        cut_z, cut_pht, cut_pjt = (float(x) for x in args.triple_cut)

    accepted = 0
    total_gen = 0

    pbar = tqdm(total=n_target, desc="Replay v3 + z_LC", unit="evt")
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
        p4_lab, _oq = hard_subprocess_outgoing_quark_lab_p4_and_index(proc)
        if p4_lab is None:
            continue

        LT = try_build_lt_from_event(ev)
        if LT is None:
            continue

        beams = extract_beams_from_event(ev)
        if beams is None:
            continue
        _e_in, _e_sc, p_in = beams
        Ep = float(p_in[0])
        pp = np.asarray(p_in[1:4], dtype=np.float64)
        pnorm = float(np.linalg.norm(pp))
        if pnorm <= 1e-18:
            n_bad_Pplus += 1
            accepted += 1
            pbar.update(1)
            continue
        n_hat = pp / pnorm
        P_plus = Ep + float(np.dot(pp, n_hat))
        if not np.isfinite(P_plus) or P_plus <= 1e-12:
            n_bad_Pplus += 1
            accepted += 1
            pbar.update(1)
            continue

        i_pi = _leading_target_pion_lab_index(ev, LT)
        if i_pi is None:
            n_no_pion += 1
            accepted += 1
            pbar.update(1)
            continue

        p = ev[i_pi]
        ph = np.array([p.px(), p.py(), p.pz()], dtype=np.float64)
        h_plus = float(p.e() + float(np.dot(ph, n_hat)))
        z_lc = h_plus / P_plus

        p4_pi_lab = np.array([p.e(), p.px(), p.py(), p.pz()], dtype=np.float64)
        p4_pi_b = LT @ flip_z(p4_pi_lab, FLIP_Z_PTREL)
        p_hT = float(np.hypot(float(p4_pi_b[1]), float(p4_pi_b[2])))

        p4_k_b = LT @ flip_z(p4_lab, FLIP_Z_PTREL)
        p_jT = float(np.hypot(float(p4_k_b[1]), float(p4_k_b[2])))

        if np.isfinite(z_lc) and z_lc >= 0.0:
            z_list.append(z_lc)

        if (
            cut_z is not None
            and np.isfinite(z_lc)
            and z_lc > cut_z
            and p_hT > cut_pht
            and p_jT > cut_pjt
        ):
            n_triple_cut += 1
            if args.write_triple_cut_event_ids is not None:
                triple_cut_event_ids.append(int(accepted))

        accepted += 1
        pbar.update(1)

    pbar.close()

    z = np.asarray(z_list, dtype=np.float64)
    print(f"accepted={accepted}  generated_tries={total_gen}  z_LC entries={len(z)}")
    print(f"no_leading_pion={n_no_pion}  bad_Pplus={n_bad_Pplus}")
    if cut_z is not None:
        print(
            f"triple_cut (z_LC>{cut_z}, P_hT>{cut_pht}, P_jT>{cut_pjt} GeV Breit): {n_triple_cut}"
        )
    if args.write_triple_cut_event_ids is not None:
        out_ids = args.write_triple_cut_event_ids.expanduser().resolve()
        out_ids.parent.mkdir(parents=True, exist_ok=True)
        eid_arr = np.asarray(triple_cut_event_ids, dtype=np.int64)
        np.save(out_ids, eid_arr)
        print(f"Wrote {len(eid_arr)} event_ids -> {out_ids}")
    if z.size == 0:
        raise SystemExit("no z_LC values")

    for p in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 100):
        print(f"  p{p:g}: {np.percentile(z, p):.6g}")
    print(f"  mean: {np.mean(z):.6g}  std: {np.std(z):.6g}")

    if args.out_npy is not None:
        args.out_npy = args.out_npy.expanduser().resolve()
        args.out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out_npy, z.astype(np.float32))
        print(f"Wrote {args.out_npy}")


if __name__ == "__main__":
    main()
