#!/usr/bin/env python3
"""
One-event test: replace final-state (ud) diquark remnant with a 3-body split,
then Level-0 reinjection via event.append(id, status, col, acol, px, py, pz, E, m).

Reads the fixed control snapshot:
  outputs/dis_isr_parton_dataset/control_event_full_record.txt

No full history replay, no LHE, no junctions.

Color assignment (required):
  Original diquark: col=0, acol=B.
  B = original diquark anticolor; A = a new tag not used elsewhere in the record.
  After split:
    stripped quark:    col=A, acol=0
    new diquark:       col=0, acol=A
    created antiquark: col=0, acol=B
  Moves B from the old diquark endpoint onto the new antiquark; opens string A between
  stripped quark and new diquark.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import pythia8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pythia8 import failed: {exc}") from exc

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

DEFAULT_RECORD = outputs_dir() / "dis_isr_parton_dataset" / "control_event_full_record.txt"

# Light constituent masses (GeV), same spirit as minimal reinjection tests
M_LIGHT = 0.33

# Pythia8 diquark IDs (no separate "du" entry in ParticleData.xml; ud_1 used for channel A)
DIQUARK_UD0 = 2101
DIQUARK_UD1 = 2103
DIQUARK_DD1 = 1103
DIQUARK_UU1 = 2203

# Final-state hadronizing status (works with direct append in this project)
STATUS_INJECT = 23

CONTROL_DIQUARK_INDEX = 32


@dataclass
class ParticleRow:
    index: int
    pid: int
    status: int
    mother1: int
    mother2: int
    daughter1: int
    daughter2: int
    px: float
    py: float
    pz: float
    e: float
    m: float
    col: int
    acol: int

    def is_final_colored_qcd(self) -> bool:
        if self.status <= 0 or self.daughter1 != 0 or self.daughter2 != 0:
            return False
        ap = abs(self.pid)
        if ap == 21:
            return True
        if 1 <= ap <= 6:
            return True
        if 1000 <= ap < 10000 and (ap // 10) % 10 == 0:
            return True
        return False


def load_one_event(path: Path) -> List[ParticleRow]:
    rows: List[ParticleRow] = []
    text = path.read_text(encoding="utf-8")
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("index "):
            continue
        parts = s.split()
        if len(parts) != 14:
            continue
        rows.append(
            ParticleRow(
                index=int(parts[0]),
                pid=int(parts[1]),
                status=int(parts[2]),
                mother1=int(parts[3]),
                mother2=int(parts[4]),
                daughter1=int(parts[5]),
                daughter2=int(parts[6]),
                px=float(parts[7]),
                py=float(parts[8]),
                pz=float(parts[9]),
                e=float(parts[10]),
                m=float(parts[11]),
                col=int(parts[12]),
                acol=int(parts[13]),
            )
        )
    rows.sort(key=lambda r: r.index)
    return rows


def find_final_state_ud_diquark(rows: Sequence[ParticleRow]) -> ParticleRow:
    cands = [r for r in rows if r.is_final_colored_qcd() and abs(r.pid) == DIQUARK_UD0]
    if not cands:
        raise RuntimeError("No final-state ud_0 (2101) diquark found in event record")
    by_idx = {r.index: r for r in cands}
    if CONTROL_DIQUARK_INDEX in by_idx:
        return by_idx[CONTROL_DIQUARK_INDEX]
    cands.sort(key=lambda r: r.index)
    return cands[0]


def choose_split_channel(rng: random.Random) -> str:
    return rng.choice(["A", "B", "C", "D"])


def channel_particles(ch: str) -> Tuple[int, int, int]:
    """Returns (stripped_quark_id, daughter_diquark_id, antiquark_id)."""
    if ch == "A":
        # (ud) -> u + (du) + ubar ; Pythia has no separate du diquark -> use ud_1
        return 2, DIQUARK_UD1, -2
    if ch == "B":
        return 2, DIQUARK_DD1, -1
    if ch == "C":
        return 1, DIQUARK_UU1, -2
    if ch == "D":
        return 1, DIQUARK_UD0, -1
    raise ValueError(ch)


def mass_for_pid(pythia: "pythia8.Pythia", pid: int) -> float:
    ap = abs(pid)
    if ap in (1, 2, 3):
        return M_LIGHT
    if 1000 <= ap < 10000:
        return float(pythia.particleData.m0(ap))
    return M_LIGHT


def build_split_daughters(
    pythia: "pythia8.Pythia",
    diquark: ParticleRow,
    channel: str,
) -> Tuple[ParticleRow, ParticleRow, ParticleRow]:
    stripped_id, dq_id, aq_id = channel_particles(channel)
    px, py, pz = diquark.px, diquark.py, diquark.pz
    f0, f1, f2 = 0.5, 0.05, 0.45
    ps = (f0 * px, f0 * py, f0 * pz)
    pa = (f1 * px, f1 * py, f1 * pz)
    pd = (f2 * px, f2 * py, f2 * pz)

    ms = mass_for_pid(pythia, stripped_id)
    ma = mass_for_pid(pythia, aq_id)
    md = mass_for_pid(pythia, dq_id)

    def e_on_shell(p: Tuple[float, float, float], m: float) -> float:
        x, y, z = p
        return math.sqrt(max(0.0, x * x + y * y + z * z + m * m))

    es, ea, ed = e_on_shell(ps, ms), e_on_shell(pa, ma), e_on_shell(pd, md)

    # Synthetic indices for bookkeeping (not passed to append)
    base = 9000
    stripped = ParticleRow(
        base,
        stripped_id,
        STATUS_INJECT,
        0,
        0,
        0,
        0,
        ps[0],
        ps[1],
        ps[2],
        es,
        ms,
        0,
        0,
    )
    antiq = ParticleRow(
        base + 1,
        aq_id,
        STATUS_INJECT,
        0,
        0,
        0,
        0,
        pa[0],
        pa[1],
        pa[2],
        ea,
        ma,
        0,
        0,
    )
    new_dq = ParticleRow(
        base + 2,
        dq_id,
        STATUS_INJECT,
        0,
        0,
        0,
        0,
        pd[0],
        pd[1],
        pd[2],
        ed,
        md,
        0,
        0,
    )
    return stripped, antiq, new_dq


def assign_split_colors(
    diquark: ParticleRow,
    stripped: ParticleRow,
    antiq: ParticleRow,
    new_dq: ParticleRow,
    new_tag_a: int,
) -> Tuple[ParticleRow, ParticleRow, ParticleRow]:
    """
    B = original diquark acol; A = fresh unused tag.
    stripped: col=A acol=0; new_dq: col=0 acol=A; antiquark: col=0 acol=B.
    """
    B = int(diquark.acol)
    if B <= 0:
        raise RuntimeError("Expected diquark with positive acol=B (open anticolor endpoint)")
    A = int(new_tag_a)
    if A <= 0:
        raise RuntimeError("New color tag A must be positive")
    s, q, d = stripped, antiq, new_dq
    return (
        ParticleRow(
            s.index, s.pid, s.status, 0, 0, 0, 0, s.px, s.py, s.pz, s.e, s.m, A, 0
        ),
        ParticleRow(
            q.index, q.pid, q.status, 0, 0, 0, 0, q.px, q.py, q.pz, q.e, q.m, 0, B
        ),
        ParticleRow(
            d.index, d.pid, d.status, 0, 0, 0, 0, d.px, d.py, d.pz, d.e, d.m, 0, A
        ),
    )


def replace_in_final_state_list(
    rows: Sequence[ParticleRow],
    diquark_index: int,
    stripped: ParticleRow,
    antiq: ParticleRow,
    new_dq: ParticleRow,
) -> List[ParticleRow]:
    out: List[ParticleRow] = []
    for r in rows:
        if not r.is_final_colored_qcd():
            continue
        if r.index == diquark_index:
            out.extend([stripped, antiq, new_dq])
        else:
            out.append(r)
    out.sort(key=lambda r: r.index)
    return out


def normalize_reinject_status(rows: Sequence[ParticleRow]) -> List[ParticleRow]:
    """Level-0 direct append: use uniform status=23 for all colored final partons."""
    return [
        ParticleRow(
            r.index,
            r.pid,
            STATUS_INJECT,
            0,
            0,
            0,
            0,
            r.px,
            r.py,
            r.pz,
            r.e,
            r.m,
            r.col,
            r.acol,
        )
        for r in rows
    ]


def collect_tags(rows: Sequence[ParticleRow]) -> Dict[int, Tuple[int, int]]:
    """tag -> (count_as_col, count_as_acol)."""
    d: Dict[int, Tuple[int, int]] = {}
    for r in rows:
        if r.col > 0:
            c, a = d.get(r.col, (0, 0))
            d[r.col] = (c + 1, a)
        if r.acol > 0:
            c, a = d.get(r.acol, (0, 0))
            d[r.acol] = (c, a + 1)
    return d


def validate_color_flow(rows: Sequence[ParticleRow]) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    tags = collect_tags(rows)
    ok = True
    for t, (nc, na) in sorted(tags.items()):
        line = f"tag {t}: n_col={nc} n_acol={na}"
        msgs.append(line)
        if nc != 1 or na != 1:
            ok = False
            msgs.append(f"  -> imbalance for tag {t}")
    return ok, msgs


def validate_momentum_subset(
    diquark: ParticleRow,
    stripped: ParticleRow,
    antiq: ParticleRow,
    new_dq: ParticleRow,
) -> Tuple[bool, str]:
    sx = stripped.px + antiq.px + new_dq.px - diquark.px
    sy = stripped.py + antiq.py + new_dq.py - diquark.py
    sz = stripped.pz + antiq.pz + new_dq.pz - diquark.pz
    err = max(abs(sx), abs(sy), abs(sz))
    if err > 1e-9:
        return False, f"3-momentum closure max|delta|={err:.3e}"
    return True, f"3-momentum closure ok (max|delta|<1e-9)"


def reinject_with_pythia(rows: Sequence[ParticleRow], fsr: bool = True) -> Tuple[bool, str, int, int]:
    p = pythia8.Pythia()
    p.readString("Beams:idA = 2212")
    p.readString("Beams:idB = 2212")
    p.readString("Beams:eCM = 200.0")
    p.readString("ProcessLevel:all = off")
    p.readString("PartonLevel:ISR = off")
    p.readString(f"PartonLevel:FSR = {'on' if fsr else 'off'}")
    p.readString("PartonLevel:MPI = off")
    p.readString("HadronLevel:all = on")
    p.readString("Print:quiet = on")
    if not p.init():
        return False, "pythia.init() failed", 0, 0

    p.event.reset()
    err = ""
    try:
        for r in rows:
            p.event.append(
                int(r.pid),
                int(r.status),
                int(r.col),
                int(r.acol),
                float(r.px),
                float(r.py),
                float(r.pz),
                float(r.e),
                float(r.m),
            )
        ok = bool(p.next())
    except Exception as exc:
        ok = False
        err = f"{type(exc).__name__}: {exc}"

    n_final = n_had = 0
    if ok:
        for i in range(p.event.size()):
            pp = p.event[i]
            if pp.isFinal():
                n_final += 1
                if pp.isHadron():
                    n_had += 1
    return ok, err, n_final, n_had


def next_free_color_tag(rows: Sequence[ParticleRow]) -> int:
    """Smallest integer > 0 not appearing as col or acol in the given rows."""
    used: set[int] = set()
    for r in rows:
        if r.col > 0:
            used.add(r.col)
        if r.acol > 0:
            used.add(r.acol)
    if not used:
        return 501
    m = max(used)
    cand = m + 1
    while cand in used:
        cand += 1
    return cand


def print_particles(title: str, rows: Sequence[ParticleRow]) -> None:
    print(f"\n=== {title} ===")
    print("id status col acol px py pz E m")
    for r in rows:
        print(
            f"{r.pid:5d} {r.status:4d} {r.col:4d} {r.acol:4d} "
            f"{r.px: .6f} {r.py: .6f} {r.pz: .6f} {r.e: .6f} {r.m: .6f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="One-event diquark 3-body split + Level-0 reinject")
    ap.add_argument(
        "--record",
        type=Path,
        default=DEFAULT_RECORD,
        help="Path to control_event_full_record.txt",
    )
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed for channel choice")
    ap.add_argument(
        "--channel",
        choices=["A", "B", "C", "D"],
        default=None,
        help="Force split channel (default: random from seed)",
    )
    ap.add_argument(
        "--no-fsr",
        action="store_true",
        help="Turn PartonLevel:FSR off (still hadronization on)",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_rows = load_one_event(args.record)
    diquark = find_final_state_ud_diquark(all_rows)
    channel = args.channel if args.channel else choose_split_channel(rng)

    ptmp = pythia8.Pythia()
    ptmp.readString("Print:quiet = on")
    ptmp.init()

    stripped0, antiq0, new_dq0 = build_split_daughters(ptmp, diquark, channel)
    tag_a = next_free_color_tag(all_rows)
    stripped, antiq, new_dq = assign_split_colors(
        diquark, stripped0, antiq0, new_dq0, new_tag_a=tag_a
    )

    inject_list = normalize_reinject_status(
        replace_in_final_state_list(all_rows, diquark.index, stripped, antiq, new_dq)
    )

    print("=== ORIGINAL DIQUARK (from record) ===")
    print(
        f"index={diquark.index} id={diquark.pid} status={diquark.status} "
        f"col={diquark.col} acol={diquark.acol}"
    )
    print(
        f"momentum px,py,pz,E,m = {diquark.px}, {diquark.py}, {diquark.pz}, {diquark.e}, {diquark.m}"
    )

    print("\n=== SPLIT ===")
    print(f"channel={channel} (seed={args.seed})")
    sq, dq_id, aq_id = channel_particles(channel)
    print(f"stripped_quark_id={sq} daughter_diquark_id={dq_id} antiquark_id={aq_id}")
    if channel == "A":
        print("note: Pythia8 has no separate (du) diquark in ParticleData.xml; using ud_1 (2103).")

    print(
        f"color: B=original_diquark_acol={diquark.acol} "
        f"A=new_unused_tag={tag_a}"
    )
    print("momentum fractions: stripped=0.5 antiquark=0.05 daughter_diquark=0.45")
    print("\nDaughters (after color assign):")
    for lab, r in ("stripped", stripped), ("antiq", antiq), ("daughter_diquark", new_dq):
        print(
            f"  {lab}: id={r.pid} status={r.status} col={r.col} acol={r.acol} "
            f"p=({r.px},{r.py},{r.pz}) E={r.e} m={r.m}"
        )

    mom_ok, mom_msg = validate_momentum_subset(diquark, stripped, antiq, new_dq)
    print(f"\n=== MOMENTUM ===\n{mom_msg}")

    col_ok, col_msgs = validate_color_flow(inject_list)
    print("\n=== COLOR TAG SUMMARY (n_col / n_acol per tag) ===")
    for ln in col_msgs:
        print(ln)
    print(f"color_flow_ok={col_ok}")

    print_particles("PARTICLES APPENDED (Level-0 reinjection list)", inject_list)

    if not col_ok:
        print("\npythia.next() skipped because color validation failed.")
        return

    ok, err, n_final, n_had = reinject_with_pythia(inject_list, fsr=not args.no_fsr)
    print(f"\n=== PYTHIA ===\npythia.next() success={ok}")
    if err:
        print(f"exception: {err}")
    print(f"n_final_particles={n_final} n_final_hadrons={n_had}")
    if not ok and not err:
        print("(Check stderr for PYTHIA Error/Warning lines from forceHadronLevel.)")


if __name__ == "__main__":
    main()
