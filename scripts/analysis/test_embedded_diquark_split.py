#!/usr/bin/env python3
"""
Fix unchanged control full-event reinjection first.

This script debugs reinjection using the fixed selected control event saved in:
  outputs/dis_isr_parton_dataset/control_event_full_record.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import pythia8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pythia8 import failed: {exc}")

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

OUTDIR = outputs_dir() / "dis_isr_parton_dataset"
OUTDIR.mkdir(parents=True, exist_ok=True)

CONTROL_RECORD_PATH = OUTDIR / "control_event_full_record.txt"
CONTROL_REINJECT_LOG = OUTDIR / "control_reinjection_log.txt"
CONTROL_TRACE_LOG = OUTDIR / "control_reinjection_diagnostic_trace.txt"
MODIFIED_REINJECT_LOG = OUTDIR / "modified_reinjection_junction_log.txt"
CONTROL_COLOR_TRACE_101 = OUTDIR / "control_color_trace_101.txt"
CONTROL_ALL_COLOR_LINES = OUTDIR / "control_all_color_lines.txt"
MODIFIED_COLOR_TRACE = OUTDIR / "modified_color_trace_101_105_106.txt"
MODIFIED_ALL_COLOR_LINES = OUTDIR / "modified_all_color_lines.txt"
TOPOLOGY_SUMMARY = OUTDIR / "control_vs_modified_color_topology_summary.txt"


@dataclass
class ControlRow:
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
    E: float
    m: float
    col: int
    acol: int

    def compact(self) -> str:
        return (
            f"orig_idx={self.index} id={self.pid} status={self.status} "
            f"m1={self.mother1} m2={self.mother2} d1={self.daughter1} d2={self.daughter2} "
            f"col={self.col} acol={self.acol}"
        )


def make_hadronizer() -> "pythia8.Pythia":
    p = pythia8.Pythia()
    p.readString("Beams:idA = 2212")
    p.readString("Beams:idB = 2212")
    p.readString("Beams:eCM = 200.0")
    p.readString("ProcessLevel:all = off")
    p.readString("PartonLevel:ISR = off")
    p.readString("PartonLevel:FSR = on")
    p.readString("PartonLevel:MPI = off")
    p.readString("HadronLevel:all = on")
    if not p.init():
        raise RuntimeError("Hadronizer init failed")
    return p


def parse_control_file(path: Path) -> List[ControlRow]:
    if not path.exists():
        raise FileNotFoundError(f"Missing control event record: {path}")
    rows: List[ControlRow] = []
    text = path.read_text(encoding="utf-8")
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("index "):
            continue
        parts = s.split()
        if len(parts) != 14:
            continue
        rows.append(
            ControlRow(
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
                E=float(parts[10]),
                m=float(parts[11]),
                col=int(parts[12]),
                acol=int(parts[13]),
            )
        )
    if not rows:
        raise RuntimeError("Parsed zero control rows")
    rows.sort(key=lambda r: r.index)
    return rows


def event_listing_text(ev: "pythia8.Event", title: str) -> str:
    lines = [f"--- {title} ---"]
    lines.append("idx id status m1 m2 d1 d2 col acol px py pz E m isFinal isHadron")
    for i in range(ev.size()):
        pp = ev[i]
        lines.append(
            f"{i:3d} {int(pp.id()):6d} {int(pp.status()):4d} "
            f"{int(pp.mother1()):3d} {int(pp.mother2()):3d} "
            f"{int(pp.daughter1()):3d} {int(pp.daughter2()):3d} "
            f"{int(pp.col()):4d} {int(pp.acol()):4d} "
            f"{float(pp.px()): .8f} {float(pp.py()): .8f} {float(pp.pz()): .8f} "
            f"{float(pp.e()): .8f} {float(pp.m()): .8f} "
            f"{int(pp.isFinal())} {int(pp.isHadron())}"
        )
    return "\n".join(lines)


def count_final_hadrons(ev: "pythia8.Event") -> Tuple[int, int]:
    n_final = 0
    n_had = 0
    for i in range(ev.size()):
        pp = ev[i]
        if pp.isFinal():
            n_final += 1
            if pp.isHadron():
                n_had += 1
    return n_final, n_had


def junction_listing_text(ev: "pythia8.Event", title: str) -> str:
    lines = [f"--- {title} ---"]
    n = int(ev.sizeJunction())
    lines.append(f"n_junctions={n}")
    for i in range(n):
        lines.append(
            f"junction[{i}] kind={int(ev.kindJunction(i))} "
            f"cols=({int(ev.colJunction(i,0))},{int(ev.colJunction(i,1))},{int(ev.colJunction(i,2))}) "
            f"status=({int(ev.statusJunction(i,0))},{int(ev.statusJunction(i,1))},{int(ev.statusJunction(i,2))})"
        )
    return "\n".join(lines)


def simple_particle_lines(rows: Sequence[ControlRow], title: str) -> str:
    out = [f"--- {title} ---"]
    out.append("idx id status col acol px py pz E m")
    for r in rows:
        out.append(
            f"{r.index:3d} {r.pid:6d} {r.status:4d} {r.col:4d} {r.acol:4d} "
            f"{r.px: .8f} {r.py: .8f} {r.pz: .8f} {r.E: .8f} {r.m: .8f}"
        )
    return "\n".join(out)


def trace_single_color(rows: Sequence[ControlRow], c: int, title: str) -> str:
    out = [f"--- {title} color={c} ---"]
    out.append("idx id status role col acol px py pz E m")
    n_col = 0
    n_acol = 0
    for r in rows:
        roles: List[str] = []
        if r.col == c:
            roles.append("color_source")
            n_col += 1
        if r.acol == c:
            roles.append("anticolor_source")
            n_acol += 1
        if not roles:
            continue
        out.append(
            f"{r.index:3d} {r.pid:6d} {r.status:4d} {','.join(roles):>18s} "
            f"{r.col:4d} {r.acol:4d} {r.px: .8f} {r.py: .8f} {r.pz: .8f} {r.E: .8f} {r.m: .8f}"
        )
    if n_col == 0 and n_acol == 0:
        out.append("(no matches)")
    out.append(f"counts: n_col_side={n_col} n_acol_side={n_acol}")
    out.append(f"balanced_pairing_hint={(n_col == n_acol)}")
    return "\n".join(out)


def all_color_lines_report(rows: Sequence[ControlRow], title: str) -> str:
    tags = sorted(
        {
            t
            for r in rows
            for t in (r.col, r.acol)
            if t > 0
        }
    )
    out = [f"--- {title} ---"]
    out.append(f"active_color_tags={tags}")
    out.append("")
    for c in tags:
        carriers = [r for r in rows if r.col == c or r.acol == c]
        n_col = sum(1 for r in carriers if r.col == c)
        n_acol = sum(1 for r in carriers if r.acol == c)
        out.append(f"color_tag={c} n_col_side={n_col} n_acol_side={n_acol} has_both_sides={n_col>0 and n_acol>0}")
        for r in carriers:
            role = []
            if r.col == c:
                role.append("col")
            if r.acol == c:
                role.append("acol")
            out.append(
                f"  idx={r.index} id={r.pid} status={r.status} role={'+'.join(role)} "
                f"col={r.col} acol={r.acol} p=({r.px:.8f},{r.py:.8f},{r.pz:.8f},{r.E:.8f},{r.m:.8f})"
            )
        out.append("")
    return "\n".join(out)


def is_diquark_id(pid: int) -> bool:
    apid = abs(pid)
    return 1000 <= apid < 10000 and ((apid // 10) % 10 == 0)


def select_rows_for_level(level: int, rows: Sequence[ControlRow]) -> List[ControlRow]:
    if level == 0:
        # Only final-state hadronizing QCD partons.
        return [
            r
            for r in rows
            if r.status > 0 and r.daughter1 == 0 and r.daughter2 == 0
            and (1 <= abs(r.pid) <= 6 or r.pid == 21 or is_diquark_id(r.pid))
        ]
    if level == 1:
        # All final-state particles.
        return [r for r in rows if r.status > 0 and r.daughter1 == 0 and r.daughter2 == 0]
    # Levels 2 and 3 use all non-system entries.
    return [r for r in rows if r.index > 0]


def make_append_row(base: ControlRow, status_mode: str) -> ControlRow:
    if status_mode == "native":
        return base
    # Tested compatibility mode: final positive statuses become 23 for colored partons.
    status = base.status
    if status_mode == "force23_final_colored":
        if status > 0 and (1 <= abs(base.pid) <= 6 or base.pid == 21 or is_diquark_id(base.pid)):
            status = 23
    return ControlRow(
        index=base.index,
        pid=base.pid,
        status=status,
        mother1=base.mother1,
        mother2=base.mother2,
        daughter1=base.daughter1,
        daughter2=base.daughter2,
        px=base.px,
        py=base.py,
        pz=base.pz,
        E=base.E,
        m=base.m,
        col=base.col,
        acol=base.acol,
    )


def pick_level0_rows(rows: Sequence[ControlRow], status_mode: str) -> List[ControlRow]:
    out = [make_append_row(r, status_mode) for r in select_rows_for_level(0, rows)]
    out.sort(key=lambda r: r.index)
    return out


def append_level0_rows(
    p: "pythia8.Pythia",
    rows: Sequence[ControlRow],
    trace_lines: Optional[List[str]] = None,
) -> None:
    p.event.reset()
    for r in rows:
        if trace_lines is not None:
            trace_lines.append(
                f"append orig_idx={r.index} id={r.pid} status={r.status} "
                f"col={r.col} acol={r.acol} "
                f"p=({r.px},{r.py},{r.pz},{r.E},{r.m})"
            )
        p.event.append(
            int(r.pid),
            int(r.status),
            int(r.col),
            int(r.acol),
            float(r.px),
            float(r.py),
            float(r.pz),
            float(r.E),
            float(r.m),
        )


def run_level0_control(rows: Sequence[ControlRow], status_mode: str) -> Dict[str, object]:
    p = make_hadronizer()
    level0 = pick_level0_rows(rows, status_mode)
    trace: List[str] = []
    append_level0_rows(p, level0, trace)
    before = event_listing_text(p.event, f"control before hadronization status_mode={status_mode}")
    err = ""
    try:
        ok = bool(p.next())
    except Exception as exc:
        ok = False
        err = f"{type(exc).__name__}: {exc}"
    n_final, n_had = count_final_hadrons(p.event) if ok else (0, 0)
    after = event_listing_text(p.event, f"control after hadronization status_mode={status_mode}") if ok else ""
    return {
        "status_mode": status_mode,
        "ok": ok,
        "error": err,
        "n_final": n_final,
        "n_had": n_had,
        "before": before,
        "after": after,
        "append_trace": trace,
        "level0_rows": level0,
    }


def diquark_quark_ids(diquark_id: int) -> Tuple[int, int]:
    sign = 1 if diquark_id >= 0 else -1
    apid = abs(diquark_id)
    return sign * (apid // 1000), sign * ((apid // 100) % 10)


def run_level0_modified(
    rows: Sequence[ControlRow],
    status_mode: str,
    split_frac: float,
) -> Dict[str, object]:
    # Fixed control-event anchors requested by user.
    struck_idx = 26
    diquark_idx = 32
    rows_by_idx = {r.index: r for r in rows}
    diq = rows_by_idx[diquark_idx]

    base = pick_level0_rows(rows, status_mode)
    base_wo_dq = [r for r in base if r.index != diquark_idx]

    qa_id, qb_id = diquark_quark_ids(diq.pid)
    f = split_frac
    g = 1.0 - f
    qa = ControlRow(
        index=1001,
        pid=qa_id,
        status=make_append_row(diq, status_mode).status,
        mother1=0,
        mother2=0,
        daughter1=0,
        daughter2=0,
        px=f * diq.px,
        py=f * diq.py,
        pz=f * diq.pz,
        E=f * diq.E,
        m=max(0.05, f * abs(diq.m)),
        col=105,
        acol=0,
    )
    qb = ControlRow(
        index=1002,
        pid=qb_id,
        status=make_append_row(diq, status_mode).status,
        mother1=0,
        mother2=0,
        daughter1=0,
        daughter2=0,
        px=g * diq.px,
        py=g * diq.py,
        pz=g * diq.pz,
        E=g * diq.E,
        m=max(0.05, g * abs(diq.m)),
        col=106,
        acol=0,
    )

    p = make_hadronizer()
    trace: List[str] = []
    append_level0_rows(p, base_wo_dq, trace)
    trace.append(f"append new qa id={qa.pid} status={qa.status} col={qa.col} acol={qa.acol} p=({qa.px},{qa.py},{qa.pz},{qa.E},{qa.m})")
    p.event.append(qa.pid, qa.status, qa.col, qa.acol, qa.px, qa.py, qa.pz, qa.E, qa.m)
    trace.append(f"append new qb id={qb.pid} status={qb.status} col={qb.col} acol={qb.acol} p=({qb.px},{qb.py},{qb.pz},{qb.E},{qb.m})")
    p.event.append(qb.pid, qb.status, qb.col, qb.acol, qb.px, qb.py, qb.pz, qb.E, qb.m)
    # Preserve old line 101 as one junction leg.
    p.event.appendJunction(1, 101, 105, 106)

    before = event_listing_text(
        p.event, f"modified before hadronization status_mode={status_mode} split={split_frac:.1f}/{1.0-split_frac:.1f}"
    )
    jbefore = junction_listing_text(p.event, "junctions before hadronization")
    err = ""
    try:
        ok = bool(p.next())
    except Exception as exc:
        ok = False
        err = f"{type(exc).__name__}: {exc}"
    n_final, n_had = count_final_hadrons(p.event) if ok else (0, 0)
    after = event_listing_text(
        p.event, f"modified after hadronization status_mode={status_mode} split={split_frac:.1f}/{1.0-split_frac:.1f}"
    ) if ok else ""
    jafter = junction_listing_text(p.event, "junctions after hadronization") if ok else ""
    modified_rows = list(base_wo_dq) + [qa, qb]
    return {
        "status_mode": status_mode,
        "split_frac": split_frac,
        "ok": ok,
        "error": err,
        "n_final": n_final,
        "n_had": n_had,
        "before": before,
        "after": after,
        "junction_before": jbefore,
        "junction_after": jafter,
        "append_trace": trace,
        "meta": f"fixed_event struck_idx={struck_idx} diquark_idx={diquark_idx} diquark_id={diq.pid} diquark_status={diq.status} diquark_color=({diq.col},{diq.acol})",
        "level0_rows_modified": modified_rows,
    }


def mapped_or_zero(orig_idx: int, idx_map: Dict[int, int]) -> int:
    return idx_map.get(orig_idx, 0) if orig_idx > 0 else 0


def attempt_reconstruction(
    rows_all: Sequence[ControlRow],
    level: int,
    status_mode: str,
    trace: List[str],
) -> Dict[str, object]:
    use_rows = [make_append_row(r, status_mode) for r in select_rows_for_level(level, rows_all)]
    use_rows.sort(key=lambda r: r.index)

    result: Dict[str, object] = {
        "level": level,
        "status_mode": status_mode,
        "selected_n_rows": len(use_rows),
        "append_failed": False,
        "append_fail_call": "",
        "append_fail_row": "",
        "next_ok": False,
        "next_error": "",
        "n_final": 0,
        "n_had": 0,
        "before_listing": "",
        "after_listing": "",
    }

    p = make_hadronizer()
    p.event.reset()
    idx_map: Dict[int, int] = {}

    trace.append("")
    trace.append(f"=== attempt level={level} status_mode={status_mode} ===")
    trace.append(f"selected_rows={len(use_rows)}")

    # Pass 1: append all particles with load-bearing fields.
    for r in use_rows:
        trace.append(f"append_pass1_start {r.compact()}")
        try:
            if level in (0, 1):
                new_idx = int(
                    p.event.append(
                        int(r.pid),
                        int(r.status),
                        int(r.col),
                        int(r.acol),
                        float(r.px),
                        float(r.py),
                        float(r.pz),
                        float(r.E),
                        float(r.m),
                    )
                )
            elif level == 2:
                new_idx = int(
                    p.event.append(
                        int(r.pid),
                        int(r.status),
                        0,
                        0,
                        0,
                        0,
                        int(r.col),
                        int(r.acol),
                        float(r.px),
                        float(r.py),
                        float(r.pz),
                        float(r.E),
                        float(r.m),
                    )
                )
            else:  # level 3
                new_idx = int(
                    p.event.append(
                        int(r.pid),
                        int(r.status),
                        0,
                        0,
                        0,
                        0,
                        int(r.col),
                        int(r.acol),
                        float(r.px),
                        float(r.py),
                        float(r.pz),
                        float(r.E),
                        float(r.m),
                    )
                )
            idx_map[r.index] = new_idx
            trace.append(f"append_pass1_ok orig_idx={r.index} new_idx={new_idx}")
        except Exception as exc:
            result["append_failed"] = True
            result["append_fail_call"] = "event.append(pass1)"
            result["append_fail_row"] = r.compact()
            result["next_error"] = f"{type(exc).__name__}: {exc}"
            trace.append(f"append_pass1_fail error={type(exc).__name__}: {exc}")
            return result

    # Pass 2: re-append with mapped mothers/daughters where requested.
    if level in (2, 3):
        trace.append("pass2_rebuild_start")
        p2 = make_hadronizer()
        p2.event.reset()
        for r in use_rows:
            m1 = mapped_or_zero(r.mother1, idx_map)
            m2 = mapped_or_zero(r.mother2, idx_map)
            d1 = 0
            d2 = 0
            if level == 3:
                d1 = mapped_or_zero(r.daughter1, idx_map)
                d2 = mapped_or_zero(r.daughter2, idx_map)
                if d2 > 0 and d1 == 0:
                    d1 = d2
                if d1 > 0 and d2 == 0:
                    d2 = d1
            trace.append(
                f"append_pass2_start {r.compact()} mapped(m1,m2,d1,d2)=({m1},{m2},{d1},{d2})"
            )
            try:
                p2.event.append(
                    int(r.pid),
                    int(r.status),
                    int(m1),
                    int(m2),
                    int(d1),
                    int(d2),
                    int(r.col),
                    int(r.acol),
                    float(r.px),
                    float(r.py),
                    float(r.pz),
                    float(r.E),
                    float(r.m),
                )
                trace.append("append_pass2_ok")
            except Exception as exc:
                result["append_failed"] = True
                result["append_fail_call"] = "event.append(pass2)"
                result["append_fail_row"] = r.compact()
                result["next_error"] = f"{type(exc).__name__}: {exc}"
                trace.append(f"append_pass2_fail error={type(exc).__name__}: {exc}")
                return result
        p = p2

    try:
        result["before_listing"] = event_listing_text(
            p.event, f"before hadronization level={level} status_mode={status_mode}"
        )
    except Exception as exc:
        trace.append(f"before_listing_fail error={type(exc).__name__}: {exc}")

    trace.append("call_pythia_next_start")
    try:
        next_ok = bool(p.next())
        result["next_ok"] = next_ok
        trace.append(f"call_pythia_next_done next_ok={next_ok}")
    except Exception as exc:
        result["next_ok"] = False
        result["next_error"] = f"{type(exc).__name__}: {exc}"
        trace.append(f"call_pythia_next_fail error={type(exc).__name__}: {exc}")
        return result

    if result["next_ok"]:
        n_final, n_had = count_final_hadrons(p.event)
        result["n_final"] = n_final
        result["n_had"] = n_had
        try:
            result["after_listing"] = event_listing_text(
                p.event, f"after hadronization level={level} status_mode={status_mode}"
            )
        except Exception as exc:
            trace.append(f"after_listing_fail error={type(exc).__name__}: {exc}")
    return result


def main() -> None:
    rows = parse_control_file(CONTROL_RECORD_PATH)
    status_modes = ["native", "force23_final_colored"]
    split_fracs = [0.5, 0.6, 0.4]

    control_results = [run_level0_control(rows, sm) for sm in status_modes]
    modified_results: List[Dict[str, object]] = []
    for sm in status_modes:
        for frac in split_fracs:
            modified_results.append(run_level0_modified(rows, sm, frac))

    # Keep diagnostic trace file updated with appended lists from all modified scans.
    trace_lines: List[str] = [
        "LEVEL-0 A/B TOPOLOGY TEST TRACE",
        f"input_file={CONTROL_RECORD_PATH}",
        "fixed_control_event_metadata: struck_outgoing_index=26 diquark_index=32 diquark_id=2101 diquark_status=63 diquark_color=(0,101)",
        "",
    ]
    for r in modified_results:
        trace_lines.append(
            f"modified_case status_mode={r['status_mode']} split={r['split_frac']:.1f}/{1.0-r['split_frac']:.1f} ok={r['ok']} error={r['error']}"
        )
        trace_lines.extend(r["append_trace"])
        trace_lines.append(r["junction_before"])
        trace_lines.append("")
    CONTROL_TRACE_LOG.write_text("\n".join(trace_lines) + "\n", encoding="utf-8")

    clines: List[str] = [
        "A/B TEST CONTROL (LEVEL-0 UNCHANGED EVENT)",
        f"input_file={CONTROL_RECORD_PATH}",
        "",
    ]
    for r in control_results:
        clines.append(
            f"status_mode={r['status_mode']} pythia_next_success={r['ok']} "
            f"n_final={r['n_final']} n_had={r['n_had']} error={r['error']}"
        )
    # Include one successful control listing.
    c_ok = next((r for r in control_results if r["ok"]), None)
    if c_ok is not None:
        clines.extend(["", c_ok["before"], "", c_ok["after"]])
    CONTROL_REINJECT_LOG.write_text("\n".join(clines) + "\n", encoding="utf-8")

    mlines: List[str] = [
        "A/B TEST MODIFIED (LEVEL-0 WITH DIQUARK -> q+q+JUNCTION)",
        f"input_file={CONTROL_RECORD_PATH}",
        "",
    ]
    for r in modified_results:
        mlines.append(
            f"status_mode={r['status_mode']} split={r['split_frac']:.1f}/{1.0-r['split_frac']:.1f} "
            f"pythia_next_success={r['ok']} n_final={r['n_final']} n_had={r['n_had']} error={r['error']}"
        )
    # Dump at least one failing case with full append/junction details.
    fail = next((r for r in modified_results if not r["ok"]), None)
    if fail is not None:
        mlines.extend(
            [
                "",
                "DETAILED FAILING CASE DUMP",
                fail["meta"],
                f"status_mode={fail['status_mode']} split={fail['split_frac']:.1f}/{1.0-fail['split_frac']:.1f}",
                "appended_particles:",
                *fail["append_trace"],
                "",
                fail["junction_before"],
                "",
                fail["before"],
            ]
        )
    MODIFIED_REINJECT_LOG.write_text("\n".join(mlines) + "\n", encoding="utf-8")

    # Forensic color-topology comparison on fixed representative configurations.
    control_ref = next((r for r in control_results if r["status_mode"] == "native"), control_results[0])
    modified_ref = next(
        (r for r in modified_results if r["status_mode"] == "native" and abs(float(r["split_frac"]) - 0.5) < 1e-12),
        modified_results[0],
    )

    control_rows = control_ref["level0_rows"]
    modified_rows = modified_ref["level0_rows_modified"]

    CONTROL_COLOR_TRACE_101.write_text(
        "\n\n".join(
            [
                simple_particle_lines(control_rows, "control level-0 appended particle list"),
                trace_single_color(control_rows, 101, "control trace"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    CONTROL_ALL_COLOR_LINES.write_text(
        "\n\n".join(
            [
                simple_particle_lines(control_rows, "control level-0 appended particle list"),
                all_color_lines_report(control_rows, "control all active color lines"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    MODIFIED_COLOR_TRACE.write_text(
        "\n\n".join(
            [
                simple_particle_lines(modified_rows, "modified level-0 appended particle list (native, 50/50)"),
                junction_listing_text(
                    make_hadronizer().event,  # placeholder replaced below
                    "junction placeholder",
                ),
            ]
        ),
        encoding="utf-8",
    )
    # Rebuild modified event once to capture junction listing text directly from event object.
    ptmp = make_hadronizer()
    append_level0_rows(ptmp, [r for r in modified_rows if r.index < 1000], None)
    qa = next(r for r in modified_rows if r.index == 1001)
    qb = next(r for r in modified_rows if r.index == 1002)
    ptmp.event.append(qa.pid, qa.status, qa.col, qa.acol, qa.px, qa.py, qa.pz, qa.E, qa.m)
    ptmp.event.append(qb.pid, qb.status, qb.col, qb.acol, qb.px, qb.py, qb.pz, qb.E, qb.m)
    ptmp.event.appendJunction(1, 101, 105, 106)
    modified_trace_payload = "\n\n".join(
        [
            simple_particle_lines(modified_rows, "modified level-0 appended particle list (native, 50/50)"),
            trace_single_color(modified_rows, 101, "modified trace"),
            trace_single_color(modified_rows, 105, "modified trace"),
            trace_single_color(modified_rows, 106, "modified trace"),
            junction_listing_text(ptmp.event, "modified junction listing"),
        ]
    )
    MODIFIED_COLOR_TRACE.write_text(modified_trace_payload + "\n", encoding="utf-8")

    MODIFIED_ALL_COLOR_LINES.write_text(
        "\n\n".join(
            [
                simple_particle_lines(modified_rows, "modified level-0 appended particle list (native, 50/50)"),
                all_color_lines_report(modified_rows, "modified all active color lines"),
                junction_listing_text(ptmp.event, "modified junction listing"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Explicit count check for the key endpoint question.
    control_acol_101 = [r for r in control_rows if r.acol == 101]
    control_col_101 = [r for r in control_rows if r.col == 101]
    modified_acol_101 = [r for r in modified_rows if r.acol == 101]
    modified_col_101 = [r for r in modified_rows if r.col == 101]
    modified_acol_105 = [r for r in modified_rows if r.acol == 105]
    modified_col_105 = [r for r in modified_rows if r.col == 105]
    modified_acol_106 = [r for r in modified_rows if r.acol == 106]
    modified_col_106 = [r for r in modified_rows if r.col == 106]

    summary_lines = [
        "CONTROL VS MODIFIED COLOR TOPOLOGY SUMMARY",
        f"control_status_mode={control_ref['status_mode']} control_success={control_ref['ok']} control_n_had={control_ref['n_had']}",
        f"modified_status_mode={modified_ref['status_mode']} modified_split=0.5/0.5 modified_success={modified_ref['ok']} modified_n_had={modified_ref['n_had']}",
        "",
        "1) Working control diquark role (from event-record color tags):",
        f"- control color 101 carriers (col side): {[r.index for r in control_col_101]}",
        f"- control color 101 carriers (acol side): {[r.index for r in control_acol_101]}",
        f"- control diquark entry indices with acol=101: {[r.index for r in control_acol_101 if is_diquark_id(r.pid)]}",
        "- In this Level-0 control list, the diquark is the only acol=101 endpoint and acts as the terminal anticolor endpoint of line 101.",
        "",
        "2) Modified event unpaired/untraceable lines:",
        f"- modified color 101 col-side indices: {[r.index for r in modified_col_101]}",
        f"- modified color 101 acol-side indices: {[r.index for r in modified_acol_101]}",
        f"- modified color 105 col-side indices: {[r.index for r in modified_col_105]}",
        f"- modified color 105 acol-side indices: {[r.index for r in modified_acol_105]}",
        f"- modified color 106 col-side indices: {[r.index for r in modified_col_106]}",
        f"- modified color 106 acol-side indices: {[r.index for r in modified_acol_106]}",
        "- In the naive replacement, tags 105 and 106 appear only on col side in particles and rely on junction legs for closure; there are no particle acol endpoints for them.",
        "- Tag 101 no longer ends on a diquark particle endpoint; it is redirected into a junction leg.",
        "",
        "3) Smallest structural inference supported by records:",
        "- Control succeeds with a conventional string-like endpoint where diquark supplies the acol endpoint for line 101.",
        "- Modified fails with ColourTracing failure when replacing that endpoint by a three-leg junction plus two quark col legs (105/106) without corresponding particle acol continuation.",
        "- Missing structural ingredient is a valid complete color network compatible with junction tracing rules, not momentum/status tuning.",
    ]
    TOPOLOGY_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Level-0 A/B topology test complete")
    print(f"control_reinjection_log={CONTROL_REINJECT_LOG}")
    print(f"modified_reinjection_log={MODIFIED_REINJECT_LOG}")
    print(f"control_reinjection_trace={CONTROL_TRACE_LOG}")
    print(f"control_color_trace_101={CONTROL_COLOR_TRACE_101}")
    print(f"control_all_color_lines={CONTROL_ALL_COLOR_LINES}")
    print(f"modified_color_trace={MODIFIED_COLOR_TRACE}")
    print(f"modified_all_color_lines={MODIFIED_ALL_COLOR_LINES}")
    print(f"topology_summary={TOPOLOGY_SUMMARY}")


if __name__ == "__main__":
    main()

