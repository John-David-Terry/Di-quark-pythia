#!/usr/bin/env python3
"""
Technical reinjection trial for split-diquark events (100-event subset).

This trial:
- reads split-diquark event-record CSVs
- converts each selected event to a tiny single-event LHE
- reinjects into PYTHIA with ISR off, FSR on, HadronLevel on
- records per-event success/failure diagnostics
"""

from __future__ import annotations

import csv
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

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

IN_SPLIT_EVENT = OUTDIR / "dis_isr_full_event_record_diquark_split.csv"
IN_SPLIT_META = OUTDIR / "dis_isr_diquark_split_metadata.csv"

OUT_LOG = OUTDIR / "reinjection_split_diquark_colorfix_trial_100events.txt"
OUT_CSV = OUTDIR / "reinjection_split_diquark_colorfix_trial_100events.csv"

N_TRIAL = 100


def pick_incoming_particles(df_event: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    e_in = df_event[(df_event["pdg_id"] == 11) & (df_event["status"] < 0)].copy()
    q_in = df_event[(df_event["pdg_id"].abs() <= 6) & (df_event["status"] < 0)].copy()
    if e_in.empty or q_in.empty:
        raise ValueError("missing incoming electron or incoming quark")
    e_row = e_in.loc[e_in["pz"].idxmax()]
    q_row = q_in.loc[q_in["pz"].abs().idxmax()]
    return e_row, q_row


def pick_outgoing_particles(df_event: pd.DataFrame) -> pd.DataFrame:
    # Minimal externally-injected final state for FSR+hadronization:
    # scattered e + final quarks/gluons (including split remnant u,d).
    out = df_event[
        (df_event["isFinal"] == 1)
        & ((df_event["pdg_id"] == 11) | (df_event["pdg_id"].abs() <= 6) | (df_event["pdg_id"] == 21))
    ].copy()
    if out.empty:
        raise ValueError("no outgoing particles selected")
    return out.sort_values("particle_index")


def apply_remnant_color_fix(df_event: pd.DataFrame, new_u_index: int, new_d_index: int) -> Tuple[pd.DataFrame, bool, int, Dict[int, int], Dict[int, int]]:
    g = df_event.copy()
    mask_u = g["particle_index"] == int(new_u_index)
    mask_d = g["particle_index"] == int(new_d_index)
    if not mask_u.any() or not mask_d.any():
        return g, False, -1, {}, {}

    tags = set()
    for c in g["col"].tolist() + g["acol"].tolist():
        ic = int(c)
        if ic > 0:
            tags.add(ic)
    new_tag = (max(tags) + 1) if tags else 501

    g.loc[mask_u, "col"] = new_tag
    g.loc[mask_u, "acol"] = 0
    g.loc[mask_d, "col"] = 0
    g.loc[mask_d, "acol"] = new_tag

    col_counts: Dict[int, int] = {}
    acol_counts: Dict[int, int] = {}
    for x in g["col"].tolist():
        ix = int(x)
        if ix > 0:
            col_counts[ix] = col_counts.get(ix, 0) + 1
    for x in g["acol"].tolist():
        ix = int(x)
        if ix > 0:
            acol_counts[ix] = acol_counts.get(ix, 0) + 1
    return g, True, new_tag, col_counts, acol_counts


def build_single_event_lhe_text(df_event: pd.DataFrame) -> str:
    e_in, q_in = pick_incoming_particles(df_event)
    out = pick_outgoing_particles(df_event)

    particles: List[Tuple[int, int, int, int, int, int, float, float, float, float, float]] = []
    # incoming entries
    particles.append((int(e_in["pdg_id"]), -1, 0, 0, 0, 0, float(e_in["px"]), float(e_in["py"]), float(e_in["pz"]), float(e_in["E"]), float(e_in["m"])))
    particles.append((int(q_in["pdg_id"]), -1, 0, 0, int(q_in["col"]), int(q_in["acol"]), float(q_in["px"]), float(q_in["py"]), float(q_in["pz"]), float(q_in["E"]), float(q_in["m"])))
    # outgoing entries: mother pointers to incoming pair (1,2)
    for _, r in out.iterrows():
        particles.append(
            (
                int(r["pdg_id"]),
                1,
                1,
                2,
                int(r["col"]),
                int(r["acol"]),
                float(r["px"]),
                float(r["py"]),
                float(r["pz"]),
                float(r["E"]),
                float(r["m"]),
            )
        )

    nup = len(particles)
    lines = [
        '<LesHouchesEvents version="1.0">',
        "<init>",
        # IDBMUP, EBMUP, PDFGUP/PDFSUP, IDWTUP, NPRUP
        "11 2212 18.0 275.0 0 0 0 0 3 1",
        "1.0 1.0 1.0 1",
        "</init>",
        "<event>",
        f"{nup} 1 1.0 0.0 0.118 0.0072973525693",
    ]
    for pid, status, m1, m2, col, acol, px, py, pz, E, m in particles:
        lines.append(
            f"{pid:8d} {status:3d} {m1:3d} {m2:3d} {col:4d} {acol:4d} "
            f"{px:.10e} {py:.10e} {pz:.10e} {E:.10e} {m:.10e} 0.0 9.0"
        )
    lines += ["</event>", "</LesHouchesEvents>"]
    return "\n".join(lines) + "\n"


def run_reinjection_one_event(lhe_path: Path) -> Tuple[bool, bool, bool, int, int, str, str]:
    p = pythia8.Pythia()
    p.readString("Beams:frameType = 4")
    p.readString(f"Beams:LHEF = {lhe_path}")
    p.readString("LesHouches:matchInOut = on")
    p.readString("ProcessLevel:all = off")
    p.readString("PartonLevel:ISR = off")
    p.readString("PartonLevel:FSR = on")
    p.readString("PartonLevel:MPI = off")
    p.readString("HadronLevel:all = on")
    init_ok = bool(p.init())
    if not init_ok:
        return True, False, False, 0, 0, "initialization_error", "pythia.init() failed"

    next_ok = bool(p.next())
    if not next_ok:
        return True, False, False, 0, 0, "pythia_next_failed", "pythia.next() failed"

    n_final = 0
    n_had = 0
    for i in range(p.event.size()):
        pp = p.event[i]
        if pp.isFinal():
            n_final += 1
            if pp.isHadron():
                n_had += 1
    hadrons_produced = n_had > 0
    return True, True, hadrons_produced, n_final, n_had, "", ""


def main() -> None:
    ev = pd.read_csv(IN_SPLIT_EVENT).sort_values(["event_id", "particle_index"]).reset_index(drop=True)
    meta = pd.read_csv(IN_SPLIT_META)
    trial_ids = meta["event_id"].drop_duplicates().sort_values().head(N_TRIAL).tolist()
    meta_by_id = meta.set_index("event_id")

    rows: List[Dict[str, object]] = []
    log_lines: List[str] = []

    for idx, event_id in enumerate(trial_ids, start=1):
        g0 = ev[ev["event_id"] == event_id].copy()
        failure_mode = ""
        error_message = ""
        build_ok = False
        color_fix_applied = False
        pythia_ok = False
        had_ok = False
        n_final = 0
        n_had = 0
        notes = ""
        debug_lines: List[str] = []
        m = meta_by_id.loc[event_id] if event_id in meta_by_id.index else None
        new_u_idx = int(m["new_u_index"]) if m is not None else -1
        new_d_idx = int(m["new_d_index"]) if m is not None else -1
        try:
            g, color_fix_applied, new_tag, col_counts, acol_counts = apply_remnant_color_fix(g0, new_u_idx, new_d_idx)
            if not color_fix_applied:
                notes = "color_fix_not_applied_missing_new_indices"
            else:
                notes = f"new_tag={new_tag}"
            if idx <= 5:
                dq = g0[(g0["isFinal"] == 1) & (g0["pdg_id"].abs() >= 1000) & (g0["pdg_id"].abs() < 10000)]
                dq_row = dq.loc[dq["E"].idxmax()] if not dq.empty else None
                debug_lines.append(f"event_id={event_id}")
                if dq_row is not None:
                    debug_lines.append(
                        f"original_diquark idx={int(dq_row['particle_index'])} col={int(dq_row['col'])} acol={int(dq_row['acol'])}"
                    )
                uu = g[g["particle_index"] == new_u_idx]
                dd = g[g["particle_index"] == new_d_idx]
                if not uu.empty:
                    ur = uu.iloc[0]
                    debug_lines.append(f"new_u idx={new_u_idx} col={int(ur['col'])} acol={int(ur['acol'])}")
                if not dd.empty:
                    dr = dd.iloc[0]
                    debug_lines.append(f"new_d idx={new_d_idx} col={int(dr['col'])} acol={int(dr['acol'])}")
                all_tags = sorted(set(list(col_counts.keys()) + list(acol_counts.keys())))
                tag_line = []
                twice_ok = True
                for t in all_tags:
                    ncol = col_counts.get(t, 0)
                    nacol = acol_counts.get(t, 0)
                    if not (ncol == 1 and nacol == 1):
                        twice_ok = False
                    tag_line.append(f"{t}:col={ncol},acol={nacol}")
                debug_lines.append("color_tags=" + "; ".join(tag_line))
                debug_lines.append(f"color_pairing_exactly_once_each_side={twice_ok}")

            lhe_text = build_single_event_lhe_text(g)
            build_ok = True
            with tempfile.NamedTemporaryFile("w", suffix=".lhe", delete=False) as tf:
                tf.write(lhe_text)
                lhe_path = Path(tf.name)
            try:
                _, pythia_ok, had_ok, n_final, n_had, failure_mode, error_message = run_reinjection_one_event(lhe_path)
                if pythia_ok and not had_ok:
                    failure_mode = "no_hadrons"
                    error_message = "next() succeeded but no final-state hadrons found"
                if idx <= 5:
                    debug_lines.append(f"pythia_next={pythia_ok} n_final={n_final} n_had={n_had}")
            finally:
                try:
                    lhe_path.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception as exc:
            failure_mode = "reinjection_build_error"
            error_message = str(exc)

        if not build_ok and not failure_mode:
            failure_mode = "reinjection_build_error"
            error_message = "unknown build failure"
        if build_ok and not pythia_ok and not failure_mode:
            failure_mode = "unknown"
            error_message = "pythia failed without classified mode"

        rows.append(
            {
                "event_id": int(event_id),
                "color_fix_applied": int(color_fix_applied),
                "reinjection_build_success": int(build_ok),
                "pythia_accept_success": int(pythia_ok),
                "hadrons_produced": int(had_ok),
                "n_final_particles": int(n_final),
                "n_final_hadrons": int(n_had),
                "failure_mode": failure_mode,
                "error_message": error_message,
                "notes": notes + (" | " + " || ".join(debug_lines) if idx <= 5 and debug_lines else ""),
                "struck_outgoing_index_selected": int(m["struck_outgoing_index_selected"]) if m is not None else -1,
                "new_u_index": int(m["new_u_index"]) if m is not None else -1,
                "new_d_index": int(m["new_d_index"]) if m is not None else -1,
            }
        )
        log_lines.append(
            f"[{idx}/{len(trial_ids)}] event_id={event_id} color_fix_applied={color_fix_applied} "
            f"build={build_ok} init_next={pythia_ok} hadrons={had_ok} "
            f"n_final={n_final} n_had={n_had} failure_mode={failure_mode} error={error_message}"
        )
        if idx <= 5 and debug_lines:
            log_lines.extend(["  " + x for x in debug_lines])

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    attempted = len(out_df)
    built = int(out_df["reinjection_build_success"].sum())
    accepted = int(out_df["pythia_accept_success"].sum())
    hadrons = int(out_df["hadrons_produced"].sum())
    frac = accepted / attempted if attempted else float("nan")
    fail_counts = Counter([x for x in out_df["failure_mode"].tolist() if x])
    example_ids: Dict[str, List[int]] = defaultdict(list)
    for _, r in out_df.iterrows():
        fm = str(r["failure_mode"])
        if fm and len(example_ids[fm]) < 5:
            example_ids[fm].append(int(r["event_id"]))

    summary = [
        "Split-diquark reinjection color-fix trial (100 events)",
        f"attempted={attempted}",
        f"built_successfully={built}",
        f"accepted_by_pythia={accepted}",
        f"produced_hadrons={hadrons}",
        f"acceptance_fraction={frac:.6f}",
        "failure_mode_counts=" + str(dict(fail_counts)),
        "failure_mode_examples=" + str(dict(example_ids)),
        "",
    ]
    OUT_LOG.write_text("\n".join(summary + log_lines), encoding="utf-8")

    print("\n".join(summary))
    print(f"log={OUT_LOG}")
    print(f"csv={OUT_CSV}")


if __name__ == "__main__":
    main()

