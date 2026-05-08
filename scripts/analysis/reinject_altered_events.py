#!/usr/bin/env python3
"""
Reinject altered split_90_10 CSV events into PYTHIA (final colored partons only),
verify color balance on that subset, hadronize, write reinjection_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

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

DEFAULT_ALTERED = outputs_dir() / "dis_isr_parton_dataset" / "split_90_10" / "altered"
DEFAULT_OUT_JSON = outputs_dir() / "dis_isr_parton_dataset" / "split_90_10" / "reinjection_summary.json"

STATUS_INJECT = 23


def row_is_final_colored_qcd(row: pd.Series) -> bool:
    if int(row["isFinal"]) != 1:
        return False
    if int(row["status"]) <= 0:
        return False
    if int(row["daughter1"]) != 0 or int(row["daughter2"]) != 0:
        return False
    ap = abs(int(row["pdg_id"]))
    if ap == 21:
        return True
    if 1 <= ap <= 6:
        return True
    if 1000 <= ap < 10000 and (ap // 10) % 10 == 0:
        return True
    return False


def final_colored_partons(df: pd.DataFrame) -> pd.DataFrame:
    m = df.apply(row_is_final_colored_qcd, axis=1)
    out = df[m].sort_values("particle_index").reset_index(drop=True)
    return out


def color_balance_ok(df: pd.DataFrame) -> Tuple[bool, str]:
    """Each tag > 0 must appear exactly once as col and once as acol within df."""
    tags: Dict[int, Tuple[int, int]] = {}
    for _, r in df.iterrows():
        for tag, is_col in [(int(r["col"]), True), (int(r["acol"]), False)]:
            if tag <= 0:
                continue
            nc, na = tags.get(tag, (0, 0))
            if is_col:
                nc += 1
            else:
                na += 1
            tags[tag] = (nc, na)
    for t, (nc, na) in sorted(tags.items()):
        if nc != 1 or na != 1:
            return False, f"tag {t}: n_col={nc} n_acol={na}"
    return True, "ok"


def run_pythia_on_partons(partons: pd.DataFrame) -> Tuple[bool, str, int, int]:
    p = pythia8.Pythia()
    p.readString("Beams:idA = 11")
    p.readString("Beams:idB = 2212")
    p.readString("Beams:eA = 18.0")
    p.readString("Beams:eB = 275.0")
    p.readString("Beams:frameType = 2")
    p.readString("PDF:lepton = off")
    p.readString("ProcessLevel:all = off")
    p.readString("PartonLevel:ISR = on")
    p.readString("PartonLevel:FSR = on")
    p.readString("PartonLevel:MPI = off")
    p.readString("HadronLevel:all = on")
    p.readString("Print:quiet = on")
    if not p.init():
        return False, "init failed", 0, 0

    p.event.reset()
    err = ""
    try:
        for _, r in partons.iterrows():
            p.event.append(
                int(r["pdg_id"]),
                STATUS_INJECT,
                int(r["col"]),
                int(r["acol"]),
                float(r["px"]),
                float(r["py"]),
                float(r["pz"]),
                float(r["E"]),
                float(r["m"]),
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Reinject altered CSVs into PYTHIA.")
    ap.add_argument("--altered-dir", type=Path, default=DEFAULT_ALTERED)
    ap.add_argument("--max-events", type=int, default=15)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = ap.parse_args()

    altered_dir: Path = args.altered_dir
    if not altered_dir.is_dir():
        raise SystemExit(f"altered dir not found: {altered_dir}")

    csv_files = sorted(altered_dir.glob("event_*.csv"))[: max(0, args.max_events)]

    results: List[Dict[str, Any]] = []
    color_pass = 0
    next_ok = 0
    hadrons_when_ok: List[int] = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path).sort_values("particle_index")
        partons = final_colored_partons(df)
        c_ok, c_msg = color_balance_ok(partons)
        rec: Dict[str, Any] = {
            "csv": str(csv_path.name),
            "n_partons_appended": int(len(partons)),
            "color_ok": c_ok,
            "color_message": c_msg,
            "pythia_next_ok": False,
            "pythia_error": "",
            "n_final_particles": 0,
            "n_final_hadrons": 0,
        }
        if c_ok:
            color_pass += 1
            ok, err, nf, nh = run_pythia_on_partons(partons)
            rec["pythia_next_ok"] = ok
            rec["pythia_error"] = err
            rec["n_final_particles"] = nf
            rec["n_final_hadrons"] = nh
            if ok:
                next_ok += 1
                hadrons_when_ok.append(nh)
        results.append(rec)

    n_tested = len(csv_files)
    avg_had = sum(hadrons_when_ok) / len(hadrons_when_ok) if hadrons_when_ok else 0.0

    summary: Dict[str, Any] = {
        "altered_dir": str(altered_dir),
        "max_events_requested": args.max_events,
        "n_altered_events_tested": n_tested,
        "color_checks_passed": color_pass,
        "pythia_next_succeeded": next_ok,
        "average_final_hadrons_on_success": avg_had,
        "per_event": results,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== reinject_altered_events ===")
    print(f"  altered events tested:     {n_tested}")
    print(f"  color checks passed:       {color_pass}")
    print(f"  pythia.next() succeeded: {next_ok}")
    print(f"  avg final hadrons (ok):  {avg_had:.3f}")
    print(f"  wrote: {args.out_json}")


if __name__ == "__main__":
    main()
