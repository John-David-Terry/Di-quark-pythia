#!/usr/bin/env python3
"""
Reinject N events from dis_isr_full_event_record.csv (Breit-frame momenta) into PYTHIA.

Uses the same pipeline as reinject_altered_events.py: take final-state colored QCD partons
(quarks, gluons, diquarks), require closed color tags, append with status 23, then
pythia.next() for ISR + FSR + hadronization (MPI off). Beams match DIS dataset generation
(18×275 GeV e–p, ``PDF:lepton = off``).

Input CSV is typically from generate_dis_isr_parton_dataset.py (Breit output by default).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    import pythia8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pythia8 import failed: {exc}") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

DEFAULT_FULL_CSV = outputs_dir() / "dis_isr_parton_dataset" / "dis_isr_full_event_record.csv"
DEFAULT_OUT_JSON = outputs_dir() / "dis_isr_parton_dataset" / "reinject_breit_csv_summary.json"


def _load_reinject_helpers():
    path = Path(__file__).resolve().parent / "reinject_altered_events.py"
    spec = importlib.util.spec_from_file_location("_reinject_altered_events", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reinject first N events from Breit-frame DIS ISR full-event CSV into PYTHIA."
    )
    ap.add_argument(
        "--full-event-csv",
        type=Path,
        default=DEFAULT_FULL_CSV,
        help=f"Path to dis_isr_full_event_record.csv (default: {DEFAULT_FULL_CSV})",
    )
    ap.add_argument(
        "--n-events",
        type=int,
        default=100,
        help="Number of distinct event_id blocks to process (from lowest event_id upward).",
    )
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON, help="Summary JSON path.")
    args = ap.parse_args()

    csv_path = args.full_event_csv.resolve()
    if not csv_path.is_file():
        print(f"error: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    mod = _load_reinject_helpers()
    final_colored_partons = mod.final_colored_partons
    color_balance_ok = mod.color_balance_ok

    p = pythia8.Pythia()
    p.readString("Print:quiet = on")
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
    if not p.init():
        print("error: pythia.init() failed", file=sys.stderr)
        sys.exit(1)

    def run_pythia_on_partons(partons: pd.DataFrame):
        """Append partons to existing hadronizer, one pythia.next() per call."""
        p.event.reset()
        err = ""
        try:
            for _, r in partons.iterrows():
                p.event.append(
                    int(r["pdg_id"]),
                    int(mod.STATUS_INJECT),
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

    ev = pd.read_csv(csv_path).sort_values(["event_id", "particle_index"]).reset_index(drop=True)
    all_ids = sorted(ev["event_id"].unique().tolist())
    n_take = max(0, int(args.n_events))
    trial_ids = all_ids[:n_take]

    results: List[Dict[str, Any]] = []
    color_pass = 0
    next_ok = 0
    hadrons_when_ok: List[int] = []

    for event_id in trial_ids:
        g = ev[ev["event_id"] == event_id]
        partons = final_colored_partons(g)
        c_ok, c_msg = color_balance_ok(partons)
        rec: Dict[str, Any] = {
            "event_id": int(event_id),
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

    avg_had = sum(hadrons_when_ok) / len(hadrons_when_ok) if hadrons_when_ok else 0.0
    summary: Dict[str, Any] = {
        "full_event_csv": str(csv_path),
        "n_events_requested": n_take,
        "n_event_ids_in_csv": len(all_ids),
        "n_event_ids_processed": len(trial_ids),
        "color_checks_passed": color_pass,
        "pythia_next_succeeded": next_ok,
        "average_final_hadrons_on_success": avg_had,
        "ecm_gev": float(args.ecm),
        "per_event": results,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=== reinject_dis_isr_breit_csv ===")
    print(f"  csv:                      {csv_path}")
    print(f"  events processed:         {len(trial_ids)}")
    print(f"  color checks passed:      {color_pass}")
    print(f"  pythia.next() succeeded:  {next_ok}")
    print(f"  avg final hadrons (ok):   {avg_had:.3f}")
    print(f"  wrote:                    {args.out_json}")


if __name__ == "__main__":
    main()
