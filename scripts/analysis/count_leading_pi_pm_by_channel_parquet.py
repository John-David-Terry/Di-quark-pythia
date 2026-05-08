#!/usr/bin/env python3
"""
Per split channel (A/B), summarize the **single** jet–hadron row per event from
``jet_hadron_observables_split_pi_pm.process_event_dataframe`` (leading charged π in
target hemisphere, x_L cut): how often ``ok``, and whether the leading hadron was π⁺ vs π⁻.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANAL = Path(__file__).resolve().parent
for _p in (_PROJECT_ROOT, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jet_hadron_observables_split_pi_pm import (  # noqa: E402
    build_pythia_reinjector,
    process_event_dataframe,
)


def _load_event_meta(altered_root: Path) -> pd.DataFrame:
    mdir = altered_root / "altered_metadata"
    if not mdir.is_dir():
        raise FileNotFoundError(f"missing altered_metadata under {altered_root}")
    parts: List[pd.DataFrame] = []
    cols = [
        "event_id",
        "split_channel",
        "alteration_succeeded",
        "struck_quark_index",
    ]
    for pq in sorted(mdir.glob("shard_*.parquet")):
        parts.append(pd.read_parquet(pq, columns=cols))
    df = pd.concat(parts, ignore_index=True)
    sub = df[(df["alteration_succeeded"] == 1) & (df["split_channel"].isin(["A", "B", "C"]))]
    return sub.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Leading charged π observable (one row/event) by split_channel (A/B/C)."
    )
    ap.add_argument(
        "--altered-root",
        type=Path,
        required=True,
        help="altered_100k_parquet directory (particles + altered_metadata).",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write summary JSON.",
    )
    args = ap.parse_args()

    altered_root = args.altered_root.expanduser().resolve()
    meta = _load_event_meta(altered_root)
    meta_by_eid = meta.set_index("event_id")

    pythia = build_pythia_reinjector()

    by_ch: Dict[str, Dict[str, Any]] = {
        "A": {
            "n_events": 0,
            "n_ok": 0,
            "n_ok_leading_pi_plus": 0,
            "n_ok_leading_pi_minus": 0,
            "fail_reason_counts": {},
        },
        "B": {
            "n_events": 0,
            "n_ok": 0,
            "n_ok_leading_pi_plus": 0,
            "n_ok_leading_pi_minus": 0,
            "fail_reason_counts": {},
        },
    }
    skipped_struck = 0

    particles_dir = altered_root / "particles"
    if not particles_dir.is_dir():
        raise FileNotFoundError(f"missing particles under {altered_root}")

    for pshard in sorted(particles_dir.glob("shard_*.parquet")):
        pdf = pd.read_parquet(pshard)
        for eid, g in pdf.groupby("event_id", sort=True):
            eid_i = int(eid)
            if eid_i not in meta_by_eid.index:
                continue
            mrow = meta_by_eid.loc[eid_i]
            if isinstance(mrow, pd.DataFrame):
                mrow = mrow.iloc[0]
            ch = str(mrow["split_channel"])
            if ch not in ("A", "B"):
                continue
            struck = int(mrow["struck_quark_index"])
            if struck < 0:
                skipped_struck += 1
                continue

            g = g.sort_values("particle_index").reset_index(drop=True)
            rows = process_event_dataframe(
                g,
                struck,
                sample=ch,
                csv_label=str(pshard.name),
                csv_momenta_breit=True,
                pythia_reuse=pythia,
            )
            if len(rows) != 1:
                raise RuntimeError(f"expected 1 row for event {eid_i}, got {len(rows)}")
            r = rows[0]
            bucket = by_ch[ch]
            bucket["n_events"] += 1
            if r.get("ok"):
                bucket["n_ok"] += 1
                if r.get("pion") == "piplus":
                    bucket["n_ok_leading_pi_plus"] += 1
                elif r.get("pion") == "piminus":
                    bucket["n_ok_leading_pi_minus"] += 1
            else:
                reason = str(r.get("reason", ""))
                bucket["fail_reason_counts"][reason] = bucket["fail_reason_counts"].get(reason, 0) + 1

    summary: Dict[str, Any] = {
        "altered_root": str(altered_root),
        "skipped_bad_struck_index": skipped_struck,
        "per_channel": {},
    }
    for ch in ("A", "B"):
        b = by_ch[ch]
        ne = max(1, int(b["n_events"]))
        nok = int(b["n_ok"])
        summary["per_channel"][ch] = {
            "label": "(ud)->[d]+(uu)+ubar" if ch == "A" else "(ud)->[d]+(ud)+dbar",
            "n_events": int(b["n_events"]),
            "n_observable_ok": int(b["n_ok"]),
            "frac_observable_ok": float(b["n_ok"]) / ne,
            "among_ok_leading_was_pi_plus": int(b["n_ok_leading_pi_plus"]),
            "among_ok_leading_was_pi_minus": int(b["n_ok_leading_pi_minus"]),
            "frac_ok_that_are_pi_plus": (float(b["n_ok_leading_pi_plus"]) / nok) if nok else 0.0,
            "frac_ok_that_are_pi_minus": (float(b["n_ok_leading_pi_minus"]) / nok) if nok else 0.0,
            "fail_reason_counts": dict(sorted(b["fail_reason_counts"].items(), key=lambda kv: -kv[1])),
        }

    text = json.dumps(summary, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
