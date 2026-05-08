#!/usr/bin/env python3
"""
Jet–hadron observables (same physics as ``jet_hadron_observables_split_pi_pm``) for a
**subset** of altered Parquet events selected by ``split_channel`` in ``altered_metadata``.

Typical use: **signal** ``altered_100k_parquet`` uses hard channel **C**
``(ud)→[d]+(us)+sbar``.  This driver keeps events with ``alteration_succeeded == 1`` and the
requested ``split_channel`` (default **C**), reinjects from the parton table, and writes one CSV
row per altered event (leading charged π + cuts).  Optional ``--editable-parent`` adds paired
``sample=unchanged`` rows from ``editable_source_v1`` for the same ``event_id`` set (for π±
comparison plots).

Example::

  python scripts/analysis/jet_hadron_observables_altered_parquet_filtered.py \\
    --altered-root ~/Data/dis_isr_editable_altered_100k/altered_100k_parquet \\
    --split-channel C \\
    --editable-parent ~/Data/dis_isr_editable_source_100k \\
    --out-csv ~/Data/.../jet_hadron_observables_signal_channel_C.csv

Requires: pythia8, pandas, numpy, tqdm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tqdm is required (see requirements.txt)") from exc

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANAL = Path(__file__).resolve().parent
for _p in (_PROJECT_ROOT, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from editable_source_parquet import EDITABLE_SOURCE_V1_DIRNAME  # noqa: E402

from jet_hadron_observables_split_pi_pm import (  # noqa: E402
    build_pythia_reinjector,
    process_event_dataframe,
)


def _load_filter_map(
    altered_root: Path, want_channel: Optional[str]
) -> Tuple[Dict[int, int], Dict[int, str], Dict[int, str]]:
    """struck_index, channel_description, and split_channel letter per event_id."""
    mdir = altered_root / "altered_metadata"
    if not mdir.is_dir():
        raise FileNotFoundError(f"missing altered_metadata under {altered_root}")
    struck: Dict[int, int] = {}
    descr: Dict[int, str] = {}
    ch_letter: Dict[int, str] = {}
    cols = ["event_id", "alteration_succeeded", "split_channel", "struck_quark_index"]
    has_desc = False
    probe = next(iter(mdir.glob("shard_*.parquet")), None)
    if probe is not None:
        pc = set(pd.read_parquet(probe, engine="pyarrow").columns)
        if "channel_description" in pc:
            has_desc = True
            cols.append("channel_description")

    for pq in sorted(mdir.glob("shard_*.parquet")):
        df = pd.read_parquet(pq, columns=cols)
        ok = df["alteration_succeeded"] == 1
        if want_channel is None:
            sub = df[ok]
        else:
            sub = df[ok & (df["split_channel"].astype(str) == want_channel)]
        for _, r in sub.iterrows():
            eid = int(r["event_id"])
            struck[eid] = int(r["struck_quark_index"])
            ch_letter[eid] = str(r.get("split_channel", ""))
            if has_desc:
                descr[eid] = str(r.get("channel_description", ""))
    return struck, descr, ch_letter


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Jet–hadron observables for altered Parquet events filtered by split_channel."
    )
    ap.add_argument(
        "--altered-root",
        type=Path,
        required=True,
        help="Directory with particles/, events/, altered_metadata/ (e.g. altered_100k_parquet).",
    )
    ap.add_argument(
        "--split-channel",
        choices=("C",),
        default="C",
        help='Hard topology filter on metadata split_channel (default "C" = (ud)→[d]+(us)+sbar). Ignored if --all-channels.',
    )
    ap.add_argument(
        "--all-channels",
        action="store_true",
        help="Include every alteration_succeeded event (any split_channel).",
    )
    ap.add_argument(
        "--editable-parent",
        type=Path,
        default=None,
        help=(
            "Parent directory containing editable_source_v1/ (same pool as altered). "
            "If set, append unchanged reinjection rows for the same filtered event_ids."
        ),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output observables CSV.",
    )
    ap.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="If >0, stop after this many matching events (debug).",
    )
    ap.add_argument(
        "--sample-label",
        type=str,
        default="altered",
        help='Value written in the ``sample`` column (default "altered").',
    )
    args = ap.parse_args()

    altered_root = args.altered_root.expanduser().resolve()
    want_filter: Optional[str] = None if bool(args.all_channels) else str(args.split_channel)
    want_label = "all" if want_filter is None else want_filter
    struck_map, desc_map, ch_map = _load_filter_map(altered_root, want_filter)
    allowed: Set[int] = set(struck_map.keys())
    if not allowed:
        raise SystemExit(
            f"no matching altered events under {altered_root} (filter={want_label!r})"
        )

    pythia = build_pythia_reinjector()
    all_rows: List[Dict[str, Any]] = []
    n_seen = 0
    particles_dir = altered_root / "particles"
    if not particles_dir.is_dir():
        raise FileNotFoundError(f"missing particles under {altered_root}")

    cap = int(args.max_events)
    total_bar = len(allowed) if cap <= 0 else min(len(allowed), cap)
    stop = False
    with tqdm(total=total_bar, unit="ev", desc=f"channel {want_label}", mininterval=0.2) as pbar:
        for pshard in sorted(particles_dir.glob("shard_*.parquet")):
            if stop:
                break
            pdf = pd.read_parquet(pshard)
            for eid, g in pdf.groupby("event_id", sort=True):
                eid_i = int(eid)
                if eid_i not in allowed:
                    continue
                if cap > 0 and n_seen >= cap:
                    stop = True
                    break
                struck = struck_map.get(eid_i, -1)
                if struck < 0:
                    continue
                g = g.sort_values("particle_index").reset_index(drop=True)
                rows = process_event_dataframe(
                    g,
                    struck,
                    sample=str(args.sample_label),
                    csv_label=str(pshard.name),
                    csv_momenta_breit=True,
                    pythia_reuse=pythia,
                )
                if len(rows) != 1:
                    raise RuntimeError(f"event {eid_i}: expected 1 observable row, got {len(rows)}")
                row = dict(rows[0])
                row["hard_split_channel"] = ch_map.get(eid_i, want_label)
                row["hard_channel_description"] = desc_map.get(eid_i, "")
                all_rows.append(row)
                n_seen += 1
                pbar.update(1)

    altered_eids_done = {
        int(r["event_id"])
        for r in all_rows
        if str(r.get("sample", "")) == str(args.sample_label)
    }

    if args.editable_parent is not None:
        ep = args.editable_parent.expanduser().resolve()
        part_dir = ep / EDITABLE_SOURCE_V1_DIRNAME / "particles"
        if not part_dir.is_dir():
            raise FileNotFoundError(f"missing editable particles dir: {part_dir}")
        n_un = 0
        with tqdm(
            total=len(altered_eids_done), unit="ev", desc="unchanged (editable)", mininterval=0.2
        ) as pbar_u:
            for pshard in sorted(part_dir.glob("shard_*.parquet")):
                pdf = pd.read_parquet(pshard)
                for eid, g in pdf.groupby("event_id", sort=True):
                    eid_i = int(eid)
                    if eid_i not in altered_eids_done:
                        continue
                    struck = struck_map.get(eid_i, -1)
                    if struck < 0:
                        continue
                    g = g.sort_values("particle_index").reset_index(drop=True)
                    rows = process_event_dataframe(
                        g,
                        struck,
                        sample="unchanged",
                        csv_label=f"editable:{pshard.name}",
                        csv_momenta_breit=True,
                        pythia_reuse=pythia,
                    )
                    if len(rows) != 1:
                        raise RuntimeError(
                            f"unchanged event {eid_i}: expected 1 observable row, got {len(rows)}"
                        )
                    row = dict(rows[0])
                    row["hard_split_channel"] = ch_map.get(eid_i, want_label)
                    row["hard_channel_description"] = desc_map.get(eid_i, "")
                    all_rows.append(row)
                    n_un += 1
                    pbar_u.update(1)
        if n_un != len(altered_eids_done):
            raise RuntimeError(
                f"unchanged pairing: expected {len(altered_eids_done)} rows, wrote {n_un}"
            )

    out = Path(args.out_csv).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    n_ok = sum(1 for r in all_rows if r.get("ok"))
    print(
        json.dumps(
            {
                "altered_root": str(altered_root),
                "split_channel": want_label,
                "n_rows_written": len(all_rows),
                "n_altered_events": len(altered_eids_done),
                "n_ok": int(n_ok),
                "editable_parent": str(args.editable_parent) if args.editable_parent else "",
                "out_csv": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
