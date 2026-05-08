#!/usr/bin/env python3
"""Print a clean console summary from split_90_10/split_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

DEFAULT_SUMMARY = outputs_dir() / "dis_isr_parton_dataset" / "split_90_10" / "split_summary.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Print split_90_10 summary statistics.")
    ap.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Path to split_summary.json",
    )
    args = ap.parse_args()
    path: Path = args.summary
    if not path.exists():
        raise SystemExit(f"Summary not found: {path}")

    s = json.loads(path.read_text(encoding="utf-8"))

    total = s.get("total_input_events", s.get("total_ids", "n/a"))
    w_unch = s.get("written_unchanged", "n/a")
    w_alt = s.get("written_altered", "n/a")
    att = s.get("attempted_altered", s.get("planned_altered", "n/a"))
    failed = s.get("failed_altered", None)
    if failed is None and att != "n/a" and w_alt != "n/a":
        failed = int(att) - int(w_alt)
    ch_c = s.get("altered_channel_C", s.get("altered_channel_A", "n/a"))

    print("=== split_90_10 summary ===")
    print(f"  file: {path}")
    print(f"  total input events:     {total}")
    print(f"  unchanged written:      {w_unch}")
    print(f"  altered written:        {w_alt}")
    print(f"  attempted altered:      {att}")
    print(f"  failed altered:         {failed}")
    print(f"  altered channel C:      {ch_c}")
    if "fallback_unchanged_no_ud_diquark" in s:
        print(
            f"  (detail) fallback no ud(2101): {s['fallback_unchanged_no_ud_diquark']}, "
            f"struck_fail: {s.get('fallback_unchanged_struck_fail', 0)}, "
            f"validation: {s.get('fallback_unchanged_validation_fail', 0)}"
        )


if __name__ == "__main__":
    main()
