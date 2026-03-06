#!/usr/bin/env python3
"""
Compare OLD (generate_pdf_plots_new.py cached) vs NEW (analyze_events_raw.py) pTrel dumps.
Both must read the SAME shards. Run after:
  OLD: python3.11 generate_pdf_plots_new.py --use_cached_shards --label ISRFSR_ON (and ISRFSR_OFF)
  NEW: python3.11 analyze_events_raw.py  (with DEBUG_STOP_AT_1234=False for full run)

Expects:
  used_event_ids_{label}_OLD.npy, used_event_triplets_{label}_OLD.npy, debug_records_{label}_OLD.npy
  used_event_ids_{label}_NEW.npy, used_event_triplets_{label}_NEW.npy, debug_records_{label}_NEW.npy

Usage:
  python3.11 diff_old_new_ptrel.py ISRFSR_ON
  python3.11 diff_old_new_ptrel.py ISRFSR_OFF

Interpretation:
  - event_ids/triplets FAIL -> selection mismatch (cuts or pion choice)
  - debug_records FAIL -> kinematics/object mismatch (LT / vectors / pTrel)
  - all PASS but plots differ -> plotting/fit/normalization bug
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

import hashlib

import numpy as np


def md5(x: np.ndarray) -> str:
    return hashlib.md5(np.asarray(x).tobytes()).hexdigest()


def load(path):
    return np.load(path, allow_pickle=False)


def compare_event_ids(label):
    """Compare used_event_ids (N,2): (shard_idx, local_idx)."""
    root = _PROJECT_ROOT
    a = load(root / f"used_event_ids_{label}_OLD.npy").astype(np.int64)
    b = load(root / f"used_event_ids_{label}_NEW.npy").astype(np.int64)
    print(f"[{label}] OLD event_ids count={a.shape[0]} hash={md5(a)}")
    print(f"[{label}] NEW event_ids count={b.shape[0]} hash={md5(b)}")
    if md5(a) != md5(b):
        n = min(a.shape[0], b.shape[0])
        for i in range(n):
            if not np.all(a[i] == b[i]):
                print(f"FIRST EVENT_ID MISMATCH at i={i}: OLD={a[i]} NEW={b[i]}")
                break
        else:
            print("One is a prefix of the other.")
        return False
    return True


def compare_triplets(label):
    """Compare used_event_triplets (N,3): (shard_idx, local_idx, pid)."""
    root = _PROJECT_ROOT
    a = load(root / f"used_event_triplets_{label}_OLD.npy").astype(np.int64)
    b = load(root / f"used_event_triplets_{label}_NEW.npy").astype(np.int64)
    print(f"[{label}] OLD triplets hash={md5(a)}")
    print(f"[{label}] NEW triplets hash={md5(b)}")
    if md5(a) != md5(b):
        n = min(a.shape[0], b.shape[0])
        for i in range(n):
            if not np.all(a[i] == b[i]):
                print(f"FIRST TRIPLET MISMATCH at row={i}: OLD={a[i]} NEW={b[i]}")
                break
        else:
            print("One is a prefix of the other.")
        return False
    return True


def compare_debug_records(label, tol=1e-9):
    """Compare debug_records; first two columns are (shard_idx, local_idx). Sort by (col0, col1)."""
    root = _PROJECT_ROOT
    a = load(root / f"debug_records_{label}_OLD.npy")
    b = load(root / f"debug_records_{label}_NEW.npy")
    order_a = np.lexsort((a[:, 1], a[:, 0]))
    order_b = np.lexsort((b[:, 1], b[:, 0]))
    a = a[order_a]
    b = b[order_b]
    print(f"[{label}] OLD debug hash={md5(a)}")
    print(f"[{label}] NEW debug hash={md5(b)}")
    if a.shape != b.shape:
        print(f"DEBUG SHAPE DIFF: OLD {a.shape} vs NEW {b.shape}")
        return False
    diff = np.abs(a - b)
    maxdiff = diff.max()
    print(f"[{label}] max abs diff = {maxdiff:.3e}")
    if maxdiff > tol:
        i, j = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"LARGEST DIFF at row={i} col={j}: OLD={a[i,j]} NEW={b[i,j]} absdiff={diff[i,j]}")
        print("Row OLD:", a[i])
        print("Row NEW:", b[i])
        return False
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3.11 diff_old_new_ptrel.py <ISRFSR_OFF|ISRFSR_ON>")
        sys.exit(1)
    label = sys.argv[1]
    ok1 = compare_event_ids(label)
    ok2 = compare_triplets(label)
    ok3 = compare_debug_records(label, tol=1e-8)
    print(f"[{label}] RESULT:", "PASS" if (ok1 and ok2 and ok3) else "FAIL")
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)
