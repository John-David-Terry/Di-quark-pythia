"""
Sharded Parquet output from ``split_dis_sample_diquark_kick_parquet.py`` (90/10 split).

Layout under ``<out-root>/split_90_10_parquet/``::

    unchanged/particles/shard_XXXXXX.parquet
    unchanged/events/shard_XXXXXX.parquet
    unchanged/manifest.parquet
    altered/particles/shard_XXXXXX.parquet
    altered/events/shard_XXXXXX.parquet
    altered/altered_metadata/shard_XXXXXX.parquet
    altered/manifest.parquet
    split_summary.json

**All-editable transform** (no 90/10 inside the 100k pool) lives under
``<out-root>/altered_100k_parquet/`` from ``transform_editable_all_altered_parquet.py``::

    particles/shard_XXXXXX.parquet
    events/shard_XXXXXX.parquet
    altered_metadata/shard_XXXXXX.parquet
    manifest.parquet
    run_summary.json

Loaders reconstruct the same per-event ``DataFrame`` shape as legacy ``event_*.csv`` for
reinjection / ``jet_hadron_observables_split_pi_pm``-style pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd

from editable_source_parquet import PARTICLE_CSV_COLUMN_ORDER, EVENT_TABLE_COLUMNS

SPLIT_SUBDIR = "split_90_10_parquet"
# ``transform_editable_all_altered_parquet.py`` (all editable events transformed)
ALTERED_ALL_SUBDIR = "altered_100k_parquet"


def split_root(output_parent: Path) -> Path:
    return output_parent.resolve() / SPLIT_SUBDIR


def branch_particles_dir(output_parent: Path, branch: str) -> Path:
    return split_root(output_parent) / branch / "particles"


def iter_branch_particle_shards(output_parent: Path, branch: str) -> Iterator[Path]:
    d = branch_particles_dir(output_parent, branch)
    if d.is_dir():
        yield from sorted(d.glob("shard_*.parquet"))


def _events_per_shard_from_split_summary(output_parent: Path, branch: str) -> int:
    p = split_root(output_parent) / "split_summary.json"
    if p.is_file():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            b = d.get("branch_shard_settings", {}).get(branch, {})
            return int(b.get("events_per_shard", d.get("events_per_output_shard", 10_000)))
        except (json.JSONDecodeError, TypeError, ValueError, KeyError):
            pass
    return 10_000


def _manifest_for_branch(output_parent: Path, branch: str) -> Optional[pd.DataFrame]:
    mp = split_root(output_parent) / branch / "manifest.parquet"
    if not mp.is_file():
        return None
    return pd.read_parquet(mp)


def _find_shard_for_event(man: pd.DataFrame, event_id: int, dataset: str) -> Optional[str]:
    sub = man[(man["dataset"] == dataset) & (man["first_event_id"] <= event_id) & (man["last_event_id"] >= event_id)]
    if sub.empty:
        return None
    return str(sub.iloc[0]["shard_path"])


def load_split_particles_dataframe(
    output_parent: Path,
    branch: str,
    event_id: int,
) -> pd.DataFrame:
    """Same columns/order as legacy ``event_XXXXXX.csv`` for ``branch`` (``altered`` | ``unchanged``)."""
    root = split_root(output_parent)
    man = _manifest_for_branch(output_parent, branch)
    if man is None:
        raise FileNotFoundError(f"missing manifest for branch={branch}")
    rel = _find_shard_for_event(man, int(event_id), "particles")
    if rel is None:
        raise KeyError(f"event_id {event_id} not found in {branch} particle manifest")
    path = root / branch / rel.replace("\\", "/")
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    g = df[df["event_id"] == int(event_id)].sort_values("particle_index").reset_index(drop=True)
    if g.empty:
        raise KeyError(f"no rows for event_id={event_id} in {path}")
    return g[PARTICLE_CSV_COLUMN_ORDER].copy()


def load_split_event_metadata_row(
    output_parent: Path,
    branch: str,
    event_id: int,
) -> pd.Series:
    root = split_root(output_parent)
    man = _manifest_for_branch(output_parent, branch)
    if man is None:
        raise FileNotFoundError(f"missing manifest for branch={branch}")
    rel = _find_shard_for_event(man, int(event_id), "events")
    if rel is None:
        raise KeyError(f"event_id {event_id} not found in {branch} events manifest")
    path = root / branch / rel.replace("\\", "/")
    edf = pd.read_parquet(path)
    rows = edf[edf["event_id"] == int(event_id)]
    if rows.empty:
        raise KeyError(f"no event row for event_id={event_id}")
    return rows.iloc[0]


def load_altered_surgery_meta_row(output_parent: Path, event_id: int) -> Optional[pd.Series]:
    """One row from ``altered/altered_metadata`` shards; ``None`` if event was not altered."""
    root = split_root(output_parent) / "altered" / "altered_metadata"
    if not root.is_dir():
        return None
    for pq in sorted(root.glob("shard_*.parquet")):
        df = pd.read_parquet(pq)
        if "event_id" not in df.columns:
            continue
        sub = df[df["event_id"] == int(event_id)]
        if not sub.empty:
            return sub.iloc[0]
    return None


def altered_all_root(output_parent: Path) -> Path:
    return output_parent.resolve() / ALTERED_ALL_SUBDIR


def _manifest_altered_all(output_parent: Path) -> Optional[pd.DataFrame]:
    mp = altered_all_root(output_parent) / "manifest.parquet"
    if not mp.is_file():
        return None
    return pd.read_parquet(mp)


def load_altered_all_particles_dataframe(output_parent: Path, event_id: int) -> pd.DataFrame:
    """Particles for one event from ``altered_100k_parquet`` (full altered transform output)."""
    root = altered_all_root(output_parent)
    man = _manifest_altered_all(output_parent)
    if man is None:
        raise FileNotFoundError(f"missing {root / 'manifest.parquet'}")
    rel = _find_shard_for_event(man, int(event_id), "particles")
    if rel is None:
        raise KeyError(f"event_id {event_id} not in altered_all manifest")
    path = root / rel.replace("\\", "/")
    df = pd.read_parquet(path)
    g = df[df["event_id"] == int(event_id)].sort_values("particle_index").reset_index(drop=True)
    if g.empty:
        raise KeyError(f"no particles for event_id={event_id}")
    return g[PARTICLE_CSV_COLUMN_ORDER].copy()


def load_altered_all_event_row(output_parent: Path, event_id: int) -> pd.Series:
    root = altered_all_root(output_parent)
    man = _manifest_altered_all(output_parent)
    if man is None:
        raise FileNotFoundError(f"missing manifest")
    rel = _find_shard_for_event(man, int(event_id), "events")
    if rel is None:
        raise KeyError(f"event_id {event_id} not in events manifest")
    edf = pd.read_parquet(root / rel.replace("\\", "/"))
    rows = edf[edf["event_id"] == int(event_id)]
    if rows.empty:
        raise KeyError(f"no event row for event_id={event_id}")
    return rows.iloc[0]


def load_altered_all_metadata_row(output_parent: Path, event_id: int) -> pd.Series:
    """Surgery / outcome row (includes ``alteration_succeeded``, ``failure_reason``)."""
    root = altered_all_root(output_parent)
    man = _manifest_altered_all(output_parent)
    if man is None:
        raise FileNotFoundError(f"missing manifest")
    rel = _find_shard_for_event(man, int(event_id), "altered_metadata")
    if rel is None:
        raise KeyError(f"event_id {event_id} not in altered_metadata manifest")
    mdf = pd.read_parquet(root / rel.replace("\\", "/"))
    rows = mdf[mdf["event_id"] == int(event_id)]
    if rows.empty:
        raise KeyError(f"no metadata for event_id={event_id}")
    return rows.iloc[0]


def metadata_row_for_reinject(meta_row: pd.Series) -> Dict[str, Any]:
    """Subset compatible with ``dis_isr_event_metadata.csv`` / struck resolution."""
    keys = [
        "event_id",
        "Q2",
        "xB",
        "x",
        "Q",
        "qT",
        "y",
        "S",
        "phiq",
        "struck_incoming_index",
        "struck_outgoing_index",
        "transverse_kick_applied",
        "kick_kx_gev",
        "kick_ky_gev",
    ]
    return {k: meta_row[k] for k in keys if k in meta_row.index}
