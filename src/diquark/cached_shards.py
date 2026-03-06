"""
Shared shard reader for pythia_finalstate_raw/<LABEL>/shard_XXXXXX/.
Used by analyze_events_raw.py (NEW) and generate_pdf_plots_new.py (OLD cached mode)
so both see identical events when running the diff harness.
"""
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = _PROJECT_ROOT / "pythia_finalstate_raw"


def _flip_z(v, do_flip: bool):
    """Negate pz to convert idA=11,idB=2212 frame to idA=2212,idB=11 frame."""
    if not do_flip:
        return v
    out = np.asarray(v, dtype=float).copy()
    if out.ndim == 1:
        out[3] = -out[3]
    else:
        out[..., 3] = -out[..., 3]
    return out


def list_shards(label: str):
    """Return sorted list of shard directories for label."""
    d = DATA_ROOT / label
    if not d.exists():
        return []
    subs = [x for x in d.iterdir() if x.is_dir() and x.name.startswith("shard_")]
    subs.sort(key=lambda x: x.name)
    return subs


def load_shard(shard_dir: Path):
    """Load one shard; return dict with event_* (Ne,4), offsets (Ne+1), pid (Np), p4 (Np,4)."""
    return {
        "event_e_in": np.load(shard_dir / "event_e_in.npy"),
        "event_p_in": np.load(shard_dir / "event_p_in.npy"),
        "event_e_sc": np.load(shard_dir / "event_e_sc.npy"),
        "event_k_out": np.load(shard_dir / "event_k_out.npy"),
        "offsets": np.load(shard_dir / "offsets.npy"),
        "pid": np.load(shard_dir / "pid.npy"),
        "p4": np.load(shard_dir / "p4.npy"),
    }


def iter_events_from_shards(label: str, flip_z: bool = False):
    """
    Yield (shard_idx, local_idx, event_data) for each event in each shard.
    event_data is a dict with:
      event_e_in, event_p_in, event_e_sc, event_k_out: 4-vectors (flipped if flip_z)
      offsets, pid, p4: full shard arrays (use offsets[ie]:offsets[ie+1] to slice)
    """
    shards = list_shards(label)
    for shard_idx, shard_path in enumerate(shards):
        data = load_shard(shard_path)
        e_in = data["event_e_in"]
        p_in = data["event_p_in"]
        e_sc = data["event_e_sc"]
        k_out = data["event_k_out"]
        Ne = e_in.shape[0]
        for ie in range(Ne):
            e_in_ev = _flip_z(np.asarray(e_in[ie], dtype=float), flip_z)
            p_in_ev = _flip_z(np.asarray(p_in[ie], dtype=float), flip_z)
            e_sc_ev = _flip_z(np.asarray(e_sc[ie], dtype=float), flip_z)
            k_out_ev = _flip_z(np.asarray(k_out[ie], dtype=float), flip_z)
            yield shard_idx, ie, {
                "event_e_in": e_in_ev,
                "event_p_in": p_in_ev,
                "event_e_sc": e_sc_ev,
                "event_k_out": k_out_ev,
                "offsets": data["offsets"],
                "pid": data["pid"],
                "p4": data["p4"],
            }
