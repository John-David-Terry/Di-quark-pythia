#!/usr/bin/env python3
"""
Generate raw PYTHIA events and store in sharded format for cached analysis.
Output: pythia_finalstate_raw/<LABEL>/shard_XXXXXX/

Run from project root: python scripts/generation/generate_events_raw.py
"""
import argparse
import json
from pathlib import Path
import numpy as np

import pythia8

# -----------------------------
# Configuration
# -----------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = _PROJECT_ROOT / "pythia_finalstate_raw"
OUTDIR.mkdir(parents=True, exist_ok=True)

N_EVENTS_PER_CONFIG = 1_000_000  # Set to 20_000 for quick test; comment out ISRFSR_* in main() to generate only ETA_ON_CRON
EVENTS_PER_SHARD = 10_000

E_E = 18.0
E_P = 275.0
Q2_MIN = 16.0

BASE_SEED = 12345

DEBUG = False
DEBUG_ATTEMPTS = 10_000

DT_FLOAT = np.float32
DT_INT = np.int32
DT_OFF = np.int64


# -----------------------------
# Helpers
# -----------------------------
def p4_from_particle(p):
    return np.array([p.e(), p.px(), p.py(), p.pz()], dtype=DT_FLOAT)


def get_scattered_electron(ev):
    """Prefer status 44; else highest-energy e- with status>0."""
    electrons = []
    for i in range(ev.size()):
        p = ev[i]
        if p.id() == 11 and p.status() > 0:
            electrons.append(p)
    if not electrons:
        return None
    for e in electrons:
        if e.status() == 44:
            return e
    return max(electrons, key=lambda x: x.e())


def find_incoming_beams(ev):
    """Find incoming electron and proton by id and status (independent of beam order)."""
    e_in = None
    p_in = None
    for i in range(ev.size()):
        p = ev[i]
        if p.id() == 11 and p.status() == -12:
            e_in = p4_from_particle(p)
        if p.id() == 2212 and p.status() < 0:
            p_in = p4_from_particle(p)
    return e_in, p_in


def find_k_out(ev):
    """Find outgoing struck quark. Accept ±23, ±63..±68, or any quark with status!=0."""
    quark_ids = (1, 2, 3, 4, 5, 6)
    for i in range(ev.size()):
        p = ev[i]
        if abs(p.id()) in quark_ids and abs(p.status()) == 23:
            return p4_from_particle(p)
    for i in range(ev.size()):
        p = ev[i]
        if abs(p.id()) in quark_ids and (63 <= abs(p.status()) <= 68):
            return p4_from_particle(p)
    best = None
    bestE = -1.0
    for i in range(ev.size()):
        p = ev[i]
        if abs(p.id()) in quark_ids and p.status() != 0:
            if p.e() > bestE:
                bestE = p.e()
                best = p
    if best is None:
        return None
    return p4_from_particle(best)


def shard_dir(label: str, shard_idx: int) -> Path:
    return OUTDIR / label / f"shard_{shard_idx:06d}"


def shard_complete(label: str, shard_idx: int) -> bool:
    d = shard_dir(label, shard_idx)
    return (d / "meta.json").exists() and (d / "offsets.npy").exists() and (d / "pid.npy").exists() and (d / "p4.npy").exists()


def next_shard_to_write(label: str) -> int:
    idx = 0
    while shard_complete(label, idx):
        idx += 1
    return idx


def write_shard(
    label: str,
    shard_idx: int,
    e_in_arr,
    p_in_arr,
    e_sc_arr,
    k_out_arr,
    offsets,
    pid,
    p4,
    meta: dict,
):
    d = shard_dir(label, shard_idx)
    d.mkdir(parents=True, exist_ok=True)

    np.save(d / "event_e_in.npy", e_in_arr.astype(DT_FLOAT, copy=False))
    np.save(d / "event_p_in.npy", p_in_arr.astype(DT_FLOAT, copy=False))
    np.save(d / "event_e_sc.npy", e_sc_arr.astype(DT_FLOAT, copy=False))
    np.save(d / "event_k_out.npy", k_out_arr.astype(DT_FLOAT, copy=False))
    np.save(d / "offsets.npy", offsets.astype(DT_OFF, copy=False))
    np.save(d / "pid.npy", pid.astype(DT_INT, copy=False))
    np.save(d / "p4.npy", p4.astype(DT_FLOAT, copy=False))

    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def build_pythia(
    idA: int,
    idB: int,
    eA: float,
    eB: float,
    colour_reconnect_on: bool,
    isr_fsr_on: bool,
    seed: int,
):
    """Build PYTHIA with configurable beams and options."""
    p = pythia8.Pythia()

    p.readString(f"Beams:idA = {idA}")
    p.readString(f"Beams:idB = {idB}")
    p.readString(f"Beams:eA = {eA}")
    p.readString(f"Beams:eB = {eB}")
    p.readString("Beams:frameType = 2")
    p.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    p.readString("HardQCD:all = off")
    p.readString("PDF:lepton = off")
    p.readString(f"PhaseSpace:Q2Min = {Q2_MIN}")

    p.readString("HadronLevel:all = on")
    p.readString(f"ColourReconnection:reconnect = {'on' if colour_reconnect_on else 'off'}")

    if isr_fsr_on:
        p.readString("PartonLevel:ISR = on")
        p.readString("PartonLevel:FSR = on")
    else:
        p.readString("PartonLevel:ISR = off")
        p.readString("PartonLevel:FSR = off")

    p.readString("Random:setSeed = on")
    p.readString(f"Random:seed = {seed}")

    p.init()
    return p


def _config_fingerprint(cfg: dict) -> dict:
    """Serializable fingerprint for a label's PYTHIA config."""
    return {
        "idA": int(cfg["idA"]),
        "idB": int(cfg["idB"]),
        "eA": float(cfg["eA"]),
        "eB": float(cfg["eB"]),
        "isr_fsr_on": bool(cfg["isr_fsr_on"]),
        "colour_reconnect_on": bool(cfg["colour_reconnect_on"]),
        "seed_offset": int(cfg["seed_offset"]),
        "Q2_MIN": float(Q2_MIN),
        "E_E": float(E_E),
        "E_P": float(E_P),
    }


def _check_or_write_fingerprint(label: str, cfg: dict) -> None:
    """If config_fingerprint.json exists, assert it matches cfg; else write it."""
    label_dir = OUTDIR / label
    label_dir.mkdir(parents=True, exist_ok=True)
    fp_path = label_dir / "config_fingerprint.json"
    fingerprint = _config_fingerprint(cfg)
    if fp_path.exists():
        with open(fp_path) as f:
            existing = json.load(f)
        if existing != fingerprint:
            raise RuntimeError(
                f"[{label}] config_fingerprint.json does not match current settings. "
                f"Existing: {existing}. Current: {fingerprint}. "
                "Remove or update the fingerprint to avoid mixing incompatible shards."
            )
    else:
        with open(fp_path, "w") as f:
            json.dump(fingerprint, f, indent=2, sort_keys=True)
        print(f"[{label}] wrote config_fingerprint.json")


def generate_config(
    label: str,
    cfg: dict,
    n_events: int,
    events_per_shard: int,
    config_meta: dict = None,
):
    _check_or_write_fingerprint(label, cfg)
    pythia_builder = lambda: build_pythia(
        idA=cfg["idA"],
        idB=cfg["idB"],
        eA=cfg["eA"],
        eB=cfg["eB"],
        colour_reconnect_on=cfg["colour_reconnect_on"],
        isr_fsr_on=cfg["isr_fsr_on"],
        seed=BASE_SEED + cfg["seed_offset"],
    )
    shard_idx = next_shard_to_write(label)
    already_done = shard_idx * events_per_shard
    if already_done >= n_events:
        print(f"[{label}] Already complete ({already_done} >= {n_events}).")
        return

    pythia = pythia_builder()

    if already_done > 0:
        print(f"[{label}] Resuming: skipping {already_done} events to reach shard {shard_idx}.")
        skipped = 0
        while skipped < already_done:
            if not pythia.next():
                continue
            skipped += 1

    produced_total = already_done
    while produced_total < n_events:
        Ne = min(events_per_shard, n_events - produced_total)

        e_in_arr = np.zeros((Ne, 4), dtype=DT_FLOAT)
        p_in_arr = np.zeros((Ne, 4), dtype=DT_FLOAT)
        e_sc_arr = np.zeros((Ne, 4), dtype=DT_FLOAT)
        k_out_arr = np.zeros((Ne, 4), dtype=DT_FLOAT)

        offsets = np.zeros(Ne + 1, dtype=DT_OFF)
        pid_list = []
        p4_list = []

        kept_events = 0
        attempts = 0

        while kept_events < Ne:
            attempts += 1
            if DEBUG and attempts >= DEBUG_ATTEMPTS:
                break
            if not pythia.next():
                continue

            ev = pythia.event

            e_in, p_in = find_incoming_beams(ev)
            if e_in is None or p_in is None:
                continue

            e_sc = get_scattered_electron(ev)
            if e_sc is None:
                continue

            k_out = find_k_out(ev)
            if k_out is None:
                continue

            e_in_arr[kept_events] = e_in
            p_in_arr[kept_events] = p_in
            e_sc_arr[kept_events] = p4_from_particle(e_sc)
            k_out_arr[kept_events] = k_out

            n_this = 0
            for i in range(ev.size()):
                p = ev[i]
                if p.isFinal():
                    pid_list.append(int(p.id()))
                    p4_list.append([p.e(), p.px(), p.py(), p.pz()])
                    n_this += 1

            offsets[kept_events + 1] = offsets[kept_events] + n_this
            kept_events += 1

        if DEBUG:
            print(f"[{label}] DEBUG: attempts={attempts}, kept={kept_events}")
            return

        pid = np.asarray(pid_list, dtype=DT_INT)
        p4 = np.asarray(p4_list, dtype=DT_FLOAT)

        meta = {
            "label": label,
            "E_e": float(E_E),
            "E_p": float(E_P),
            "Q2_min": float(Q2_MIN),
            "seed": int(pythia.settings.mode("Random:seed")),
            **(config_meta or {}),
            "events_in_shard": int(Ne),
            "attempts_for_shard": int(attempts),
            "particles_in_shard": int(pid.shape[0]),
            "event_index_start": int(produced_total),
            "event_index_end_exclusive": int(produced_total + Ne),
            "format": {
                "event_vectors": "float32 (E,px,py,pz)",
                "p4": "float32 (E,px,py,pz)",
                "pid": "int32",
                "offsets": "int64 prefix sum into pid/p4",
                "final_state_only": True,
            },
        }

        write_shard(label, shard_idx, e_in_arr, p_in_arr, e_sc_arr, k_out_arr, offsets, pid, p4, meta)

        produced_total += Ne
        print(f"[{label}] shard {shard_idx}: {Ne} events (total={produced_total}/{n_events}), acceptance={Ne/attempts:.4f}")
        shard_idx += 1

    print(f"[{label}] done.")
    pythia.stat()


# Label -> (idA, idB, eA, eB, isr_fsr_on, colour_reconnect_on, seed_offset)
# Used for CLI --labels and config_fingerprint; defaults to all three.
LABEL_CONFIGS = {
    "ETA_ON_CRON": {
        "idA": 2212,
        "idB": 11,
        "eA": E_P,
        "eB": E_E,
        "isr_fsr_on": True,
        "colour_reconnect_on": True,
        "seed_offset": 100,
        "config_meta": {"Beams:idA": 2212, "Beams:idB": 11, "ColourReconnection": "on"},
    },
    "ISRFSR_ON": {
        "idA": 11,
        "idB": 2212,
        "eA": E_E,
        "eB": E_P,
        "isr_fsr_on": True,
        "colour_reconnect_on": False,
        "seed_offset": 1,
        "config_meta": {"Beams:idA": 11, "Beams:idB": 2212, "ColourReconnection": "off"},
    },
    "ISRFSR_OFF": {
        "idA": 11,
        "idB": 2212,
        "eA": E_E,
        "eB": E_P,
        "isr_fsr_on": False,
        "colour_reconnect_on": False,
        "seed_offset": 0,
        "config_meta": {"Beams:idA": 11, "Beams:idB": 2212, "ColourReconnection": "off"},
    },
}

DEFAULT_LABELS = list(LABEL_CONFIGS.keys())


def main():
    parser = argparse.ArgumentParser(description="Generate raw PYTHIA shards per label.")
    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(DEFAULT_LABELS),
        help=f"Comma-separated labels to generate (default: all). Choices: {DEFAULT_LABELS}",
    )
    parser.add_argument(
        "--n_events",
        type=int,
        default=N_EVENTS_PER_CONFIG,
        help=f"Target events per selected label (default: {N_EVENTS_PER_CONFIG})",
    )
    parser.add_argument(
        "--events_per_shard",
        type=int,
        default=EVENTS_PER_SHARD,
        help=f"Events per shard (default: {EVENTS_PER_SHARD})",
    )
    args = parser.parse_args()

    requested = [s.strip() for s in args.labels.split(",") if s.strip()]
    unknown = [l for l in requested if l not in LABEL_CONFIGS]
    if unknown:
        parser.error(f"Unknown label(s): {unknown}. Valid: {list(LABEL_CONFIGS.keys())}")

    n_events = args.n_events
    events_per_shard = args.events_per_shard
    print(f"Labels: {requested}, n_events={n_events}, events_per_shard={events_per_shard}")

    for label in requested:
        cfg = LABEL_CONFIGS[label].copy()
        config_meta = cfg.pop("config_meta", None)
        generate_config(
            label=label,
            cfg=cfg,
            n_events=n_events,
            events_per_shard=events_per_shard,
            config_meta=config_meta,
        )


if __name__ == "__main__":
    main()
