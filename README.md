# Di-quark-pythia

PYTHIA 8–based analysis of **di-hadron and TMD (Transverse Momentum Dependent) physics** in Deep Inelastic Scattering (DIS) for EIC (Electron–Ion Collider) kinematics.

## Overview

This project generates and analyzes DIS events using PYTHIA 8 with EIC-like beam energies (18 GeV electron × 275 GeV proton). It studies hadron pseudorapidity, transverse momentum distributions, and related observables in the Breit frame.

### Physics observables

- **Hadron pseudorapidity** (η<sub>h</sub>) and EIC detector coverage
- **p<sub>T,rel</sub>** — pion p<sub>T</sub> relative to the remnant axis (with Gaussian fits)
- **Transverse sums/differences**: S<sub>Rπ</sub> = |p<sub>T,π</sub> + p<sub>T,rem</sub>|, D<sub>Rπ</sub> = |p<sub>T,π</sub> − p<sub>T,rem</sub>|, S<sub>Jπ</sub>, D<sub>Jπ</sub>
- **Jacobian-corrected distributions** for Gaussian behavior tests
- **Jet azimuth** (φ<sub>J</sub>) and hadron azimuth relative to the jet in the Breit frame

## Requirements

- **Python** 3.8+
- **PYTHIA 8** (Python bindings: `pythia8`)
- **NumPy**
- **Matplotlib**
- **SciPy** (optional, for `curve_fit` in Gaussian fits; some scripts require it)

### Installation

```bash
pip install numpy matplotlib scipy pythia8
# Optional: install project in editable mode for development
pip install -e .
```

PYTHIA 8 may need to be built from source with Python bindings; see the [PYTHIA 8 documentation](https://pythia.org/).

**Note**: All scripts must be run from the **project root** directory.

## Data directory (storage hygiene)

**Convention:** keep **source code** in your normal project tree (e.g. `~/Documents/Projects/Di-quark-pythia/`) and keep **large simulation outputs** elsewhere. This repository is intended to hold **source only**; generated data should live in a separate directory.

**Do not** store large outputs under **`~/Documents`** or **`~/Desktop`** if those folders are synced to iCloud — that pattern has caused heavy CloudDocs / FileProvider activity and editor slowdowns in practice.

**Recommended data root:** `~/Data/Di-quark-pythia-nosync/`

The `-nosync` suffix is a common macOS reminder that the folder is not meant for iCloud Drive. The code default matches this path when `DIQUARK_DATA_ROOT` is unset.

### Quick setup (optional)

From the project root:

```bash
chmod +x scripts/setup_data_root.sh   # once
./scripts/setup_data_root.sh
```

This creates `~/Data/Di-quark-pythia-nosync` (or `DIQUARK_DATA_ROOT` if set) and, on macOS, runs `tmutil addexclusion` on that directory so Time Machine skips it (large simulation trees are poor backup targets). If `tmutil` fails or is unavailable, create the directory manually and set `DIQUARK_DATA_ROOT` yourself.

### Configuration

- **Environment**: set `DIQUARK_DATA_ROOT` (or `DIQUARK_OUTPUT_ROOT`) to an absolute path. If unset, the default is **`~/Data/Di-quark-pythia-nosync/`**.
- If the resolved path lies under **`~/Documents`** or **`~/Desktop`**, Python code in `diquark.paths` emits a **one-time** `UserWarning` — you should point `DIQUARK_DATA_ROOT` at a safer location.

**If you already used the old default (`~/Data/Di-quark-pythia/`):** the codebase no longer uses that path by default. New runs will write to **`~/Data/Di-quark-pythia-nosync/`** unless you override the environment. To keep using your existing tree, either **move or rename** the directory to match the new default (e.g. rename to `Di-quark-pythia-nosync`) or set **`DIQUARK_DATA_ROOT`** explicitly to `~/Data/Di-quark-pythia` so nothing “disappears” — it is simply under a different path than new jobs.

**Typical layout under the data root:**

  - `pythia_finalstate_raw/<LABEL>/shard_XXXXXX/` — sharded PYTHIA dumps from `generate_events_raw.py`
  - `outputs/` — pipeline outputs (DIS CSV/LHE experiments, benchmarks, POPF tools, etc.)
  - `outputs/analysis/` — η / pTrel PDFs, transverse `.npy` arrays, Jacobian plot inputs, and similar analysis products
  - `run_manifests/` — small JSON summaries from some jobs (resolved data root, dirs touched, approximate file counts)

### Repo vs Cursor / git

Heavy file types and output directory names are listed in **`.gitignore`** and **`.cursorignore`** so accidental copies of data **inside** the clone are not tracked or indexed. Your primary data directory should still live **outside** the repo (the default paths above).

**Follow-up:** a few auxiliary tools (for example some C/C++ helpers under `scripts/cpp/`) may still use project-local paths; audit those separately if you rely on them for production outputs.

## Project structure

```
Di-quark-pythia/
├── src/diquark/               # Core package
│   ├── paths.py               # DIQUARK_DATA_ROOT + output path helpers
│   ├── cached_shards.py       # Shard reader for event data
│   └── analyze_events_raw.py  # Main analysis logic
├── scripts/
│   ├── generation/            # Event generation
│   │   ├── generate_events_raw.py
│   │   └── generate_pdf_plots_new.py
│   ├── analysis/              # Analysis scripts
│   │   ├── analyze_events_raw.py
│   │   └── compute_transverse_observables.py
│   ├── plots/                 # Plotting scripts
│   │   ├── plot_observables_isrfsr_on.py
│   │   ├── plot_S_Rpi_jacobian_corrected.py
│   │   ├── plot_D_Rpi_jacobian_corrected.py
│   │   ├── phi_J_breit_ISRFSR_ON.py
│   │   └── phi_h_relative_to_jet_ISRFSR_ON.py
│   └── tests/                 # Tests and validation
│       ├── test_tmd_narrow_bin.py
│       ├── test_ptrel_remnant_narrow_bin.py
│       ├── diff_old_new_ptrel.py
│       └── debug_compare_cached_vs_pythia_eta.py
├── notebooks/                 # Jupyter notebooks
└── ETA_CONFIG_CHECKLIST.md    # Config notes (ColourReconnection, beam order)

# Sharded data and heavy outputs live under DIQUARK_DATA_ROOT (default ~/Data/Di-quark-pythia-nosync/), e.g.:
#   pythia_finalstate_raw/<LABEL>/shard_XXXXXX/
```

## Usage

Run all commands from the **project root** directory.

### 1. Generate events

Generate raw PYTHIA events and write them to sharded NumPy files:

```bash
# Generate all labels (ISRFSR_ON, ISRFSR_OFF, ETA_ON_CRON)
python scripts/generation/generate_events_raw.py

# Generate specific labels
python scripts/generation/generate_events_raw.py --labels ISRFSR_ON,ISRFSR_OFF
python scripts/generation/generate_events_raw.py --labels ETA_ON_CRON
```

Output (under `DIQUARK_DATA_ROOT`): `pythia_finalstate_raw/<LABEL>/shard_XXXXXX/` with `event_e_in.npy`, `event_p_in.npy`, `event_e_sc.npy`, `event_k_out.npy`, `offsets.npy`, `pid.npy`, `p4.npy`, `meta.json`.

### 2. Run main analysis (cached shards)

Analyze events from cached shards (no live PYTHIA):

```bash
python scripts/analysis/analyze_events_raw.py
```

Produces (under `DIQUARK_DATA_ROOT/outputs/analysis/`):

- `eta_hadron_<e>x<p>_xQ_regions_3x2.pdf` — four figures for `(e,p)` = (5,41), (9,41), (9,100), (9,275) GeV (requires `ETA_XQ_*` shards; see **Beam-specific eta grids** below)
- `eta_hadron_xQ_regions_summary.json` and `eta_hadron_xQ_regions_summary.csv` — per-panel event counts and bin metadata
- `eta_hadron_EIC_hardware_QCD_regions.pdf`
- `pTrel_target_pion_wrt_remnant_axis_2D_fit_gaussian_comparison.pdf`

**Beam-specific eta grids:** generate cached shards for the additional beam settings (same workflow as other labels):

```bash
python scripts/generation/generate_events_raw.py \
  --labels ETA_XQ_5x41,ETA_XQ_9x41,ETA_XQ_9x100,ETA_XQ_9x275 --n_events 100000
```

Then run `python scripts/analysis/analyze_events_raw.py` as above (no need for `PYTHONPATH=src`: both scripts prepend `src/` themselves; forcing `PYTHONPATH=src` can break third-party imports such as NumPy on some setups).

### 3. Generate PDFs (live or cached)

```bash
# Use cached shards
python scripts/generation/generate_pdf_plots_new.py --use_cached_shards --label ISRFSR_ON

# Run PYTHIA directly (no cached shards)
python scripts/generation/generate_pdf_plots_new.py
```

### 4. Compute transverse observables

```bash
python scripts/analysis/compute_transverse_observables.py
```

Produces `S_Rpi_ISRFSR_ON.npy`, `D_Rpi_ISRFSR_ON.npy`, `S_Jpi_ISRFSR_ON.npy`, `D_Jpi_ISRFSR_ON.npy` under `DIQUARK_DATA_ROOT/outputs/analysis/`.

### 5. Plotting scripts

```bash
python scripts/plots/plot_observables_isrfsr_on.py
python scripts/plots/plot_S_Rpi_jacobian_corrected.py
python scripts/plots/plot_D_Rpi_jacobian_corrected.py
python scripts/plots/phi_J_breit_ISRFSR_ON.py
python scripts/plots/phi_h_relative_to_jet_ISRFSR_ON.py
```

### 6. TMD / narrow-bin tests

```bash
python scripts/tests/test_tmd_narrow_bin.py
python scripts/tests/test_ptrel_remnant_narrow_bin.py
```

### 7. Validation

```bash
python scripts/tests/diff_old_new_ptrel.py ISRFSR_ON
python scripts/tests/diff_old_new_ptrel.py ISRFSR_OFF
```

## Configuration

### Kinematics

- **E<sub>e</sub>** = 18 GeV, **E<sub>p</sub>** = 275 GeV (EIC-like)
- **Q²<sub>min</sub>** = 16 GeV²
- **Frame**: Breit-like via Lorentz transformation

### Shard labels

| Label        | Description                          |
|--------------|--------------------------------------|
| `ISRFSR_ON`  | ISR/FSR on, ColourReconnection off  |
| `ISRFSR_OFF` | ISR/FSR off                          |
| `ETA_ON_CRON`| ColourReconnection on (for η analysis)|

**Important**: Cached shards (`ISRFSR_ON`/`ISRFSR_OFF`) use different beam order and ColourReconnection settings than the original η analysis. See `ETA_CONFIG_CHECKLIST.md` for details and how to regenerate with matching config.

## Notebooks

Notebooks are in the `notebooks/` directory:

- **`generate_pdf_plots.ipynb`** — Original PDF generation (η, pTrel)
- **`generate_pdf_plots_new.ipynb`** — Updated version of the script
- **`di-hadron.ipynb`** — Di-hadron analysis with DIS kinematics
- **`Decorrelation.ipynb`** — Breit-frame kinematics and transformation matrices

Run Jupyter from the project root so that imports and data paths resolve correctly.

## License

See repository for license information.
