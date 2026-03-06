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

## Project structure

```
Di-quark-pythia/
├── src/diquark/               # Core package
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
├── ETA_CONFIG_CHECKLIST.md    # Config notes (ColourReconnection, beam order)
└── pythia_finalstate_raw/     # Sharded event data
    ├── ISRFSR_OFF/
    ├── ISRFSR_ON/
    └── ETA_ON_CRON/
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

Output: `pythia_finalstate_raw/<LABEL>/shard_XXXXXX/` with `event_e_in.npy`, `event_p_in.npy`, `event_e_sc.npy`, `event_k_out.npy`, `offsets.npy`, `pid.npy`, `p4.npy`, `meta.json`.

### 2. Run main analysis (cached shards)

Analyze events from cached shards (no live PYTHIA):

```bash
python scripts/analysis/analyze_events_raw.py
```

Produces:

- `eta_hadron_EIC_hardware_QCD_regions.pdf`
- `pTrel_target_pion_wrt_remnant_axis_2D_fit_gaussian_comparison.pdf`

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

Produces `S_Rpi_ISRFSR_ON.npy`, `D_Rpi_ISRFSR_ON.npy`, `S_Jpi_ISRFSR_ON.npy`, `D_Jpi_ISRFSR_ON.npy`.

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
