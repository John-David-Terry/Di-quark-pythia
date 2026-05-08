# Backward trace: `dis_us_sbar_1M` → event generation (for LLM context)

This document summarizes how the **analysis folder `analysis/dis_us_sbar_1M/`** connects backward to **PYTHIA event generation** in the **Di-quark-pythia** repository. It is written to be pasted into ChatGPT (or similar) as pipeline context.

**Repository root:** project containing `src/`, `scripts/analysis/`, and this path.

---

## 1. What exists under `analysis/dis_us_sbar_1M/`

| Artifact | Role | Notes |
|----------|------|--------|
| `editable_source_v1/run_summary.json` | Summary of the **first** generation pass | Records **1M accepted** events (`n_accepted: 1000000`), Parquet shards, `pythia_seed: 902452`, Breit frame, no kick on this pool (`kick_fraction: 0`). |
| `altered_100k_parquet/run_summary.json` | Summary of the **diquark alteration** step | **1M events processed**; **487950** written as successful **channel C** alterations; **512050** fallback **unchanged**; mode `diquark_kick`, `delta_px: 0.4`, seed **42**; paths point at `editable_source_v1` as input and `altered_100k_parquet` as output. |
| `jet_hadron_pi_pm_observables_signal54217_plus_unchanged487950_manifest.json` | **Post hoc** CSV merge | References `jet_hadron_pi_pm_observables_us_sbar_1M.csv` → merged output CSV; **54217** sampled “signal” altered rows, **487950** unchanged rows, **542167** total rows, **`random_seed: 42`**. **The merge script is not in the repo** (likely a one-off `pandas` sample + concat). |
| `figures_pi_pm_signal54217/jet_hadron_transverse_angle_target_leading_pion_Breit_combined_pi_pm_side_by_side.png` | A **π-related** comparison figure | **Filename only** in repo; the exact plotting command that produced this PNG is **not** committed. |

Large CSV/Parquet products (e.g. `jet_hadron_pi_pm_observables_us_sbar_1M.csv`) are often **gitignored**; they may exist only on the author’s machine under the same directory layout.

---

## 2. Forward pipeline (causal order, for intuition)

1. **Generate** 1M **struck-u** DIS events with PYTHIA, **parton-level / no hadronization in the stored pass**, write **sharded Parquet** under `editable_source_v1/`.
2. **Alter** each event: diquark split + transverse kick, **channel C** topology \((ud) \to d + (us) + \bar s\) when possible; write `altered_100k_parquet/` (name is historical; this run is **1M** events).
3. **Re-inject** selected parton states into **PYTHIA hadronization** and compute **jet–hadron** observables (including **φ\_{hJ}**-style angle and π⁺/π⁻ splits) into a large **CSV** (e.g. `jet_hadron_pi_pm_observables_us_sbar_1M.csv`).
4. **Optionally subsample/merge** rows (manifest: 54 217 altered + 487 950 unchanged) for specific plots.
5. **Plot** with `plot_jet_hadron_pi_pm_observables.py` + `jet_hadron_pi_pm_figures.py` (or related Parquet plotters — see §6).

---

## 3. Step A — First generation: `editable_source_v1/`

**Script:** `scripts/analysis/generate_dis_editable_source_parquet.py`

**What it does:** Same **physics/selection** family as `generate_dis_isr_parton_dataset.py`, but output is **sharded Parquet** (`editable_source_v1/particles/`, `editable_source_v1/events/`) instead of a monolithic CSV.

**PYTHIA configuration (from `build_pythia_source` in that script):** default **18 GeV e⁻ on 275 GeV proton** (EIC-style), DIS, parton/ISR/FSR flags as in that function, **`HadronLevel:all = off`** so the **stored** record is **pre-hadronization** (editable for later steps).

**Acceptance (important):** the main loop **only accepts events where the incoming struck quark is a u quark** (`abs(PDG)==2`). So the pool is **struck-u DIS**, not “all flavors.”

**This run’s parameters (from `run_summary.json`):** `n_accepted: 1000000`, `n_generated_tried: 1423678`, `pythia_seed: 902452` (via `--seed` to match), `use_breit_frame: true`, `kick_fraction: 0`.

---

## 4. Step B — Alteration: `altered_100k_parquet/`

**Script:** `scripts/analysis/transform_editable_all_altered_parquet.py`

**What it does:** Reads **every** event in `editable_source_v1/`, runs **`split_dis_sample_diquark_kick.process_one_event_split`**, hard topology **channel C** (see script docstring: \((ud) \to [d] + (us) + \bar s\)), writes **altered** parton data + **altered_metadata** under `altered_100k_parquet/`.

**This run (from `run_summary.json`):** `total_processed: 1000000`, `written_altered: 487950` (channel C), `written_unchanged: 512050` (fallback when the ud diquark path is not available or validation fails as per that pipeline), `delta_px: 0.4`, `seed: 42`, `editable_source` and `output_root` under `analysis/dis_us_sbar_1M/`.

**Folder name `dis_us_sbar_1M`:** “**us** + **s̄**” matches the **altered** **channel C** flavor story; the **initial** generation is still **struck-**u (PDG 2), not “us” in the initial hard process.

**Hadronization:** **not** in this step; that comes at **reinject** (next section).

---

## 5. Step C — Observables CSV: `jet_hadron_pi_pm_observables_*.csv`

**Primary in-repo driver for “altered Parquet + channel filter + optional unchanged twin rows”:**  
`scripts/analysis/jet_hadron_observables_altered_parquet_filtered.py`

- Reads `altered_100k_parquet/` (particles + `altered_metadata`), filters **`split_channel == C`** and `alteration_succeeded == 1` (unless `--all-channels`).
- Uses **`jet_hadron_observables_split_pi_pm.build_pythia_reinjector`** and **`process_event_dataframe`**: **reinject parton event → PYTHIA hadronization** → compute **leading charged π** in target hemisphere, **φ\_{hJ}**, etc., one row per processed event.
- If **`--editable-parent`** points to the parent directory that contains **`editable_source_v1/`**, it appends **unchanged** rows for the **same `event_id`** set (for π⁺/π⁻ comparison).

**The specific file** `jet_hadron_pi_pm_observables_us_sbar_1M.csv` is **referenced** by the merge manifest but the **exact command line** that produced a full **~1M-row** table is **not** stored in the repository (likely batching, filters, or concat of several runs).

**Core physics / code path** for the observables is still:  
`jet_hadron_observables_split_pi_pm.py` (reinject + Breit/LT handling shared with the split-CSV workflow).

---

## 6. Step D — Subsample / merge (manifest only)

**File:** `jet_hadron_pi_pm_observables_signal54217_plus_unchanged487950_manifest.json`

- **Source:** `…/jet_hadron_pi_pm_observables_us_sbar_1M.csv`
- **Output:** `…/jet_hadron_pi_pm_observables_signal54217_plus_unchanged487950.csv`
- **Counts:** `n_signal_sampled: 54217`, `n_unchanged_all: 487950`, `n_rows: 542167`, `random_seed: 42`

**Implementation:** not in repo; logically **random subsample of altered rows** + **all** chosen unchanged rows (or similar).

---

## 7. Step E — Plots (φ\_{hJ}, R\_{π⁻/π⁺}, etc.)

**From π± observables CSV (columns like `phi_hJ`, `sample`, `pion`, `ok`):**  
- `scripts/analysis/plot_jet_hadron_pi_pm_observables.py`  
- `scripts/analysis/jet_hadron_pi_pm_figures.py` — `write_combined_pi_pm_comparison_pdfs`, **`obs_key="phi_hJ"`**, normalized step histograms + **R\_{π⁻/π⁺}** ratio panel.

**Alternative path from Parquet jet–hadron tables** (columns like `obs_angle_rad`):  
- `scripts/analysis/plot_jet_hadron_transverse_from_parquet.py`

The saved PNG under `figures_pi_pm_signal54217/` may come from either workflow or a **composite** of two PDFs; **not provenanced** in git.

---

## 8. One-page backward table (prompt-friendly)

| Closest artifact | Step | In-repo script(s) |
|------------------|------|---------------------|
| PNG / PDF figures | Plotting | `plot_jet_hadron_pi_pm_observables.py`, `jet_hadron_pi_pm_figures.py` |
| Merged CSV + manifest | Subsample/merge | **Not in repo** (manifest only) |
| `jet_hadron_pi_pm_observables_us_sbar_1M.csv` | Reinject + observables | `jet_hadron_observables_altered_parquet_filtered.py` **pattern** + `jet_hadron_observables_split_pi_pm.py` |
| `altered_100k_parquet/` | Diquark kick / channel C | `transform_editable_all_altered_parquet.py`, `split_dis_sample_diquark_kick.py` |
| `editable_source_v1/` | PYTHIA DIS, parton store, struck-u | `generate_dis_editable_source_parquet.py` |

---

## 9. Known gaps (do not assume without user confirmation)

1. Exact **shell history** for creating `jet_hadron_pi_pm_observables_us_sbar_1M.csv` and the **`side_by_side` PNG**.
2. The **Python snippet** that wrote `jet_hadron_pi_pm_observables_signal54217_plus_unchanged487950_manifest.json` / merged CSV.
3. Whether a given figure used **`phi_hJ`** from the π± CSV path vs **`obs_angle_rad`** from the final-state Parquet path.

---

## 10. Naming cheat sheet

| Name | Meaning |
|------|--------|
| **Struck-u pool** | Initial PYTHIA acceptance uses **incoming quark = u** (PDG 2). |
| **Channel C / us / s̄** | **Alteration** topology targeting **(ud) → … (us) … + s̄** flavor/color configuration — **not** the beam energy `5×41` / `9×275` ETA grids elsewhere in the project. |
| **18×275** | Default **electron–proton beam energies** in `generate_dis_editable_source_parquet.py`’s `build_pythia_source`, matching typical **editable-source** DIS generation for this pipeline. |

---

*Generated as a consolidated handoff document for LLM chat context; align any CLI with local `run_summary.json` and manifests under `analysis/dis_us_sbar_1M/`.*
