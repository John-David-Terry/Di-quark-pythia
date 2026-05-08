# Jet–hadron transverse observables (final-state Parquet) — summary for review

## Goal

Compute three **Breit-frame** observables from a **jet** (struck outgoing quark proxy or true parton) and a **leading charged pion** in the **target hemisphere** (`pz_Breit > 0`), for the combined DIS sample:

- ~**900,000** background events: `dis_isr_background_final_state` → `final_state_v1` (final hadrons only in Breit).
- ~**48,877** altered reinject events: editable partons → PYTHIA reinject → `dis_isr_altered_reinject_100k/final_state_v1`.

**Total rows written:** **948,877** (one per event).

## The three observables (per event)

Using transverse 2-vectors \(\mathbf{p}_{T,\mathrm{jet}} = (p_x, p_y)\), \(\mathbf{p}_{T,h}\) in the same Breit frame:

1. **Azimuthal angle** \(\phi_{hJ} \in [0, \pi]\) between \(\mathbf{p}_{T,\mathrm{jet}}\) and \(\mathbf{p}_{T,h}\).
2. **Sum magnitude:** \(|\mathbf{p}_{T,\mathrm{jet}} + \mathbf{p}_{T,h}|\) (GeV).
3. **Difference magnitude:** \(|\mathbf{p}_{T,\mathrm{jet}} - \mathbf{p}_{T,h}|\) (GeV).

Same definitions as `analyze_jet_hadron_transverse_observables.py` / `unchanged_direct_jet_hadron_core.transverse_three_from_k_out_and_pion_breit`.

## Hadron selection

- Among **final hadrons** in the stored Breit frame, require `|pdg| == 211`.
- **Target hemisphere:** `pz > 0`.
- **Leading** by largest **energy** \(E\) among those candidates.
- Column `pion_pdg` is **+211 (π⁺)** or **−211 (π⁻)** for downstream π⁺/π⁻ plots.

## Jet definition (important distinction)

| Arm | Jet source | Notes |
|-----|------------|--------|
| **altered_reinject** | **Exact** outgoing struck-quark four-momentum from editable **parton** Parquet (already Breit), joined on `event_id` to reinjected final state. | Same MC-truth jet idea as the split CSV / full-event pipeline. |
| **background** | **No** final-state struck quark in the Parquet after hadronization. Jet = **LO collinear proxy:** \(q\) forced to \(q_T = 0\) in the flipped-lab frame, \(k \approx x_B P + q\), Breit transform via `build_LT` from \((Q^2, x_B, y=Q^2/(Sx))\) with beams sampled once from PYTHIA (`--pythia-seed`, default **12345**). | Documented in `source_lineage` as `background_final_state_v1_lo_collinear_qt0_jet_proxy`. **Not** guaranteed to match the true per-event Breit map used when hadrons were written; use for bulk shape only, not for strict “truth jet” comparisons. |

## On-disk artifacts

**Producer script:** `scripts/analysis/produce_dis_final_state_jet_hadron_transverse.py`

- **Parquet:** `~/Data/dis_jet_hadron_from_final_state_v1/jet_hadron_transverse_v1/rows.parquet` (or `--out-dir` variant).
- **Schema:** `scripts/analysis/unchanged_direct_schema.py` → `UNCHANGED_DIRECT_JET_HADRON_V1_COLUMNS` (`event_id`, `arm`, `source_lineage`, `ok`, `failure_reason`, DIS vars, `k_out_breit_*`, `pion_pdg`, `pion_breit_*`, three `obs_*`, `n_final_hadrons_used`).

**Full run options used (representative):** `--mode both --skip-xQ-window --pythia-seed 12345`  
(`--skip-xQ-window` skips the narrow pTrel \(Q,x\) cut so most failures are only “no pion”.)

**Outcome counts (that run):**

- **911,883** rows with `ok == True` (observables filled).
- **36,994** rows with `ok == False`, **all** `failure_reason == no_pion_candidate`.

## Plots

**Script:** `scripts/analysis/plot_jet_hadron_transverse_from_parquet.py`

- Reads Parquet, keeps **`ok == True`**. **Default** pools all arms; use **`--arm background`** for the DIS background subsample only (π⁻/π⁺ ratio plots with filenames containing `..._background_pi_ratio_...`).
- **Output:** three PDFs per run. Filename tag is `..._combined_pi_ratio_...` (default, all arms) or `..._background_pi_ratio_...` / `..._altered_reinject_pi_ratio_...`. Each figure has:
  - **Top:** π⁺ vs π⁻ **normalized** \((1/N)\,\mathrm{d}N/\mathrm{d}x\) step histograms (same bins as `analyze_jet_hadron_transverse_observables.py`).
  - **Bottom:** **π⁻/π⁺** per bin, \(N_{\pi^-}(b)/N_{\pi^+}(b)\), with Poisson-style error bars; dashed line at 1. Global \(N_{\pi^-}/N_{\pi^+}\) is annotated on the top panel.
- Optional **`--by-arm`:** extra legacy PDFs that compare **background** vs **altered_reinject** in π⁺/π⁻ two-panel layout (for debugging only).

**Example combined counts:** \(N_{\pi^+}+N_{\pi^-}=911{,}883\) ok rows; global \(N_{\pi^-}/N_{\pi^+}\approx 0.94\) (exact values depend on the Parquet).

## What to tell ChatGPT to watch for

1. **Background jet is a proxy** — interpret background–altered differences cautiously; altered arm has the consistent parton-level jet.
2. **π⁺ vs π⁻** — same leading-\(E\) rule in target hemisphere; charge comes from the selected hadron’s PDG.
3. **Weights** — Parquet v1 row does not include MC weights; plots are unweighted normalized densities.
4. For **strict** phase-space matching to older pTrel studies, rerun the producer **without** `--skip-xQ-window` and/or filter on `Q`, `xB` offline.
