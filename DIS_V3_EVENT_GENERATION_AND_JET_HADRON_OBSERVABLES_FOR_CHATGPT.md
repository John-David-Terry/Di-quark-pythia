# DIS background (final_state_v3) and jet–hadron transverse observables — handoff for ChatGPT

This document describes **how events are generated** (PYTHIA v3 final-state Parquet) and **how the three jet–hadron transverse observables are constructed** (background path with `events_k_out`). It is meant to be pasted or attached for another model that needs implementation-level detail.

**Primary code locations**

| Stage | Script |
|--------|--------|
| Event generation (v1 / v3) | `scripts/analysis/generate_dis_background_final_state_parquet.py` |
| DIS / Breit helpers | `scripts/analysis/generate_dis_isr_parton_dataset.py` |
| Lab `flip_z` convention | `src/diquark/analyze_events_raw.py` (`FLIP_Z_PTREL`, `flip_z`) |
| Jet–hadron row production | `scripts/analysis/produce_dis_final_state_jet_hadron_transverse.py` |
| Observable formulas (shared) | `scripts/analysis/analyze_jet_hadron_transverse_observables.py` |
| Thin wrapper used by producer | `scripts/analysis/unchanged_direct_jet_hadron_core.py` |
| Parquet column contract | `scripts/analysis/unchanged_direct_schema.py` |
| Leading pion rule | `scripts/analysis/split_kinematics_extract.py` (`leading_target_pion_breit`) |
| Plots from Parquet | `scripts/analysis/plot_jet_hadron_transverse_from_parquet.py` |

**Optional follow-ons (subset / extra kinematics)**

| Task | Script |
|------|--------|
| Replay v3 acceptance, `z_LC`, triple-cut counts / event IDs | `scripts/analysis/compute_zlc_replay_v3_background.py` |
| Filter `rows.parquet` by `event_id` list | `scripts/analysis/filter_jet_hadron_rows_by_event_ids.py` |

---

## Part A — PYTHIA setup (`build_pythia_background`)

Implemented in `generate_dis_background_final_state_parquet.py` as `build_pythia_background(seed, hadron_level=True)` for production.

**Beams and process**

- `Beams:idA = 11` (electron), `Beams:idB = 2212` (proton)
- `Beams:eA = 18.0`, `Beams:eB = 275.0` (GeV)
- `Beams:frameType = 2`
- DIS NC-style photon exchange: `WeakBosonExchange:ff2ff(t:gmZ) = on`
- `PhaseSpace:Q2Min = 16.0` (GeV²)
- `ProcessLevel:all = on`
- `PDF:lepton = off`
- `PartonLevel:ISR = on`, `PartonLevel:FSR = on`, `PartonLevel:MPI = off`, `PartonLevel:Remnants = on`
- **Hadronization:** `HadronLevel:all = on` for v1 and v3 production
- `Random:seed`, `Random:setSeed = on`, `Print:quiet = on`
- `pythia.init()` must succeed

**Units:** PYTHIA native momenta and energies are **GeV**.

---

## Part B — v3 event generation (recommended for `k_out`)

**Goal:** One **fully hadronized** event per accepted sample, with **true hard-subprocess outgoing quark** lab four-momentum read from `pythia.process`, transformed to the **same Breit frame** as the hadronic final state, and written to Parquet. **Do not** use a two-step `HadronLevel:off` → read → `forceHadronLevel` → second `next()` pattern (old “v2” idea); it does not preserve normal hadronization for the same physical event.

### B.1 Event loop (v3)

For each target accepted event:

1. Call **`pythia.next()` once** with `HadronLevel:all = on`.
2. **Incoming struck quark filter (generator acceptance):**
   - `pick_incoming_quark_index(ev)` — among **incoming** (`status < 0`) **quarks** (`|id| ≤ 6`), pick the line with **largest** `|pz|` in the **lab** event record.
   - Require **`|id| == 2`** (struck **u** quark) on that line.
3. **Hard subprocess (same `next()` call, post-hadronization):**
   - Read `proc = pythia.process`.
   - **Outgoing struck quark lab four-vector** via `hard_subprocess_outgoing_quark_lab_p4_and_index(proc)`:
     - Scan `proc[i]` for **`status == 23`** and **`1 ≤ |id| ≤ 6`**.
     - If several, keep the **highest energy** quark.
     - Return `p4_lab` as **`numpy.array([E, px, py, pz], float64)`** and process index `oq`.
   - Also record `hard_subprocess_incoming_quark_process_index(proc)` → `iq` (indices refer to **`pythia.process`**, not `pythia.event`).
   - If no such outgoing quark: count `hard_subprocess_miss` and **reject** this acceptance attempt.
4. **Breit transform `LT` (4×4):** `LT = try_build_lt_from_event(ev)` from `generate_dis_isr_parton_dataset.py`.
   - Uses **beams** extracted from the **hadronic** `event` (incoming e⁻, scattered e⁻, incoming proton), applies the same **`flip_z(..., FLIP_Z_PTREL)`** convention as the rest of the DIS analysis, builds DIS kinematics, then **`build_LT`**.
   - If `LT is None`: Breit rejection, skip event.
5. **`k_out` in Breit (jet proxy for observables):**
   - `p4_b = LT @ flip_z(p4_lab, FLIP_Z_PTREL)` where `p4_lab` is the **status-23 outgoing quark** from step 3.
   - Store `k_out_breit_E, px, py, pz` from `p4_b` (component `0` = energy in the convention used consistently downstream).
6. **Event metadata:** `Q2`, `xB` from `pythia.infoPython()` (`Q2Fac()`, `x2()`), optional event weight best-effort.
7. **Final-state hadrons only (Breit):** `collect_final_state_breit(ev, LT, event_id)`:
   - Loop `pythia.event`, `p.isFinal()`, skip non-positive energy, etc.
   - For each final particle: `p4_lab = [E,px,py,pz]`, then `p4_lab = flip_z(p4_lab, FLIP_Z_PTREL)`, then **`p4_b = LT @ p4_lab`**.
   - Store per-particle rows: `event_id`, dense `particle_index`, `pdg_id`, **`px, py, pz, E` in Breit** (names `px`…`E` in Parquet; all Breit).
8. **`event_id`:** equals the **0-based sequential accepted index** at write time (first accepted event → `0`, …).
9. **Sharding:** default `EVENTS_PER_SHARD = 10_000`; manifest lists `particles/` and `events/` shards.

**Outputs (v3)**

- Parent directory default: `~/Data/dis_isr_background_final_state_v3/`
- Dataset folder: `final_state_v3/`
  - `particles/shard_*.parquet`
  - `events/shard_*.parquet` — includes `n_final`, `Q2`, `xB`, `weight`, **`k_out_breit_*`**, `struck_incoming_index`, `struck_outgoing_index`
  - `manifest.parquet`, `run_summary.json` (includes `hard_subprocess_miss` for v3)

**CLI (typical production)**

```bash
python scripts/analysis/generate_dis_background_final_state_parquet.py \
  --final-state-variant v3 \
  --n-accepted 900000 \
  --seed 12345
```

**Validation:** `generate_dis_background_final_state_parquet.py --validate-only <parent_dir>` auto-picks `final_state_v3/` vs `final_state_v1/` when only one exists under the parent.

---

## Part C — Jet–hadron row construction (`produce_dis_final_state_jet_hadron_transverse.py`)

### C.1 Inputs

- **Background mode:** read paired **events** and **particles** shards from `manifest.parquet`.
- **Jet mode for v3:** `--background-jet events_k_out` with `--background-final-state-subdir final_state_v3` (and `--background-root` pointing at the parent of `final_state_v3/`).
- Events shards must contain **`k_out_breit_px,py,pz,E`** (checked via `events_table_has_k_out_breit`).

### C.2 Per-event processing (`_process_background_shard_events_k_out`)

For each `event_id` row in the events shard (joined to particles by `event_id`):

1. **Hadron list:** all final particles with `is_hadron(pdg_id)` from `diquark.analyze_events_raw`.
2. **Four-momentum for each hadron in Breit:** `[E, px, py, pz]` using the Parquet columns (already Breit).
3. **Leading target charged pion:** `leading_target_pion_breit(hadrons)` (`split_kinematics_extract.py`):
   - Only `|pdg| == 211` (π⁺ or π⁻).
   - Require **`pz > 0`** in that Breit frame (target / “current” hemisphere convention used across the project).
   - Among survivors, choose the hadron with **largest Breit energy** `E`.
   - If none: row `ok=False`, `failure_reason=no_pion_candidate`.
4. **DIS scalars:** `Q2`, `xB` from events row; `Q = sqrt(Q2)` when valid.
5. **Optional x/Q window:** unless `--skip-xQ-window`, apply the same `xmin_ptrel`, `xmax_ptrel`, `Qmin_ptrel`, `Qmax_ptrel` as in `jet_hadron_observables_split_pi_pm` / producer imports.
6. **Jet four-vector:** read `k_out_breit_*` from the **events** row (not recomputed in this mode).
7. **Observables:** `transverse_three_from_k_out_and_pion_breit(k_out_breit, pion_breit)` → see Part D.

**Output**

- `jet_hadron_transverse_v1/rows.parquet` under `--out-dir`
- Columns per `UNCHANGED_DIRECT_JET_HADRON_V1_COLUMNS` in `unchanged_direct_schema.py`
- `source_lineage` e.g. `background_final_state_v3_events_table_k_out_breit`

---

## Part D — The three observables (exact definitions)

All use **Breit-frame** four-vectors **`[E, px, py, pz]`** for jet and pion. Transverse 2-vectors use **only** `(px, py)`.

### D.1 Transverse vectors

```text
pT_jet = (k_out_breit_px, k_out_breit_py)
pT_h   = (pion_breit_px, pion_breit_py)
```

(Code: `_pT_vec_breit` in `analyze_jet_hadron_transverse_observables.py`.)

### D.2 `obs_angle_rad` — opening angle in **[0, π]**, **not** “Δφ” in the azimuthal sense

\[
\theta = \arccos\!\left(\mathrm{clip}\!\left(\frac{\mathbf{p}_{T,\mathrm{jet}}\cdot \mathbf{p}_{T,h}}{|\mathbf{p}_{T,\mathrm{jet}}|\,|\mathbf{p}_{T,h}|},\,[-1,1]\right)\right)
\]

- **θ = 0:** transverse vectors are **parallel** (aligned).
- **θ = π:** transverse vectors are **antiparallel** (back-to-back **in the 2D transverse plane**).

If either \(|\mathbf{p}_T| < \texttt{PT\_MIN\_ANGLE}\) with **`PT_MIN_ANGLE = 1e-6`**, the angle is **`NaN`** (`None` from helper → stored as `nan`).

**Important naming caveat:** `plot_jet_hadron_transverse_from_parquet.py` labels the x-axis **`$\phi_{hJ}$`**, but the plotted column is **`obs_angle_rad`** = **θ** above. It is **not** \(\phi_h - \phi_J\) computed as a wrapped azimuthal difference unless you add a separate column.

### D.3 `obs_sum_mag_GeV`

\[
|\mathbf{p}_{T,\mathrm{jet}} + \mathbf{p}_{T,h}|
\]

(Vector sum first, then Euclidean norm in 2D.)

### D.4 `obs_diff_mag_GeV`

\[
|\mathbf{p}_{T,\mathrm{jet}} - \mathbf{p}_{T,h}|
\]

(Vector difference first, then norm.)

**Implementation reference**

```python
# unchanged_direct_jet_hadron_core.transverse_three_from_k_out_and_pion_breit
# calls analyze_jet_hadron_transverse_observables._pT_vec_breit and _angle_between_2d
```

---

## Part E — Plotting

`plot_jet_hadron_transverse_from_parquet.py`:

- Reads `rows.parquet`, keeps **`ok == True`**, optionally **`--arm background`**.
- For each observable column, splits **π⁺** vs **π⁻** by `pion_pdg`, histograms with shared binning, bottom panel **π⁻/π⁺** ratio with Poisson-derived uncertainty.

---

## Part F — Optional: `z_LC` and triple-cut subsets (replay, not in base Parquet)

**`z_LC` (proton-aligned “+” component ratio)** is **not** stored in `final_state_v3` or default `rows.parquet`. It is computed by **replaying** the same PYTHIA acceptance with `scripts/analysis/compute_zlc_replay_v3_background.py`:

- Same `build_pythia_background` + acceptance as v3.
- Incoming proton **lab** three-vector **n̂** from `extract_beams_from_event`.
- \(z_{\mathrm{LC}} = (E_h + \mathbf{p}_h\!\cdot\!\hat{\mathbf{n}}) / (E_p + |\mathbf{p}_p|)\) with **hadron** = same Breit-leading target pion as jet–hadron (re-identified in the replay).
- **`--triple-cut z pht pjt`** counts events with **`z_LC > z`**, **`P_{hT} > pht`**, **`P_{jT} > pjt`** where **`P_{hT}`**, **`P_{jT}`** are **Breit** \(\sqrt{p_x^2+p_y^2}\) for the selected pion and for **`k_out`** built as in generation.

**`--write-triple-cut-event-ids`** saves `event_id` (accepted index) for each passing event. **`filter_jet_hadron_rows_by_event_ids.py`** subsets an existing `rows.parquet` to those IDs without recomputing observables.

---

## Part G — Quick sanity expectations

- **v3** should give **~20–25** final charged+neutral particles per accepted event on average (hadronization on), not ~6 (broken two-`next()` path).
- **`hard_subprocess_miss`** should be **0** (or extremely rare) if `pythia.process` always exposes the DIS outgoing status-23 quark after a normal `next()`.

---

*End of handoff document.*
