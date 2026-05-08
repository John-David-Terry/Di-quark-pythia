# Breit-frame hemisphere analysis — summary

## High-level summary

### What was added

The remnant-branch tracing script `scripts/generation/test_hadron_progenitor_tracing.py` was extended with **Breit-frame hemisphere analysis** so that we can test whether hadrons classified as remnant-branch-reachable also lie in the **Breit target hemisphere (pz_breit > 0)**, which is the expected region for remnant fragmentation.

- **Breit transform:** A helper `build_breit_transform(e_in, e_sc, p_in)` was added. It replicates the analysis codebase’s `build_LT` logic: from incoming/scattered electron and proton 4-vectors it builds the photon momentum q = l − l′, computes Q², x, y, qT, phiq, S, and returns the 4×4 Lorentz matrix that takes LAB 4-vectors to the Breit frame (q along −z, q⁰ ≈ 0). Internal checks (optional debug) verify that the transformed q has q_T ≈ 0 and q⁰ ≈ 0 and that the Minkowski norm is preserved.

- **Hadron kinematics in Breit:** For every hadron selected for tracing, the script now transforms its 4-momentum to the Breit frame and stores `pz_lab`, `pz_breit`, `pT_breit`, and the booleans `is_forward_lab = (pz_lab > 0)` and `is_target_breit = (pz_breit > 0)`.

- **Hemisphere selections:** Two optional filters are available via CLI:
  - **LAB forward:** `--forward-lab-pz-only` → require `pz_lab > 0`.
  - **Breit target:** `--breit-target-only` → require `pz_breit > 0`.
  If both are set, both conditions must hold. If neither is set, all selected hadrons are traced.

- **Cross-tabulations and correlation stats:** The script now reports:
  - Counts of remnant_branch_reachable hadrons with pz_breit > 0 vs pz_breit < 0.
  - Counts of struck_branch_reachable hadrons with pz_breit > 0 vs pz_breit < 0.
  - Same for both_branches_reachable.
  - The same breakdown for π⁻ only.
  - For forward π⁻ only (pz_lab > 0): how many also have pz_breit > 0, and the fraction P(pz_breit > 0 | pz_lab > 0) as a measure of how well the LAB forward region tracks the Breit target hemisphere.

- **Validation and diagnostics:** Printed checks include: (1) Breit transform sanity (number of events where LT failed); (2) hemisphere consistency (fraction of remnant-branch hadrons in target, fraction of struck-branch hadrons in current); (3) LAB vs Breit correlation for π⁻. Optional summary statistics (mean, std, min, max) of the pz_breit distribution are reported separately for remnant_branch_reachable, struck_branch_reachable, and both_branches_reachable.

- **JSONL schema:** Each hadron record in the JSONL output now includes the fields: `pz_lab`, `pz_breit`, `pT_breit`, `is_forward_lab`, `is_target_breit`.

### Hemisphere classification and remnant definition

- **LAB forward:** `pz_lab > 0` (unchanged).
- **Breit target:** `pz_breit > 0` (target/remnant side in the Breit frame; current/fragmentation side is pz_breit < 0).

The remnant-branch definition itself is unchanged: remnant seeds are still “proton daughters not on the struck branch,” and the remnant-branch node set is built by walking mothers from those seeds. The new logic only adds kinematic hemisphere labels and cross-tabulations; it does not change how branches are identified.

### Interpretation

- **Remnant-branch hadrons in target hemisphere:** If the proof-of-principle is physically sensible, a large fraction of hadrons whose ancestry reaches the remnant branch should have `pz_breit > 0`. The script reports this fraction; if it is high, that supports the interpretation that remnant-branch reachability corresponds to target-side (remnant) fragmentation.

- **Struck-branch hadrons in current hemisphere:** Similarly, hadrons that reach only the struck branch should predominantly have `pz_breit < 0` (current hemisphere). The script reports that fraction.

- **LAB forward vs Breit target:** The conditional probability P(pz_lab > 0 | pz_breit > 0) for π⁻ (and the forward-π⁻ subsample) indicates how well the simple “LAB forward” cut approximates the Breit target hemisphere. High correlation is expected but not guaranteed, depending on kinematics and acceptance.

### Caveats

- The Breit transform is built from the same event kinematics (e_in, e_sc, p_in) used in the analysis codebase; if `build_LT` is unavailable (import failure), no Breit quantities are computed and hemisphere stats are skipped for those runs.
- No event kinematics are modified; this is a diagnostic only. No momentum reinjection or production optimizations were added.

---

## Low-level summary for ChatGPT

### Files changed

- **Modified:** `scripts/generation/test_hadron_progenitor_tracing.py`
  - Added sys path and optional import of `build_LT` from `diquark.analyze_events_raw`.
  - Added `build_breit_transform(e_in, e_sc, p_in, debug=False)`.
  - Extended `HadronTraceResult` with optional fields: `pz_lab`, `pz_breit`, `pT_breit`, `is_forward_lab`, `is_target_breit`.
  - Extended `run_tracing_for_label` with parameters `forward_lab_pz_only` and `breit_target_only`; build LT per event; for each hadron compute LAB/Breit kinematics, apply hemisphere filters, attach hemisphere fields to the trace result, and accumulate cross-tabulation and pz_breit list for histogram summary.
  - Extended JSONL hadron record with `pz_lab`, `pz_breit`, `pT_breit`, `is_forward_lab`, `is_target_breit`.
  - Added CLI flags `--forward-lab-pz-only` and `--breit-target-only`.
  - Added printed block “Breit-frame and hemisphere analysis” with Validation 1–3, cross-tabulations A/B/C, hemisphere correlation statistics, and pz_breit distribution summary.

### New/updated functions

- **`build_breit_transform(e_in, e_sc, p_in, debug=False)`**  
  - Inputs: `e_in`, `e_sc`, `p_in` as length-4 arrays (E, px, py, pz) in the LAB frame.
  - Computes q = e_in − e_sc, Q² = −q², P·q, x = Q²/(2 P·q), S = 4 Ee Ep, y = Q²/(S x), qT = |q_T|, phiq = atan2(qy, qx).
  - Calls `build_LT(Ee, Ep, q, x, y, qT, phiq, S)` from `diquark.analyze_events_raw` and returns the 4×4 numpy array LT, or None if any step fails.
  - If `debug=True`, checks transformed q: warns if |q⁰| or |q_T| is large compared to Q, or if the Minkowski norm of q is not preserved.

- **`run_tracing_for_label(..., forward_lab_pz_only=False, breit_target_only=False)`**  
  - For each accepted event, builds `LT = build_breit_transform(e_in_4, e_sc_4, p_in_4)` from the event’s incoming/scattered electron and proton.
  - For each candidate hadron index: gets 4-momentum in LAB, computes `pz_lab`, and if LT is not None computes `pz_breit`, `pT_breit`, `is_forward_lab`, `is_target_breit`. Skips the hadron if `forward_lab_pz_only` and not `is_forward_lab`, or if `breit_target_only` and (LT is None or not `is_target_breit`).
  - After `trace_ancestry`, sets on the result the fields `pz_lab`, `pz_breit`, `pT_breit`, `is_forward_lab`, `is_target_breit`.
  - Accumulates: `n_remnant_target`, `n_remnant_current`, `n_struck_target`, `n_struck_current`, `n_both_target`, `n_both_current`; for π⁻: `n_remnant_pi`, `n_remnant_pi_target`, `n_struck_pi`, `n_struck_pi_current`; for forward π⁻: `n_forward_pi`, `n_forward_pi_target_breit`; and lists `pz_breit_remnant`, `pz_breit_struck`, `pz_breit_both` for histogram summary.
  - Writes hadron records to JSONL including the new hemisphere fields.

### Breit transform location

- The actual Lorentz matrix construction is in **`src/diquark/analyze_events_raw.py`** (function `build_LT`). The generation script imports it and uses it inside `build_breit_transform` after computing kinematics from (e_in, e_sc, p_in).

### JSONL schema changes

- Each element of `hadron_traces` now has:
  - `pz_lab`: float or null
  - `pz_breit`: float or null
  - `pT_breit`: float or null
  - `is_forward_lab`: bool
  - `is_target_breit`: bool  
  All previous hadron fields (hadron, trace_chain, termination_label, reached_struck_branch, reached_remnant_branch, struck_hits, remnant_hits, branch_classification, ambiguity_notes) are unchanged.

### Hemisphere-selection logic

- **No flags:** All hadrons from `get_selected_hadrons` are traced.
- **`--forward-lab-pz-only`:** Keep only hadrons with `pz_lab > 0`.
- **`--breit-target-only`:** Keep only hadrons with `pz_breit > 0` (requires a valid LT for the event).
- **Both:** Keep only hadrons that satisfy both conditions.

### Validation tests (printed)

1. **Validation 1 — Breit transform sanity:** Number (and fraction) of events for which `build_breit_transform` returned None. Warnings if the fraction is non-negligible.
2. **Validation 2 — Hemisphere consistency:**  
   - Fraction of remnant_branch_reachable hadrons with pz_breit > 0.  
   - Fraction of struck_branch_reachable hadrons with pz_breit < 0.  
   Expected: most remnant in target, most struck in current.
3. **Cross-tabulation A:** Counts (all hadrons) for remnant/struck/both × pz_breit>0 / pz_breit<0.
4. **Cross-tabulation B:** For π⁻ only: remnant in target, struck in current.
5. **Cross-tabulation C:** Forward π⁻ (pz_lab > 0): how many have pz_breit > 0 and the resulting P(pz_breit>0 | pz_lab>0).
6. **Hemisphere correlation:** fraction_remnant_in_target, fraction_struck_in_current.
7. **Validation 3 — LAB vs Breit:** P(pz_lab > 0 | pz_breit > 0) for π⁻ (and forward π⁻) as stated in the output.
8. **pz_breit distribution:** For each of remnant_branch_reachable, struck_branch_reachable, both_branches_reachable: n, mean, std, min, max of pz_breit (if any).

### Notable anomalies to watch

- **Remnant hadrons in current hemisphere (pz_breit < 0):** Some remnant-branch-reachable hadrons can still have pz_breit < 0 (e.g. backward-going in Breit). The fraction is reported; if large, it may indicate acceptance effects or that the remnant-branch definition is picking up more than “target fragmentation” in the strict sense.
- **Struck hadrons in target hemisphere (pz_breit > 0):** Similarly, a non-zero count of struck-branch-reachable hadrons with pz_breit > 0 can appear; the script reports it.
- **Breit transform failures:** If many events have LT = None (e.g. missing or incompatible `build_LT`), hemisphere stats will be incomplete; the script reports how many events failed.
