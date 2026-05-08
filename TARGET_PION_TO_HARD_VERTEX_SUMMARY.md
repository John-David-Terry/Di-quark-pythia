# Target-pion-to-hard-vertex tracing — summary

## High-level summary

### Script created

**`scripts/generation/trace_target_pion_to_hard_vertex.py`**

A dedicated proof-of-principle that:

1. Builds the Breit frame from incoming/scattered electron and proton (same kinematics and `build_LT` as the analysis codebase).
2. Selects a **single tagged hadron**: the **hardest π⁻ in the Breit target hemisphere** (PID = -211, final state, pz_breit > 0, maximum E_breit).
3. Traces that pion’s ancestry **backward** through the PYTHIA mother graph.
4. **Stops** when the ancestry reaches the **hard-interaction node set** (struck-quark candidates plus their immediate mothers and daughters).
5. Writes one JSONL record per accepted event that has a tagged pion, and prints validation stats and debug examples.

No remnant definition from proton daughters; no event modification or reinjection.

### Tagged-pion selection

- **Definition:** Among final-state particles with PID = -211 and pz_breit > 0 (after transforming with the event’s Breit matrix), choose the one with **largest energy in the Breit frame**.
- If there is no π⁻ with pz_breit > 0, the event is skipped (no record in JSONL).
- The script reports: accepted DIS events, and how many had a tagged hardest target-region π⁻.

### Hard-interaction stopping rule

- **Hard-interaction node set** is built per event as:
  - **Struck-quark candidates:** from existing logic (status 23, then 63–68, then highest-energy non-zero-status quark).
  - **Mothers** of each struck candidate (mother1, mother2).
  - **Daughters** of each struck candidate (daughter1..daughter2 range).
- The backward walk from the tagged pion **stops** as soon as it visits any node in this set.
- **Stop reason** is recorded as: `reached_hard_interaction`, or `no_mother`, `cycle_detected`, `invalid_mother`, `max_depth`, `ambiguous_branch`, `invalid_index`, `unresolved`.
- Exactly which node(s) were hit is stored in `stop_node_indices`; the script also classifies stop nodes (e.g. struck_quark_candidate, mother_of_struck, daughter_of_struck, exchanged_boson_candidate, other_hard_node).

### Does the proof-of-principle work?

**Yes.** In a short test (ISRFSR_OFF, 80 accepted events, 24 with a tagged target π⁻):

- **22/24** tagged events had ancestry that **reached the hard-interaction node set**; the trace stopped at a node in that set.
- **2/24** hit **cycle_detected** (ancestry loop) before reaching the hard set.
- Among the 22 that reached the hard set, the stopping node was classified as **daughter_of_struck** in all 22 (the trace reached a daughter of the struck-quark candidate, which is part of the hard-interaction neighborhood).

So we can take the hardest π⁻ in the Breit target hemisphere and trace its ancestry back to the DIS hard-interaction layer in a well-defined, reproducible way. The stopping rule is explicit and the result is inspectable (JSONL + debug dump).

### Main caveats

- **Cycles:** A few events hit a cycle in the mother graph before reaching the hard set; those are reported as `cycle_detected` and do not yield a hard-vertex stop node.
- **Stopping at “daughter of struck”:** The hard set includes daughters of the struck quark, so the trace often stops at a parton in the first shower step (e.g. status -62) rather than at the struck quark (status 23) itself. That is intentional: the “hard interaction layer” is defined as the struck candidate plus its immediate mothers/daughters.
- **Single path vs full tree:** The tracer does a BFS backward; the stored `ancestry_trace` is the list of all nodes **visited** until the stop. It is not necessarily a single linear chain from the pion to the stop node; it can include siblings. For a single path, post-processing or a DFS variant would be needed.
- **No remnant-side definition:** This script does not define or use a remnant branch; it only traces from one tagged hadron to the hard-interaction set.

---

## Low-level summary for ChatGPT

### File created

- **`scripts/generation/trace_target_pion_to_hard_vertex.py`** (new, standalone).

Reused logic is inlined (no global remnant or branch sets from proton daughters). Uses `build_LT` from `diquark.analyze_events_raw` for the Breit transform.

### Helper functions

| Function | Purpose |
|----------|--------|
| `p4_from_particle(p)` | (E, px, py, pz) from a PYTHIA particle. |
| `minkowski_norm(e,px,py,pz)` | Minkowski squared norm. |
| `setup_pythia(label)` | Build PYTHIA with DIS config (E_E, E_P, Q2_MIN, WeakBosonExchange, etc.). |
| `find_incoming_beams(ev)` | Return (e_idx, p_idx) for incoming electron and proton. |
| `get_scattered_electron_idx(ev)` | Prefer status 44; else highest-E electron. |
| `compute_q2_and_x(ev, e_in_idx, e_sc_idx, p_in_idx)` | Q² and Bjorken x from q = l−l′, P·q. |
| `build_breit_transform(e_in, e_sc, p_in)` | 4×4 Lorentz matrix to Breit frame via `build_LT`. |
| `identify_struck_quark_candidates(ev)` | List of indices: status 23, else 63–68, else highest-E quark. |
| `mothers_of(p)` | List [mother1, mother2] (deduped, valid only). |
| **`find_hardest_target_pi_minus(ev, LT)`** | Among final π⁻ with pz_breit > 0, pick argmax E_breit. Returns (idx, p4_lab, p4_breit, E_breit, pz_breit) or None. |
| **`identify_hard_interaction_nodes(ev)`** | Hard set = struck candidates ∪ their mothers ∪ their daughters. Returns (hard_set, node_summaries_with_tags, struck_summaries). |
| **`trace_hadron_to_hard_vertex(ev, start_idx, hard_nodes, max_depth)`** | BFS backward from start_idx; stop on first visit to any node in hard_nodes, or on no_mother / cycle / invalid_mother / max_depth / ambiguous_branch. Returns (trace, stop_reason, stop_node_indices). |
| `run_label(...)` | Event loop: DIS selection, Breit build, tag hardest target π⁻, build hard set, trace, write JSONL, accumulate validation stats, print debug. |

### Tagged-pion definition

- **PID:** -211 (π⁻).
- **Final state:** `p.isFinal()`.
- **Breit target hemisphere:** pz_breit > 0 (from LT @ p4).
- **Hardest:** Among those, the one with **largest E_breit** (p4_breit[0]).
- At most **one** tagged pion per event.

### Hard-interaction node set

- **Struck candidates:** `identify_struck_quark_candidates(ev)` (status 23 → 63–68 → fallback).
- **Mothers:** For each struck index i, add `mother1(i)` and `mother2(i)` if ≥ 0.
- **Daughters:** For each struck index i, add all indices in `[daughter1(i), daughter2(i)]` when that range is valid.
- **Set:** `hard_nodes = struck ∪ mothers ∪ daughters`.
- Node summaries include tags: `struck_candidate`, `mother_of_struck`, `daughter_of_struck`, `exchanged_boson_candidate` (if PID 22), `hard_interaction_node`.

### Stopping rule

- During the backward BFS from the tagged pion:
  - **Stop and set reason `reached_hard_interaction`** when the current node index is in `hard_nodes`; record that index in `stop_node_indices`.
  - **Stop and set `no_mother`** when the current node has no valid mothers.
  - **Stop and set `cycle_detected`** when the current node was already in `visited`.
  - **Stop and set `invalid_index`** if the node index is out of range.
  - **Stop and set `invalid_mother`** if a mother index is out of range.
  - **Stop and set `max_depth`** if depth > max_depth.
  - **Stop and set `ambiguous_branch`** if the current node has more than two mothers.
- Only the **first** hit of the hard set is recorded (first time we pop a node that is in hard_nodes).

### JSONL schema (one record per event with a tagged pion)

- `label`, `event_number`, `global_event_index`, `q2`, `x_bj`
- `tagged_pion`: `{ idx, pid, status, p4_lab, p4_breit, energy_breit, pz_breit }`
- `hard_interaction_nodes`: list of `{ idx, pid, status, mother1, mother2, daughter1, daughter2, e, px, py, pz, tags }`
- `struck_candidates`: list of `{ idx, pid, status, e, px, py, pz }`
- `ancestry_trace`: list of `{ idx, pid, status, mothers }` (all nodes visited in BFS order until stop)
- `stop_reason`, `stop_node_indices`, `ambiguity_notes`

### Validation tests (printed)

1. **Validation 1 — Breit-frame selection:** Generated events, accepted DIS events, number of events with a tagged hardest target-region π⁻.
2. **Validation 2 — Hard-interaction stopping:** Number (and fraction) of tagged events where ancestry reached the hard-interaction set vs did not.
3. **Validation 3 — Trace integrity:** Counts of cycle_detected, invalid_mother/index, max_depth; and full stop_reason distribution.
4. **Validation 4 — Debug examples:** For a few events: event number, Q², x, tagged pion idx and Breit E/pz, hard-interaction neighborhood (idx, pid, status, tags), ancestry trace (first and last steps), stop_reason and stop_node_indices.
5. **Validation 5 — Stopping-node types:** Across the sample, counts of stop nodes classified as: struck_quark_candidate, mother_of_struck, daughter_of_struck, exchanged_boson_candidate, other_hard_node.

### Test run (ISRFSR_OFF, 80 accepted events)

- Tagged events: **24** (events with at least one π⁻ in Breit target).
- Reached hard interaction: **22**; cycle_detected: **2**.
- Fraction reached: **22/24 ≈ 0.917**.
- Stop-node type: **daughter_of_struck: 22** (all 22 that reached the hard set stopped at a daughter of the struck quark).

### Notable ambiguities

- **Cycle in ancestry:** In a few events the mother graph has a cycle (e.g. through the event record “system” or repeated indices); the walk stops with `cycle_detected` and does not reach the hard set.
- **BFS vs single path:** The stored trace is the set of nodes visited in BFS order, not necessarily one path; multiple branches are explored until one node in the hard set is popped.
- **Stopping at daughter of struck:** The definition of “hard interaction” includes the first shower step (daughters of the struck quark). So the trace often stops at a status -62 (or similar) parton rather than the status 23 struck quark. Both are part of the same hard-interaction neighborhood.
