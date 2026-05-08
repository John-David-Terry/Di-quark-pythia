## High-level summary

- **Script:** `scripts/generation/test_hadron_progenitor_tracing.py`
- **Purpose:**  
  Proof-of-principle ancestry tracing of final-state hadrons in PYTHIA 8 DIS events, asking whether a hadron’s ancestry reaches:
  - the **struck-side branch** (outgoing hard parton line), and/or
  - the **remnant-side branch** (beam remnant after the struck parton is removed).
- **Setup:**  
  Reuses the same DIS configuration as `generate_events_raw.py`:
  - Beams: \(E_e = 18\) GeV, \(E_p = 275\) GeV.
  - Process: `WeakBosonExchange:ff2ff(t:gmZ) = on`, `HardQCD:all = off`, `PDF:lepton = off`.
  - Cut: `PhaseSpace:Q2Min = 16 GeV^2`.
  - ISR/FSR and ColourReconnection settings per label: `ISRFSR_ON`, `ISRFSR_OFF`, `ETA_ON_CRON`.
- **What it does per event:**
  1. Finds incoming electron/proton and scattered electron; computes \(Q^2\) and Bjorken \(x\) and applies the same DIS-style cut.
  2. Identifies a **DIS neighborhood**:
     - **Struck-branch seeds:** quark candidates using the same status logic as `find_k_out` (status 23, then 63–68, then any non-zero-status quark).
     - Builds a **struck-branch node set** by walking upward along mothers from these seeds until the incoming proton index.
     - Uses the proton’s daughters (via `daughter1()/daughter2()`) as the local proton-side products at the interaction.
     - Any proton daughter *not* on the struck branch is treated as a **remnant-branch seed**; from these, it builds a **remnant-branch node set**.
  3. Selects final-state hadrons to trace (default: \(\pi^-\); options for charged pions or all hadrons).
  4. For each hadron:
     - Walks the ancestry graph backward through mothers (BFS), recording each step and protecting against cycles and runaway depth.
     - Records whether the ancestry hits:
       - any node in the struck-branch node set, and/or
       - any node in the remnant-branch node set.
     - Assigns a **top-level branch classification**:
       - `remnant_branch_reachable`
       - `struck_branch_reachable`
       - `both_branches_reachable`
       - `neither_branch_reachable`
       - `string_only` (terminated on string/junction)
       - `ambiguous` (e.g. branch ambiguity, cycles, max depth).
     - Keeps the low-level termination label (e.g. `reached_beam`, `reached_remnant`, `reached_string_or_junction`, `no_mother`, `max_depth`, `cycle_detected`) and a short ambiguity note.
- **Outputs:**
  - A JSONL file with one record per accepted event, containing:
    - DIS kinematics (\(Q^2, x\)), beam and scattered-electron snapshots.
    - Struck- and remnant-branch seed indices.
    - A primary struck-branch seed snapshot (if any).
    - For each hadron: full ancestry chain and branch-reachability classification.
  - A human-readable summary with:
    - Event-record sanity metrics (generated vs accepted vs events with struck/remnant branches).
    - Branch-seed consistency and local DIS-neighborhood dumps around the proton and branch seeds.
    - Hadron branch-reachability fractions for all selected hadrons and for \(\pi^-\) only.
    - Mother-link integrity checks (cycles, invalid mothers).
- **Status / caveats:**
  - The remnant side is **not** forced into a single “remnant quark”; it is treated as one or more remnant-seed indices plus their upstream branch node set.
  - Branch reachability is about whether the ancestry graph intersects these node sets, **not** about unique quark-level parentage.
  - Strings and junctions (ids 92/93/94) terminate traces as `string_only` or `ambiguous`, making explicit where the event record structure is too messy for clean interpretation.

## Low-level technical summary

- **File changed:**
  - `scripts/generation/test_hadron_progenitor_tracing.py`

- **Remnant- and struck-branch identification:**
  - `identify_struck_quark_candidates(ev) -> List[int]` (unchanged logic):
    - Step 1: quarks with \(|\text{id}| \in \{1,\dots,6\}\) and \(|\text{status}| = 23\).
    - Step 2: quarks with \(63 \le |\text{status}| \le 68\).
    - Step 3: any non-zero-status quark, with highest-energy first.
  - `identify_dis_neighborhood(ev, e_idx, p_idx, e_sc_idx)`:
    - `struck_seeds = identify_struck_quark_candidates(ev)`.
    - `struck_branch_nodes = build_branch_nodes(ev, struck_seeds, stop_at={p_idx})` walks upward along mothers until reaching the incoming proton index.
    - Proton daughters from `ev[p_idx].daughter1()/daughter2()` define the local proton-side neighborhood.
    - Any daughter not in `struck_branch_nodes` is a **remnant_seed**.
    - `remnant_branch_nodes = build_branch_nodes(ev, remnant_seeds, stop_at={p_idx})`.
  - Both struck and remnant branches are modeled as **sets of node indices**, not as single ancestors.

- **Ancestry tracing and branch classification:**
  - `trace_ancestry(ev, hadron_idx, struck_branch_nodes, remnant_branch_nodes, max_depth=100)`:
    - BFS over mothers starting from `hadron_idx`, with:
      - `visited` set to avoid cycles.
      - `trace_chain` capturing `(idx, id, status, mothers)` at each step.
      - `termination_label` describing why the walk stopped (`reached_beam`, `reached_remnant`, `reached_string_or_junction`, `no_mother`, `ambiguous_branch`, `cycle_detected`, `max_depth`, or `other`).
      - `struck_hits`: indices where the ancestry intersects `struck_branch_nodes`.
      - `remnant_hits`: indices where the ancestry intersects `remnant_branch_nodes`.
    - Uses `classify_branch_reachability(reached_struck, reached_remnant, termination_label)` to produce a high-level `branch_classification` and `ambiguity_notes`.
      - If both struck and remnant are reached, classification is `both_branches_reachable`.
      - If neither is reached, classification is `neither_branch_reachable`, `string_only`, or `ambiguous`, depending on the low-level termination.

- **JSONL schema (per event):**
  - `label`, `event_number`, `global_event_index`, `q2`, `x_bj`.
  - `has_struck_branch`, `has_remnant_branch`.
  - `struck_branch_seed_indices`, `remnant_branch_seed_indices`.
  - `struck_branch_primary`: snapshot of the first struck seed (if present).
  - `scattered_electron`, `incoming_electron`, `incoming_proton`: `ParticleSnapshot`.
  - `hadron_traces`: list of:
    - `hadron`: `ParticleSnapshot`.
    - `trace_chain`: list of `TraceStep` `{idx, id, status, mothers}`.
    - `termination_label`: low-level termination classification.
    - `reached_struck_branch`, `reached_remnant_branch`: booleans.
    - `struck_hits`, `remnant_hits`: index lists.
    - `branch_classification`: one of `remnant_branch_reachable`, `struck_branch_reachable`, `both_branches_reachable`, `neither_branch_reachable`, `string_only`, `ambiguous`.
    - `ambiguity_notes`: short text explanation.

- **Validation tests (printed):**
  - **Test 1 – DIS neighborhood sanity:**
    - Counts of generated vs accepted events and events with struck/remnant branches.
  - **Test 2 – Mother-chain consistency:**
    - Cycles detected, invalid mother references.
  - **Test 3 – Branch seeds and local neighborhood:**
    - For a few debug events: prints struck/remnant seeds and a local neighborhood (`idx:id(status)->mothers,daughters`) around proton and seeds.
  - **Test 4 – Hadron branch reachability:**
    - For all selected hadrons: counts/fractions by `branch_classification`.
  - **Test 5 – Forward-pion sanity:**
    - Same stats but restricted to \(\pi^-\); explicitly notes that there is currently **no forward cut** (no \(\eta\)/\(p_z\) selection yet).
  - **Test 6 – Mother-link integrity:**
    - Summary of cycles and invalid mothers.

- **Caveats:**
  - Remnant-side structure in PYTHIA (spectator quark/diquark/string endpoints) can be messy, so the script does **not** claim a unique remnant quark; it explicitly keeps all candidiate remnant seeds and full branch node sets.
  - Hitting a branch node set means that the ancestry graph touches that branch; it does **not** imply a unique quark-level ancestor.
  - String/junction-dominated ancestries are flagged as `string_only` or `ambiguous`, not forced into either branch. 

