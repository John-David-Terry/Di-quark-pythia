## Next-step remnant-branch validation / refinement plan

This file captures the design for the next diagnostic step on the remnant/struck branch tracing study. It focuses on auditing whether the current remnant-branch definition is physically sensible in the PYTHIA DIS event record, and on testing the stability of hadron branch classifications under variations in the remnant definition.

See `REMNANT_BRANCH_TRACING_OVERVIEW_FOR_CHATGPT.md` for the current implementation summary of `scripts/generation/test_hadron_progenitor_tracing.py`.

### Core idea

The existing tracing script defines:

- **Struck branch:** via status-based struck-quark candidates (status 23, then 63–68, then any non-zero-status quark) and uses their ancestry to form a struck-branch node set.
- **Remnant branch (Definition A):** proton daughters that are not on the struck branch, then ancestors of these daughters up to the proton.

This plan adds:

1. A dedicated DIS-neighborhood inspection mode around the proton and its daughters.
2. Remnant-seed type classification.
3. Alternative remnant definitions (A vs B, optionally C).
4. A classification-stability study of hadron branch reachability under these definitions.
5. Optional forward-\(\pi^-\) selection.

All of this remains diagnostic; there is no event modification or re-injection.

### New diagnostic script

- **Script:** `scripts/generation/inspect_dis_remnant_structure.py`
- **Imports / reuse:**
  - From `test_hadron_progenitor_tracing.py`:
    - `setup_pythia`, `LABEL_CONFIGS`
    - `find_incoming_beams`, `get_scattered_electron_idx`, `compute_q2_and_x`
    - `identify_struck_quark_candidates`
    - `build_branch_nodes`, `mothers_of`
    - `trace_ancestry` (can be reused or slightly wrapped)

- **CLI options:**
  - `--labels`: which PYTHIA labels to run (e.g. `ISRFSR_OFF,ISRFSR_ON`).
  - `--n-events`: number of accepted DIS events to inspect.
  - `--max-debug-events`: number of events for full DIS-neighborhood dumps.
  - `--trace-all-hadrons`, `--trace-charged-pions`: hadron selection controls.
  - `--forward-lab-pz-only`: restrict hadrons to `p_z > 0` in the LAB.
  - `--out`: JSONL for machine-readable results (similar to existing).

### Remnant-seed type classification

Add a helper:

```python
def classify_remnant_seed_type(pid: int, status: int) -> str:
    """
    Classify a proton-side daughter as one of:
      - spectator_quark_like
      - diquark_like
      - remnant_hadronic_like
      - string_or_junction_like
      - unknown
    """
```

Suggested rules:

- **spectator_quark_like**
  - \(|\text{pid}| \in \{1,2,3,4,5,6\}\) (quarks), `status >= 0`.
- **diquark_like**
  - Known diquark PDG codes (e.g. 1103, 2101, 2103, 2203, 3101, etc.). These can be listed explicitly for clarity.
- **remnant_hadronic_like**
  - Proton/neutron-like or similar:
    - `pid` in {2212, 2112} (p, n) or small excited nucleon states.
    - `status > 0` and not final state (`not isFinal()`), interpreted as remnant objects.
- **string_or_junction_like**
  - `pid` in {92, 93, 94} (string, junction, cluster-like).
- **unknown**
  - Everything else not caught above.

These labels are used for:

- Per-event remnant-seed audits.
- Building stricter remnant definitions that exclude `string_or_junction_like` and possibly `unknown`.

### Alternative remnant definitions

For each event:

1. Find:
   - proton index `p_idx`
   - struck-branch seeds `struck_seeds = identify_struck_quark_candidates(ev)`
   - struck-branch nodes `struck_nodes = build_branch_nodes(ev, struck_seeds, stop_at={p_idx})`
   - proton daughters from `ev[p_idx].daughter1()`/`daughter2()`:
     - `proton_daughters = list(range(d1, d2+1))` if `d1 > 0` and `d2 >= d1`.

2. **Definition A (current):**
   - `remnant_seeds_A = [d for d in proton_daughters if d not in struck_nodes]`
   - `remnant_nodes_A = build_branch_nodes(ev, remnant_seeds_A, stop_at={p_idx})`

3. **Definition B (stricter, parton/remnant oriented):**
   - For each `d in remnant_seeds_A`:
     - Compute `seed_type = classify_remnant_seed_type(pid, status)`.
     - Keep only types:
       - `"spectator_quark_like"`, `"diquark_like"`, `"remnant_hadronic_like"`.
   - `remnant_seeds_B = [d for d in remnant_seeds_A if seed_type(d) in {...}]`
   - `remnant_nodes_B = build_branch_nodes(ev, remnant_seeds_B, stop_at={p_idx})`

4. **(Optional) Definition C (closest leftover):**
   - From `remnant_seeds_B`, keep only the highest-energy one per event:
     - `best = max(remnant_seeds_B, key=lambda i: ev[i].e())`
   - `remnant_seeds_C = [best]` (or `[]` if none).
   - `remnant_nodes_C = build_branch_nodes(ev, remnant_seeds_C, stop_at={p_idx})`

Definitions A and B are the main comparison; C is optional for additional tests.

### Local DIS graph extractor

Purpose: visualize the local event-record graph near the DIS interaction, with explicit marks for:
  - proton
  - proton daughters
  - struck seeds / branch nodes
  - remnant seeds / branch nodes
  - beams

Steps per debug event:

1. Build a starting seed set:
   - `seeds = {p_idx} | set(struck_seeds) | set(remnant_seeds_A) | set(remnant_seeds_B)`

2. Expand to a radius-1 or radius-2 neighborhood in the mother/daughter graph:

   ```python
   local_nodes = set(seeds)
   frontier = list(seeds)
   for _ in range(radius):
       new_frontier = []
       for i in frontier:
           if i < 0 or i >= ev.size():
               continue
           p = ev[i]
           moms = mothers_of(p)
           d1, d2 = p.daughter1(), p.daughter2()
           daughters = list(range(d1, d2+1)) if d1 > 0 and d2 >= d1 else []
           for j in moms + daughters:
               if 0 <= j < ev.size() and j not in local_nodes:
                   local_nodes.add(j)
                   new_frontier.append(j)
       frontier = new_frontier
   ```

3. For each `idx in sorted(local_nodes)`:
   - Get `p = ev[idx]`, `pid`, `status`, `mothers`, `daughters`, `E, px, py, pz`.
   - Mark tags:
     - `P` if `idx == p_idx`.
     - `struck_seed` / `struck_branch` if in `struck_seeds` / `struck_nodes`.
     - `remnant_seed_A` / `remnant_branch_A`.
     - `remnant_seed_B` / `remnant_branch_B`.
     - `beam` if `status < 0` and `pid` in {11, 2212}.
   - Use a simple PID→name helper or PYTHIA `particleData.name(pid)` if available.
   - Print:

     ```text
     idx: pid=... (name) status=... [tags] mothers=[...] daughters=[...] E=..., px=..., py=..., pz=...
     ```

This supports manual inspection of the DIS neighborhood and satisfies the “local graph sanity” and “seed-neighborhood inspection” requirements.

### Remnant-seed composition audit

Over all inspected events:

- Track:
  - Number of proton daughters per event.
  - `len(remnant_seeds_A)` and `len(remnant_seeds_B)` per event.
  - For each `d in remnant_seeds_A`:
    - `type_A = classify_remnant_seed_type(pid, status)`:
      - Increment `type_counts_A[type_A]`.
      - Increment `pid_status_counts_A[(pid, status)]`.
  - Similarly, for `remnant_seeds_B`.
  - For `struck_seeds`:
    - Compute type with `classify_remnant_seed_type` as a sanity check on how “hard” seeds look.

Print summary distributions:

```text
Proton daughters per event: histogram or min/mean/max

Remnant seeds (Definition A) by type:
  spectator_quark_like      : N (fraction)
  diquark_like              : N (fraction)
  remnant_hadronic_like     : N (fraction)
  string_or_junction_like   : N (fraction)
  unknown                   : N (fraction)

Remnant seeds (Definition B) by type:
  ...

Struck seeds by type:
  ...
```

This checks whether remnant seeds are mostly spectator-like quarks/diquarks, remnant hadrons, or string bookkeeping.

### Classification stability A vs B

For each selected hadron (default: \(\pi^-\), options for charged pions / all hadrons):

1. Trace ancestry with Definition A:

   ```python
   resA = trace_ancestry(ev, hidx, struck_nodes, remnant_nodes_A, max_depth)
   clsA = resA.branch_classification
   ```

2. Trace ancestry with Definition B:

   ```python
   resB = trace_ancestry(ev, hidx, struck_nodes, remnant_nodes_B, max_depth)
   clsB = resB.branch_classification
   ```

3. Update stability stats:

   - For **all hadrons**:

     ```python
     total_all += 1
     if clsA == clsB:
         unchanged_all += 1
     transitions_all[(clsA, clsB)] += 1
     ```

   - For **pi- only** (`pid == -211`):
     - Similarly track `total_pi`, `unchanged_pi`, `transitions_pi`.

   - For **forward pi- only** (if `p_z > 0` in LAB and `pid == -211`):
     - Similarly track `total_forward_pi`, `unchanged_forward_pi`, `transitions_forward_pi`.

4. Print:

```text
Stability A→B (all hadrons):
  unchanged fraction: unchanged_all / total_all
  transitions:
    (A -> B): count (fraction)

Stability A→B (pi- only):
  ...

Stability A→B (forward pi- only):
  ...
```

This quantifies how robust hadron classifications are under remnant-definition changes.

### Branch-reachability summary per definition

Maintain separate branch-counts:

- For Definition A:

  ```python
  branch_counts_A[clsA] += 1
  branch_counts_pi_A[clsA] += 1 if pid == -211 else 0
  ```

- For Definition B:

  ```python
  branch_counts_B[clsB] += 1
  branch_counts_pi_B[clsB] += 1 if pid == -211 else 0
  ```

Print, for each definition (A, B), for all hadrons and for \(\pi^-\):

```text
Branch reachability (Definition A, all hadrons):
  remnant_branch_reachable       : ...
  struck_branch_reachable        : ...
  both_branches_reachable        : ...
  neither_branch_reachable       : ...
  string_only                    : ...
  ambiguous                      : ...

Branch reachability (Definition B, all hadrons):
  ...

Branch reachability (Definition A, pi-):
  ...

Branch reachability (Definition B, pi-):
  ...
```

Optionally provide the same for forward pi- only.

### Forward-pion selection

Simple choice:

- Forward in LAB if `p_z > 0`.

Implementation options:

- CLI flag `--forward-lab-pz-only`:
  - If set, restrict hadron selection in `get_selected_hadrons` to `p.pz() > 0`.
- Regardless, internally:
  - Tag forward pi- as `pid == -211 and p.pz() > 0` for stability statistics.

Print explicitly:

```text
Forward selection (if enabled): pz_lab > 0 (no eta cut).
```

### Validation tests to implement

1. **Local graph sanity**  
   - For several debug events:
     - Print the full local DIS neighborhood (`idx:id(status)->mothers,daughters,tags`) with branch/tag markers.
   - Ensure we see:
     - Proton.
     - Proton daughters.
     - Struck seeds.
     - Remnant seeds (A, B).

2. **Remnant-seed composition**  
   - Over a sample:
     - Print type-distributions of remnant seeds under A and B, plus struck-seed types.
   - Explicitly show:
     - spectator_quark_like
     - diquark_like
     - remnant_hadronic_like
     - string_or_junction_like
     - unknown

3. **Definition-stability test**  
   - Report A→B classification stability:
     - all hadrons,
     - pi- only,
     - forward pi- only (if implemented).

4. **Branch-reachability summary**  
   - As above, branch-classification fractions for A and B separately.

5. **Seed-neighborhood inspection**  
   - For debug events, explicitly show examples where:
     - remnant seeds are clean / intuitive (quark/diquark-like).
     - remnant seeds are mixed / ambiguous.
     - remnant seeds are dominated by string/junction-like objects.

### Caveats and interpretation

- The remnant side in PYTHIA is encoded with a mix of quarks, diquarks, nucleon remnants, and string/junction objects. Any “remnant branch” definition is inherently **model-dependent and somewhat heuristic**.
- The goal of this validation step is to:
  - Expose clearly what the current remnant definition is picking up.
  - Test how sensitive our hadron branch-reachability judgments are to reasonable changes in that definition.
- Even when a hadron is classified as `remnant_branch_reachable`, this should be interpreted as:
  - “its ancestry graph intersects a reasonable approximation of the remnant-side branch,”
  - **not** “we have identified a unique true remnant quark ancestor.” 

