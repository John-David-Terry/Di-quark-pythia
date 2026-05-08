# Ancestry tracing algorithm for the tagged target-region π⁻

This document describes **exactly** how `scripts/generation/trace_target_pion_to_hard_vertex.py` traces the tagged target-region π⁻ backward through the PYTHIA event record, identifies **hard-side** and **remnant-side** partons, and classifies remnant paths.

---

## 1. Event preparation

1. **DIS event generation**
   - The script calls `setup_pythia(label)` with `label ∈ {ISRFSR_OFF, ISRFSR_ON, ETA_ON_CRON}`.
   - For `ISRFSR_OFF` (the main diagnostic mode):
     - `Beams:idA = 11` (electron), `Beams:idB = 2212` (proton).
     - `Beams:eA = 18 GeV`, `Beams:eB = 275 GeV`.
     - `Beams:frameType = 2` (fixed-target / collider setup with explicit beam energies).
     - `WeakBosonExchange:ff2ff(t:gmZ) = on` (DIS via γ*/Z exchange).
     - `PhaseSpace:Q2Min = 16 GeV^2` (Q² cut).
     - `PartonLevel:ISR = off`, `PartonLevel:FSR = off`.
     - `HadronLevel:all = on` (hadronization enabled).
     - A reproducible seed is set via `Random:setSeed = on` and `Random:seed = BASE_SEED + seed_offset`.

2. **Finding beam and scattered electron indices**
   - After each `pythia.next()` call, the script identifies:
     - Incoming electron index `e_idx` as `id == 11` and `status == -12` (beam electron).
     - Proton index `p_idx` as `id == 2212` and `status < 0` (incoming proton beam).
     - Scattered electron index `e_sc_idx` as the electron with `id == 11` and `status > 0` and, if present, `status == 44`; otherwise the highest-energy positive-status electron.

3. **DIS selection (Q² and x)**
   - Using `compute_q2_and_x(ev, e_idx, e_sc_idx, p_idx)`:
     - Construct `q = l_in - l_sc` from `e_in` and `e_sc` 4-vectors.
     - Compute `Q² = -q²` via `minkowski_norm` (metric diag(+,-,-,-)).
     - Compute `P·q` from proton 4-vector and `q`.
     - Require:
       - `Q² > Q2_MIN` (e.g. 16 GeV²).
       - `P·q > 0`.
       - `x = Q² / (2 P·q) > 0` and finite.

4. **Breit frame construction**
   - `build_breit_transform(e_in_4, e_sc_4, p_in_4)`:
     - Recomputes `q`, `Q²`, `x`, `S = 4 E_e E_p`, `y = Q²/(S x)`, `q_T`, `φ_q`.
     - Forms `qµ = (q⁰, qˣ, qʸ, qᶻ)` in the lab frame.
     - Calls `build_LT(Ee, Ep, qµ, x, y, qT, phiq, S)` from `diquark.analyze_events_raw` to obtain a **4×4 Lorentz matrix `LT`** that defines the Breit(-like) frame.

5. **Tagged target-region π⁻ selection**
   - For each accepted event:
     - Loop over all particles `i` in the event:
       - Require `p.isFinal()` and `p.id() == -211` (π⁻).
       - Form lab 4-vector `p4 = (E, px, py, pz)`.
       - Compute Breit 4-vector `p4_b = LT @ p4`.
       - Require `pz_breit = p4_b[3] > 0` (Breit **target hemisphere**).
     - Among all π⁻ satisfying this, pick the **hardest**:
       - Maximize `E_breit = p4_b[0]`.
       - Store:
         - `tag_idx` (pion index), `p4_lab`, `p4_breit`, `E_breit`, `pz_breit`.


---

## 2. Hard-interaction identification

1. **Struck candidates (`identify_struck_quark_candidates`)**
   - Quark IDs considered: `|pid| ∈ {1,2,3,4,5,6}`.
   - Pass 1: return the **first** particle with quark ID and `|status| == 23` (hard-scattered parton in PYTHIA’s convention).
   - If none found, pass 2: return first quark with `63 ≤ |status| ≤ 68` (alternative hard-process codes).
   - If still none found:
     - Fall back to: among all quarks with `status != 0`, choose the **highest-energy** quark as primary struck candidate, and include all other quark indices as secondary candidates.
   - Result: a **list `struck_list`** of indices (`[primary, secondary1, ...]`).

2. **Hard-interaction set (`identify_hard_interaction_nodes`)**
   - Initialize `hard_nodes = {}`.
   - For each struck index `i` in `struck_list`:
     - Add `i` to `hard_nodes`.
     - Add its mothers `mother1`, `mother2` (if `≥ 0`) to `hard_nodes`.
     - Add all daughters in the contiguous range `[daughter1, daughter2]` (if valid) to `hard_nodes`.
   - Build **summary objects** for struck candidates and all `hard_nodes`:
     - Each entry includes `idx, pid, status, mother1, mother2, daughter1, daughter2, 4-momentum components`, and `tags`:
       - `struck_candidate` if the node is in `struck_list`.
       - `mother_of_struck` if this node is a mother of any struck candidate.
       - `daughter_of_struck` if this node is in the daughter range of any struck candidate.
       - `hard_interaction_node` if in `hard_nodes` but not tagged above.
       - `exchanged_boson_candidate` if `pid == 22` (photon-like).


---

## 3. Remnant candidate identification

1. **Proton index**
   - From `find_incoming_beams(ev)`, `p_idx` is the index with `id == 2212` and `status < 0`.
   - This is treated as the **incoming proton**.

2. **Proton descendants (with mother-scan fallback)**
   - Direct daughters using the event’s *forward* links:
     - `proton_daughters = daughters_of(ev[proton_idx])`, using `daughter1/daughter2`.
   - If `proton_daughters` is empty (common in these DIS events):
     - Use a **mother scan fallback**:
       - `get_children_by_mother_scan(ev, proton_idx)` returns all `i` such that `proton_idx in mothers_of(ev[i])`.
       - This picks out nodes like index 7 with `mothers=[2, 0]` where `2` is the proton.
   - Seeds for forward traversal:

   ```python
   proton_daughters = daughters_of(ev[proton_idx])
   proton_seeds = proton_daughters if proton_daughters else get_children_by_mother_scan(ev, proton_idx)
   descendants_proton = {proton_idx} | get_descendants_forward(ev, proton_seeds, set())
   ```

   - `get_descendants_forward` performs a **forward BFS along daughter links** from these seeds, adding each visited index to `descendants_proton`.

3. **Struck branch descendants**
   - `descendants_struck = get_descendants_forward(ev, struck_list, set())` collects everything that is downstream of the struck line.

4. **Remnant candidate set**
   - Logic:

   ```python
   remnant_indices = set()
   for i in descendants_proton:
       if i in hard_nodes or i in descendants_struck:
           continue  # hard or struck-side
       if i < 0 or i >= ev.size():
           continue
       p = ev[i]
       if is_parton_like(p.id()):  # |pid| in {1,2,3,4,5,6,21}
           remnant_indices.add(i)
   ```

   - At this point `remnant_indices` typically contains the **beam/remnant-side parton(s)** such as index 7 (status -61, mother=[proton]).

5. **Per-node remnant tagging**
   - For each **parton-like** `i` in `descendants_proton`:
     - Build a record with `idx, pid, status, mothers, daughters` and a `tag`:
       - `excluded_struck` if `i ∈ struck_list`.
       - `excluded_hard_neighbor` if `i ∈ hard_nodes` (but not in `struck_list`).
       - `remnant_candidate` if `i ∈ remnant_indices`.
       - `unknown` otherwise.
   - This list is stored as `remnant_candidate_nodes` in the JSONL.


---

## 4. Backward ancestry tracing (BFS)

The core backward traversal is implemented in `trace_backward_bfs`.

1. **Initialization**
   - Input:
     - `start_idx = tag_idx` (tagged π⁻ index).
     - `hard_nodes` and `remnant_nodes` (the `remnant_indices` set).
     - `max_depth` (default 100).
   - State:

   ```python
   trace = []                 # list of steps (in BFS order)
   visited = set()            # indices already processed
   parent_map = {}            # child_idx -> parent_idx that discovered it
   queue = [(start_idx, 0)]   # BFS queue: (index, depth)
   reached_remnant = set()
   reached_hard = set()
   first_remnant_hit_step = None
   first_hard_hit_step = None
   stop_reason = "completed"
   ```

2. **BFS loop (backward through mothers)**

   **Pseudocode:**

   ```python
   while queue:
       idx, depth = queue.pop(0)
       if idx in visited:
           stop_reason = "cycle_detected"; break
       if depth > max_depth:
           stop_reason = "max_depth"; break
       visited.add(idx)
       if idx < 0 or idx >= ev.size():
           stop_reason = "invalid_index"; break

       p = ev[idx]
       moms = mothers_of(p)  # [mother1, mother2] filtered for >= 0

       step_idx = len(trace)
       trace.append({"idx": idx, "pid": p.id(), "status": p.status(), "mothers": moms})

       # Record remnant / hard hits
       if idx in remnant_nodes:
           reached_remnant.add(idx)
           if first_remnant_hit_step is None:
               first_remnant_hit_step = step_idx
       if idx in hard_nodes:
           reached_hard.add(idx)
           if first_hard_hit_step is None:
               first_hard_hit_step = step_idx

       # Termination conditions if no more mothers or structure is odd
       if not moms:
           stop_reason = "no_mother"; break
       if len(moms) > 2:
           stop_reason = "ambiguous_branch"; break

       # Enqueue mothers
       for m in moms:
           if m < 0 or m >= ev.size():
               stop_reason = "invalid_mother"; break
           if m not in visited and m not in parent_map:
               parent_map[m] = idx   # record that we reached m via idx
           if m not in visited:
               queue.append((m, depth + 1))
   ```

3. **What `ancestry_trace` stores**
   - `trace` holds **every visited node** in BFS order:
     - Each entry: `{ "idx", "pid", "status", "mothers" }`.
   - This is the **entire visited ancestor set** from the tagged π⁻ up to the stopping condition (hard node, remnant, no mothers, invalid ref, cycle, max depth).
   - It is **not** limited to a single linear chain; it includes all branches that BFS explores.


---

## 5. Path reconstruction

1. **Parent pointers during BFS**
   - Whenever the BFS enqueues a new mother `m` from child `idx`, it records:

   ```python
   if m not in visited and m not in parent_map:
       parent_map[m] = idx  # child idx discovered mother m
   ```

   - This means: for each discovered node `m`, `parent_map[m]` holds **the child index that led to it** in the BFS tree.

2. **Reconstructing a path (`reconstruct_path`)**

   - To reconstruct a path from `start_idx` (tagged pion) to some hit index `hit_idx` (e.g. a remnant candidate or a hard node):

   **Pseudocode:**

   ```python
   def reconstruct_path(ev, parent_map, start_idx, hit_idx):
       path = []
       cur = hit_idx
       while True:
           if cur < 0 or cur >= ev.size():
               break
           p = ev[cur]
           path.append({
               "idx": cur,
               "pid": int(p.id()),
               "status": int(p.status()),
               "mothers": mothers_of(p),
           })
           if cur == start_idx:
               break
           cur = parent_map.get(cur)
           if cur is None:
               break
       path.reverse()
       return path
   ```

   - Because BFS filled `parent_map` along **backward** steps (child → mother), the reconstructed path is a **consistent ancestry chain** that exists within the visited graph.

3. **Stored paths**
   - For each event the script stores:
     - `path_to_first_hard_hit`: path to the earliest hard node seen in BFS.
     - `path_to_first_remnant_hit`: path to the earliest remnant candidate seen in BFS.
     - `paths_to_remnant_hits`: paths to **all** reached remnant-candidate indices.


---

## 6. Hit detection (hard vs remnant)

1. **Hits during BFS**
   - As seen above, during BFS the script checks each visited index `idx` against:
     - `idx in remnant_nodes` → add to `reached_remnant`, update `first_remnant_hit_step` if unset.
     - `idx in hard_nodes` → add to `reached_hard`, update `first_hard_hit_step` if unset.

2. **Event-level hit flags**
   - After BFS finishes:

   ```python
   reached_remnant = bool(reached_remnant_indices)
   reached_hard = bool(reached_hard_indices)
   classification = classify_trace_result(reached_remnant, reached_hard)
   ```

   where `reached_remnant_indices = reached_remnant`, `reached_hard_indices = reached_hard`.

   - `classify_trace_result` returns:
     - `both` if both sets are non-empty,
     - `remnant_only` if only remnant reached,
     - `hard_only` if only hard reached,
     - `neither` otherwise.

   - These are the **coarse-grained** classifications used earlier in the study.


---

## 7. Remnant path classification (via hard vs avoiding hard)

This is the **new diagnostic** that directly answers whether the remnant is reached **only through the hard line** or can be reached along a path that **avoids hard nodes**.

1. **All remnant paths per event**
   - After BFS and `reached_remnant_indices` are known, the script builds:

   ```python
   paths_to_remnant_hits = [
       reconstruct_path(ev, parent_map, tag_idx, r)
       for r in reached_remnant_indices
   ]
   ```

   - Each element is a path:
     - `path[0]` is the tagged π⁻.
     - `path[-1]` is a remnant-candidate node (`idx in remnant_nodes`).

2. **Path classification w.r.t. hard set**

   **Pseudocode (`classify_remnant_paths`)**:

   ```python
   def classify_remnant_paths(paths_to_remnant, hard_nodes):
       paths_through = []
       paths_avoiding = []
       for path in paths_to_remnant:
           if not path:
               continue
           # Tagged pion -> ... -> remnant; check all nodes *before* the remnant
           touches_hard = any(n["idx"] in hard_nodes for n in path[:-1])
           if touches_hard:
               paths_through.append(path)
           else:
               paths_avoiding.append(path)

       has_remnant = len(paths_to_remnant) > 0
       has_through = len(paths_through) > 0
       has_avoiding = len(paths_avoiding) > 0

       if not has_remnant:
           classification = "no_remnant_path"
       elif has_avoiding and has_through:
           classification = "both_types"
       elif has_avoiding:
           classification = "remnant_avoids_hard"
       else:
           classification = "remnant_via_hard"

       return (
           has_remnant,
           has_avoiding,
           has_through,
           classification,
           paths_avoiding,
           paths_through,
       )
   ```

3. **Event-level remnant path fields**

   For each tagged event, the JSONL record includes:

   - `has_remnant_path`: `True` iff at least one remnant path exists.
   - `has_remnant_path_avoiding_hard`: `True` iff at least one remnant path has **no** hard node before the remnant.
   - `has_remnant_path_through_hard`: `True` iff at least one remnant path **does** go through the hard set.
   - `remnant_path_classification ∈ {"no_remnant_path", "remnant_via_hard", "remnant_avoids_hard", "both_types"}`.
   - `all_reached_remnant_node_indices`: all remnant candidates reached by BFS.
   - `paths_to_remnant_hits`: all remnant paths.
   - `paths_to_remnant_hits_avoiding_hard`: subset of paths that avoid hard nodes.
   - `paths_to_remnant_hits_through_hard`: subset of paths that pass through hard nodes.

4. **Global counts and examples**
   - The script accumulates `remnant_path_classification_counts` over all tagged events, and `example_by_remnant_path` gives a representative event for each category.
   - The markdown report **explicitly prints**:
     - Counts for `no_remnant_path`, `remnant_via_hard`, `remnant_avoids_hard`, `both_types`.
     - Example paths for each non-empty category.


---

## 8. Final answer: what the algorithm measures

Putting all the pieces together, the algorithm answers the question:

> **“Which remnant-side partons present after the hard interaction lie on an ancestry path to the tagged π⁻?”**

Concretely:

1. It identifies **remnant-side partons** as parton-like nodes that:
   - Descend from the **incoming proton** (using daughter links or mother-scan fallback),
   - Are **not** on the struck branch and **not** part of the immediate hard-interaction neighborhood.

2. It performs a **backward BFS** from the tagged target-region π⁻ across **all mother links**, recording every visited node and whether it is in the hard set or remnant set.

3. Using parent pointers, it reconstructs **all ancestry paths** from the pion to each reached remnant candidate.

4. It then determines, for each remnant-side parton candidate:
   - Whether there exists a path from the pion to that remnant candidate that **passes through** any hard-interaction node (`remnant_via_hard`),
   - Whether there exists a path that **completely avoids** the hard set (`remnant_avoids_hard`),
   - Or whether **both types** of paths exist (`both_types`), or **no remnant path** exists at all.

Thus, for every tagged target-region π⁻, the code explicitly maps out **which remnant-side partons are on its ancestry graph**, and whether that connection requires going **through the hard DIS line** or can proceed via an **independent remnant-side branch**.
