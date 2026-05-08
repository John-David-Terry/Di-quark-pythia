## Overview

This note explains, in technical detail, how the current scripts use the **PYTHIA 8 event record and methods** to:

- generate DIS events,
- construct the Breit frame,
- identify hard and remnant partons,
- trace ancestry via mother links, and
- diagnose remnant-side connectivity using **colour flow** (including the remnant-filtered, struck-veto construction).

The focus is on how we use PYTHIA’s own interfaces (`pythia8.Pythia`, `pythia8.Event`, `pythia8.Particle`) and their methods (`id()`, `status()`, `mothers`, `daughters`, `col()`, `acol()`, etc.).

The main implementation lives in:

- `scripts/generation/trace_target_pion_to_hard_vertex.py`
- `scripts/generation/debug_pythia_color_flow.py`

Throughout, `ev` denotes a `pythia8.Event` instance, and `p` denotes a `pythia8.Particle`.

---

## 1. DIS event generation with PYTHIA

We set up and run PYTHIA 8 directly in Python using the `pythia8` bindings:

```python
p = pythia8.Pythia()
p.readString("Beams:idA = 11")          # electron
p.readString("Beams:idB = 2212")        # proton
p.readString("Beams:eA = 18.0")
p.readString("Beams:eB = 275.0")
p.readString("Beams:frameType = 2")     # fixed-target / collider frame

p.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")  # DIS
p.readString("HardQCD:all = off")
p.readString("PDF:lepton = off")

p.readString("PhaseSpace:Q2Min = 16.0")             # DIS Q² cut
p.readString("HadronLevel:all = on")

## Toggle ISR/FSR:
p.readString("PartonLevel:ISR = on/off")
p.readString("PartonLevel:FSR = on/off")

## Colour reconnection toggle for some labels:
p.readString("ColourReconnection:reconnect = on/off")

p.readString("Random:setSeed = on")
p.readString(f"Random:seed = {BASE_SEED + seed_offset}")

p.init()
```

We then generate events in a loop using:

```python
while n_accepted < target:
    if not p.next():
        continue
    ev = p.event
    # analyse ev ...
```

All subsequent analysis uses the **live PYTHIA event record** in `ev`.

---

## 2. Accessing particles: id, status, mothers, daughters

In PYTHIA 8, the event record is a flat array of `Particle` objects:

- `ev.size()` – number of particles.
- `ev[i]` – the `i`‑th `pythia8.Particle`.

For each `Particle p = ev[i]` we use:

- `p.id()` – PDG code (e.g. 11 = e⁻, 2212 = p, ±211 = π±, 1–6 quarks, 21 gluon).
- `p.status()` – PYTHIA status code (negative for initial/beam-like, 23/63/… for intermediate partons, 91/−91 for cluster/string fragments, 1 / final‑state stable hadrons, etc.).
- `p.mother1()`, `p.mother2()` – indices of the first and second mother (0 or negative means no mother).
- `p.daughter1()`, `p.daughter2()` – index range of daughters (0 or negative means “no explicit daughters stored”).
- `p.e()`, `p.px()`, `p.py()`, `p.pz()` – 4‑momentum components.
- `p.col()`, `p.acol()` – **colour** and **anticolour** tags for partons / string endpoints.

We wrap the mother/daughter access in helpers:

```python
def mothers_of(p: pythia8.Particle) -> List[int]:
    m1, m2 = p.mother1(), p.mother2()
    out = []
    if m1 > 0:
        out.append(m1)
    if m2 > 0 and m2 != m1:
        out.append(m2)
    return out

def daughters_of(p: pythia8.Particle) -> List[int]:
    d1, d2 = p.daughter1(), p.daughter2()
    if d1 <= 0 or d2 <= 0:
        return []
    return list(range(d1, d2 + 1))
```

We also define a “parton-like” predicate:

```python
PARTON_PIDS = {1, 2, 3, 4, 5, 6, 21}
def is_parton_like(pid: int) -> bool:
    return abs(pid) in PARTON_PIDS
```

These are the basic PYTHIA methods we repeatedly use to:

- find beams and scattered leptons,
- build the hard-interaction and remnant sets,
- traverse ancestry and descendants,
- and inspect colour flow.

---

## 3. Finding beams, DIS kinematics, and the Breit transform

### 3.1 Incoming and scattered leptons, proton beam

We identify the incoming electron and proton, and the scattered electron, using `(p.id(), p.status())`:

```python
def find_incoming_beams(ev: pythia8.Event) -> Tuple[Optional[int], Optional[int]]:
    e_idx, p_idx = None, None
    for i in range(ev.size()):
        p = ev[i]
        if p.id() == 11 and p.status() == -12:   # incoming e⁻
            e_idx = i
        if p.id() == 2212 and p.status() < 0:    # proton beam (negative status)
            p_idx = i
    return e_idx, p_idx

def get_scattered_electron_idx(ev: pythia8.Event) -> Optional[int]:
    candidates = [i for i in range(ev.size()) if ev[i].id() == 11 and ev[i].status() > 0]
    # Prefer status=44 if present, else the highest-energy final e⁻
    for i in candidates:
        if ev[i].status() == 44:
            return i
    return max(candidates, key=lambda i: ev[i].e()) if candidates else None
```

Using these indices, we build PYTHIA‑frame 4‑vectors:

```python
e_in = ev[e_idx]
e_sc = ev[e_sc_idx]
P_in = ev[p_idx]

e_in_4 = np.array([e_in.e(), e_in.px(), e_in.py(), e_in.pz()])
e_sc_4 = np.array([e_sc.e(), e_sc.px(), e_sc.py(), e_sc.pz()])
P_in_4 = np.array([P_in.e(), P_in.px(), P_in.py(), P_in.pz()])
```

### 3.2 Q² and x\_Bj from the event record

`trace_target_pion_to_hard_vertex.py` computes \(Q^2\) and \(x_{\mathrm{Bj}}\) directly from these vectors using the Minkowski metric, relying only on PYTHIA’s 4‑momenta:

```python
def minkowski_norm(e, px, py, pz):
    return e*e - px*px - py*py - pz*pz

def compute_q2_and_x(ev, e_in_idx, e_sc_idx, p_in_idx):
    e_in, e_sc, p_in = ev[e_in_idx], ev[e_sc_idx], ev[p_in_idx]
    Ein, pxin, pyin, pzin = e_in.e(), e_in.px(), e_in.py(), e_in.pz()
    Esc, pxsc, pysc, pzsc = e_sc.e(), e_sc.px(), e_sc.py(), e_sc.pz()
    Pin, ppx, ppy, ppz = p_in.e(), p_in.px(), p_in.py(), p_in.pz()

    q_e  = Ein - Esc
    q_px = pxin - pxsc
    q_py = pyin - pysc
    q_pz = pzin - pzsc

    q2 = -minkowski_norm(q_e, q_px, q_py, q_pz)
    Pdotq = Pin*q_e - ppx*q_px - ppy*q_py - ppz*q_pz
    xbj = q2 / (2.0 * Pdotq) if Pdotq > 0 else float("nan")
    return q2, xbj
```

### 3.3 Breit transform

We reuse the same **Breit-frame Lorentz transform** that is used in the main analysis pipeline. The transform is built once per event (using electron and proton 4‑vectors from PYTHIA) and then applied as a matrix to any 4‑vector:

```python
LT = build_breit_transform(e_in_4, e_sc_4, P_in_4)
if LT is None:
    # Breit construction failed for this event
    return None

def apply_LT(LT, v4):
    return LT @ v4
```

The tagged target-region π⁻ is found by scanning over all final-state particles in `ev`, selecting π⁻ with `pz_breit > 0` and maximal `E_breit`:

```python
def find_hardest_target_pi_minus(ev: pythia8.Event, LT: np.ndarray):
    best_idx, best_Eb, best_p4_lab, best_p4_breit, best_pz_breit = None, -1.0, None, None, None
    for i in range(ev.size()):
        p = ev[i]
        if p.id() != -211 or p.status() <= 0:
            continue
        p4_lab = np.array([p.e(), p.px(), p.py(), p.pz()])
        p4_breit = LT @ p4_lab
        E_b, px_b, py_b, pz_b = p4_breit
        if pz_b <= 0.0:       # target (positive-z) hemisphere in this Breit convention
            continue
        if E_b > best_Eb:
            best_idx = i
            best_Eb = E_b
            best_p4_lab = p4_lab
            best_p4_breit = p4_breit
            best_pz_breit = pz_b
    if best_idx is None:
        return None
    return best_idx, best_p4_lab, best_p4_breit, best_Eb, best_pz_breit
```

All of this uses only PYTHIA’s `Particle` methods and a linear algebra layer for the boosts.

---

## 4. Hard-interaction and struck candidates

### 4.1 Identifying struck-quark candidates

`trace_target_pion_to_hard_vertex.py` defines **struck candidates** using PYTHIA’s status codes and quark PIDs. The helper `identify_struck_quark_candidates(ev)` (imported in `debug_pythia_color_flow.py`) scans the event for:

- quark-like PIDs (±1..±6),
- status codes consistent with outgoing hard partons in DIS (e.g. −23, −63..−68, etc.).

In pseudocode:

```python
def identify_struck_quark_candidates(ev):
    struck = []
    for i in range(ev.size()):
        p = ev[i]
        if not is_parton_like(p.id()):
            continue
        st = p.status()
        # DIS-like outgoing struck quarks:
        if st in (-23, -63, -64, -65, -66, -67, -68) or (st > 0 and some additional checks):
            struck.append(i)
    return struck
```

### 4.2 Hard-interaction node set

Using these struck candidates, `identify_hard_interaction_nodes(ev)` builds a **hard node set** that includes:

- the struck candidate indices,
- their mothers (e.g. virtual photon, proton-side pre-splitter),
- their daughters (immediate downstream partons).

This is done entirely with PYTHIA’s `mothers` and `daughters`:

```python
def identify_hard_interaction_nodes(ev: pythia8.Event) -> Tuple[Set[int], List[Dict], List[Dict]]:
    struck = identify_struck_quark_candidates(ev)
    hard: Set[int] = set(struck)
    for i in struck:
        # include mothers
        for m in mothers_of(ev[i]):
            hard.add(m)
        # include daughters
        for d in daughters_of(ev[i]):
            hard.add(d)
    # Build summaries for later reporting (pid, status, e, px, py, pz)
    ...
    return hard, node_summaries, struck_summaries
```

These definitions are used both in the ancestry-tracing script and in the colour-flow diagnostic.

---

## 5. Remnant-candidate identification from the proton side

The **remnant candidate set** is built by walking forward from the proton through its descendants, then excluding anything on the struck branch or in the hard set.

### 5.1 Proton descendants with mother-scan fallback

PYTHIA does not always populate `p.daughter1()/daughter2()` for the incoming proton. To recover proton children, we also **scan for particles whose mother is the proton**:

```python
def get_children_by_mother_scan(ev: pythia8.Event, parent_idx: int) -> List[int]:
    return [i for i in range(ev.size()) if parent_idx in mothers_of(ev[i])]
```

We then build the proton-descendant seeds:

```python
proton = ev[proton_idx]
proton_daughters = daughters_of(proton)
proton_seeds = proton_daughters if proton_daughters else get_children_by_mother_scan(ev, proton_idx)
```

and walk **forward** with `get_descendants_forward`:

```python
def get_descendants_forward(ev, seed_indices, exclude=None, max_nodes=2000):
    exclude = exclude or set()
    out: Set[int] = set()
    queue = list(seed_indices)
    while queue and len(out) < max_nodes:
        idx = queue.pop(0)
        if idx in out or idx in exclude or idx < 0 or idx >= ev.size():
            continue
        out.add(idx)
        for d in daughters_of(ev[idx]):
            if d not in out:
                queue.append(d)
    return out

descendants_proton = {proton_idx} | get_descendants_forward(ev, proton_seeds, set())
```

### 5.2 Excluding the struck branch

We also build **struck-branch descendants** from the struck candidates:

```python
descendants_struck = get_descendants_forward(ev, struck_list, set())
```

The **remnant candidate set** is then:

- nodes that are proton descendants,
- are parton-like (quark/gluon),
- are not in the hard set,
- are not struck-branch descendants.

This uses only PYTHIA’s `id()`, `status()`, and ancestry information:

```python
remnant_candidate_set: Set[int] = set()
for i in descendants_proton:
    p = ev[i]
    if not is_parton_like(p.id()):
        continue
    if i in hard_nodes or i in descendants_struck:
        continue
    remnant_candidate_set.add(i)
```

We also construct a **node table** with tags like `"remnant_candidate"`, `"excluded_struck"`, `"excluded_hard_neighbor"`, etc., for later inspection.

---

## 6. Backward BFS ancestry tracing (mother graph)

The basic ancestry tracer uses a **breadth-first search (BFS) over PYTHIA’s mother links** starting from the tagged π⁻ index:

```python
def trace_backward_bfs(
    ev: pythia8.Event,
    start_idx: int,
    hard_nodes: Set[int],
    remnant_nodes: Set[int],
    max_depth: int = 100,
):
    queue = [(start_idx, 0)]
    visited: Set[int] = set()
    parent: Dict[int, int] = {}       # child_idx -> mother_idx

    ancestry_trace: List[Dict] = []   # for full record
    reached_remnant_indices: Set[int] = set()
    reached_hard_indices: Set[int] = set()
    first_remnant_hit_step = None
    first_hard_hit_step = None

    while queue:
        idx, depth = queue.pop(0)
        if idx in visited or idx < 0 or idx >= ev.size():
            continue
        visited.add(idx)

        p = ev[idx]
        ancestry_trace.append({
            "idx": idx,
            "pid": int(p.id()),
            "status": int(p.status()),
            "mothers": mothers_of(p),
            "depth": depth,
        })

        if idx in remnant_nodes:
            reached_remnant_indices.add(idx)
            if first_remnant_hit_step is None:
                first_remnant_hit_step = len(ancestry_trace) - 1

        if idx in hard_nodes:
            reached_hard_indices.add(idx)
            if first_hard_hit_step is None:
                first_hard_hit_step = len(ancestry_trace) - 1

        if depth >= max_depth:
            continue

        for m in mothers_of(p):
            if m not in visited:
                parent[m] = idx
                queue.append((m, depth + 1))

    return ancestry_trace, parent, reached_remnant_indices, reached_hard_indices, first_remnant_hit_step, first_hard_hit_step, "ok"
```

This uses only PYTHIA’s `mothers` and `status`/`id` fields, and produces a full **set of visited ancestor nodes** (`ancestry_trace`), plus intersection information with the hard and remnant sets.

### 6.1 Reconstructing a specific path

Given the `parent` map from BFS, we can reconstruct **one ancestry chain** from the pion to some hit node:

```python
def reconstruct_path(ev: pythia8.Event, parent_map: Dict[int, int], start_idx: int, hit_idx: int) -> List[Dict]:
    path_indices = [hit_idx]
    cur = hit_idx
    while cur != start_idx and cur in parent_map:
        cur = parent_map[cur]
        path_indices.append(cur)
    path_indices.reverse()
    out: List[Dict] = []
    for idx in path_indices:
        p = ev[idx]
        out.append({
            "idx": idx,
            "pid": int(p.id()),
            "status": int(p.status()),
            "mothers": mothers_of(p),
            "daughters": daughters_of(p),
        })
    return out
```

Again, everything is driven by PYTHIA’s built-in `mother1/2` and `daughter1/2` links.

---

## 7. Colour flow via `col()` / `acol()`

The colour-flow diagnostic uses the `col()` and `acol()` tags that PYTHIA stores on partons and string endpoints:

```python
def collect_color_tags_for_node(ev: pythia8.Event, idx: int) -> Tuple[int, int]:
    if idx < 0 or idx >= ev.size():
        return 0, 0
    p = ev[idx]
    return int(p.col()), int(p.acol())
```

### 7.1 Remnant colour tags

For each **remnant candidate** index `i` in `remnant_candidate_set`, we read its colour and anticolour:

```python
remnant_color_tags: Set[int] = set()
remnant_idx_to_colors: Dict[int, Set[int]] = {}

for n in remnant_candidate_nodes:   # each n contains 'idx', 'pid', 'status', etc.
    idx = n["idx"]
    c, ac = collect_color_tags_for_node(ev, idx)
    tags = set()
    if c != 0:  tags.add(c)
    if ac != 0: tags.add(ac)
    remnant_idx_to_colors[idx] = tags
    remnant_color_tags |= tags
```

We then build a reverse map:

```python
color_to_remnant_indices: Dict[int, List[int]] = {}
for idx, tags in remnant_idx_to_colors.items():
    for t in tags:
        color_to_remnant_indices.setdefault(t, []).append(idx)
```

### 7.2 Ancestry/neighbourhood colour tags

We define a **neighbourhood** of indices around the tagged pion using ancestry and path information:

```python
def build_neighborhood_indices(tagged_idx, ancestry_trace, path_to_first_hard_hit, paths_to_remnant_hits, remnant_indices, hard_nodes, ev):
    neigh = {tagged_idx}
    neigh |= {step["idx"] for step in ancestry_trace}
    neigh |= {step["idx"] for step in path_to_first_hard_hit}
    for path in paths_to_remnant_hits:
        neigh |= {step["idx"] for step in path}
    neigh |= remnant_indices
    neigh |= hard_nodes
    # plus immediate mothers/daughters of all these
    extra = set()
    for idx in list(neigh):
        if 0 <= idx < ev.size():
            p = ev[idx]
            extra.update(mothers_of(p))
            extra.update(daughters_of(p))
    neigh |= extra
    return {i for i in neigh if 0 <= i < ev.size()}
```

For each node `idx` in this neighbourhood we look up its `col()` and `acol()` and check whether any of those tags match remnant colours:

```python
ancestry_color_tags: Set[int] = set()
color_flow_matches: List[Dict] = []
color_connected_remnant_node_indices: Set[int] = set()

for idx in neigh_indices:
    p = ev[idx]
    c, ac = collect_color_tags_for_node(ev, idx)
    for t in (c, ac):
        if t == 0:
            continue
        ancestry_color_tags.add(t)
        if t in color_to_remnant_indices:
            for r_idx in color_to_remnant_indices[t]:
                if r_idx == idx:
                    continue   # skip trivial self-matches
                color_connected_remnant_node_indices.add(r_idx)
                color_flow_matches.append({
                    "tag": t,
                    "node_idx": idx,
                    "node_pid": int(p.id()),
                    "remnant_idx": r_idx,
                    "remnant_pid": next((n["pid"] for n in remnant_candidate_nodes if n["idx"] == r_idx), None),
                })
```

This defines the **unfiltered** colour-flow remnant connectivity:

- `has_color_flow_to_remnant = (len(color_flow_matches) > 0)`,
- `n_color_connected_remnant_partons = len(color_connected_remnant_node_indices)`.

All of this uses only PYTHIA’s **colour tags** and the previously defined remnant-candidate set.

---

## 8. Struck-side veto and remnant-filtered colour history

To isolate **remnant-side colour flow**, we build a **struck veto set** that collects everything associated with the struck line:

```python
hard_nodes, hard_node_summaries, struck_summaries = identify_hard_interaction_nodes(ev)
struck_list = [s["idx"] for s in struck_summaries]

struck_veto_set: Set[int] = set(struck_list) | hard_nodes
if struck_list:
    struck_veto_set |= get_descendants_forward(ev, struck_list, set())
```

This uses PYTHIA daughter links to include **all descendants of the struck candidates**; combined with `hard_nodes`, it defines the **entire struck-side branch** in the event record.

### 8.1 Filtered colour matches

We then take the **unfiltered** colour matches and discard any where the ancestry node or the remnant node lies in the struck veto set:

```python
filtered_color_connected_remnant_node_indices: Set[int] = set()
filtered_color_flow_matches: List[Dict] = []

for m in color_flow_matches:
    node_idx = m["node_idx"]
    remnant_idx = m["remnant_idx"]
    if node_idx in struck_veto_set or remnant_idx in struck_veto_set:
        continue
    filtered_color_connected_remnant_node_indices.add(remnant_idx)
    filtered_color_flow_matches.append(m)

n_filtered_color_connected_remnant_partons = len(filtered_color_connected_remnant_node_indices)
has_filtered_color_connection_to_remnant = (n_filtered_color_connected_remnant_partons > 0)
```

This yields a **remnant-filtered** colour-history:

- **`n_filtered_color_connected_remnant_partons`** is now the primary quantity,
- it counts distinct remnant candidates that share a colour tag with the tagged π⁻ neighbourhood **without touching the struck-side branch**.

Again, this uses only:

- `col()`, `acol()` for colour tags,
- `mothers` / `daughters` to build the struck veto via descendants,
- the remnant candidate set from proton descendants.

---

## 9. Summary: How PYTHIA methods underpin the analysis

All steps in the ancestry and colour-flow remnant analysis are built directly on top of PYTHIA’s own API:

- **Event generation and DIS kinematics** use `Pythia.readString`, `Pythia.next()`, `Event`, and `Particle` 4‑momenta.
- **Beam, scattered lepton, and proton identification** rely on `p.id()` and `p.status()`.
- **Hard-interaction and struck candidates** are defined from **status codes and quark PIDs**, plus mother/daughter relations.
- **Remnant candidates** come from proton descendants (using both `daughter1/2` and a mother-scan), excluding struck-branch descendants and hard nodes.
- **Backward ancestry tracing** is a BFS over `mother1/2` links.
- **Colour-flow connectivity** is entirely determined by `p.col()` and `p.acol()`, via colour-tag matches between the tagged-π⁻ neighbourhood and remnant candidates.
- The **struck-side veto** uses daughter links (`daughter1/2`) to gather all struck descendants and removes any colour connection that touches those nodes.

In this way, the entire **remnant connection picture**—including the remnant-filtered colour history—is constructed using standard PYTHIA 8 methods and data structures, with no external reconstruction of the event graph beyond what PYTHIA already encodes.

