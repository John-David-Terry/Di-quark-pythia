# Example ancestry histories for the tagged target-region π⁻

Generated from `scripts/generation/trace_target_pion_to_hard_vertex.py`
with ISRFSR_OFF, 120 accepted DIS events, 37 tagged events (hardest π⁻ in Breit target, pz_breit > 0).

---

## A. Clean successful case

### Event header
- **Label:** ISRFSR_OFF
- **Accepted event number:** 2
- **Global event index:** 16
- **Q²:** 49.82 GeV²
- **x_bj:** 0.0059

### Tagged pion summary
- **Index:** 14
- **PID:** -211 (π⁻)
- **Status:** 83
- **LAB 4-vector (E, px, py, pz):** (0.6037, 0.0243, 0.4625, 0.3612) GeV
- **Breit 4-vector (E, px, py, pz):** (0.8151, 0.3329, -0.1996, 0.7031) GeV
- **E_breit:** 0.8151 GeV
- **pz_breit:** 0.7031 GeV

### Hard-interaction neighborhood summary
- **idx=3**  pid=11 (e⁻)  status=-21  mothers=[1, 0]  daughters=[5, 6]  **tags:** mother_of_struck
- **idx=4**  pid=1 (d)  status=-21  mothers=[7, 7]  daughters=[5, 6]  **tags:** mother_of_struck
- **idx=6**  pid=1 (d)  status=-23  mothers=[3, 4]  daughters=[8, 8]  **tags:** struck_candidate
- **idx=8**  pid=1 (d)  status=-62  mothers=[6, 6]  daughters=[11, 22]  **tags:** daughter_of_struck

### Ancestry trace (visited nodes in BFS order)
The trace is the set of nodes visited during the backward walk until the stop. Order is BFS (breadth-first), not necessarily a single linear path.

- **Step 0:** idx=14  pid=-211 (π⁻)  status=83  mothers=[8, 9] ← **starting pion**
- **Step 1:** idx=8  pid=1 (d)  status=-62  mothers=[6] ← **stopping node**

**Stop reason:** reached_hard_interaction
**Stop node indices:** [8]

#### Interpretation
This pion has a very short ancestry: it traces backward through one step to a **daughter of the struck quark** (idx=8, status -62). So it is directly connected to the hard-interaction layer with minimal steps.

---

## B. More complicated successful case

### Event header
- **Label:** ISRFSR_OFF
- **Accepted event number:** 30
- **Global event index:** 102
- **Q²:** 43.21 GeV²
- **x_bj:** 0.0067

### Tagged pion summary
- **Index:** 47
- **PID:** -211 (π⁻)
- **Status:** 91
- **LAB 4-vector (E, px, py, pz):** (0.5764, -0.0719, -0.2499, -0.4952) GeV
- **Breit 4-vector (E, px, py, pz):** (0.6681, -0.6060, 0.2350, 0.0660) GeV
- **E_breit:** 0.6681 GeV
- **pz_breit:** 0.0660 GeV

### Hard-interaction neighborhood summary
- **idx=3**  pid=11 (e⁻)  status=-21  mothers=[1, 0]  daughters=[5, 6]  **tags:** mother_of_struck
- **idx=4**  pid=-4 (c̄)  status=-21  mothers=[7, 7]  daughters=[5, 6]  **tags:** mother_of_struck
- **idx=6**  pid=-4 (c̄)  status=-23  mothers=[3, 4]  daughters=[8, 8]  **tags:** struck_candidate
- **idx=8**  pid=-4 (c̄)  status=-62  mothers=[6, 6]  daughters=[12, 12]  **tags:** daughter_of_struck

### Ancestry trace (visited nodes in BFS order)
The trace is the set of nodes visited during the backward walk until the stop. Order is BFS (breadth-first), not necessarily a single linear path.

- **Step 0:** idx=47  pid=-211 (π⁻)  status=91  mothers=[27, 0] ← **starting pion**
- **Step 1:** idx=27  pid=310 (K_S⁰)  status=-91  mothers=[25]
- **Step 2:** idx=0  pid=90 (system)  status=-11  mothers=[0]
- **Step 3:** idx=25  pid=311 (K⁰)  status=-91  mothers=[20, 0]
- **Step 4:** idx=20  pid=323 (K*⁺)  status=-84  mothers=[11, 12]
- **Step 5:** idx=11  pid=2 (u)  status=-71  mothers=[10]
- **Step 6:** idx=12  pid=-4 (c̄)  status=-71  mothers=[8]
- **Step 7:** idx=10  pid=2 (u)  status=-63  mothers=[2, 0]
- **Step 8:** idx=8  pid=-4 (c̄)  status=-62  mothers=[6] ← **stopping node**

**Stop reason:** reached_hard_interaction
**Stop node indices:** [8]

#### Interpretation
This pion traces backward through several intermediate partons and resonances (π⁻ → K_S⁰ → K⁰ → K*⁺ → u/c̄ chain). The ancestry eventually reaches a **daughter of the struck quark** (idx=8, c̄ with status -62). The path is longer and shows branching (BFS visits multiple ancestors), but one branch connects to the hard-interaction neighborhood.

---

## C. Failure / messy case

### Event header
- **Label:** ISRFSR_OFF
- **Accepted event number:** 0
- **Global event index:** 4
- **Q²:** 28.17 GeV²
- **x_bj:** 0.0058

### Tagged pion summary
- **Index:** 36
- **PID:** -211 (π⁻)
- **Status:** 91
- **LAB 4-vector (E, px, py, pz):** (2.8016, -2.0376, 1.1824, 1.5099) GeV
- **Breit 4-vector (E, px, py, pz):** (2.2394, 1.6855, -0.0362, 1.4674) GeV
- **E_breit:** 2.2394 GeV
- **pz_breit:** 1.4674 GeV

### Hard-interaction neighborhood summary
- **idx=3**  pid=11 (e⁻)  status=-21  mothers=[1, 0]  daughters=[5, 6]  **tags:** mother_of_struck
- **idx=4**  pid=4 (c)  status=-21  mothers=[7, 7]  daughters=[5, 6]  **tags:** mother_of_struck
- **idx=6**  pid=4 (c)  status=-23  mothers=[3, 4]  daughters=[8, 8]  **tags:** struck_candidate
- **idx=8**  pid=4 (c)  status=-62  mothers=[6, 6]  daughters=[11, 18]  **tags:** daughter_of_struck

### Ancestry trace (visited nodes in BFS order)
The trace is the set of nodes visited during the backward walk until the stop. Order is BFS (breadth-first), not necessarily a single linear path.

- **Step 0:** idx=36  pid=-211 (π⁻)  status=91  mothers=[27, 0] ← **starting pion**
- **Step 1:** idx=27  pid=113 (ρ⁰)  status=-91  mothers=[11, 0]
- **Step 2:** idx=0  pid=90 (system)  status=-11  mothers=[0]
- **Step 3:** idx=11  pid=411 (D⁺)  status=-83  mothers=[8, 9]

**Stop reason:** cycle_detected
**Stop node indices:** []

#### Interpretation
This case enters a **cycle** in the mother graph (the same node was visited twice) before reaching the hard-interaction set. The tracer stops with `cycle_detected` and does not assign a stopping node. The ancestry is therefore unresolved for this event.

---

## Summary

- **What do the ancestor sets usually look like?**  They are the set of nodes visited in a backward BFS from the tagged π⁻. In successful cases the trace reaches the hard-interaction set (struck-quark candidates plus their mothers and daughters) after a small to moderate number of steps.

- **Are they typically short or long?**  In this sample, successful traces often have 2–5 steps (pion → one or two intermediates → hard node). Some events have longer traces (up to ~9 steps) when the pion comes from a longer decay/parton chain.

- **Do they usually stop on daughters of the struck quark or on the struck quark itself?**  In the current implementation they **almost always stop on a daughter of the struck quark** (status -62, first shower step), because the hard-interaction set includes those daughters. Stopping on the struck quark (status -23) itself would require the ancestry to hit that index before hitting any of its daughters.

- **Most common messy/failure modes?**  In this run, the only failure mode seen was **cycle_detected** (2 out of 37 tagged events). The mother graph sometimes contains a cycle (e.g. through the event record or repeated indices), and the walk stops without reaching the hard set.
