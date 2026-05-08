# Target π⁻ remnant ancestry report

Generated from `scripts/generation/trace_target_pion_to_hard_vertex.py` (remnant + hard set tracing).

**Caveats:** Remnant-candidate set is a first-pass operational definition; this does not prove unique physical parentage. It identifies whether the tagged π⁻ ancestor graph intersects the candidate remnant-side parton set and/or the hard set.

## Summary

- **Label:** ISRFSR_OFF
- **Total accepted DIS events:** 120
- **Total tagged π⁻ events (hardest in Breit target):** 37

### Classification counts

| Classification | Count | Fraction |
|-----------------|-------|----------|
| remnant_only | 0 | 0.000 |
| hard_only | 7 | 0.189 |
| both | 28 | 0.757 |
| neither | 2 | 0.054 |

### Remnant path classification (does any path to remnant avoid the hard set?)

| Remnant path classification | Count | Fraction |
|-------------------------------|-------|----------|
| no_remnant_path | 9 | 0.243 |
| remnant_via_hard | 4 | 0.108 |
| remnant_avoids_hard | 12 | 0.324 |
| both_types | 12 | 0.324 |

### Validation note

Remnant-candidate set = parton-like (quark/gluon) nodes that are proton descendants but not on the struck branch. Proton descendants are now built using a mother-scan fallback when the proton's daughter links are empty (see PYTHIA_REMNANT_DEBUG.md). With this fix, **remnant_candidate_nodes are non-empty in ISRFSR_OFF**; nodes like idx=7 (status -61, beam/remnant-side parton) are included as remnant_candidate.

## Representative examples

### remnant_only
(No example in this run.)

### hard_only

- **Event number:** 2  |  **Q²:** 49.82 GeV²  |  **x_bj:** 0.0059
- **Tagged pion:** idx=14, E_breit=0.815, pz_breit=0.703
- **Remnant candidate nodes:**
  - idx=4 pid=1 status=-21 tag=excluded_hard_neighbor mothers=[7] daughters=[5, 6]
  - idx=6 pid=1 status=-23 tag=excluded_struck mothers=[3, 4] daughters=[8]
  - idx=7 pid=1 status=-61 tag=remnant_candidate mothers=[2, 0] daughters=[4]
  - idx=8 pid=1 status=-62 tag=excluded_hard_neighbor mothers=[6] daughters=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
- **Hard-interaction nodes (first 10):**
  - idx=3 pid=11 status=-21 tags=['mother_of_struck']
  - idx=4 pid=1 status=-21 tags=['mother_of_struck']
  - idx=6 pid=1 status=-23 tags=['struck_candidate']
  - idx=8 pid=1 status=-62 tags=['daughter_of_struck']
- **Ancestry trace:** 8 nodes
- **Path to first remnant hit:** (none)
- **Path to first hard hit:** idx=14(pid=-211) → idx=8(pid=1)

### both

- **Event number:** 1  |  **Q²:** 25.76 GeV²  |  **x_bj:** 0.0174
- **Tagged pion:** idx=21, E_breit=4.567, pz_breit=0.014
- **Remnant candidate nodes:**
  - idx=4 pid=1 status=-21 tag=excluded_hard_neighbor mothers=[7] daughters=[5, 6]
  - idx=6 pid=1 status=-23 tag=excluded_struck mothers=[3, 4] daughters=[8]
  - idx=7 pid=1 status=-61 tag=remnant_candidate mothers=[2, 0] daughters=[4]
  - idx=8 pid=1 status=-62 tag=excluded_hard_neighbor mothers=[6] daughters=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
- **Hard-interaction nodes (first 10):**
  - idx=3 pid=11 status=-21 tags=['mother_of_struck']
  - idx=4 pid=1 status=-21 tags=['mother_of_struck']
  - idx=6 pid=1 status=-23 tags=['struck_candidate']
  - idx=8 pid=1 status=-62 tags=['daughter_of_struck']
- **Ancestry trace:** 11 nodes
- **Path to first remnant hit:** idx=21(pid=-211) → idx=14(pid=-213) → idx=8(pid=1) → idx=6(pid=1) → idx=4(pid=1) → idx=7(pid=1)
- **Path to first hard hit:** idx=21(pid=-211) → idx=14(pid=-213) → idx=8(pid=1)

### neither

- **Event number:** 0  |  **Q²:** 28.17 GeV²  |  **x_bj:** 0.0058
- **Tagged pion:** idx=36, E_breit=2.239, pz_breit=1.467
- **Remnant candidate nodes:**
  - idx=4 pid=4 status=-21 tag=excluded_hard_neighbor mothers=[7] daughters=[5, 6]
  - idx=6 pid=4 status=-23 tag=excluded_struck mothers=[3, 4] daughters=[8]
  - idx=7 pid=4 status=-61 tag=remnant_candidate mothers=[2, 0] daughters=[4]
  - idx=8 pid=4 status=-62 tag=excluded_hard_neighbor mothers=[6] daughters=[11, 12, 13, 14, 15, 16, 17, 18]
- **Hard-interaction nodes (first 10):**
  - idx=3 pid=11 status=-21 tags=['mother_of_struck']
  - idx=4 pid=4 status=-21 tags=['mother_of_struck']
  - idx=6 pid=4 status=-23 tags=['struck_candidate']
  - idx=8 pid=4 status=-62 tags=['daughter_of_struck']
- **Ancestry trace:** 4 nodes
- **Path to first remnant hit:** (none)
- **Path to first hard hit:** (none)

## Remnant path examples (hard vs avoiding hard)

### remnant_via_hard

- **Event number:** 1  |  **Q²:** 25.76  |  **x_bj:** 0.0174
- **remnant_path_classification:** remnant_via_hard
- **Example path through hard (pion → ... → remnant):**
  idx=21(pid=-211) → idx=14(pid=-213) → idx=8(pid=1) → idx=6(pid=1) → idx=4(pid=1) → idx=7(pid=1)

### remnant_avoids_hard

- **Event number:** 3  |  **Q²:** 23.64  |  **x_bj:** 0.0099
- **remnant_path_classification:** remnant_avoids_hard
- **Example path avoiding hard (pion → ... → remnant):**
  idx=16(pid=-211) → idx=11(pid=2) → idx=9(pid=2)

### both_types

- **Event number:** 5  |  **Q²:** 54.88  |  **x_bj:** 0.0084
- **remnant_path_classification:** both_types
- **Example path through hard (pion → ... → remnant):**
  idx=28(pid=-211) → idx=26(pid=310) → idx=23(pid=-311) → idx=15(pid=-411) → idx=12(pid=-4) → idx=8(pid=-4) → idx=6(pid=-4) → idx=4(pid=-4) → idx=7(pid=-4)
- **Example path avoiding hard (pion → ... → remnant):**
  idx=28(pid=-211) → idx=26(pid=310) → idx=23(pid=-311) → idx=15(pid=-411) → idx=11(pid=2) → idx=9(pid=2)

---

## Validation (remnant path diagnostic)

1. **How many tagged events have any remnant path?** 28 (events with at least one path to a remnant-candidate node).

2. **How many have a remnant path avoiding hard nodes?** 24 (12 classified `remnant_avoids_hard` + 12 classified `both_types` have at least one path that does not touch the hard set before the remnant hit).

3. **How many only reach remnant through hard nodes?** 4 (classified `remnant_via_hard` — every path to a remnant node goes through the hard-interaction set first).

4. **One explicit example path for each non-empty category:**
   - **remnant_via_hard:** Event 1 — path through hard: idx=21 → idx=14 → idx=8 → idx=6 → idx=4 → idx=7 (pion → ρ⁻ → struck daughter → struck → incoming parton → remnant parton 7).
   - **remnant_avoids_hard:** Event 3 — path avoiding hard: idx=16 → idx=11 → idx=9 (pion → u quark → remnant node 9).
   - **both_types:** Event 5 — one path through hard (…→12→8→6→4→7), one path avoiding hard (…→11→9).
