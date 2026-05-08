# PYTHIA proton-remnant encoding debug

Focused dump of event-record neighborhoods for selected tagged events to see where proton-side leftover partons live.

Label: ISRFSR_OFF. Generated until we had tagged events 0, 1, 30 (or as many as available).

---

## Event number 0 (classification = no_remnant_candidates)

- Q² = 28.17 GeV², x_bj = 0.0058
- **Proton index:** 2
- **Proton daughters (direct from event record):** []
- **Proton children (by mother scan):** [7, 9, 10]
- **Proton descendants up to depth 5:** 42 nodes: [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]

- **Struck candidate index:** 6 (all struck: [6])
- **Struck daughter tree up to depth 4:** 23 nodes: [6, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 35, 36]

### Node table (neighborhood)

| idx | pid | status | mothers | daughters | parton? | in_hard? | remnant_cand? |
|-----|-----|--------|---------|-----------|---------|----------|---------------|
| 0 | 90 | -11 | [0] | [] | no | no | no |
| 1 | 11 | -12 | [0] | [] | no | no | no |
| 2 | 2212 | -12 | [0] | [] | no | no | no |
| 3 | 11 | -21 | [1, 0] | [5, 6] | no | yes | no |
| 4 | 4 | -21 | [7] | [5, 6] | yes | yes | no |
| 5 | 11 | 23 | [3, 4] | [] | no | no | no |
| 6 | 4 | -23 | [3, 4] | [8] | yes | yes | no |
| 7 | 4 | -61 | [2, 0] | [4] | yes | no | no |
| 8 | 4 | -62 | [6] | [11, 12, 13, 14, 15, 16, 17, 18] | yes | yes | no |
| 9 | 2203 | -63 | [2, 0] | [11, 12, 13, 14, 15, 16, 17, 18] | no | no | no |
| 10 | -411 | -63 | [2, 0] | [23, 24] | no | no | no |
| 11 | 411 | -83 | [8, 9] | [25, 26, 27] | no | no | no |
| 12 | 111 | -83 | [8, 9] | [28, 29] | no | no | no |
| 13 | -211 | 83 | [8, 9] | [] | no | no | no |
| 14 | 2224 | -83 | [8, 9] | [19, 20] | no | no | no |
| 15 | -211 | 83 | [8, 9] | [] | no | no | no |
| 16 | -2212 | 83 | [8, 9] | [] | no | no | no |
| 17 | 2212 | 84 | [8, 9] | [] | no | no | no |
| 18 | 213 | -84 | [8, 9] | [21, 22] | no | no | no |
| 19 | 2212 | 91 | [14, 0] | [] | no | no | no |
| 20 | 211 | 91 | [14, 0] | [] | no | no | no |
| 21 | 211 | 91 | [18, 0] | [] | no | no | no |
| 22 | 111 | -91 | [18, 0] | [30, 31] | no | no | no |
| 23 | -20213 | -91 | [10, 0] | [32, 33] | no | no | no |
| 24 | -311 | -91 | [10, 0] | [34] | no | no | no |
| 25 | -13 | 91 | [11, 0] | [] | no | no | no |
| 26 | 14 | 91 | [11, 0] | [] | no | no | no |
| 27 | 113 | -91 | [11, 0] | [35, 36] | no | no | no |
| 28 | 22 | 91 | [12, 0] | [] | no | no | no |
| 29 | 22 | 91 | [12, 0] | [] | no | no | no |
| 30 | 22 | 91 | [22, 0] | [] | no | no | no |
| 31 | 22 | 91 | [22, 0] | [] | no | no | no |
| 32 | -213 | -91 | [23, 0] | [37, 38] | no | no | no |
| 33 | 111 | -91 | [23, 0] | [39, 40] | no | no | no |
| 34 | 310 | -91 | [24] | [41, 42] | no | no | no |
| 35 | 211 | 91 | [27, 0] | [] | no | no | no |
| 36 | -211 | 91 | [27, 0] | [] | no | no | no |
| 37 | -211 | 91 | [32, 0] | [] | no | no | no |
| 38 | 111 | -91 | [32, 0] | [43, 44] | no | no | no |
| 39 | 22 | 91 | [33, 0] | [] | no | no | no |
| 40 | 22 | 91 | [33, 0] | [] | no | no | no |
| 41 | 211 | 91 | [34, 0] | [] | no | no | no |
| 42 | -211 | 91 | [34, 0] | [] | no | no | no |
| 43 | 22 | 91 | [38, 0] | [] | no | no | no |
| 44 | 22 | 91 | [38, 0] | [] | no | no | no |

### Parton-like nodes in neighborhood (detail)
- **idx=4** pid=4 status=-21 mothers=[7] daughters=[5, 6]  [in hard set]
- **idx=6** pid=4 status=-23 mothers=[3, 4] daughters=[8]  [in hard set]
- **idx=7** pid=4 status=-61 mothers=[2, 0] daughters=[4]  [—]
- **idx=8** pid=4 status=-62 mothers=[6] daughters=[11, 12, 13, 14, 15, 16, 17, 18]  [in hard set]

---

## Event number 1 (classification = no_remnant_candidates)

- Q² = 25.76 GeV², x_bj = 0.0174
- **Proton index:** 2
- **Proton daughters (direct from event record):** []
- **Proton children (by mother scan):** [7, 9, 10]
- **Proton descendants up to depth 5:** 34 nodes: [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

- **Struck candidate index:** 6 (all struck: [6])
- **Struck daughter tree up to depth 4:** 26 nodes: [6, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

### Node table (neighborhood)

| idx | pid | status | mothers | daughters | parton? | in_hard? | remnant_cand? |
|-----|-----|--------|---------|-----------|---------|----------|---------------|
| 0 | 90 | -11 | [0] | [] | no | no | no |
| 1 | 11 | -12 | [0] | [] | no | no | no |
| 2 | 2212 | -12 | [0] | [] | no | no | no |
| 3 | 11 | -21 | [1, 0] | [5, 6] | no | yes | no |
| 4 | 1 | -21 | [7] | [5, 6] | yes | yes | no |
| 5 | 11 | 23 | [3, 4] | [] | no | no | no |
| 6 | 1 | -23 | [3, 4] | [8] | yes | yes | no |
| 7 | 1 | -61 | [2, 0] | [4] | yes | no | no |
| 8 | 1 | -62 | [6] | [11, 12, 13, 14, 15, 16, 17, 18, 19, 20] | yes | yes | no |
| 9 | 2203 | -63 | [2, 0] | [11, 12, 13, 14, 15, 16, 17, 18, 19, 20] | no | no | no |
| 10 | 111 | -63 | [2, 0] | [25, 26] | no | no | no |
| 11 | -211 | 83 | [8, 9] | [] | no | no | no |
| 12 | 3222 | -83 | [8, 9] | [27, 28] | no | no | no |
| 13 | -3212 | -83 | [8, 9] | [29, 30] | no | no | no |
| 14 | -213 | -83 | [8, 9] | [21, 22] | no | no | no |
| 15 | 213 | -84 | [8, 9] | [23, 24] | no | no | no |
| 16 | -211 | 84 | [8, 9] | [] | no | no | no |
| 17 | 211 | 84 | [8, 9] | [] | no | no | no |
| 18 | -211 | 84 | [8, 9] | [] | no | no | no |
| 19 | 2212 | 84 | [8, 9] | [] | no | no | no |
| 20 | 211 | 84 | [8, 9] | [] | no | no | no |
| 21 | -211 | 91 | [14, 0] | [] | no | no | no |
| 22 | 111 | -91 | [14, 0] | [31, 32] | no | no | no |
| 23 | 211 | 91 | [15, 0] | [] | no | no | no |
| 24 | 111 | -91 | [15, 0] | [33, 34] | no | no | no |
| 25 | 22 | 91 | [10, 0] | [] | no | no | no |
| 26 | 22 | 91 | [10, 0] | [] | no | no | no |
| 27 | 2112 | 91 | [12, 0] | [] | no | no | no |
| 28 | 211 | 91 | [12, 0] | [] | no | no | no |
| 29 | -3122 | -91 | [13, 0] | [35, 36] | no | no | no |
| 30 | 22 | 91 | [13, 0] | [] | no | no | no |
| 31 | 22 | 91 | [22, 0] | [] | no | no | no |
| 32 | 22 | 91 | [22, 0] | [] | no | no | no |
| 33 | 22 | 91 | [24, 0] | [] | no | no | no |
| 34 | 22 | 91 | [24, 0] | [] | no | no | no |
| 35 | -2212 | 91 | [29, 0] | [] | no | no | no |
| 36 | 211 | 91 | [29, 0] | [] | no | no | no |

### Parton-like nodes in neighborhood (detail)
- **idx=4** pid=1 status=-21 mothers=[7] daughters=[5, 6]  [in hard set]
- **idx=6** pid=1 status=-23 mothers=[3, 4] daughters=[8]  [in hard set]
- **idx=7** pid=1 status=-61 mothers=[2, 0] daughters=[4]  [—]
- **idx=8** pid=1 status=-62 mothers=[6] daughters=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  [in hard set]

---

## Event number 30 (classification = no_remnant_candidates)

- Q² = 43.21 GeV², x_bj = 0.0067
- **Proton index:** 2
- **Proton daughters (direct from event record):** []
- **Proton children (by mother scan):** [7, 9, 10]
- **Proton descendants up to depth 5:** 62 nodes: [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52] ...

- **Struck candidate index:** 6 (all struck: [6])
- **Struck daughter tree up to depth 4:** 28 nodes: [6, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]

### Node table (neighborhood)

| idx | pid | status | mothers | daughters | parton? | in_hard? | remnant_cand? |
|-----|-----|--------|---------|-----------|---------|----------|---------------|
| 0 | 90 | -11 | [0] | [] | no | no | no |
| 1 | 11 | -12 | [0] | [] | no | no | no |
| 2 | 2212 | -12 | [0] | [] | no | no | no |
| 3 | 11 | -21 | [1, 0] | [5, 6] | no | yes | no |
| 4 | -4 | -21 | [7] | [5, 6] | yes | yes | no |
| 5 | 11 | 23 | [3, 4] | [] | no | no | no |
| 6 | -4 | -23 | [3, 4] | [8] | yes | yes | no |
| 7 | -4 | -61 | [2, 0] | [4] | yes | no | no |
| 8 | -4 | -62 | [6] | [12] | yes | yes | no |
| 9 | 4122 | -63 | [2, 0] | [28, 29, 30] | no | no | no |
| 10 | 2 | -63 | [2, 0] | [11] | yes | no | no |
| 11 | 2 | -71 | [10] | [13, 14, 15, 16, 17, 18, 19, 20, 21] | yes | no | no |
| 12 | -4 | -71 | [8] | [13, 14, 15, 16, 17, 18, 19, 20, 21] | yes | no | no |
| 13 | 321 | 83 | [11, 12] | [] | no | no | no |
| 14 | -311 | -83 | [11, 12] | [22] | no | no | no |
| 15 | 221 | -83 | [11, 12] | [31, 32, 33] | no | no | no |
| 16 | 111 | -83 | [11, 12] | [34, 35] | no | no | no |
| 17 | 221 | -83 | [11, 12] | [36, 37] | no | no | no |
| 18 | -213 | -83 | [11, 12] | [23, 24] | no | no | no |
| 19 | 221 | -84 | [11, 12] | [38, 39] | no | no | no |
| 20 | 323 | -84 | [11, 12] | [25, 26] | no | no | no |
| 21 | -433 | -84 | [11, 12] | [40, 41] | no | no | no |
| 22 | 310 | -91 | [14] | [42, 43] | no | no | no |
| 23 | -211 | 91 | [18, 0] | [] | no | no | no |
| 24 | 111 | -91 | [18, 0] | [44, 45] | no | no | no |
| 25 | 311 | -91 | [20, 0] | [27] | no | no | no |
| 26 | 211 | 91 | [20, 0] | [] | no | no | no |
| 27 | 310 | -91 | [25] | [46, 47] | no | no | no |
| 28 | 211 | 91 | [9, 0] | [] | no | no | no |
| 29 | 223 | -91 | [9, 0] | [48, 49, 50] | no | no | no |
| 30 | 2112 | 91 | [9, 0] | [] | no | no | no |
| 31 | 111 | -91 | [15, 0] | [51, 52] | no | no | no |
| 32 | 111 | -91 | [15, 0] | [53, 54] | no | no | no |
| 33 | 111 | -91 | [15, 0] | [55, 56] | no | no | no |
| 34 | 22 | 91 | [16, 0] | [] | no | no | no |
| 35 | 22 | 91 | [16, 0] | [] | no | no | no |
| 36 | 22 | 91 | [17, 0] | [] | no | no | no |
| 37 | 22 | 91 | [17, 0] | [] | no | no | no |
| 38 | 22 | 91 | [19, 0] | [] | no | no | no |
| 39 | 22 | 91 | [19, 0] | [] | no | no | no |
| 40 | -431 | -91 | [21, 0] | [57, 58, 59] | no | no | no |
| 41 | 22 | 91 | [21, 0] | [] | no | no | no |
| 42 | 211 | 91 | [22, 0] | [] | no | no | no |
| 43 | -211 | 91 | [22, 0] | [] | no | no | no |
| 44 | 22 | 91 | [24, 0] | [] | no | no | no |
| 45 | 22 | 91 | [24, 0] | [] | no | no | no |
| 46 | 211 | 91 | [27, 0] | [] | no | no | no |
| 47 | -211 | 91 | [27, 0] | [] | no | no | no |
| 48 | 211 | 91 | [29, 0] | [] | no | no | no |
| 49 | -211 | 91 | [29, 0] | [] | no | no | no |
| 50 | 111 | -91 | [29, 0] | [60, 61] | no | no | no |
| 51 | 22 | 91 | [31, 0] | [] | no | no | no |
| 52 | 22 | 91 | [31, 0] | [] | no | no | no |
| 53 | 22 | 91 | [32, 0] | [] | no | no | no |
| 54 | 22 | 91 | [32, 0] | [] | no | no | no |
| 55 | 22 | 91 | [33, 0] | [] | no | no | no |
| 56 | 22 | 91 | [33, 0] | [] | no | no | no |
| 57 | 13 | 91 | [40, 0] | [] | no | no | no |
| 58 | -14 | 91 | [40, 0] | [] | no | no | no |
| 59 | 221 | -91 | [40, 0] | [62, 63, 64] | no | no | no |
| 60 | 22 | 91 | [50, 0] | [] | no | no | no |
| 61 | 22 | 91 | [50, 0] | [] | no | no | no |
| 62 | 111 | -91 | [59, 0] | [65, 66] | no | no | no |
| 63 | 111 | -91 | [59, 0] | [67, 68] | no | no | no |
| 64 | 111 | -91 | [59, 0] | [69, 70] | no | no | no |

### Parton-like nodes in neighborhood (detail)
- **idx=4** pid=-4 status=-21 mothers=[7] daughters=[5, 6]  [in hard set]
- **idx=6** pid=-4 status=-23 mothers=[3, 4] daughters=[8]  [in hard set]
- **idx=7** pid=-4 status=-61 mothers=[2, 0] daughters=[4]  [—]
- **idx=8** pid=-4 status=-62 mothers=[6] daughters=[12]  [in hard set]
- **idx=10** pid=2 status=-63 mothers=[2, 0] daughters=[11]  [—]
- **idx=11** pid=2 status=-71 mothers=[10] daughters=[13, 14, 15, 16, 17, 18, 19, 20, 21]  [—]
- **idx=12** pid=-4 status=-71 mothers=[8] daughters=[13, 14, 15, 16, 17, 18, 19, 20, 21]  [—]

---

## Validation summary

- **Event(s) dumped:** 0, 1, 30
- **At least one hard_only-type event:** present (events 1 and 30 show hard-interaction nodes; event 0 may be neither if cycle_detected).
- **At least one neither-type event:** event 0 often has cycle_detected and no hard hit — see ancestry in main tracer.
- **Complicated hard_only:** event 30 (longer ancestry) if in dump.

**With ISR/FSR off:** Do any non-struck proton-descendant partons appear in the record?
 In the events dumped above, **no** node is classified as remnant_candidate by the current definition (remnant_candidate set is empty).

**Where do proton-side leftover partons live?** In all three events, **Proton children (by mother scan)** are [7, 9, 10]. Node **idx=7** is parton-like (quark), has mother [2, 0] (proton), and is **not** in the hard set: it is the beam/remnant-side parton that produced the incoming hard parton (idx=4). So with ISR/FSR off, **non-struck proton-descendant partons do appear**: idx=7 (status -61) is the natural remnant-side parton. The current remnant-candidate definition is empty because it uses only the proton's `daughter1`/`daughter2` links to find descendants; in PYTHIA's record the proton often has no daughters stored, so we must use a mother-scan (particles whose mother is the proton) to get proton children, then build the descendant set from those. Including idx=7 (and any other proton-child partons not on the struck branch) in the remnant-candidate set would fix the definition.
