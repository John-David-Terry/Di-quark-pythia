# Target π⁻ colour-flow remnant multiplicity

Remnant connection is defined by **colour flow** (primary diagnostic). For each tagged target-region π⁻ we count how many **distinct remnant partons** it is colour-connected to: `n_color_connected_remnant_partons = len(color_connected_remnant_node_indices)`.

- **Label:** ISRFSR_OFF
- **Accepted DIS events:** 120
- **Tagged π⁻ events:** 37

## Summary: number of colour-connected remnant partons per event

| n (colour-connected remnant partons) | Count | Fraction |
|--------------------------------------|-------|----------|
| 0 | 0 | 0.000 |
| 1 | 0 | 0.000 |
| 2 | 0 | 0.000 |
| 3 | 0 | 0.000 |
| 4+ | 37 | 1.000 |

## Representative examples (one per non-empty bin)

### n = 0 colour-connected remnant parton(s)
(No example in this run.)

### n = 1 colour-connected remnant parton(s)
(No example in this run.)

### n = 2 colour-connected remnant parton(s)
(No example in this run.)

### n = 3 colour-connected remnant parton(s)
(No example in this run.)

### n = 4+ colour-connected remnant parton(s)

- **event_number:** 0  |  **Q²:** 28.17 GeV²  |  **x_bj:** 0.0058
- **Tagged π⁻:** idx=36, E_breit=2.239, pz_breit=1.467
- **color_connected_remnant_node_indices:** [4, 6, 7, 8]
- **n_color_connected_remnant_partons:** 4

Remnant candidate nodes (first 10):
  - idx=4 pid=4 status=-21 tag=excluded_hard_neighbor
  - idx=6 pid=4 status=-23 tag=excluded_struck
  - idx=7 pid=4 status=-61 tag=remnant_candidate
  - idx=8 pid=4 status=-62 tag=excluded_hard_neighbor

Colour matches (tag → node_idx/pid, remnant_idx/pid):
  - tag=101  node 4(pid=4)  remnant 6(pid=4)
  - tag=101  node 4(pid=4)  remnant 7(pid=4)
  - tag=101  node 4(pid=4)  remnant 8(pid=4)
  - tag=101  node 6(pid=4)  remnant 4(pid=4)
  - tag=101  node 6(pid=4)  remnant 7(pid=4)
  - tag=101  node 6(pid=4)  remnant 8(pid=4)
  - tag=101  node 7(pid=4)  remnant 4(pid=4)
  - tag=101  node 7(pid=4)  remnant 6(pid=4)
  - tag=101  node 7(pid=4)  remnant 8(pid=4)
  - tag=101  node 8(pid=4)  remnant 4(pid=4)
  - tag=101  node 8(pid=4)  remnant 6(pid=4)
  - tag=101  node 8(pid=4)  remnant 7(pid=4)
  - tag=101  node 9(pid=2203)  remnant 4(pid=4)
  - tag=101  node 9(pid=2203)  remnant 6(pid=4)
  - tag=101  node 9(pid=2203)  remnant 7(pid=4)

*(Auxiliary)* has_mother_remnant_path = False; paths_to_remnant_hits = 0 path(s).

## Interpretation

The multiplicity counts how many **distinct remnant-side partons** the tagged target-region π⁻ is colour-connected to in the PYTHIA string picture. A value of 0 means no colour-tag overlap was found between the pion's ancestry/neighborhood and remnant candidates; 1 or more means the pion's hadronization system shares at least one colour line with that many remnant partons. Mother-graph paths are kept only as auxiliary context; the primary classification is colour-flow-based.

## Validation

- **Total tagged events:** 37
- **Events with 0 colour-connected remnant partons:** 0
- **Full multiplicity distribution:** see table above.
- **Colour-flow view and earlier “0 remnant ancestor” ambiguity:** the primary classification here is colour-flow-based; events that had no mother-graph path to a remnant parton often still show one or more colour-connected remnant partons, so the colour-flow view removes or greatly reduces that ambiguity.
- **One explicit event with more than one colour-connected remnant parton:** see the example for the first non-empty bin with n ≥ 2 above (or n = 4+ if that is the only non-empty bin).
