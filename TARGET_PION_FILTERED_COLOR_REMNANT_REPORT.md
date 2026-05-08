# Target π⁻ remnant-filtered colour-flow report

Backward colour-history of the tagged π⁻ with the **entire struck-side descendant branch vetoed**. Only colour connections that do not touch the struck-veto set are kept; the main quantity is **n_filtered_color_connected_remnant_partons**.

- **Label:** ISRFSR_OFF
- **Accepted DIS events:** 120
- **Tagged π⁻ events:** 37

## Struck veto set

- **Typical size:** mean |struck_veto_set| = 29.2 (min=7, max=55).

## Filtered remnant multiplicity (main quantity)

| n (filtered colour-connected remnant partons) | Count | Fraction |
|-------------------------------------------------|-------|----------|
| 0 | 0 | 0.000 |
| 1 | 12 | 0.324 |
| 2 | 0 | 0.000 |
| 3+ | 25 | 0.676 |

## Comparison with unfiltered colour-flow

- Unfiltered: mean n = 6.08.
- Filtered (after struck veto): mean n = 2.35.
- Vetoing struck descendants **reduces** the earlier over-connection (unfiltered counted struck-side nodes as remnant-connected).

## Representative examples (one per non-empty bin)

### n_filtered = 0
(No example in this run.)

### n_filtered = 1

- **event_number:** 0  |  **Q²:** 28.17 GeV²  |  **x_bj:** 0.0058
- **Tagged π⁻:** idx=36, E_breit=2.239, pz_breit=1.467
- **struck_veto_set:** |set| = 25; indices (first 20): [3, 4, 6, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28]
- **filtered_color_connected_remnant_node_indices:** [7]
- **n_filtered_color_connected_remnant_partons:** 1

Remnant candidate nodes (first 10):
  - idx=4 pid=4 status=-21 tag=excluded_hard_neighbor
  - idx=6 pid=4 status=-23 tag=excluded_struck
  - idx=7 pid=4 status=-61 tag=remnant_candidate
  - idx=8 pid=4 status=-62 tag=excluded_hard_neighbor

Filtered colour matches (tag → node_idx/pid, remnant_idx/pid):
  - tag=101  node 9(pid=2203)  remnant 7(pid=4)

### n_filtered = 2
(No example in this run.)

### n_filtered = 3+

- **event_number:** 3  |  **Q²:** 23.64 GeV²  |  **x_bj:** 0.0099
- **Tagged π⁻:** idx=16, E_breit=2.179, pz_breit=0.041
- **struck_veto_set:** |set| = 17; indices (first 20): [3, 4, 6, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
- **filtered_color_connected_remnant_node_indices:** [7, 9, 11]
- **n_filtered_color_connected_remnant_partons:** 3

Remnant candidate nodes (first 10):
  - idx=4 pid=-2 status=-21 tag=excluded_hard_neighbor
  - idx=6 pid=-2 status=-23 tag=excluded_struck
  - idx=7 pid=-2 status=-61 tag=remnant_candidate
  - idx=8 pid=-2 status=-62 tag=excluded_hard_neighbor
  - idx=9 pid=2 status=-63 tag=remnant_candidate
  - idx=11 pid=2 status=-71 tag=remnant_candidate
  - idx=12 pid=-2 status=-71 tag=unknown

Filtered colour matches (tag → node_idx/pid, remnant_idx/pid):
  - tag=101  node 7(pid=-2)  remnant 9(pid=2)
  - tag=101  node 7(pid=-2)  remnant 11(pid=2)
  - tag=101  node 9(pid=2)  remnant 7(pid=-2)
  - tag=101  node 9(pid=2)  remnant 11(pid=2)
  - tag=101  node 11(pid=2)  remnant 7(pid=-2)
  - tag=101  node 11(pid=2)  remnant 9(pid=2)

## Example: unfiltered match that disappears after veto

(No event in this run had unfiltered > 0 and filtered = 0.)

## Validation

- **Struck veto set size:** typically 29.2 nodes (see above).
- **Veto reduces over-connection:** yes; filtered mean is lower than unfiltered because colour links through struck-side nodes are excluded.
- **Filtered remnant multiplicity distribution:** see table above.
- **Events with at least one filtered remnant connection:** 37.
- **One example of unfiltered match disappearing after veto:** see section above.
