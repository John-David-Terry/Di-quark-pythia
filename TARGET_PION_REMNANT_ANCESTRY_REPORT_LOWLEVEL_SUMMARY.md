# Target π⁻ remnant ancestry — low-level summary

## Command
```
scripts/generation/trace_target_pion_to_hard_vertex.py --labels ISRFSR_OFF --n-events 120
```

## Counts
- Accepted DIS events: 120
- Tagged π⁻ events: 37
- Classification counts: {'neither': 2, 'both': 28, 'hard_only': 7}
- Remnant path classification counts: {'no_remnant_path': 9, 'remnant_via_hard': 4, 'remnant_avoids_hard': 12, 'both_types': 12}

## One example per class

### remnant_only
(No example in this run.)

### hard_only
- event_number=2, Q²=49.82, x_bj=0.0059
- classification=hard_only, reached_remnant=False, reached_hard=True
- remnant_candidate_nodes: 4 entries
- path_to_first_remnant_hit: 0 steps
- path_to_first_hard_hit: 2 steps

### both
- event_number=1, Q²=25.76, x_bj=0.0174
- classification=both, reached_remnant=True, reached_hard=True
- remnant_candidate_nodes: 4 entries
- path_to_first_remnant_hit: 6 steps
- path_to_first_hard_hit: 3 steps

### neither
- event_number=0, Q²=28.17, x_bj=0.0058
- classification=neither, reached_remnant=False, reached_hard=False
- remnant_candidate_nodes: 4 entries
- path_to_first_remnant_hit: 0 steps
- path_to_first_hard_hit: 0 steps
