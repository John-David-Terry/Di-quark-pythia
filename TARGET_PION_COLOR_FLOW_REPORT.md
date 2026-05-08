# Target π⁻ colour-flow remnant diagnostic

Label: ISRFSR_OFF

- Total generated events: 120
- Accepted DIS events: 120
- Tagged π⁻ events (hardest in Breit target): 37

## Summary table

### Mother-graph remnant paths

- With mother-graph remnant path: 28
- With no mother-graph remnant path: 9

### Colour-flow remnant connections

- has_color_flow_to_remnant = True: 37
- has_color_flow_to_remnant = False: 0

### Among events with NO mother-graph remnant path

- Count: 9
- With colour-flow connection to remnant: 9
- With no remnant indication (mothers + colour): 0

### Remnant path classification (from main tracer, if available)

- : 37

## Example: no mother-graph remnant path, but colour-flow remnant connection

- event_number = 0, global_event_index = 4
- Q² = 28.17 GeV², x_bj = 0.0058
- Tagged π⁻: idx=36, status=91, E_breit=2.239, pz_breit=1.467
- has_mother_remnant_path = False
- has_color_flow_to_remnant = True
- has_mother_or_color_connection_to_remnant = True
- color_flow_diagnostic_note = no mother-graph remnant path; colour-flow match to remnant

### Remnant candidate nodes
- idx=4 pid=4 status=-21 tag=excluded_hard_neighbor mothers=[7] daughters=[5, 6]
- idx=6 pid=4 status=-23 tag=excluded_struck mothers=[3, 4] daughters=[8]
- idx=7 pid=4 status=-61 tag=remnant_candidate mothers=[2, 0] daughters=[4]
- idx=8 pid=4 status=-62 tag=excluded_hard_neighbor mothers=[6] daughters=[11, 12, 13, 14, 15, 16, 17, 18]

### Hard-interaction nodes (first 10)

- hard_nodes = [3, 4, 6, 8]

### Colour matches

- tag=101  node_idx=4(pid=4)  remnant_idx=6(pid=4)
- tag=101  node_idx=4(pid=4)  remnant_idx=7(pid=4)
- tag=101  node_idx=4(pid=4)  remnant_idx=8(pid=4)
- tag=101  node_idx=6(pid=4)  remnant_idx=4(pid=4)
- tag=101  node_idx=6(pid=4)  remnant_idx=7(pid=4)
- tag=101  node_idx=6(pid=4)  remnant_idx=8(pid=4)
- tag=101  node_idx=7(pid=4)  remnant_idx=4(pid=4)
- tag=101  node_idx=7(pid=4)  remnant_idx=6(pid=4)
- tag=101  node_idx=7(pid=4)  remnant_idx=8(pid=4)
- tag=101  node_idx=8(pid=4)  remnant_idx=4(pid=4)

### Node colour table (subset)

- idx=0 pid=90 status=-11 mothers=[0] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=1 pid=11 status=-12 mothers=[0] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=2 pid=2212 status=-12 mothers=[0] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=3 pid=11 status=-21 mothers=[1, 0] daughters=[5, 6] col=0 acol=0 parton_like=False in_hard=True in_remnant=False
- idx=4 pid=4 status=-21 mothers=[7] daughters=[5, 6] col=101 acol=0 parton_like=True in_hard=True in_remnant=False
- idx=5 pid=11 status=23 mothers=[3, 4] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=6 pid=4 status=-23 mothers=[3, 4] daughters=[8] col=101 acol=0 parton_like=True in_hard=True in_remnant=False
- idx=7 pid=4 status=-61 mothers=[2, 0] daughters=[4] col=101 acol=0 parton_like=True in_hard=False in_remnant=True
- idx=8 pid=4 status=-62 mothers=[6] daughters=[11, 12, 13, 14, 15, 16, 17, 18] col=101 acol=0 parton_like=True in_hard=True in_remnant=False
- idx=9 pid=2203 status=-63 mothers=[2, 0] daughters=[11, 12, 13, 14, 15, 16, 17, 18] col=0 acol=101 parton_like=False in_hard=False in_remnant=False
- idx=11 pid=411 status=-83 mothers=[8, 9] daughters=[25, 26, 27] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=12 pid=111 status=-83 mothers=[8, 9] daughters=[28, 29] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=13 pid=-211 status=83 mothers=[8, 9] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=14 pid=2224 status=-83 mothers=[8, 9] daughters=[19, 20] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=15 pid=-211 status=83 mothers=[8, 9] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=16 pid=-2212 status=83 mothers=[8, 9] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=17 pid=2212 status=84 mothers=[8, 9] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=18 pid=213 status=-84 mothers=[8, 9] daughters=[21, 22] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=25 pid=-13 status=91 mothers=[11, 0] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=26 pid=14 status=91 mothers=[11, 0] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False

## Example: both mother-graph and colour-flow remnant connection

- event_number = 1, global_event_index = 11
- Q² = 25.76 GeV², x_bj = 0.0174
- Tagged π⁻: idx=21, status=91, E_breit=4.567, pz_breit=0.014
- has_mother_remnant_path = True
- has_color_flow_to_remnant = True
- has_mother_or_color_connection_to_remnant = True
- color_flow_diagnostic_note = colour-flow match to remnant

### Remnant candidate nodes
- idx=4 pid=1 status=-21 tag=excluded_hard_neighbor mothers=[7] daughters=[5, 6]
- idx=6 pid=1 status=-23 tag=excluded_struck mothers=[3, 4] daughters=[8]
- idx=7 pid=1 status=-61 tag=remnant_candidate mothers=[2, 0] daughters=[4]
- idx=8 pid=1 status=-62 tag=excluded_hard_neighbor mothers=[6] daughters=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

### Hard-interaction nodes (first 10)

- hard_nodes = [3, 4, 6, 8]

### Colour matches

- tag=101  node_idx=4(pid=1)  remnant_idx=6(pid=1)
- tag=101  node_idx=4(pid=1)  remnant_idx=7(pid=1)
- tag=101  node_idx=4(pid=1)  remnant_idx=8(pid=1)
- tag=101  node_idx=6(pid=1)  remnant_idx=4(pid=1)
- tag=101  node_idx=6(pid=1)  remnant_idx=7(pid=1)
- tag=101  node_idx=6(pid=1)  remnant_idx=8(pid=1)
- tag=101  node_idx=7(pid=1)  remnant_idx=4(pid=1)
- tag=101  node_idx=7(pid=1)  remnant_idx=6(pid=1)
- tag=101  node_idx=7(pid=1)  remnant_idx=8(pid=1)
- tag=101  node_idx=8(pid=1)  remnant_idx=4(pid=1)

### Node colour table (subset)

- idx=0 pid=90 status=-11 mothers=[0] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=1 pid=11 status=-12 mothers=[0] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=2 pid=2212 status=-12 mothers=[0] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=3 pid=11 status=-21 mothers=[1, 0] daughters=[5, 6] col=0 acol=0 parton_like=False in_hard=True in_remnant=False
- idx=4 pid=1 status=-21 mothers=[7] daughters=[5, 6] col=101 acol=0 parton_like=True in_hard=True in_remnant=False
- idx=5 pid=11 status=23 mothers=[3, 4] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=6 pid=1 status=-23 mothers=[3, 4] daughters=[8] col=101 acol=0 parton_like=True in_hard=True in_remnant=False
- idx=7 pid=1 status=-61 mothers=[2, 0] daughters=[4] col=101 acol=0 parton_like=True in_hard=False in_remnant=True
- idx=8 pid=1 status=-62 mothers=[6] daughters=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20] col=101 acol=0 parton_like=True in_hard=True in_remnant=False
- idx=9 pid=2203 status=-63 mothers=[2, 0] daughters=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20] col=0 acol=101 parton_like=False in_hard=False in_remnant=False
- idx=11 pid=-211 status=83 mothers=[8, 9] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=12 pid=3222 status=-83 mothers=[8, 9] daughters=[27, 28] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=13 pid=-3212 status=-83 mothers=[8, 9] daughters=[29, 30] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=14 pid=-213 status=-83 mothers=[8, 9] daughters=[21, 22] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=15 pid=213 status=-84 mothers=[8, 9] daughters=[23, 24] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=16 pid=-211 status=84 mothers=[8, 9] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=17 pid=211 status=84 mothers=[8, 9] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=18 pid=-211 status=84 mothers=[8, 9] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=19 pid=2212 status=84 mothers=[8, 9] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False
- idx=20 pid=211 status=84 mothers=[8, 9] daughters=[] col=0 acol=0 parton_like=False in_hard=False in_remnant=False

## Example: neither mother-graph nor colour-flow remnant connection
(No example in this run.)
