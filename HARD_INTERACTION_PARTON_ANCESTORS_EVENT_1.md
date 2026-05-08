# Hard-interaction parton ancestors (event_number = 1)

- **Tagged pion index:** 21
- **Path to first hard hit:** 3 steps
- **Path to first remnant hit:** 6 steps

---

## A. Hard-side parton ancestors

Parton-like nodes on `path_to_first_hard_hit`:

- **idx=8**  pid=1  status=-62  mothers=[6]  daughters=[]  tag=both_paths

## B. Remnant-side parton ancestors

Parton-like nodes on `path_to_first_remnant_hit`:

- **idx=8**  pid=1  status=-62  mothers=[6]  daughters=[6]  tag=both_paths
- **idx=6**  pid=1  status=-23  mothers=[3, 4]  daughters=[4]  tag=remnant_side
- **idx=4**  pid=1  status=-21  mothers=[7]  daughters=[7]  tag=remnant_side
- **idx=7**  pid=1  status=-61  mothers=[2, 0]  daughters=[]  tag=remnant_side

---

## Interpretation

- **First hard-side parton ancestor (on path):** idx=8 (pid=1, status=-62) — first parton where the path reaches the hard-interaction set.
- **First remnant-side parton ancestor (on path):** idx=7 (pid=1, status=-61) — parton where the path reaches the remnant-candidate set (beam/remnant-side quark).
- **Remnant-side path passes through hard-side path first:** Yes — the path to the remnant (pion→14→8→6→4→7) goes through hard node 8 and the struck line (6, 4) before reaching remnant parton 7.

---

## Validation

- Every printed node is parton-like: yes (filtered by abs(pid) in {1,2,3,4,5,6,21}).
- Every printed node appears on at least one reconstructed path: yes (all from path_to_first_hard_hit and/or path_to_first_remnant_hit).
- No hadrons or system nodes included: yes (only parton-like nodes listed).