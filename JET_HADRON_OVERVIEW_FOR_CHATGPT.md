## Project and Code Overview (for ChatGPT)

This repository uses PYTHIA 8 to generate deep inelastic scattering (DIS) \(e^- p\) events at EIC-like beam energies and then analyzes hadron and “jet” transverse-momentum observables in a Breit-like frame.

### Event generation

- **Generator script**: `scripts/generation/generate_events_raw.py`
- **Beams**:
  - Electron: 18 GeV
  - Proton: 275 GeV
- **Process**:
  - DIS via `WeakBosonExchange:ff2ff(t:gmZ) = on`
  - `PhaseSpace:Q2Min = 16.0` GeV\(^2\)
  - Hadronization on (`HadronLevel:all = on`)
- **Configurations (labels)**:
  - `ISRFSR_ON`: ISR/FSR on, colour reconnection off, beams (e, p)
  - `ISRFSR_OFF`: ISR/FSR off, colour reconnection off, beams (e, p)
  - `ETA_ON_CRON`: ISR/FSR on, colour reconnection on, beams (p, e) for η studies
- **Output format**:
  - Sharded NumPy files under `pythia_finalstate_raw/<LABEL>/shard_XXXXXX/`
  - Per event:
    - `event_e_in.npy`: incoming electron 4-vector
    - `event_p_in.npy`: incoming proton 4-vector
    - `event_e_sc.npy`: scattered electron 4-vector
    - `event_k_out.npy`: struck-quark candidate 4-vector
    - `pid.npy`, `p4.npy`, `offsets.npy`: all final-state particles (PID + 4-momenta) with indexing
    - `meta.json`: PYTHIA and shard metadata

### Core shard-based analysis

- **Main analysis module**: `src/diquark/analyze_events_raw.py`
- Loads cached shards from `pythia_finalstate_raw/` (no live PYTHIA).
- Applies DIS cuts in \(x\) and \(Q^2\), including a `flip_z` helper to reconcile different beam orders.
- Constructs a **Breit-like Lorentz transform** (`build_LT`) that:
  - Rotates so that the virtual photon transverse momentum \(q_T\) lies along +x.
  - Boosts so that in the transformed frame \(q^0 \approx 0\) and \(q_T \approx 0\).
  - Preserves Minkowski norms and 4-vector scalar products (explicitly checked in diagnostics).
- Produces baseline observables such as:
  - Hadron pseudorapidity distributions vs EIC hardware regions.
  - \(p_{T, \text{rel}}\) of pions relative to the remnant axis, with Gaussian fits.

### Jet–hadron transverse observables

- **Analysis script**: `scripts/analysis/analyze_jet_hadron_transverse_observables.py`
- Loads `ISRFSR_ON` and `ISRFSR_OFF` shards and applies the same \(x, Q, x_L\) cuts and Breit-like transform as the core analysis.

#### Hadron definition

- “Target-leading pion”:
  - Highest-energy hadron in the **target hemisphere** with
    - `pz_breit > 0`
    - `|pid| == 211` (charged pions)
  - Hadron transverse momentum: \(\vec{p}_{T,h} = (p_x, p_y)\) in the Breit frame.

#### Jet (incoming parton) definition — original vs. fixed

- **Original (now deprecated) definition**:
  - \(k_\text{in,old} = k_\text{out} - q\), where \(k_\text{out}\) is taken from PYTHIA status=23 (or similar) and \(q = l - l'\) is the virtual photon.
  - Detailed audits showed this is **unphysical**:
    - \(k_\text{in,old}^2 \sim -200\ \text{GeV}^2\) (deeply spacelike).
    - Sometimes negative energy.
    - Produced a spurious \(\mathcal{O}(2\ \text{GeV})\) transverse momentum in the Breit frame and strong x-axis locking of the “jet” azimuth.

- **Current (fixed) benchmark collinear definition**:
  - \(k_\text{in,ref} = x P\) in the LAB frame, where
    - \(x\) is Bjorken \(x = Q^2 / (2 P \cdot q)\),
    - \(P\) is the proton 4-vector.
  - Properties:
    - Timelike with \(k_\text{in,ref}^2 = x^2 m_p^2\).
    - Positive energy.
    - After boosting to the Breit frame, has essentially zero transverse momentum:
      - \(\langle |p_{T,\text{jet}}| \rangle \sim 3 \times 10^{-4}\ \text{GeV}\) instead of \(\sim 2.3\ \text{GeV}\).
  - This represents the **collinear parton model** limit. Any observed transverse momentum in the final state then comes from radiation, hadronization, and fragmentation.

#### Jet–hadron observables (Breit frame)

With the fixed jet definition (\(\vec{p}_{T,\text{jet}} \approx 0\)), the script defines three observables:

1. **Angle observable \(\phi_{hJ}\)**  
   Angle between the two transverse vectors:
   \[
   \phi_{hJ} = \angle(\vec{p}_{T,\text{jet}}, \vec{p}_{T,h}) \in [0, \pi]
   \]
   Computed via a numerically stable 2D dot-product-based arccos with clipping.

2. **Sum magnitude \(\bar{P}_{hJ}\)**  
   Form the vector sum first, then take the magnitude:
   \[
   \bar{P}_{hJ} = \big|\vec{p}_{T,\text{jet}} + \vec{p}_{T,h}\big|
   \]

3. **Difference magnitude \(\Delta P_{hJ}\)**  
   Form the vector difference first, then take the magnitude:
   \[
   \Delta P_{hJ} = \big|\vec{p}_{T,\text{jet}} - \vec{p}_{T,h}\big|
   \]

In all cases, the magnitude is taken **only after** vector addition or subtraction; the code never uses \(|p_T^\text{jet}| + |p_T^h|\) or \(|p_T^\text{jet}| - |p_T^h|\).

The analysis produces ISR/FSR ON vs. OFF comparison PDFs:
- `jet_hadron_transverse_angle_target_leading_pion_Breit_comparison.pdf`
- `jet_hadron_transverse_sum_mag_target_leading_pion_Breit_comparison.pdf`
- `jet_hadron_transverse_diff_mag_target_leading_pion_Breit_comparison.pdf`
as well as a jet-azimuth diagnostic:
- `jet_hadron_transverse_phi_jet_Breit_comparison.pdf`.

### Diagnostics and audits

- **Breit frame audit** (`scripts/analysis/breit_frame_audit.py`):
  - Verifies that the Lorentz transform `build_LT` is a proper Lorentz transformation:
    - Checks \(L^T g L \approx g\) (with \(g = \text{diag}(1,-1,-1,-1)\)).
    - Confirms invariance of norms and dot products for relevant 4-vectors.
  - Examines the properties of the Breit-like frame:
    - \(q_T \approx 0\), \(q^0 \approx 0\),
    - Characterizes the proton transverse momentum \(P_T\).
  - Compares the unphysical \(k_\text{in,old} = k_\text{out} - q\) and the benchmark \(x P\) in both LAB and Breit.

- **Controlled k\_in fix test** (`scripts/analysis/test_kin_fix.py`):
  - Compares:
    - OLD: \(k_\text{in,old} = k_\text{out} - q\),
    - REF: \(k_\text{in,ref} = x P\),
    for both ISR/FSR OFF and ON.
  - Shows that the large \(\mathcal{O}(2\ \text{GeV})\) transverse effect and x-axis locking disappear with the benchmark definition.

### Short summary paragraph (copy-ready)

The project uses PYTHIA 8 to generate DIS \(e^- p\) events at 18 × 275 GeV, stores them in sharded NumPy format, and analyzes them in a Breit-like frame. We originally defined the incoming parton as \(k_\text{in} = k_\text{out} - q\), but detailed Lorentz and kinematic audits showed this was unphysical (deeply off-shell, spurious \(\mathcal{O}(2\ \text{GeV})\) transverse momentum). We replaced it with a benchmark collinear definition \(k_\text{in} = x P\), which is timelike, has positive energy, and essentially zero transverse momentum in the Breit frame. Using this fixed jet definition, we compute three jet–hadron observables (angle, sum magnitude, difference magnitude of transverse momenta) for a target-leading pion, and we produce ISR/FSR ON vs. OFF comparison plots along with extensive diagnostics of the Breit transform and parton kinematics.

