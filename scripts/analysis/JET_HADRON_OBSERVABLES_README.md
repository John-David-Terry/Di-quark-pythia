# Jet–hadron transverse observables analysis

## Intent: Option A

**Jet transverse momentum = transverse momentum of the active incoming parton (k_in), not the outgoing parton after the hard scattering.**

So \( P_{\mathrm{jet}} = P_{\mathrm{proton}} - P_{\mathrm{remnant}} \) implies \( P_{\mathrm{jet}} = k_{\mathrm{in}} \). We do **not** use the outgoing parton momentum (k_out) for the jet. The language in this analysis refers to the **active incoming parton** or **active-incoming-parton transverse momentum** (k_in).

---

## Summary of definitions

### Jet (active incoming parton) identification

The **jet** is defined as the **active incoming parton**: the parton that enters the hard DIS scattering (momentum \( k_{\mathrm{in}} \) before absorbing the photon). It is reconstructed as follows:

1. From the event record we have the outgoing parton 4-momentum `k_out` (stored in the shards as `event_k_out`).
2. The **incoming** parton 4-momentum is \( k_{\mathrm{in}} = k_{\mathrm{out}} - q \), where \( q \) is the virtual photon 4-momentum.
3. The proton remnant is \( P_{\mathrm{rem}} = P_{\mathrm{proton}} - k_{\mathrm{in}} \) (lab frame).
4. The jet 4-momentum is \( P_{\mathrm{jet}} = P_{\mathrm{proton}} - P_{\mathrm{rem}} = k_{\mathrm{in}} \) in the lab frame (so the jet is the active incoming parton).
5. We boost to the **Breit frame** using the same Lorentz transform as the rest of the analysis. In that frame, \( P_{\mathrm{jet}}^{\mathrm{Breit}} \) is the jet 4-momentum.
6. **Jet transverse momentum**: \( \vec{p}_{T,\mathrm{jet}} = (P_{\mathrm{jet},x}^{\mathrm{Breit}}, P_{\mathrm{jet},y}^{\mathrm{Breit}}) \). Both the 2-vector and its magnitude are saved.

No jet algorithm is used; the “jet” is defined solely as the transverse momentum of this active incoming parton (\( k_{\mathrm{in}} \)).

### Hadron identification

The **hadron** is the **target-fragmentation leading charged pion**: the highest-energy hadron in the Breit-frame **current hemisphere** (i.e. \( p_z^{\mathrm{Breit}} > 0 \)) that has \( |\mathrm{pid}| = 211 \) (π⁺ or π⁻). This is the same convention as in the existing pTrel and transverse observables analysis. The hadron transverse momentum is \( \vec{p}_{T,h} = (P_{h,x}^{\mathrm{Breit}}, P_{h,y}^{\mathrm{Breit}}) \) in the same Breit frame. Both the 2-vector and its magnitude are saved.

### Frame

All quantities (jet and hadron 4-momenta and transverse vectors) are evaluated in the **Breit frame**. The same frame and the same DIS cuts (\( x \), \( Q^2 \), \( x_L \)) as in the existing analysis are used. No mixing of lab and Breit frame.

---

## Observables

All observables use the **same** Breit-frame transverse vectors \( \vec{p}_{T,\mathrm{jet}} \) and \( \vec{p}_{T,h} \). The magnitude is taken **only after** vector addition or subtraction. **Do not use** \( |\vec{p}_{T,\mathrm{jet}}| + |\vec{p}_{T,h}| \) and **do not use** \( |\vec{p}_{T,\mathrm{jet}}| - |\vec{p}_{T,h}| \).

1. **Observable 1: angle between the two transverse vectors**  
   - 2D angle in the transverse plane.  
   - **Convention**: \( [0, \pi] \) rad; \( 0 \) = aligned, \( \pi \) = back-to-back.  
   - Computed as \( \arccos(\hat{p}_{T,\mathrm{jet}} \cdot \hat{p}_{T,h}) \) with clipping for numerical stability. Events with \( |\vec{p}_{T,\mathrm{jet}}| \) or \( |\vec{p}_{T,h}| \) below a small threshold are given `nan` for the angle (excluded from the angle histogram).

2. **Observable 2: first form the vector sum, then take the magnitude**  
   - \( |\vec{p}_{T,\mathrm{jet}} + \vec{p}_{T,h}| \) (GeV).

3. **Observable 3: first form the vector difference, then take the magnitude**  
   - \( |\vec{p}_{T,\mathrm{jet}} - \vec{p}_{T,h}| \) (GeV).

---

## Running

From the project root:

```bash
python scripts/analysis/analyze_jet_hadron_transverse_observables.py
```

Requires the same raw shards as the rest of the analysis (`pythia_finalstate_raw/ISRFSR_OFF`, `ISRFSR_ON`).

---

## Outputs

- **Plots** (project root):  
  - `jet_hadron_transverse_angle_target_leading_pion_Breit_comparison.pdf`  
  - `jet_hadron_transverse_sum_mag_target_leading_pion_Breit_comparison.pdf`  
  - `jet_hadron_transverse_diff_mag_target_leading_pion_Breit_comparison.pdf`  
  Each compares ISR/FSR off vs on.

- **Per-event arrays** (for later inspection):  
  - `jet_hadron_transverse_angle_*.npy`, `*_sum_mag_*.npy`, `*_diff_mag_*.npy`  
  - `jet_hadron_transverse_event_ids_ISRFSR_OFF.npy`, `*_ON.npy`  
  - `jet_hadron_transverse_pT_jet_vec_*.npy`, `*_pT_hadron_vec_*.npy`

- **Debug**:  
  - First `DEBUG_N_EVENTS` events: printed event id, active-incoming-parton (jet) and hadron momenta, and the three observables.  
  - A few **sanity-check** cases printed: nearly parallel, nearly back-to-back, and one pT very small.  
  - `jet_hadron_transverse_debug_sample_ISRFSR_OFF.npy`, `*_ON.npy`: first 100 events, columns `[shard_idx, local_idx, pT_jet_x, pT_jet_y (active incoming parton), pT_hadron_x, pT_hadron_y, angle, sum_mag, diff_mag]`.

- **Metadata**: `jet_hadron_transverse_metadata.txt` (definitions, Option A, observable wording, and “do not use” sentence).
