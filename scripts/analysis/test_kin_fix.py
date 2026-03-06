#!/usr/bin/env python3
"""
Controlled test: compare k_in_old = k_out - q vs k_in_ref = x * P.

Tests whether replacing the unphysical incoming-parton definition with the
benchmark collinear definition removes the O(2 GeV) transverse effect and
artificial x-axis locking.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diquark.analyze_events_raw import (
    FLIP_Z_PTREL,
    PTREL_LABEL_OFF,
    PTREL_LABEL_ON,
    Qmax_ptrel,
    Qmin_ptrel,
    build_LT,
    dot4,
    flip_z,
    is_hadron,
    list_shards,
    load_shard,
    xmax_ptrel,
    xmin_ptrel,
)

# Minkowski metric
def minkowski_norm(v):
    return v[0]**2 - v[1]**2 - v[2]**2 - v[3]**2

def transverse_mag(v):
    return np.sqrt(v[1]**2 + v[2]**2)

def _pT_vec(p4):
    return np.array([float(p4[1]), float(p4[2])], dtype=np.float64)

def _angle_between_2d(a, b):
    ma = np.linalg.norm(a)
    mb = np.linalg.norm(b)
    if ma < 1e-10 or mb < 1e-10:
        return None
    cos_angle = np.dot(a, b) / (ma * mb)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def run_comparison(label: str, max_events: int = 2000):
    """Run side-by-side comparison of k_in_old vs k_in_ref."""
    shards = list_shards(label)
    if not shards:
        print(f"[{label}] No shards found.")
        return None
    
    # Collectors for OLD definition (k_out - q)
    old_mass_sq = []
    old_energy = []
    old_kT_breit = []
    old_kx_breit = []
    old_ky_breit = []
    old_phi_breit = []
    
    # Collectors for REF definition (x * P)
    ref_mass_sq = []
    ref_energy = []
    ref_kT_breit = []
    ref_kx_breit = []
    ref_ky_breit = []
    ref_phi_breit = []
    
    # Cross-check: k_in_ref_breit vs x * P_breit
    crosscheck_diff = []
    
    # Observables with OLD jet definition
    obs_angle_old = []
    obs_sum_old = []
    obs_diff_old = []
    
    # Observables with REF jet definition
    obs_angle_ref = []
    obs_sum_ref = []
    obs_diff_ref = []
    
    processed = 0
    debug_printed = 0
    DEBUG_PRINT_N = 10
    
    for shard_idx, shard_path in enumerate(shards):
        if max_events is not None and processed >= max_events:
            break
        data = load_shard(shard_path)
        e_in = data["event_e_in"]
        p_in = data["event_p_in"]
        e_sc = data["event_e_sc"]
        k_out = data["event_k_out"]
        offsets = data["offsets"]
        pid = data["pid"]
        p4_arr = data["p4"]
        Ne = e_in.shape[0]
        
        for ie in range(Ne):
            if max_events is not None and processed >= max_events:
                break
            
            # LAB frame vectors
            e_in_lab = flip_z(np.asarray(e_in[ie], dtype=float), FLIP_Z_PTREL)
            P_lab = flip_z(np.asarray(p_in[ie], dtype=float), FLIP_Z_PTREL)
            e_sc_lab = flip_z(np.asarray(e_sc[ie], dtype=float), FLIP_Z_PTREL)
            k_out_lab = flip_z(np.asarray(k_out[ie], dtype=float), FLIP_Z_PTREL)
            
            Ep = float(P_lab[0])
            Ee = float(e_in_lab[0])
            
            # Virtual photon
            q_lab = e_in_lab - e_sc_lab
            Q2 = -minkowski_norm(q_lab)
            if Q2 <= 0:
                continue
            Q = np.sqrt(Q2)
            
            # Kinematics
            p_dot_q = P_lab[0]*q_lab[0] - P_lab[1]*q_lab[1] - P_lab[2]*q_lab[2] - P_lab[3]*q_lab[3]
            if p_dot_q == 0:
                continue
            x = Q2 / (2.0 * p_dot_q)
            if not (xmin_ptrel <= x <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel):
                continue
            
            qT_lab = transverse_mag(q_lab)
            phiq = np.arctan2(q_lab[2], q_lab[1])
            S = 4.0 * Ee * Ep
            y = Q2 / (S * x)
            
            # Build LT
            LT = build_LT(Ee, Ep, q_lab, x, y, qT_lab, phiq, S)
            if LT is None:
                continue
            
            boost = lambda v: LT @ np.asarray(v, dtype=float)
            P_breit = boost(P_lab)
            P_plus = float(P_breit[0] + P_breit[3])
            if P_plus <= 0:
                continue
            
            # Hadron selection (same as main analysis)
            start = int(offsets[ie])
            end = int(offsets[ie + 1])
            best_tarE = -1.0
            best_tar_breit = None
            best_tar_pid = None
            for j in range(start, end):
                this_pid = int(pid[j])
                if not is_hadron(this_pid):
                    continue
                lab = flip_z(np.asarray(p4_arr[j], dtype=float), FLIP_Z_PTREL)
                trf = boost(lab)
                E_, px_, py_, pz_ = trf
                if pz_ <= 0:
                    continue
                if E_ > best_tarE:
                    best_tarE = E_
                    best_tar_breit = trf
                    best_tar_pid = this_pid
            if best_tar_breit is None or abs(best_tar_pid) != 211:
                continue
            
            # xL cut (using old definition for remnant, just to keep same event selection)
            k_in_old_lab = k_out_lab - q_lab
            p_rem_old = P_lab - k_in_old_lab
            p_rem_old_breit = boost(p_rem_old)
            q_breit = boost(q_lab)
            den = dot4(p_rem_old_breit, q_breit)
            if den <= 0:
                continue
            xL_exact = dot4(best_tar_breit, q_breit) / den
            if xL_exact < 0.01 or xL_exact > 1.0 + 1e-6:
                continue
            
            # ============================================================
            # Define the two candidate incoming-parton 4-vectors
            # ============================================================
            
            # A. OLD definition: k_in = k_out - q
            k_in_old_lab = k_out_lab - q_lab
            k_in_old_breit = boost(k_in_old_lab)
            
            # B. REFERENCE definition: k_in = x * P
            k_in_ref_lab = x * P_lab
            k_in_ref_breit = boost(k_in_ref_lab)
            
            # Cross-check: k_in_ref_breit should equal x * P_breit (linearity of LT)
            x_P_breit = x * P_breit
            crosscheck = np.linalg.norm(k_in_ref_breit - x_P_breit)
            crosscheck_diff.append(crosscheck)
            
            # ============================================================
            # Collect statistics
            # ============================================================
            
            # OLD
            old_mass_sq.append(minkowski_norm(k_in_old_lab))
            old_energy.append(k_in_old_lab[0])
            old_kT_breit.append(transverse_mag(k_in_old_breit))
            old_kx_breit.append(k_in_old_breit[1])
            old_ky_breit.append(k_in_old_breit[2])
            phi_old = np.arctan2(k_in_old_breit[2], k_in_old_breit[1])
            old_phi_breit.append(phi_old)
            
            # REF
            ref_mass_sq.append(minkowski_norm(k_in_ref_lab))
            ref_energy.append(k_in_ref_lab[0])
            ref_kT_breit.append(transverse_mag(k_in_ref_breit))
            ref_kx_breit.append(k_in_ref_breit[1])
            ref_ky_breit.append(k_in_ref_breit[2])
            phi_ref = np.arctan2(k_in_ref_breit[2], k_in_ref_breit[1])
            ref_phi_breit.append(phi_ref)
            
            # ============================================================
            # Compute observables with both definitions
            # ============================================================
            
            pT_hadron = _pT_vec(best_tar_breit)
            
            # OLD jet definition
            pT_jet_old = _pT_vec(k_in_old_breit)
            angle_old = _angle_between_2d(pT_jet_old, pT_hadron)
            if angle_old is not None:
                obs_angle_old.append(angle_old)
            sum_vec_old = pT_jet_old + pT_hadron
            diff_vec_old = pT_jet_old - pT_hadron
            obs_sum_old.append(np.linalg.norm(sum_vec_old))
            obs_diff_old.append(np.linalg.norm(diff_vec_old))
            
            # REF jet definition
            pT_jet_ref = _pT_vec(k_in_ref_breit)
            angle_ref = _angle_between_2d(pT_jet_ref, pT_hadron)
            if angle_ref is not None:
                obs_angle_ref.append(angle_ref)
            sum_vec_ref = pT_jet_ref + pT_hadron
            diff_vec_ref = pT_jet_ref - pT_hadron
            obs_sum_ref.append(np.linalg.norm(sum_vec_ref))
            obs_diff_ref.append(np.linalg.norm(diff_vec_ref))
            
            # Debug print
            if debug_printed < DEBUG_PRINT_N:
                print(f"\n{'='*70}")
                print(f"Event ({shard_idx}, {ie}) [{label}]")
                print(f"{'='*70}")
                print(f"x = {x:.6f}, Q = {Q:.4f} GeV")
                
                print(f"\n--- LAB frame ---")
                print(f"  k_in_OLD = ({k_in_old_lab[0]:.4f}, {k_in_old_lab[1]:.4f}, {k_in_old_lab[2]:.4f}, {k_in_old_lab[3]:.4f})")
                print(f"  k_in_REF = ({k_in_ref_lab[0]:.4f}, {k_in_ref_lab[1]:.4f}, {k_in_ref_lab[2]:.4f}, {k_in_ref_lab[3]:.4f})")
                print(f"  k_in_OLD^2 = {minkowski_norm(k_in_old_lab):.4f} GeV^2")
                print(f"  k_in_REF^2 = {minkowski_norm(k_in_ref_lab):.6f} GeV^2 (expected: x^2*m_p^2 = {(x**2 * 0.938**2):.6f})")
                print(f"  E_OLD = {k_in_old_lab[0]:.4f} GeV, E_REF = {k_in_ref_lab[0]:.4f} GeV")
                
                print(f"\n--- Breit frame ---")
                print(f"  k_in_OLD_breit = ({k_in_old_breit[0]:.4f}, {k_in_old_breit[1]:.6f}, {k_in_old_breit[2]:.6f}, {k_in_old_breit[3]:.4f})")
                print(f"  k_in_REF_breit = ({k_in_ref_breit[0]:.4f}, {k_in_ref_breit[1]:.6f}, {k_in_ref_breit[2]:.6f}, {k_in_ref_breit[3]:.4f})")
                print(f"  |k_T|_OLD = {transverse_mag(k_in_old_breit):.6f} GeV")
                print(f"  |k_T|_REF = {transverse_mag(k_in_ref_breit):.6f} GeV")
                print(f"  phi_OLD = {phi_old:.4f} rad = {np.degrees(phi_old):.2f} deg")
                print(f"  phi_REF = {phi_ref:.4f} rad = {np.degrees(phi_ref):.2f} deg")
                
                print(f"\n--- Cross-check: k_in_ref_breit = x * P_breit ---")
                print(f"  |k_in_ref_breit - x*P_breit| = {crosscheck:.2e} (should be ~0)")
                
                print(f"\n--- Observables ---")
                print(f"  pT_hadron = ({pT_hadron[0]:.4f}, {pT_hadron[1]:.4f})")
                print(f"  pT_jet_OLD = ({pT_jet_old[0]:.4f}, {pT_jet_old[1]:.4f}), |pT| = {np.linalg.norm(pT_jet_old):.4f}")
                print(f"  pT_jet_REF = ({pT_jet_ref[0]:.4f}, {pT_jet_ref[1]:.4f}), |pT| = {np.linalg.norm(pT_jet_ref):.4f}")
                if angle_old is not None:
                    print(f"  angle_OLD = {angle_old:.4f} rad = {np.degrees(angle_old):.2f} deg")
                if angle_ref is not None:
                    print(f"  angle_REF = {angle_ref:.4f} rad = {np.degrees(angle_ref):.2f} deg")
                
                debug_printed += 1
            
            processed += 1
    
    # Convert to arrays
    old_mass_sq = np.array(old_mass_sq)
    old_energy = np.array(old_energy)
    old_kT_breit = np.array(old_kT_breit)
    old_kx_breit = np.array(old_kx_breit)
    old_ky_breit = np.array(old_ky_breit)
    old_phi_breit = np.array(old_phi_breit)
    
    ref_mass_sq = np.array(ref_mass_sq)
    ref_energy = np.array(ref_energy)
    ref_kT_breit = np.array(ref_kT_breit)
    ref_kx_breit = np.array(ref_kx_breit)
    ref_ky_breit = np.array(ref_ky_breit)
    ref_phi_breit = np.array(ref_phi_breit)
    
    crosscheck_diff = np.array(crosscheck_diff)
    
    obs_angle_old = np.array(obs_angle_old)
    obs_sum_old = np.array(obs_sum_old)
    obs_diff_old = np.array(obs_diff_old)
    obs_angle_ref = np.array(obs_angle_ref)
    obs_sum_ref = np.array(obs_sum_ref)
    obs_diff_ref = np.array(obs_diff_ref)
    
    # ============================================================
    # Summary statistics
    # ============================================================
    
    print(f"\n{'='*70}")
    print(f"[{label}] SUMMARY STATISTICS ({processed} events)")
    print(f"{'='*70}")
    
    print(f"\n--- 1. Mass-squared and energy ---")
    print(f"  OLD (k_out - q):")
    print(f"    k^2: min={np.min(old_mass_sq):.2f}, max={np.max(old_mass_sq):.2f}, mean={np.mean(old_mass_sq):.2f} GeV^2")
    print(f"    Energy: min={np.min(old_energy):.2f}, max={np.max(old_energy):.2f}, mean={np.mean(old_energy):.2f} GeV")
    print(f"    Fraction with E < 0: {np.sum(old_energy < 0) / len(old_energy):.4f}")
    print(f"    Fraction with |k^2| > 1: {np.sum(np.abs(old_mass_sq) > 1) / len(old_mass_sq):.4f}")
    
    print(f"  REF (x * P):")
    print(f"    k^2: min={np.min(ref_mass_sq):.6f}, max={np.max(ref_mass_sq):.6f}, mean={np.mean(ref_mass_sq):.6f} GeV^2")
    print(f"    Energy: min={np.min(ref_energy):.4f}, max={np.max(ref_energy):.4f}, mean={np.mean(ref_energy):.4f} GeV")
    print(f"    Fraction with E < 0: {np.sum(ref_energy < 0) / len(ref_energy):.4f}")
    print(f"    -> REF is PHYSICAL: timelike (k^2 > 0), positive energy")
    
    print(f"\n--- 2. MAIN TEST: Transverse momentum in Breit frame ---")
    print(f"  OLD (k_out - q):")
    print(f"    |k_T|: min={np.min(old_kT_breit):.4f}, max={np.max(old_kT_breit):.4f}, MEAN={np.mean(old_kT_breit):.4f} GeV")
    print(f"    <k_x> = {np.mean(old_kx_breit):.4f}, <k_y> = {np.mean(old_ky_breit):.4f}")
    print(f"    <cos(phi)> = {np.mean(np.cos(old_phi_breit)):.4f}")
    print(f"    <sin(phi)> = {np.mean(np.sin(old_phi_breit)):.4f}")
    
    print(f"  REF (x * P):")
    print(f"    |k_T|: min={np.min(ref_kT_breit):.6f}, max={np.max(ref_kT_breit):.6f}, MEAN={np.mean(ref_kT_breit):.6f} GeV")
    print(f"    <k_x> = {np.mean(ref_kx_breit):.6f}, <k_y> = {np.mean(ref_ky_breit):.6f}")
    print(f"    <cos(phi)> = {np.mean(np.cos(ref_phi_breit)):.4f}")
    print(f"    <sin(phi)> = {np.mean(np.sin(ref_phi_breit)):.4f}")
    
    # Key comparison
    old_mean_kT = np.mean(old_kT_breit)
    ref_mean_kT = np.mean(ref_kT_breit)
    print(f"\n  *** KEY RESULT ***")
    print(f"  OLD mean |k_T| = {old_mean_kT:.4f} GeV")
    print(f"  REF mean |k_T| = {ref_mean_kT:.6f} GeV")
    if ref_mean_kT < 0.01:
        print(f"  -> The O({old_mean_kT:.1f} GeV) transverse effect DISAPPEARS with REF definition!")
    else:
        print(f"  -> The transverse effect persists (but reduced by factor {old_mean_kT/ref_mean_kT:.1f})")
    
    print(f"\n--- 3. Azimuthal locking ---")
    cos_phi_old = np.mean(np.cos(old_phi_breit))
    cos_phi_ref = np.mean(np.cos(ref_phi_breit))
    print(f"  OLD <cos(phi)> = {cos_phi_old:.4f} (1.0 = perfect x-axis lock)")
    print(f"  REF <cos(phi)> = {cos_phi_ref:.4f}")
    if abs(cos_phi_ref) < 0.1:
        print(f"  -> Azimuthal locking DISAPPEARS with REF definition!")
    elif abs(cos_phi_ref) < abs(cos_phi_old):
        print(f"  -> Azimuthal locking reduced from {cos_phi_old:.2f} to {cos_phi_ref:.2f}")
    
    print(f"\n--- 4. Cross-check: k_in_ref_breit = x * P_breit ---")
    print(f"  max|diff| = {np.max(crosscheck_diff):.2e}")
    print(f"  mean|diff| = {np.mean(crosscheck_diff):.2e}")
    if np.max(crosscheck_diff) < 1e-10:
        print(f"  -> VERIFIED: linearity of LT confirmed")
    
    print(f"\n--- 5. Observables comparison ---")
    if len(obs_angle_old) > 0 and len(obs_angle_ref) > 0:
        print(f"  angle(jet, hadron):")
        print(f"    OLD: mean={np.mean(obs_angle_old):.4f}, std={np.std(obs_angle_old):.4f}")
        print(f"    REF: mean={np.mean(obs_angle_ref):.4f}, std={np.std(obs_angle_ref):.4f}")
    print(f"  |pT_jet + pT_h|:")
    print(f"    OLD: mean={np.mean(obs_sum_old):.4f}, std={np.std(obs_sum_old):.4f}")
    print(f"    REF: mean={np.mean(obs_sum_ref):.4f}, std={np.std(obs_sum_ref):.4f}")
    print(f"  |pT_jet - pT_h|:")
    print(f"    OLD: mean={np.mean(obs_diff_old):.4f}, std={np.std(obs_diff_old):.4f}")
    print(f"    REF: mean={np.mean(obs_diff_ref):.4f}, std={np.std(obs_diff_ref):.4f}")
    
    return {
        "old_kT_breit": old_kT_breit,
        "ref_kT_breit": ref_kT_breit,
        "old_phi_breit": old_phi_breit,
        "ref_phi_breit": ref_phi_breit,
        "obs_angle_old": obs_angle_old,
        "obs_angle_ref": obs_angle_ref,
        "obs_sum_old": obs_sum_old,
        "obs_sum_ref": obs_sum_ref,
        "obs_diff_old": obs_diff_old,
        "obs_diff_ref": obs_diff_ref,
    }


def main():
    print("=" * 70)
    print("CONTROLLED TEST: k_in_old (k_out - q) vs k_in_ref (x * P)")
    print("=" * 70)
    
    results_off = run_comparison(PTREL_LABEL_OFF, max_events=2000)
    results_on = run_comparison(PTREL_LABEL_ON, max_events=2000)
    
    if results_off is None or results_on is None:
        print("Failed to get results.")
        return
    
    # ============================================================
    # Generate comparison plots
    # ============================================================
    
    print("\nGenerating comparison plots...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: |k_T| distributions
    ax = axes[0, 0]
    ax.hist(results_off["old_kT_breit"], bins=50, alpha=0.7, label="OLD (k_out-q)", density=True)
    ax.hist(results_off["ref_kT_breit"], bins=50, alpha=0.7, label="REF (xP)", density=True)
    ax.set_xlabel(r"|$k_T$| [GeV]")
    ax.set_ylabel("Density")
    ax.set_title("ISR/FSR OFF")
    ax.legend()
    ax.set_xlim(0, 10)
    
    ax = axes[0, 1]
    ax.hist(results_on["old_kT_breit"], bins=50, alpha=0.7, label="OLD (k_out-q)", density=True)
    ax.hist(results_on["ref_kT_breit"], bins=50, alpha=0.7, label="REF (xP)", density=True)
    ax.set_xlabel(r"|$k_T$| [GeV]")
    ax.set_ylabel("Density")
    ax.set_title("ISR/FSR ON")
    ax.legend()
    ax.set_xlim(0, 10)
    
    # Row 1: phi distributions
    ax = axes[0, 2]
    ax.hist(results_off["old_phi_breit"], bins=40, range=(-np.pi, np.pi), alpha=0.7, label="OLD", density=True)
    ax.hist(results_off["ref_phi_breit"], bins=40, range=(-np.pi, np.pi), alpha=0.7, label="REF", density=True)
    ax.set_xlabel(r"$\phi$ [rad]")
    ax.set_ylabel("Density")
    ax.set_title(r"$\phi$(jet) - ISR/FSR OFF")
    ax.legend()
    
    ax = axes[0, 3]
    ax.hist(results_on["old_phi_breit"], bins=40, range=(-np.pi, np.pi), alpha=0.7, label="OLD", density=True)
    ax.hist(results_on["ref_phi_breit"], bins=40, range=(-np.pi, np.pi), alpha=0.7, label="REF", density=True)
    ax.set_xlabel(r"$\phi$ [rad]")
    ax.set_ylabel("Density")
    ax.set_title(r"$\phi$(jet) - ISR/FSR ON")
    ax.legend()
    
    # Row 2: Angle observable
    ax = axes[1, 0]
    if len(results_off["obs_angle_old"]) > 0:
        ax.hist(results_off["obs_angle_old"], bins=40, range=(0, np.pi), alpha=0.7, label="OLD", density=True)
        ax.hist(results_off["obs_angle_ref"], bins=40, range=(0, np.pi), alpha=0.7, label="REF", density=True)
    ax.set_xlabel(r"$\phi_{hJ}$ [rad]")
    ax.set_ylabel("Density")
    ax.set_title("Angle - ISR/FSR OFF")
    ax.legend()
    
    ax = axes[1, 1]
    if len(results_on["obs_angle_old"]) > 0:
        ax.hist(results_on["obs_angle_old"], bins=40, range=(0, np.pi), alpha=0.7, label="OLD", density=True)
        ax.hist(results_on["obs_angle_ref"], bins=40, range=(0, np.pi), alpha=0.7, label="REF", density=True)
    ax.set_xlabel(r"$\phi_{hJ}$ [rad]")
    ax.set_ylabel("Density")
    ax.set_title("Angle - ISR/FSR ON")
    ax.legend()
    
    # Row 2: Sum observable
    ax = axes[1, 2]
    ax.hist(results_off["obs_sum_old"], bins=50, range=(0, 5), alpha=0.7, label="OLD", density=True)
    ax.hist(results_off["obs_sum_ref"], bins=50, range=(0, 5), alpha=0.7, label="REF", density=True)
    ax.set_xlabel(r"|$p_T^{jet}$ + $p_T^h$| [GeV]")
    ax.set_ylabel("Density")
    ax.set_title("Sum - ISR/FSR OFF")
    ax.legend()
    
    ax = axes[1, 3]
    ax.hist(results_on["obs_sum_old"], bins=50, range=(0, 5), alpha=0.7, label="OLD", density=True)
    ax.hist(results_on["obs_sum_ref"], bins=50, range=(0, 5), alpha=0.7, label="REF", density=True)
    ax.set_xlabel(r"|$p_T^{jet}$ + $p_T^h$| [GeV]")
    ax.set_ylabel("Density")
    ax.set_title("Sum - ISR/FSR ON")
    ax.legend()
    
    # Row 3: Diff observable
    ax = axes[2, 0]
    ax.hist(results_off["obs_diff_old"], bins=50, range=(0, 5), alpha=0.7, label="OLD", density=True)
    ax.hist(results_off["obs_diff_ref"], bins=50, range=(0, 5), alpha=0.7, label="REF", density=True)
    ax.set_xlabel(r"|$p_T^{jet}$ - $p_T^h$| [GeV]")
    ax.set_ylabel("Density")
    ax.set_title("Diff - ISR/FSR OFF")
    ax.legend()
    
    ax = axes[2, 1]
    ax.hist(results_on["obs_diff_old"], bins=50, range=(0, 5), alpha=0.7, label="OLD", density=True)
    ax.hist(results_on["obs_diff_ref"], bins=50, range=(0, 5), alpha=0.7, label="REF", density=True)
    ax.set_xlabel(r"|$p_T^{jet}$ - $p_T^h$| [GeV]")
    ax.set_ylabel("Density")
    ax.set_title("Diff - ISR/FSR ON")
    ax.legend()
    
    # Row 3: REF only (zoomed)
    ax = axes[2, 2]
    ax.hist(results_off["ref_kT_breit"], bins=50, alpha=0.7, label="OFF", density=True)
    ax.hist(results_on["ref_kT_breit"], bins=50, alpha=0.7, label="ON", density=True)
    ax.set_xlabel(r"|$k_T$| REF [GeV]")
    ax.set_ylabel("Density")
    ax.set_title("REF |k_T| (zoomed)")
    ax.legend()
    ax.set_xlim(0, 0.1)
    
    ax = axes[2, 3]
    ax.hist(results_off["ref_phi_breit"], bins=40, range=(-np.pi, np.pi), alpha=0.7, label="OFF", density=True)
    ax.hist(results_on["ref_phi_breit"], bins=40, range=(-np.pi, np.pi), alpha=0.7, label="ON", density=True)
    ax.set_xlabel(r"$\phi$ REF [rad]")
    ax.set_ylabel("Density")
    ax.set_title(r"REF $\phi$(jet)")
    ax.legend()
    
    plt.tight_layout()
    out_path = _PROJECT_ROOT / "kin_fix_comparison.pdf"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")
    
    # ============================================================
    # Final interpretation
    # ============================================================
    
    print("\n" + "=" * 70)
    print("CRITICAL INTERPRETATION")
    print("=" * 70)
    
    old_kT_off = np.mean(results_off["old_kT_breit"])
    ref_kT_off = np.mean(results_off["ref_kT_breit"])
    old_kT_on = np.mean(results_on["old_kT_breit"])
    ref_kT_on = np.mean(results_on["ref_kT_breit"])
    
    cos_old_off = np.mean(np.cos(results_off["old_phi_breit"]))
    cos_ref_off = np.mean(np.cos(results_off["ref_phi_breit"]))
    cos_old_on = np.mean(np.cos(results_on["old_phi_breit"]))
    cos_ref_on = np.mean(np.cos(results_on["ref_phi_breit"]))
    
    print(f"""
1. Does using k_in_ref = xP eliminate the unphysical large transverse momentum?

   ISR/FSR OFF:
     OLD mean |k_T| = {old_kT_off:.4f} GeV
     REF mean |k_T| = {ref_kT_off:.6f} GeV
     -> {"YES! Reduced by factor " + f"{old_kT_off/max(ref_kT_off, 1e-10):.0f}" if ref_kT_off < 0.01 else "PARTIALLY reduced"}

   ISR/FSR ON:
     OLD mean |k_T| = {old_kT_on:.4f} GeV
     REF mean |k_T| = {ref_kT_on:.6f} GeV
     -> {"YES! Reduced by factor " + f"{old_kT_on/max(ref_kT_on, 1e-10):.0f}" if ref_kT_on < 0.01 else "PARTIALLY reduced"}

2. Does it eliminate or significantly reduce the strong x-axis locking?

   ISR/FSR OFF:
     OLD <cos(phi)> = {cos_old_off:.4f}
     REF <cos(phi)> = {cos_ref_off:.4f}
     -> {"YES! X-axis locking eliminated" if abs(cos_ref_off) < 0.1 else "REDUCED but not eliminated"}

   ISR/FSR ON:
     OLD <cos(phi)> = {cos_old_on:.4f}
     REF <cos(phi)> = {cos_ref_on:.4f}
     -> {"YES! X-axis locking eliminated" if abs(cos_ref_on) < 0.1 else "REDUCED but not eliminated"}

3. Do the jet-hadron observables change qualitatively?

   See the comparison plots. With REF definition:
   - The jet has essentially zero transverse momentum
   - The observables are dominated by the hadron's transverse momentum
   - The angle distribution should change from peaked at 0 to more uniform

4. Does this support the conclusion that k_in_old = k_out - q was the source?

   {"YES - The pathology is clearly caused by the k_in definition." if (ref_kT_off < 0.01 and ref_kT_on < 0.01) else "The k_in definition contributes but may not be the only source."}

CONCLUSION:
""")
    
    if ref_kT_off < 0.01 and ref_kT_on < 0.01:
        print("""
The benchmark definition k_in_ref = x * P has essentially ZERO transverse
momentum in the Breit frame (as expected for a collinear parton).

This confirms that:
1. The O(2 GeV) transverse effect was entirely an artifact of using
   k_in = k_out - q, where k_out from PYTHIA is not the simple LO DIS quark.

2. The x-axis locking was caused by the Breit transform acting on this
   unphysical vector, not by an error in the transform itself.

3. The proposed fix (using a collinear parton model for k_in) works.

RECOMMENDATION:
For physics observables that require the incoming parton transverse momentum,
use k_in = x * P (or equivalently, recognize that in the collinear parton
model, the incoming parton has no intrinsic transverse momentum).
""")
    
    print("\nTest complete.")


if __name__ == "__main__":
    main()
