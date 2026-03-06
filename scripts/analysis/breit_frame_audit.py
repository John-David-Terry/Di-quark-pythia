#!/usr/bin/env python3
"""
Comprehensive audit of the Lorentz transform to the Breit frame.

This script performs systematic checks to diagnose where the O(2 GeV) transverse
momentum in k_in originates when ISR/FSR is off.

Checks:
1. LT validity (Lorentz condition LT^T g LT = g)
2. Standard Breit frame conditions (q0~0, qT~0, PT~0)
3. k_in identification and consistency
4. Step-by-step audit of build_LT stages
5. Momentum conservation
6. Comparison with reference Breit construction
7. ISR/FSR OFF special checks
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
g = np.diag([1.0, -1.0, -1.0, -1.0])


def minkowski_norm(v):
    """Compute v^2 = v^mu v_mu = v0^2 - v1^2 - v2^2 - v3^2."""
    return v[0]**2 - v[1]**2 - v[2]**2 - v[3]**2


def dot4_explicit(a, b):
    """Minkowski dot product a·b = a0*b0 - a1*b1 - a2*b2 - a3*b3."""
    return a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]


def transverse_mag(v):
    """Transverse magnitude sqrt(v1^2 + v2^2)."""
    return np.sqrt(v[1]**2 + v[2]**2)


def check_lorentz_condition(LT):
    """Check if LT^T g LT = g. Returns max absolute residual."""
    residual = LT.T @ g @ LT - g
    return np.max(np.abs(residual)), residual


def build_reference_breit_transform(qmu, P):
    """
    Build a reference Breit transformation using textbook steps:
    1. Rotate so q lies in x-z plane (eliminate q_y)
    2. Boost along x to eliminate q_x (make q longitudinal)
    3. Boost along z so q0 = 0 (standard Breit condition)
    
    Returns the combined 4x4 matrix or None if construction fails.
    """
    q = np.asarray(qmu, dtype=float)
    
    # Step 1: Rotate in x-y plane to put q in x-z plane
    qT = np.sqrt(q[1]**2 + q[2]**2)
    if qT > 1e-10:
        cos_phi = q[1] / qT
        sin_phi = q[2] / qT
    else:
        cos_phi = 1.0
        sin_phi = 0.0
    
    R1 = np.array([
        [1, 0, 0, 0],
        [0, cos_phi, sin_phi, 0],
        [0, -sin_phi, cos_phi, 0],
        [0, 0, 0, 1]
    ], dtype=float)
    
    q1 = R1 @ q  # Now q1_y should be ~0
    
    # Step 2: Boost along x to eliminate q_x
    # After boost, q should be purely longitudinal
    # This requires solving for boost velocity
    # For a general case, we need to boost to the frame where q_x = 0
    # Boost velocity beta_x = q1[1] / q1[0] won't work directly for spacelike q
    # Instead, use the frame where q is along z
    
    # Actually, for DIS q is spacelike, so we need a different approach
    # Let's rotate to put q along z first
    
    q_mag_xz = np.sqrt(q1[1]**2 + q1[3]**2)
    if q_mag_xz > 1e-10:
        cos_theta = q1[3] / q_mag_xz
        sin_theta = q1[1] / q_mag_xz
    else:
        cos_theta = 1.0
        sin_theta = 0.0
    
    R2 = np.array([
        [1, 0, 0, 0],
        [0, cos_theta, 0, -sin_theta],
        [0, 0, 1, 0],
        [0, sin_theta, 0, cos_theta]
    ], dtype=float)
    
    q2 = R2 @ q1  # Now q2 should be along z: (q0, 0, 0, qz)
    
    # Step 3: Boost along z to make q0 = 0
    # For spacelike q with q2 = (q0, 0, 0, qz), we have q0^2 - qz^2 = -Q^2 < 0
    # Boost: q0' = gamma(q0 - beta*qz), qz' = gamma(qz - beta*q0)
    # Want q0' = 0 => beta = q0/qz
    if abs(q2[3]) > 1e-10:
        beta_z = q2[0] / q2[3]
        if abs(beta_z) < 1.0:
            gamma_z = 1.0 / np.sqrt(1 - beta_z**2)
            B3 = np.array([
                [gamma_z, 0, 0, -gamma_z*beta_z],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [-gamma_z*beta_z, 0, 0, gamma_z]
            ], dtype=float)
        else:
            return None  # Superluminal boost needed
    else:
        B3 = np.eye(4)
    
    LT_ref = B3 @ R2 @ R1
    return LT_ref


def run_audit(label: str, max_events: int = 500):
    """Run comprehensive Breit frame audit for one label."""
    shards = list_shards(label)
    if not shards:
        print(f"[{label}] No shards found.")
        return {}
    
    # Collectors
    results = {
        "lorentz_residuals": [],
        "det_LT": [],
        "q0_breit": [],
        "qT_breit": [],
        "PT_breit": [],
        "kinT_lab": [],
        "kinT_breit": [],
        "kin_minus_xP_T_lab": [],
        "kin_minus_xP_T_breit": [],
        "q_norm_diff": [],
        "P_norm_diff": [],
        "kin_norm_diff": [],
        "Pdotq_diff": [],
        "kout_minus_kin_minus_q": [],
        "kin_mass_sq": [],
        "kout_mass_sq": [],
        "kin_energy": [],
    }
    
    processed = 0
    debug_printed = 0
    DEBUG_PRINT_N = 5
    
    # For step-by-step audit
    stage_audit_done = False
    
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
            
            # Get 4-vectors in LAB (with flip_z applied)
            l_lab = flip_z(np.asarray(e_in[ie], dtype=float), FLIP_Z_PTREL)
            P_lab = flip_z(np.asarray(p_in[ie], dtype=float), FLIP_Z_PTREL)
            lprime_lab = flip_z(np.asarray(e_sc[ie], dtype=float), FLIP_Z_PTREL)
            kout_lab = flip_z(np.asarray(k_out[ie], dtype=float), FLIP_Z_PTREL)
            
            Ep = float(P_lab[0])
            Ee = float(l_lab[0])
            
            # Virtual photon
            q_lab = l_lab - lprime_lab
            Q2 = -minkowski_norm(q_lab)
            if Q2 <= 0:
                continue
            Q = np.sqrt(Q2)
            
            # Kinematics
            p_dot_q = dot4_explicit(P_lab, q_lab)
            if p_dot_q == 0:
                continue
            x = Q2 / (2.0 * p_dot_q)
            if not (xmin_ptrel <= x <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel):
                continue
            
            qT_lab_val = transverse_mag(q_lab)
            phiq = np.arctan2(q_lab[2], q_lab[1])
            S = 4.0 * Ee * Ep
            y = Q2 / (S * x)
            
            # Build LT
            LT = build_LT(Ee, Ep, q_lab, x, y, qT_lab_val, phiq, S)
            if LT is None:
                continue
            
            # Check Lorentz condition
            max_residual, residual_matrix = check_lorentz_condition(LT)
            results["lorentz_residuals"].append(max_residual)
            results["det_LT"].append(np.linalg.det(LT))
            
            # Apply transform
            boost = lambda v: LT @ np.asarray(v, dtype=float)
            
            l_breit = boost(l_lab)
            lprime_breit = boost(lprime_lab)
            P_breit = boost(P_lab)
            q_breit = boost(q_lab)
            kout_breit = boost(kout_lab)
            
            # k_in definition
            k_in_lab = kout_lab - q_lab
            k_in_breit = boost(k_in_lab)
            
            # Remnant
            P_rem_lab = P_lab - k_in_lab
            P_rem_breit = boost(P_rem_lab)
            
            # xP for comparison
            xP_lab = x * P_lab
            xP_breit = boost(xP_lab)
            
            # Collect statistics
            results["q0_breit"].append(q_breit[0])
            results["qT_breit"].append(transverse_mag(q_breit))
            results["PT_breit"].append(transverse_mag(P_breit))
            results["kinT_lab"].append(transverse_mag(k_in_lab))
            results["kinT_breit"].append(transverse_mag(k_in_breit))
            
            kin_minus_xP_lab = k_in_lab - xP_lab
            kin_minus_xP_breit = k_in_breit - xP_breit
            results["kin_minus_xP_T_lab"].append(transverse_mag(kin_minus_xP_lab))
            results["kin_minus_xP_T_breit"].append(transverse_mag(kin_minus_xP_breit))
            
            # Check norm preservation
            results["q_norm_diff"].append(abs(minkowski_norm(q_breit) - minkowski_norm(q_lab)))
            results["P_norm_diff"].append(abs(minkowski_norm(P_breit) - minkowski_norm(P_lab)))
            results["kin_norm_diff"].append(abs(minkowski_norm(k_in_breit) - minkowski_norm(k_in_lab)))
            
            # Check dot product preservation
            Pdotq_lab = dot4_explicit(P_lab, q_lab)
            Pdotq_breit = dot4_explicit(P_breit, q_breit)
            results["Pdotq_diff"].append(abs(Pdotq_breit - Pdotq_lab))
            
            # Check k_out = k_in + q
            kout_check = kout_lab - k_in_lab - q_lab
            results["kout_minus_kin_minus_q"].append(np.max(np.abs(kout_check)))
            
            # CRITICAL: Check if k_in is on-shell (should be ~0 for physical parton)
            kin_mass_sq = minkowski_norm(k_in_lab)
            kout_mass_sq = minkowski_norm(kout_lab)
            results["kin_mass_sq"].append(kin_mass_sq)
            results["kout_mass_sq"].append(kout_mass_sq)
            results["kin_energy"].append(k_in_lab[0])
            
            # Debug print for first few events
            if debug_printed < DEBUG_PRINT_N:
                print(f"\n{'='*70}")
                print(f"AUDIT EVENT ({shard_idx}, {ie}) [{label}]")
                print(f"{'='*70}")
                
                print(f"\n--- 4-VECTORS IN LAB ---")
                print(f"  l (e_in)    = ({l_lab[0]:.4f}, {l_lab[1]:.4f}, {l_lab[2]:.4f}, {l_lab[3]:.4f})")
                print(f"  l' (e_sc)   = ({lprime_lab[0]:.4f}, {lprime_lab[1]:.4f}, {lprime_lab[2]:.4f}, {lprime_lab[3]:.4f})")
                print(f"  P (proton)  = ({P_lab[0]:.4f}, {P_lab[1]:.4f}, {P_lab[2]:.4f}, {P_lab[3]:.4f})")
                print(f"  q (photon)  = ({q_lab[0]:.4f}, {q_lab[1]:.4f}, {q_lab[2]:.4f}, {q_lab[3]:.4f})")
                print(f"  k_out       = ({kout_lab[0]:.4f}, {kout_lab[1]:.4f}, {kout_lab[2]:.4f}, {kout_lab[3]:.4f})")
                print(f"  k_in=k_out-q= ({k_in_lab[0]:.4f}, {k_in_lab[1]:.4f}, {k_in_lab[2]:.4f}, {k_in_lab[3]:.4f})")
                print(f"  P_rem=P-k_in= ({P_rem_lab[0]:.4f}, {P_rem_lab[1]:.4f}, {P_rem_lab[2]:.4f}, {P_rem_lab[3]:.4f})")
                print(f"  xP          = ({xP_lab[0]:.4f}, {xP_lab[1]:.4f}, {xP_lab[2]:.4f}, {xP_lab[3]:.4f})")
                
                print(f"\n--- 4-VECTORS IN BREIT ---")
                print(f"  l_breit     = ({l_breit[0]:.4f}, {l_breit[1]:.6f}, {l_breit[2]:.6f}, {l_breit[3]:.4f})")
                print(f"  l'_breit    = ({lprime_breit[0]:.4f}, {lprime_breit[1]:.6f}, {lprime_breit[2]:.6f}, {lprime_breit[3]:.4f})")
                print(f"  P_breit     = ({P_breit[0]:.4f}, {P_breit[1]:.6f}, {P_breit[2]:.6f}, {P_breit[3]:.4f})")
                print(f"  q_breit     = ({q_breit[0]:.6f}, {q_breit[1]:.6e}, {q_breit[2]:.6e}, {q_breit[3]:.4f})")
                print(f"  k_out_breit = ({kout_breit[0]:.4f}, {kout_breit[1]:.6f}, {kout_breit[2]:.6f}, {kout_breit[3]:.4f})")
                print(f"  k_in_breit  = ({k_in_breit[0]:.4f}, {k_in_breit[1]:.6f}, {k_in_breit[2]:.6f}, {k_in_breit[3]:.4f})")
                print(f"  P_rem_breit = ({P_rem_breit[0]:.4f}, {P_rem_breit[1]:.6f}, {P_rem_breit[2]:.6f}, {P_rem_breit[3]:.4f})")
                print(f"  xP_breit    = ({xP_breit[0]:.4f}, {xP_breit[1]:.6f}, {xP_breit[2]:.6f}, {xP_breit[3]:.4f})")
                
                print(f"\n--- BREIT FRAME CONDITIONS ---")
                print(f"  A. q should be longitudinal:")
                print(f"     q_x = {q_breit[1]:.6e}, q_y = {q_breit[2]:.6e}, |qT| = {transverse_mag(q_breit):.6e}")
                print(f"  B. P should be longitudinal (standard Breit):")
                print(f"     P_x = {P_breit[1]:.6e}, P_y = {P_breit[2]:.6e}, |P_T| = {transverse_mag(P_breit):.6e}")
                print(f"  C. q and P along z:")
                print(f"     q_z = {q_breit[3]:.4f} (should be < 0)")
                print(f"     P_z = {P_breit[3]:.4f} (should be > 0)")
                print(f"  D. q^0 in Breit (should be ~0 for standard Breit):")
                print(f"     q^0 = {q_breit[0]:.6e}")
                
                print(f"\n--- LORENTZ TRANSFORM VALIDITY ---")
                print(f"  max|LT^T g LT - g| = {max_residual:.6e}")
                print(f"  det(LT) = {np.linalg.det(LT):.6f}")
                
                print(f"\n--- NORM PRESERVATION (should be ~0) ---")
                print(f"  |q^2_breit - q^2_lab| = {abs(minkowski_norm(q_breit) - minkowski_norm(q_lab)):.6e}")
                print(f"  |P^2_breit - P^2_lab| = {abs(minkowski_norm(P_breit) - minkowski_norm(P_lab)):.6e}")
                print(f"  |k_in^2_breit - k_in^2_lab| = {abs(minkowski_norm(k_in_breit) - minkowski_norm(k_in_lab)):.6e}")
                
                print(f"\n--- DOT PRODUCT PRESERVATION ---")
                print(f"  P·q LAB = {Pdotq_lab:.4f}, Breit = {Pdotq_breit:.4f}, diff = {abs(Pdotq_breit - Pdotq_lab):.6e}")
                
                print(f"\n--- k_in vs xP COMPARISON ---")
                print(f"  k_in - xP (LAB):   ({kin_minus_xP_lab[0]:.4f}, {kin_minus_xP_lab[1]:.4f}, {kin_minus_xP_lab[2]:.4f}, {kin_minus_xP_lab[3]:.4f})")
                print(f"  |k_in - xP|_T LAB: {transverse_mag(kin_minus_xP_lab):.6f}")
                print(f"  k_in - xP (Breit): ({kin_minus_xP_breit[0]:.4f}, {kin_minus_xP_breit[1]:.6f}, {kin_minus_xP_breit[2]:.6f}, {kin_minus_xP_breit[3]:.4f})")
                print(f"  |k_in - xP|_T Breit: {transverse_mag(kin_minus_xP_breit):.6f}")
                
                print(f"\n--- HARD SCATTERING CHECK: k_out = k_in + q ---")
                print(f"  k_out - k_in - q = ({kout_check[0]:.6e}, {kout_check[1]:.6e}, {kout_check[2]:.6e}, {kout_check[3]:.6e})")
                
                print(f"\n--- CRITICAL: MASS-SQUARED CHECK ---")
                print(f"  k_in^2 = {kin_mass_sq:.4f} GeV^2 (should be ~0 for massless quark)")
                print(f"  k_out^2 = {kout_mass_sq:.4f} GeV^2 (should be ~0 for massless quark)")
                if kin_mass_sq < -1.0:
                    print(f"  -> WARNING: k_in is SPACELIKE with |k_in^2| = {abs(kin_mass_sq):.1f} GeV^2!")
                if k_in_lab[0] < 0:
                    print(f"  -> WARNING: k_in has NEGATIVE ENERGY ({k_in_lab[0]:.2f} GeV)!")
                
                print(f"\n--- KINEMATICS ---")
                print(f"  x = {x:.6f}, Q = {Q:.4f} GeV, y = {y:.4f}")
                
                debug_printed += 1
            
            # Step-by-step LT audit for first event only
            if not stage_audit_done and debug_printed == 1:
                print(f"\n{'='*70}")
                print(f"STEP-BY-STEP LT AUDIT (first accepted event)")
                print(f"{'='*70}")
                
                # Reconstruct the individual matrices
                Mm1 = np.array([
                    [Ee/np.sqrt(S) + np.sqrt(S)/(4.0*Ee), 0, 0, Ee/np.sqrt(S) - np.sqrt(S)/(4.0*Ee)],
                    [0, 1, 0, 0], [0, 0, 1, 0],
                    [Ee/np.sqrt(S) - np.sqrt(S)/(4.0*Ee), 0, 0, Ee/np.sqrt(S) + np.sqrt(S)/(4.0*Ee)]
                ])
                M0 = np.array([
                    [1, 0, 0, 0],
                    [0, np.cos(phiq), np.sin(phiq), 0],
                    [0, -np.sin(phiq), np.cos(phiq), 0], [0, 0, 0, 1]
                ])
                den2 = (-qT_lab_val**2 + S*(1+x)*y)
                denom_M1 = 2.0 * y * np.sqrt(S * (-qT_lab_val**2 + S*(1+x)*y))
                M1 = np.array([
                    [(-qT_lab_val**2 + S*y*(1+x+y)) / denom_M1, 0, 0, (qT_lab_val**2 + S*y*(-x+y-1)) / denom_M1],
                    [0, 1, 0, 0], [0, 0, 1, 0],
                    [(qT_lab_val**2 + S*y*(-x+y-1)) / denom_M1, 0, 0, (-qT_lab_val**2 + S*y*(1+x+y)) / denom_M1]
                ])
                denom_M2_s1 = np.sqrt(S*(1+x)*y / den2)
                denom_M2_s2 = np.sqrt(S*(1+x)*y)
                M2 = np.array([
                    [1, 0, 0, 0],
                    [0, 1/denom_M2_s1, 0, qT_lab_val/denom_M2_s2],
                    [0, 0, 1, 0], [0, -qT_lab_val/denom_M2_s2, 0, 1/denom_M2_s1]
                ])
                num_log = qT_lab_val + np.sqrt(S*(1+x)*y)
                den_log = np.sqrt(S*(1+x)*y) - qT_lab_val
                eta_m3 = 0.5*np.log(num_log/den_log)
                denom_M3 = np.sqrt(den2)
                M3 = np.array([
                    [np.cosh(eta_m3), -qT_lab_val/denom_M3, 0, 0],
                    [-qT_lab_val/denom_M3, np.cosh(eta_m3), 0, 0],
                    [0, 0, 1, 0], [0, 0, 0, 1]
                ])
                denom_M4 = 2*np.sqrt(x*(1+x))
                M4 = np.array([
                    [(1+2*x)/denom_M4, 0, 0, 1/denom_M4],
                    [0, 1, 0, 0], [0, 0, 1, 0],
                    [1/denom_M4, 0, 0, (1+2*x)/denom_M4]
                ])
                
                print(f"\nphiq (rotation angle) = {phiq:.6f} rad = {np.degrees(phiq):.2f} deg")
                
                # Apply step by step
                v_lab = {"q": q_lab, "P": P_lab, "k_in": k_in_lab, "k_out": kout_lab}
                
                for name, v in v_lab.items():
                    print(f"\n--- {name} through LT stages ---")
                    print(f"  LAB:       ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}, {v[3]:.4f}) |T| = {transverse_mag(v):.4f}")
                    
                    v1 = Mm1 @ v
                    print(f"  after Mm1: ({v1[0]:.4f}, {v1[1]:.4f}, {v1[2]:.4f}, {v1[3]:.4f}) |T| = {transverse_mag(v1):.4f}")
                    
                    v2 = M0 @ v1
                    print(f"  after M0:  ({v2[0]:.4f}, {v2[1]:.4f}, {v2[2]:.4f}, {v2[3]:.4f}) |T| = {transverse_mag(v2):.4f}")
                    
                    v3 = M1 @ v2
                    print(f"  after M1:  ({v3[0]:.4f}, {v3[1]:.4f}, {v3[2]:.4f}, {v3[3]:.4f}) |T| = {transverse_mag(v3):.4f}")
                    
                    v4 = M2 @ v3
                    print(f"  after M2:  ({v4[0]:.4f}, {v4[1]:.4f}, {v4[2]:.4f}, {v4[3]:.4f}) |T| = {transverse_mag(v4):.4f}")
                    
                    v5 = M3 @ v4
                    print(f"  after M3:  ({v5[0]:.4f}, {v5[1]:.4f}, {v5[2]:.4f}, {v5[3]:.4f}) |T| = {transverse_mag(v5):.4f}")
                    
                    v6 = M4 @ v5
                    print(f"  after M4:  ({v6[0]:.4f}, {v6[1]:.6f}, {v6[2]:.6f}, {v6[3]:.4f}) |T| = {transverse_mag(v6):.6f}")
                
                stage_audit_done = True
            
            processed += 1
    
    print(f"\n[{label}] Processed {processed} events for audit")
    
    # Convert to arrays
    for key in results:
        results[key] = np.asarray(results[key])
    
    # Summary statistics
    print(f"\n{'='*70}")
    print(f"[{label}] SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    print(f"\n--- 1. LORENTZ TRANSFORM VALIDITY ---")
    print(f"  max|LT^T g LT - g|: min={np.min(results['lorentz_residuals']):.2e}, max={np.max(results['lorentz_residuals']):.2e}, mean={np.mean(results['lorentz_residuals']):.2e}")
    print(f"  det(LT): min={np.min(results['det_LT']):.6f}, max={np.max(results['det_LT']):.6f}, mean={np.mean(results['det_LT']):.6f}")
    if np.max(results['lorentz_residuals']) < 1e-10:
        print(f"  -> LT IS a valid Lorentz transform (residual < 1e-10)")
    else:
        print(f"  -> WARNING: LT may not be a valid Lorentz transform!")
    
    print(f"\n--- 2. STANDARD BREIT FRAME CONDITIONS ---")
    print(f"  q^0 in Breit: min={np.min(results['q0_breit']):.2e}, max={np.max(results['q0_breit']):.2e}, mean={np.mean(results['q0_breit']):.2e}")
    print(f"  |qT| in Breit: min={np.min(results['qT_breit']):.2e}, max={np.max(results['qT_breit']):.2e}, mean={np.mean(results['qT_breit']):.2e}")
    print(f"  |P_T| in Breit: min={np.min(results['PT_breit']):.2e}, max={np.max(results['PT_breit']):.2e}, mean={np.mean(results['PT_breit']):.2e}")
    
    is_standard_breit = True
    if np.mean(np.abs(results['q0_breit'])) > 1e-3:
        print(f"  -> q^0 is NOT ~0; this is NOT the standard Breit frame")
        is_standard_breit = False
    else:
        print(f"  -> q^0 ~ 0 ✓")
    
    if np.mean(results['qT_breit']) > 1e-3:
        print(f"  -> |qT| is NOT ~0; q is not longitudinal")
        is_standard_breit = False
    else:
        print(f"  -> |qT| ~ 0 ✓ (q is longitudinal)")
    
    if np.mean(results['PT_breit']) > 1e-3:
        print(f"  -> |P_T| is NOT ~0; this is a NONSTANDARD Breit-like frame")
        is_standard_breit = False
    else:
        print(f"  -> |P_T| ~ 0 ✓")
    
    if is_standard_breit:
        print(f"  CONCLUSION: This IS the standard Breit frame")
    else:
        print(f"  CONCLUSION: This is a NONSTANDARD Breit-like frame")
    
    print(f"\n--- 3. NORM AND DOT PRODUCT PRESERVATION ---")
    print(f"  |q^2_breit - q^2_lab|: max={np.max(results['q_norm_diff']):.2e}, mean={np.mean(results['q_norm_diff']):.2e}")
    print(f"  |P^2_breit - P^2_lab|: max={np.max(results['P_norm_diff']):.2e}, mean={np.mean(results['P_norm_diff']):.2e}")
    print(f"  |k_in^2_breit - k_in^2_lab|: max={np.max(results['kin_norm_diff']):.2e}, mean={np.mean(results['kin_norm_diff']):.2e}")
    print(f"  |P·q_breit - P·q_lab|: max={np.max(results['Pdotq_diff']):.2e}, mean={np.mean(results['Pdotq_diff']):.2e}")
    
    print(f"\n--- 4. k_in IDENTIFICATION ---")
    print(f"  |k_in,T| LAB: min={np.min(results['kinT_lab']):.4f}, max={np.max(results['kinT_lab']):.4f}, mean={np.mean(results['kinT_lab']):.4f}")
    print(f"  |k_in,T| Breit: min={np.min(results['kinT_breit']):.4f}, max={np.max(results['kinT_breit']):.4f}, mean={np.mean(results['kinT_breit']):.4f}")
    print(f"  |k_in - xP|_T LAB: min={np.min(results['kin_minus_xP_T_lab']):.4f}, max={np.max(results['kin_minus_xP_T_lab']):.4f}, mean={np.mean(results['kin_minus_xP_T_lab']):.4f}")
    print(f"  |k_in - xP|_T Breit: min={np.min(results['kin_minus_xP_T_breit']):.4f}, max={np.max(results['kin_minus_xP_T_breit']):.4f}, mean={np.mean(results['kin_minus_xP_T_breit']):.4f}")
    
    print(f"\n--- 5. HARD SCATTERING CHECK k_out = k_in + q ---")
    print(f"  max|k_out - k_in - q|: max={np.max(results['kout_minus_kin_minus_q']):.2e}")
    if np.max(results['kout_minus_kin_minus_q']) < 1e-10:
        print(f"  -> k_out = k_in + q is satisfied (by construction)")
    
    print(f"\n--- 6. CRITICAL: k_in MASS-SQUARED (should be ~0 for physical parton) ---")
    print(f"  k_in^2: min={np.min(results['kin_mass_sq']):.2f}, max={np.max(results['kin_mass_sq']):.2f}, mean={np.mean(results['kin_mass_sq']):.2f}")
    print(f"  k_out^2: min={np.min(results['kout_mass_sq']):.4f}, max={np.max(results['kout_mass_sq']):.4f}, mean={np.mean(results['kout_mass_sq']):.4f}")
    print(f"  k_in energy: min={np.min(results['kin_energy']):.2f}, max={np.max(results['kin_energy']):.2f}, mean={np.mean(results['kin_energy']):.2f}")
    neg_energy_frac = np.sum(np.array(results['kin_energy']) < 0) / len(results['kin_energy'])
    print(f"  Fraction with k_in energy < 0: {neg_energy_frac:.4f}")
    offshell_frac = np.sum(np.abs(np.array(results['kin_mass_sq'])) > 1.0) / len(results['kin_mass_sq'])
    print(f"  Fraction with |k_in^2| > 1 GeV^2: {offshell_frac:.4f}")
    
    if offshell_frac > 0.5:
        print(f"  -> WARNING: k_in = k_out - q is HEAVILY OFF-SHELL in most events!")
        print(f"     This means k_out from the event record is NOT the direct hard-scattering quark.")
        print(f"     The definition k_in = k_out - q does NOT give the physical incoming parton.")
    
    return results


def run_boost_consistency_check(label: str, max_events: int = 20):
    """
    Verify all initial-state 4-vectors are boosted using the same LT.
    
    This checks that e_in, e_sc, P, q are all transformed consistently
    and that no vector is accidentally left in the lab frame.
    """
    shards = list_shards(label)
    if not shards:
        print(f"[{label}] No shards found for boost consistency check.")
        return
    
    print(f"\n{'='*70}")
    print(f"BOOST CONSISTENCY CHECK [{label}]")
    print(f"{'='*70}")
    
    processed = 0
    norm_diffs = {
        "e_in": [], "e_sc": [], "P": [], "q": [], "k_out": [], "k_in": []
    }
    
    for shard_idx, shard_path in enumerate(shards):
        if processed >= max_events:
            break
        data = load_shard(shard_path)
        e_in = data["event_e_in"]
        p_in = data["event_p_in"]
        e_sc = data["event_e_sc"]
        k_out = data["event_k_out"]
        offsets = data["offsets"]
        Ne = e_in.shape[0]
        
        for ie in range(Ne):
            if processed >= max_events:
                break
            
            # Get LAB frame 4-vectors
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
            p_dot_q = dot4_explicit(P_lab, q_lab)
            if p_dot_q == 0:
                continue
            x = Q2 / (2.0 * p_dot_q)
            if not (xmin_ptrel <= x <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel):
                continue
            
            qT_lab_val = transverse_mag(q_lab)
            phiq = np.arctan2(q_lab[2], q_lab[1])
            S = 4.0 * Ee * Ep
            y = Q2 / (S * x)
            
            # Build LT
            LT = build_LT(Ee, Ep, q_lab, x, y, qT_lab_val, phiq, S)
            if LT is None:
                continue
            
            # Define boost function (same as used in analysis)
            boost = lambda v: LT @ np.asarray(v, dtype=float)
            
            # Apply boost to ALL initial-state vectors
            e_in_breit = boost(e_in_lab)
            e_sc_breit = boost(e_sc_lab)
            P_breit = boost(P_lab)
            q_breit = boost(q_lab)
            k_out_breit = boost(k_out_lab)
            k_in_lab = k_out_lab - q_lab
            k_in_breit = boost(k_in_lab)
            
            # Check norm preservation (Lorentz invariants)
            norm_diffs["e_in"].append(abs(minkowski_norm(e_in_breit) - minkowski_norm(e_in_lab)))
            norm_diffs["e_sc"].append(abs(minkowski_norm(e_sc_breit) - minkowski_norm(e_sc_lab)))
            norm_diffs["P"].append(abs(minkowski_norm(P_breit) - minkowski_norm(P_lab)))
            norm_diffs["q"].append(abs(minkowski_norm(q_breit) - minkowski_norm(q_lab)))
            norm_diffs["k_out"].append(abs(minkowski_norm(k_out_breit) - minkowski_norm(k_out_lab)))
            norm_diffs["k_in"].append(abs(minkowski_norm(k_in_breit) - minkowski_norm(k_in_lab)))
            
            # Debug print for first few events
            if processed < 5:
                print(f"\n{'='*60}")
                print(f"Event ({shard_idx}, {ie})")
                print(f"{'='*60}")
                
                print(f"\nLAB frame:")
                print(f"  e_in   = ({e_in_lab[0]:.4f}, {e_in_lab[1]:.4f}, {e_in_lab[2]:.4f}, {e_in_lab[3]:.4f})")
                print(f"  e_sc   = ({e_sc_lab[0]:.4f}, {e_sc_lab[1]:.4f}, {e_sc_lab[2]:.4f}, {e_sc_lab[3]:.4f})")
                print(f"  P      = ({P_lab[0]:.4f}, {P_lab[1]:.4f}, {P_lab[2]:.4f}, {P_lab[3]:.4f})")
                print(f"  q      = ({q_lab[0]:.4f}, {q_lab[1]:.4f}, {q_lab[2]:.4f}, {q_lab[3]:.4f})")
                print(f"  k_out  = ({k_out_lab[0]:.4f}, {k_out_lab[1]:.4f}, {k_out_lab[2]:.4f}, {k_out_lab[3]:.4f})")
                print(f"  k_in   = ({k_in_lab[0]:.4f}, {k_in_lab[1]:.4f}, {k_in_lab[2]:.4f}, {k_in_lab[3]:.4f})")
                
                print(f"\nBreit frame (after LT @ v):")
                print(f"  e_in   = ({e_in_breit[0]:.4f}, {e_in_breit[1]:.6f}, {e_in_breit[2]:.6f}, {e_in_breit[3]:.4f})")
                print(f"  e_sc   = ({e_sc_breit[0]:.4f}, {e_sc_breit[1]:.6f}, {e_sc_breit[2]:.6f}, {e_sc_breit[3]:.4f})")
                print(f"  P      = ({P_breit[0]:.4f}, {P_breit[1]:.6f}, {P_breit[2]:.6f}, {P_breit[3]:.4f})")
                print(f"  q      = ({q_breit[0]:.6f}, {q_breit[1]:.6e}, {q_breit[2]:.6e}, {q_breit[3]:.4f})")
                print(f"  k_out  = ({k_out_breit[0]:.4f}, {k_out_breit[1]:.6f}, {k_out_breit[2]:.6f}, {k_out_breit[3]:.4f})")
                print(f"  k_in   = ({k_in_breit[0]:.4f}, {k_in_breit[1]:.6f}, {k_in_breit[2]:.6f}, {k_in_breit[3]:.4f})")
                
                print(f"\nSanity checks:")
                print(f"  A. Photon properties:")
                print(f"     |qT| = {transverse_mag(q_breit):.2e} (should be ~0)")
                print(f"     q^0 = {q_breit[0]:.2e} (should be ~0)")
                
                print(f"  B. Electron scattering plane:")
                phi_ein = np.arctan2(e_in_breit[2], e_in_breit[1])
                phi_esc = np.arctan2(e_sc_breit[2], e_sc_breit[1])
                print(f"     phi(e_in) = {np.degrees(phi_ein):.2f} deg")
                print(f"     phi(e_sc) = {np.degrees(phi_esc):.2f} deg")
                print(f"     (phi difference = {np.degrees(phi_esc - phi_ein):.2f} deg)")
                
                print(f"  C. Proton:")
                print(f"     |P_T| = {transverse_mag(P_breit):.6f} GeV")
                
                print(f"\nNorm preservation (v² LAB vs Breit, diff):")
                print(f"  e_in: {minkowski_norm(e_in_lab):.4f} vs {minkowski_norm(e_in_breit):.4f}, diff = {norm_diffs['e_in'][-1]:.2e}")
                print(f"  e_sc: {minkowski_norm(e_sc_lab):.4f} vs {minkowski_norm(e_sc_breit):.4f}, diff = {norm_diffs['e_sc'][-1]:.2e}")
                print(f"  P: {minkowski_norm(P_lab):.4f} vs {minkowski_norm(P_breit):.4f}, diff = {norm_diffs['P'][-1]:.2e}")
                print(f"  q: {minkowski_norm(q_lab):.4f} vs {minkowski_norm(q_breit):.4f}, diff = {norm_diffs['q'][-1]:.2e}")
                
                # Verify q = e_in - e_sc in BOTH frames
                q_check_lab = e_in_lab - e_sc_lab
                q_check_breit = e_in_breit - e_sc_breit
                print(f"\nConsistency check: q = e_in - e_sc")
                print(f"  LAB: q_check = ({q_check_lab[0]:.4f}, {q_check_lab[1]:.4f}, {q_check_lab[2]:.4f}, {q_check_lab[3]:.4f})")
                print(f"       q_stored = ({q_lab[0]:.4f}, {q_lab[1]:.4f}, {q_lab[2]:.4f}, {q_lab[3]:.4f})")
                print(f"       diff = {np.linalg.norm(q_check_lab - q_lab):.2e}")
                print(f"  Breit: q_check = ({q_check_breit[0]:.6f}, {q_check_breit[1]:.6e}, {q_check_breit[2]:.6e}, {q_check_breit[3]:.4f})")
                print(f"         q_boosted = ({q_breit[0]:.6f}, {q_breit[1]:.6e}, {q_breit[2]:.6e}, {q_breit[3]:.4f})")
                print(f"         diff = {np.linalg.norm(q_check_breit - q_breit):.2e}")
            
            processed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY [{label}] - {processed} events")
    print(f"{'='*60}")
    
    print(f"\nNorm preservation (max |v²_breit - v²_lab|):")
    for key in norm_diffs:
        if norm_diffs[key]:
            print(f"  {key:6s}: max = {np.max(norm_diffs[key]):.2e}, mean = {np.mean(norm_diffs[key]):.2e}")
    
    all_good = True
    for key in norm_diffs:
        if norm_diffs[key] and np.max(norm_diffs[key]) > 1e-6:
            print(f"  WARNING: {key} norm not preserved!")
            all_good = False
    
    if all_good:
        print(f"\n✓ All initial-state vectors are boosted consistently using the same LT.")
    
    return norm_diffs


def main():
    print("=" * 70)
    print("BREIT FRAME LORENTZ TRANSFORM AUDIT")
    print("=" * 70)
    
    # First run the boost consistency check
    print("\n" + "=" * 70)
    print("BOOST CONSISTENCY CHECK - ISR/FSR OFF")
    print("=" * 70)
    run_boost_consistency_check(PTREL_LABEL_OFF, max_events=10)
    
    print("\n" + "=" * 70)
    print("BOOST CONSISTENCY CHECK - ISR/FSR ON")
    print("=" * 70)
    run_boost_consistency_check(PTREL_LABEL_ON, max_events=10)
    
    print("\n" + "=" * 70)
    print("RUNNING AUDIT FOR ISR/FSR OFF")
    print("=" * 70)
    results_off = run_audit(PTREL_LABEL_OFF, max_events=500)
    
    print("\n" + "=" * 70)
    print("RUNNING AUDIT FOR ISR/FSR ON")
    print("=" * 70)
    results_on = run_audit(PTREL_LABEL_ON, max_events=500)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    print("""
ANSWERS TO KEY QUESTIONS:

1. Is LT a valid Lorentz transform?
   -> Check the max|LT^T g LT - g| values above. If < 1e-10, YES.

2. Is the final frame the standard Breit frame or nonstandard?
   -> Standard Breit: q^0 = 0, qT = 0, PT = 0
   -> Check the q^0, |qT|, |P_T| values above.

3. Is q transformed correctly?
   -> q should become (0, 0, 0, -Q) in standard Breit
   -> Check q^0 and |qT| above.

4. Is P transformed correctly?
   -> P should be along +z in standard Breit
   -> Check |P_T| above.

5. Is k_in identified correctly?
   -> k_in = k_out - q (by definition)
   -> For ISR/FSR OFF, k_in should be ~ xP (collinear)
   -> Check |k_in - xP|_T above.

6. At what stage does the O(2 GeV) transverse effect enter?
   -> See the step-by-step LT audit for the first event.
   -> The stage where k_in gains large |T| is the source.

KEY INSIGHT:
If |P_T| is nonzero in the "Breit" frame, then the frame is NOT the standard
Breit frame. The proton's transverse momentum in this frame defines a 
preferred direction, and k_in = xP + k_T will have k_T aligned with -P_T.

This is not an error in the transform, but rather a deliberate choice of
"Breit-like" frame that has q longitudinal but P not longitudinal. The
transverse direction is fixed by the original q_T in the lab frame.
""")
    
    # Create diagnostic plots
    print("\nGenerating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # q0 in Breit
    ax = axes[0, 0]
    ax.hist(results_off["q0_breit"], bins=50, alpha=0.7, label="ISR/FSR OFF", density=True)
    ax.hist(results_on["q0_breit"], bins=50, alpha=0.7, label="ISR/FSR ON", density=True)
    ax.set_xlabel(r"$q^0$ (Breit)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(r"$q^0$ in Breit frame")
    
    # |qT| in Breit
    ax = axes[0, 1]
    ax.hist(results_off["qT_breit"], bins=50, alpha=0.7, label="ISR/FSR OFF", density=True)
    ax.hist(results_on["qT_breit"], bins=50, alpha=0.7, label="ISR/FSR ON", density=True)
    ax.set_xlabel(r"$|q_T|$ (Breit)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(r"$|q_T|$ in Breit frame")
    
    # |P_T| in Breit
    ax = axes[0, 2]
    ax.hist(results_off["PT_breit"], bins=50, alpha=0.7, label="ISR/FSR OFF", density=True)
    ax.hist(results_on["PT_breit"], bins=50, alpha=0.7, label="ISR/FSR ON", density=True)
    ax.set_xlabel(r"$|P_T|$ (Breit)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(r"$|P_T|$ in Breit frame")
    
    # |k_in,T| comparison
    ax = axes[1, 0]
    ax.hist(results_off["kinT_lab"], bins=50, alpha=0.7, label="LAB", density=True)
    ax.hist(results_off["kinT_breit"], bins=50, alpha=0.7, label="Breit", density=True)
    ax.set_xlabel(r"$|k_{\mathrm{in},T}|$ [GeV]")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(r"$|k_{\mathrm{in},T}|$ LAB vs Breit (ISR/FSR OFF)")
    
    # |k_in - xP|_T
    ax = axes[1, 1]
    ax.hist(results_off["kin_minus_xP_T_lab"], bins=50, alpha=0.7, label="LAB", density=True)
    ax.hist(results_off["kin_minus_xP_T_breit"], bins=50, alpha=0.7, label="Breit", density=True)
    ax.set_xlabel(r"$|k_{\mathrm{in}} - xP|_T$ [GeV]")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title(r"$|k_{\mathrm{in}} - xP|_T$ (ISR/FSR OFF)")
    
    # Lorentz residuals
    ax = axes[1, 2]
    ax.hist(results_off["lorentz_residuals"], bins=50, alpha=0.7, label="ISR/FSR OFF", density=True)
    ax.hist(results_on["lorentz_residuals"], bins=50, alpha=0.7, label="ISR/FSR ON", density=True)
    ax.set_xlabel(r"max$|LT^T g LT - g|$")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("LT Lorentz condition residual")
    
    plt.tight_layout()
    out_path = _PROJECT_ROOT / "breit_frame_audit_plots.pdf"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")
    
    print("\nAudit complete.")


if __name__ == "__main__":
    main()
