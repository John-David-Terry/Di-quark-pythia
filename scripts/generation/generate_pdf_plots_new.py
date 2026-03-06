#!/usr/bin/env python3
"""
Generate PDF plots (script version of generate_pdf_plots_new.ipynb).
  1. eta_hadron_EIC_hardware_QCD_regions.pdf
  2. pTrel_target_pion_wrt_remnant_axis_2D_fit_gaussian_comparison.pdf
Requires: pythia8, numpy, matplotlib

Run from project root:
  python scripts/generation/generate_pdf_plots_new.py
  python scripts/generation/generate_pdf_plots_new.py --use_cached_shards --label ISRFSR_ON
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl

# Configure matplotlib
mpl.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif', 'serif'],
    'mathtext.fontset': 'stix',
})

# ---------------------------------------------------------------------------
# Helpers (shared)
# ---------------------------------------------------------------------------
def get_scattered_electron(ev):
    """
    Return scattered electron from event.
    Prefer status 44 if present.
    Otherwise return highest-energy final-state electron.
    """
    electrons = []
    for i in range(ev.size()):
        p = ev[i]
        if p.id() == 11 and p.status() > 0:
            electrons.append(p)
    if not electrons:
        return None
    for e in electrons:
        if e.status() == 44:
            return e
    return max(electrons, key=lambda x: x.e())


def is_hadron(pid):
    return abs(pid) >= 100

def run_eta_analysis_and_plot():
    import pythia8
    pythia = pythia8.Pythia()
    # Reproducibility
    pythia.readString("Random:setSeed = on")
    pythia.readString("Random:seed = 12345")
    Ee_set, Ep_set = 18.0, 275.0
    pythia.readString("Beams:idA = 2212")
    pythia.readString("Beams:idB = 11")
    pythia.readString(f"Beams:eA = {Ep_set}")
    pythia.readString(f"Beams:eB = {Ee_set}")
    pythia.readString("Beams:frameType = 2")
    pythia.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    pythia.readString("HardQCD:all = off")
    pythia.readString("PDF:lepton = off")
    pythia.readString("PartonLevel:ISR = on")
    pythia.readString("PartonLevel:FSR = on")
    pythia.readString("HadronLevel:all = on")
    pythia.readString("ColourReconnection:reconnect = on")
    pythia.readString("PhaseSpace:Q2Min = 16.0")
    pythia.init()

    Qmin, Qmax = 5.0, 15.0
    xmin, xmax = 1e-2, 0.5
    n_events = 100_000
    eta_list = []
    x_list = []
    Q_list = []

    for iEvent in range(n_events):
        if not pythia.next():
            continue
        ev = pythia.event
        e_in = p_in = None
        for p in ev:
            if p.id() == 11 and p.status() == -12:
                e_in = p
            elif p.id() == 2212 and p.status() < 0:
                p_in = p
        e_sc = get_scattered_electron(ev)
        if e_sc is None:
            continue
        if not (e_in and p_in):
            continue
        Ep = p_in.e()
        q0 = e_in.e() - e_sc.e()
        q1 = e_in.px() - e_sc.px()
        q2 = e_in.py() - e_sc.py()
        q3 = e_in.pz() - e_sc.pz()
        qmu = np.array([q0, q1, q2, q3])
        Q2 = -(q0*q0 - q1*q1 - q2*q2 - q3*q3)
        if Q2 <= 0:
            continue
        Q = float(np.sqrt(Q2))
        qT = float(np.hypot(q1, q2))
        p_dot_q = Ep*q0 - Ep*q3
        if p_dot_q == 0:
            continue
        x = Q2 / (2.0 * p_dot_q)
        if not (xmin <= x <= xmax) or not (Qmin <= Q <= Qmax):
            continue
        phiq = float(np.arctan2(q2, q1))
        S = 4.0 * e_in.e() * Ep
        y = Q2 / (S * x)
        Mm1 = np.array([
            [e_in.e()/np.sqrt(S) + np.sqrt(S)/(4.0*e_in.e()), 0, 0, e_in.e()/np.sqrt(S) - np.sqrt(S)/(4.0*e_in.e())],
            [0, 1, 0, 0], [0, 0, 1, 0],
            [e_in.e()/np.sqrt(S) - np.sqrt(S)/(4.0*e_in.e()), 0, 0, e_in.e()/np.sqrt(S) + np.sqrt(S)/(4.0*e_in.e())]
        ])
        M0 = np.array([
            [1, 0, 0, 0],
            [0, np.cos(phiq), np.sin(phiq), 0],
            [0, -np.sin(phiq), np.cos(phiq), 0], [0, 0, 0, 1]
        ])
        denom_M1 = 2.0 * y * np.sqrt(S * (-qT*qT + S*(1+x)*y))
        if denom_M1 == 0:
            continue
        M1 = np.array([
            [(-qT*qT + S*y*(1+x+y)) / denom_M1, 0, 0, (qT*qT + S*y*(-x+y-1)) / denom_M1],
            [0, 1, 0, 0], [0, 0, 1, 0],
            [(qT*qT + S*y*(-x+y-1)) / denom_M1, 0, 0, (-qT*qT + S*y*(1+x+y)) / denom_M1]
        ])
        den2 = (-qT*qT + S*(1+x)*y)
        if den2 <= 0:
            continue
        denom_M2_s1 = np.sqrt(S*(1+x)*y / den2)
        denom_M2_s2 = np.sqrt(S*(1+x)*y)
        M2 = np.array([
            [1, 0, 0, 0],
            [0, 1/denom_M2_s1, 0, qT/denom_M2_s2],
            [0, 0, 1, 0], [0, -qT/denom_M2_s2, 0, 1/denom_M2_s1]
        ])
        num_log = qT + np.sqrt(S*(1+x)*y)
        den_log = np.sqrt(S*(1+x)*y) - qT
        if num_log <= 0 or den_log <= 0:
            continue
        eta_m3 = 0.5*np.log(num_log/den_log)
        denom_M3 = np.sqrt(den2)
        M3 = np.array([
            [np.cosh(eta_m3), -qT/denom_M3, 0, 0],
            [-qT/denom_M3, np.cosh(eta_m3), 0, 0],
            [0, 0, 1, 0], [0, 0, 0, 1]
        ])
        denom_M4 = 2*np.sqrt(x*(1+x))
        if denom_M4 == 0:
            continue
        M4 = np.array([
            [(1+2*x)/denom_M4, 0, 0, 1/denom_M4],
            [0, 1, 0, 0], [0, 0, 1, 0],
            [1/denom_M4, 0, 0, (1+2*x)/denom_M4]
        ])
        # Construct approximate Breit-like frame:
        # After transformation, the virtual photon should align with -z
        # and have ~0 energy component (within numerical tolerance).
        DEBUG_FRAME = False
        LT = M4 @ M3 @ M2 @ M1 @ M0 @ Mm1
        if DEBUG_FRAME:
            q_tr = LT @ qmu
            if abs(q_tr[0]) > 1e-3:
                print("Warning: photon energy not ~0 in transformed frame:", q_tr[0])
        def boost(v):
            return LT @ np.array(v)
        best_E = -1.0
        best_lab = None
        best_trf = None
        for p in ev:
            if not p.isFinal() or not is_hadron(p.id()):
                continue
            lab = np.array([p.e(), p.px(), p.py(), p.pz()])
            trf = boost(lab)
            E_, px_, py_, pz_ = trf
            if pz_ <= 0:
                continue
            if E_ > best_E:
                best_E = E_
                best_lab = lab
                best_trf = trf
        if best_trf is None:
            continue
        E_lab, px_lab, py_lab, pz_lab = best_lab
        p_mag = np.sqrt(px_lab*px_lab + py_lab*py_lab + pz_lab*pz_lab)
        if p_mag <= 0:
            continue
        # NOTE:
        # Hadron is selected in transformed (Breit-like) frame
        # but pseudorapidity is computed in LAB frame
        den = max(p_mag - pz_lab, 1e-12)
        num = max(p_mag + pz_lab, 1e-12)
        eta = float(0.5 * np.log(num / den))
        eta_list.append(eta)
        x_list.append(x)
        Q_list.append(Q)

    # Plot eta_hadron_EIC_hardware_QCD_regions.pdf
    fontsize = 15
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['legend.fontsize'] = fontsize
    avg_x = np.mean(x_list) if len(x_list) > 0 else 0.0
    avg_Q = np.mean(Q_list) if len(Q_list) > 0 else 0.0
    fig, ax = plt.subplots()
    ax.hist(eta_list, bins=80, range=(2, 8), histtype='step', color='k', density=True, linewidth=1.5)
    ax.axvspan(4.6, 5.9, alpha=0.18, color='C0', zorder=0)
    ax.axvspan(6.0, 8.0, alpha=0.18, color='C0', zorder=0)
    ax.set_xlabel(r"$\eta_{h}$", fontsize=fontsize)
    ax.set_ylabel(r"$\dfrac{1}{\sigma} \dfrac{d\sigma}{d\eta_h}$", fontsize=fontsize)
    ax.grid(False)
    ax.tick_params(direction='in', labelsize=fontsize)
    handles = [mpatches.Patch(color='C0', alpha=0.18, label=r"EIC coverage")]
    ax.legend(handles=handles, frameon=False, loc='upper left', fontsize=fontsize)
    if len(x_list) > 0 and len(Q_list) > 0:
        text_str = f"$\\langle x \\rangle = {avg_x:.4f}$\n$\\langle Q \\rangle = {avg_Q:.2f}$ GeV"
        ax.text(0.05, 0.87, text_str, transform=ax.transAxes,
                fontsize=fontsize, verticalalignment='top', horizontalalignment='left')
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    plt.savefig(_PROJECT_ROOT / "eta_hadron_EIC_hardware_QCD_regions.pdf", format="pdf")
    plt.close(fig)
    print("Saved: eta_hadron_EIC_hardware_QCD_regions.pdf")


def p3(v):
    return np.array([v[1], v[2], v[3]], dtype=float)

def pT_rel_to_axis(p_vec, axis_vec):
    na = np.linalg.norm(axis_vec)
    if na <= 0:
        return None
    n = axis_vec / na
    p_par = np.dot(p_vec, n) * n
    p_perp = p_vec - p_par
    pTrel2 = float(np.dot(p_perp, p_perp))
    pTrel2 = max(pTrel2, 0.0)
    return float(np.sqrt(pTrel2))

def dot4(a, b):
    return a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]

def p4(p):
    return np.array([p.e(), p.px(), p.py(), p.pz()], dtype=float)


# For OLD-vs-NEW diff harness: dump indices, triplets, and debug records (saved with _OLD suffix).
DEBUG_PTREL_OLD = True
DEBUG_EVENT_LIST_OLD = {0, 1, 2, 3, 4, 5, 10, 25, 49, 1234}


def _run_pTrel_loop(pythia, n_events, Qmin, Qmax, xmin, xmax, out_pTrel, out_x, out_Q, label=None):
    """Single event loop: fill out_pTrel, out_x, out_Q. Uses remnant axis and target-leading pi+/-."""
    used_event_indices = []
    used_event_triplets = []
    debug_records = []
    for iEvent in range(n_events):
        if not pythia.next():
            continue
        ev = pythia.event
        e_in = p_in = None
        for p in ev:
            if p.id() == 11 and p.status() == -12:
                e_in = p
            elif p.id() == 2212 and p.status() < 0:
                p_in = p
        e_sc = get_scattered_electron(ev)
        if e_sc is None:
            continue
        if not (e_in and p_in):
            continue
        Ee, Ep = e_in.e(), p_in.e()
        q0 = e_in.e() - e_sc.e()
        q1 = e_in.px() - e_sc.px()
        q2 = e_in.py() - e_sc.py()
        q3 = e_in.pz() - e_sc.pz()
        qmu = np.array([q0, q1, q2, q3], dtype=float)
        Q2 = -(q0*q0 - q1*q1 - q2*q2 - q3*q3)
        if Q2 <= 0:
            continue
        Q = float(np.sqrt(Q2))
        qT = float(np.hypot(q1, q2))
        p_dot_q = Ep*q0 - Ep*q3
        if p_dot_q == 0:
            continue
        x = Q2 / (2.0 * p_dot_q)
        if not (xmin <= x <= xmax) or not (Qmin <= Q <= Qmax):
            continue
        phiq = float(np.arctan2(q2, q1))
        S = 4.0 * Ee * Ep
        y = Q2 / (S * x)
        Mm1 = np.array([
            [Ee/np.sqrt(S) + np.sqrt(S)/(4.0*Ee), 0, 0, Ee/np.sqrt(S) - np.sqrt(S)/(4.0*Ee)],
            [0, 1, 0, 0], [0, 0, 1, 0],
            [Ee/np.sqrt(S) - np.sqrt(S)/(4.0*Ee), 0, 0, Ee/np.sqrt(S) + np.sqrt(S)/(4.0*Ee)]
        ])
        denom_M1 = 2.0 * y * np.sqrt(S * (S*(1+x)*y - qT*qT))
        if denom_M1 == 0:
            continue
        M1 = np.array([
            [(S*y*(x+y+1) - qT*qT) / denom_M1, 0, 0, (qT*qT + S*y*(-x+y-1)) / denom_M1],
            [0, 1, 0, 0], [0, 0, 1, 0],
            [(qT*qT + S*y*(-x+y-1)) / denom_M1, 0, 0, (S*y*(x+y+1) - qT*qT) / denom_M1]
        ])
        M0 = np.array([
            [1, 0, 0, 0],
            [0, np.cos(phiq), np.sin(phiq), 0],
            [0, -np.sin(phiq), np.cos(phiq), 0], [0, 0, 0, 1]
        ])
        den2 = S*(1+x)*y - qT*qT
        if den2 <= 0:
            continue
        denom_M2_s1 = np.sqrt(S*(1+x)*y / den2)
        denom_M2_s2 = np.sqrt(S*(1+x)*y)
        M2 = np.array([
            [1, 0, 0, 0],
            [0, 1/denom_M2_s1, 0, qT/denom_M2_s2],
            [0, 0, 1, 0], [0, -qT/denom_M2_s2, 0, 1/denom_M2_s1]
        ])
        denom_M3 = np.sqrt(den2)
        num_log = qT + np.sqrt(S*(1+x)*y)
        den_log = np.sqrt(S*(1+x)*y) - qT
        if num_log <= 0 or den_log <= 0:
            continue
        eta = 0.5*np.log(num_log/den_log)
        M3 = np.array([
            [np.cosh(eta), -qT/denom_M3, 0, 0],
            [-qT/denom_M3, np.cosh(eta), 0, 0],
            [0, 0, 1, 0], [0, 0, 0, 1]
        ])
        denom_M4 = 2*np.sqrt(x*(1+x))
        if denom_M4 == 0:
            continue
        M4 = np.array([
            [(1+2*x)/denom_M4, 0, 0, 1/denom_M4],
            [0, 1, 0, 0], [0, 0, 1, 0],
            [1/denom_M4, 0, 0, (1+2*x)/denom_M4]
        ])
        # Construct approximate Breit-like frame:
        # After transformation, the virtual photon should align with -z
        # and have ~0 energy component (within numerical tolerance).
        DEBUG_FRAME = False
        LT = M4 @ M3 @ M2 @ M1 @ M0 @ Mm1
        if DEBUG_FRAME:
            q_tr = LT @ qmu
            if abs(q_tr[0]) > 1e-3:
                print("Warning: photon energy not ~0 in transformed frame:", q_tr[0])
        def boost(v):
            return LT @ np.array(v)
        q_breit = boost(qmu)
        p_breit = boost([p_in.e(), p_in.px(), p_in.py(), p_in.pz()])
        P_plus = float(p_breit[0] + p_breit[3])
        if P_plus <= 0:
            continue
        best_tarE = -1.0
        best_tar_breit = None
        best_tar_pid = None
        for p in ev:
            if not p.isFinal() or not is_hadron(p.id()):
                continue
            lab = np.array([p.e(), p.px(), p.py(), p.pz()])
            trf = boost(lab)
            E_, px_, py_, pz_ = trf
            if pz_ <= 0:
                continue
            if E_ > best_tarE:
                best_tarE = E_
                best_tar_breit = trf
                best_tar_pid = p.id()
        if best_tar_breit is None or abs(best_tar_pid) != 211:
            continue
        k_out = None
        for pp in ev:
            pid, status = pp.id(), abs(pp.status())
            if 1 <= abs(pid) <= 6 and status == 23:
                k_out = p4(pp)
                break
        if k_out is None:
            for pp in ev:
                pid, status = pp.id(), abs(pp.status())
                if 1 <= abs(pid) <= 6 and 63 <= status <= 68:
                    k_out = p4(pp)
                    break
        if k_out is None:
            for pp in ev:
                if 1 <= abs(pp.id()) <= 6 and pp.status() > 0:
                    k_out = p4(pp)
                    break
        if k_out is None:
            continue
        k_in = k_out - qmu
        p_rem_truth = p4(p_in) - k_in
        p_rem_truth_breit = boost(p_rem_truth)
        axis_rem = p3(p_rem_truth_breit)
        if np.linalg.norm(axis_rem) <= 0:
            continue
        pTrel_rem = pT_rel_to_axis(p3(best_tar_breit), axis_rem)
        if pTrel_rem is None:
            continue
        den = dot4(p_rem_truth_breit, q_breit)
        if den <= 0:
            continue
        xL_exact = dot4(best_tar_breit, q_breit) / den
        if xL_exact < 0.01 or xL_exact > 1.0 + 1e-6:
            continue
        event_index = len(out_pTrel)
        p_breit_arr = np.asarray(p_breit, dtype=float)
        P_jet_breit = p_breit_arr - p_rem_truth_breit
        if label and DEBUG_PTREL_OLD and event_index in DEBUG_EVENT_LIST_OLD:
            # row: event_idx, x, Q, y, pTrel_rem, p_pi_xyz, p_rem_xyz, p_jet_xyz (3-mom from 4-vec [1:4])
            debug_records.append([
                float(event_index), float(x), float(Q), float(y), float(pTrel_rem),
                float(best_tar_breit[1]), float(best_tar_breit[2]), float(best_tar_breit[3]),
                float(p_rem_truth_breit[1]), float(p_rem_truth_breit[2]), float(p_rem_truth_breit[3]),
                float(P_jet_breit[1]), float(P_jet_breit[2]), float(P_jet_breit[3]),
            ])
        out_pTrel.append(pTrel_rem)
        out_x.append(x)
        out_Q.append(Q)
        used_event_indices.append(event_index)
        used_event_triplets.append((event_index, int(best_tar_pid), 0))

    if label and DEBUG_PTREL_OLD and used_event_indices:
        used_event_indices = np.asarray(used_event_indices, dtype=np.int64)
        np.save(_PROJECT_ROOT / f"used_event_indices_{label}_OLD.npy", used_event_indices)
        used_event_triplets = np.asarray(used_event_triplets, dtype=np.int64)
        np.save(_PROJECT_ROOT / f"used_event_triplets_{label}_OLD.npy", used_event_triplets)
        if debug_records:
            debug_records = np.array(debug_records, dtype=np.float64)
            np.save(_PROJECT_ROOT / f"debug_records_{label}_OLD.npy", debug_records)


def run_ptrel_from_cached_shards(label: str):
    """
    Run pTrel computation on cached shards (same as NEW pipeline).
    Saves used_event_ids_{label}_OLD.npy, used_event_triplets_{label}_OLD.npy,
    debug_records_{label}_OLD.npy for diff_old_new_ptrel.py.
    """
    from diquark.analyze_events_raw import (
        FLIP_Z_PTREL,
        build_LT,
        dot4,
        flip_z,
        is_hadron,
        p3,
        pT_rel_to_axis,
        xmax_ptrel,
        xmin_ptrel,
        Qmax_ptrel,
        Qmin_ptrel,
    )
    from diquark.cached_shards import iter_events_from_shards

    used_event_ids = []
    used_event_triplets = []
    debug_records = []
    out_pTrel = []
    out_x = []
    out_Q = []
    processed = 0

    for shard_idx, ie, data in iter_events_from_shards(label, flip_z=FLIP_Z_PTREL):
        e_in_ev = data["event_e_in"]
        p_in_ev = data["event_p_in"]
        e_sc_ev = data["event_e_sc"]
        k_out_ev = data["event_k_out"]
        offsets = data["offsets"]
        pid = data["pid"]
        p4_arr = data["p4"]

        Ep = float(p_in_ev[0])
        Ee = float(e_in_ev[0])
        q0 = e_in_ev[0] - e_sc_ev[0]
        q1 = e_in_ev[1] - e_sc_ev[1]
        q2 = e_in_ev[2] - e_sc_ev[2]
        q3 = e_in_ev[3] - e_sc_ev[3]
        qmu = np.array([q0, q1, q2, q3], dtype=float)
        Q2 = -(q0 * q0 - q1 * q1 - q2 * q2 - q3 * q3)
        if Q2 <= 0:
            continue
        Q = float(np.sqrt(Q2))
        qT = float(np.hypot(q1, q2))
        p_dot_q = p_in_ev[0] * q0 - p_in_ev[1] * q1 - p_in_ev[2] * q2 - p_in_ev[3] * q3
        if p_dot_q == 0:
            continue
        x = Q2 / (2.0 * p_dot_q)
        if not (xmin_ptrel <= x <= xmax_ptrel) or not (Qmin_ptrel <= Q <= Qmax_ptrel):
            continue
        phiq = float(np.arctan2(q2, q1))
        S = 4.0 * Ee * Ep
        y = Q2 / (S * x)
        LT = build_LT(Ee, Ep, qmu, x, y, qT, phiq, S)
        if LT is None:
            continue
        boost = lambda v: LT @ np.asarray(v, dtype=float)
        p_breit = boost(p_in_ev)
        P_plus = float(p_breit[0] + p_breit[3])
        if P_plus <= 0:
            continue

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

        k_in = k_out_ev - qmu
        p_rem_truth = np.asarray(p_in_ev, dtype=float) - k_in
        p_rem_truth_breit = boost(p_rem_truth)
        axis_rem = p3(p_rem_truth_breit)
        if np.linalg.norm(axis_rem) <= 0:
            continue
        pTrel_rem = pT_rel_to_axis(p3(best_tar_breit), axis_rem)
        if pTrel_rem is None:
            continue
        q_breit = boost(qmu)
        den = dot4(p_rem_truth_breit, q_breit)
        if den <= 0:
            continue
        xL_exact = dot4(best_tar_breit, q_breit) / den
        if xL_exact < 0.01 or xL_exact > 1.0 + 1e-6:
            continue

        p_proton_breit = np.asarray(p_breit, dtype=float)
        P_jet_breit = p_proton_breit - p_rem_truth_breit
        event_id = (int(shard_idx), int(ie))

        if processed in DEBUG_EVENT_LIST_OLD:
            debug_records.append([
                float(event_id[0]), float(event_id[1]),
                float(x), float(Q), float(y), float(pTrel_rem),
                float(best_tar_breit[1]), float(best_tar_breit[2]), float(best_tar_breit[3]),
                float(p_rem_truth_breit[1]), float(p_rem_truth_breit[2]), float(p_rem_truth_breit[3]),
                float(P_jet_breit[1]), float(P_jet_breit[2]), float(P_jet_breit[3]),
            ])

        out_pTrel.append(pTrel_rem)
        out_x.append(x)
        out_Q.append(Q)
        used_event_ids.append(event_id)
        used_event_triplets.append((event_id[0], event_id[1], int(best_tar_pid)))
        processed += 1

    used_event_ids = np.asarray(used_event_ids, dtype=np.int64)
    used_event_triplets = np.asarray(used_event_triplets, dtype=np.int64)
    np.save(_PROJECT_ROOT / f"used_event_ids_{label}_OLD.npy", used_event_ids)
    np.save(_PROJECT_ROOT / f"used_event_triplets_{label}_OLD.npy", used_event_triplets)
    if debug_records:
        debug_records = np.array(debug_records, dtype=np.float64)
        np.save(_PROJECT_ROOT / f"debug_records_{label}_OLD.npy", debug_records)
    print(f"[{label}] cached mode: {processed} events -> used_event_ids_*_OLD.npy, debug_records_*_OLD.npy")


def run_pTrel_comparison_and_plot():
    import pythia8
    Ee_set, Ep_set = 18.0, 275.0
    Qmin, Qmax = 5.0, 15.0
    xmin, xmax = 1e-3, 0.5
    n_events = 100_000

    # OFF run
    pythia_off = pythia8.Pythia()
    # Reproducibility (different seed per configuration)
    pythia_off.readString("Random:setSeed = on")
    pythia_off.readString("Random:seed = 12345")
    pythia_off.readString("Beams:idA = 2212")
    pythia_off.readString("Beams:idB = 11")
    pythia_off.readString(f"Beams:eA = {Ep_set}")
    pythia_off.readString(f"Beams:eB = {Ee_set}")
    pythia_off.readString("Beams:frameType = 2")
    pythia_off.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    pythia_off.readString("HardQCD:all = off")
    pythia_off.readString("PDF:lepton = off")
    pythia_off.readString("PartonLevel:ISR = off")
    pythia_off.readString("PartonLevel:FSR = off")
    pythia_off.readString("HadronLevel:all = on")
    pythia_off.readString("ColourReconnection:reconnect = off")
    pythia_off.readString("PhaseSpace:Q2Min = 16.0")
    pythia_off.init()
    pTrel_to_remnant_list = []
    x_list = []
    Q_list = []
    print("Running PYTHIA ISR/FSR OFF...")
    _run_pTrel_loop(pythia_off, n_events, Qmin, Qmax, xmin, xmax,
                    pTrel_to_remnant_list, x_list, Q_list, label="ISRFSR_OFF")

    # ON run (use x_list_on, Q_list_on so comparison has both)
    pythia_on = pythia8.Pythia()
    # Reproducibility (different seed per configuration)
    pythia_on.readString("Random:setSeed = on")
    pythia_on.readString("Random:seed = 12346")
    pythia_on.readString("Beams:idA = 2212")
    pythia_on.readString("Beams:idB = 11")
    pythia_on.readString(f"Beams:eA = {Ep_set}")
    pythia_on.readString(f"Beams:eB = {Ee_set}")
    pythia_on.readString("Beams:frameType = 2")
    pythia_on.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    pythia_on.readString("HardQCD:all = off")
    pythia_on.readString("PDF:lepton = off")
    pythia_on.readString("PartonLevel:ISR = on")
    pythia_on.readString("PartonLevel:FSR = on")
    pythia_on.readString("HadronLevel:all = on")
    pythia_on.readString("ColourReconnection:reconnect = off")
    pythia_on.readString("PhaseSpace:Q2Min = 16.0")
    pythia_on.init()
    pTrel_to_remnant_list_on = []
    x_list_on = []
    Q_list_on = []
    print("Running PYTHIA ISR/FSR ON...")
    _run_pTrel_loop(pythia_on, n_events, Qmin, Qmax, xmin, xmax,
                    pTrel_to_remnant_list_on, x_list_on, Q_list_on, label="ISRFSR_ON")

    # Fit and comparison plot
    pTmax_fit = 1.0
    eps = 1e-12

    def fit_gaussian_single_width(pT_list, label_suffix=""):
        if len(pT_list) == 0:
            return None, None, None, None, None, None
        counts, edges = np.histogram(pT_list, bins=60, range=(0.01, 2.0), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        binw = edges[1:] - edges[:-1]
        y2d = counts / centers
        N = len(pT_list)
        n_i_est = np.maximum(N * counts * binw, 1.0)
        sigma_counts = np.sqrt(n_i_est) / (N * binw)
        sigma_y2d = sigma_counts / centers
        mask = (centers < pTmax_fit) & (y2d > 0) & np.isfinite(y2d) & np.isfinite(sigma_y2d)
        if np.sum(mask) < 3:
            return None, None, None, None, None, None
        X = centers[mask]**2
        Y = np.log(y2d[mask] + eps)
        sigma_lnY = sigma_y2d[mask] / (y2d[mask] + eps)
        w = 1.0 / np.maximum(sigma_lnY, 1e-6)**2
        Y_mean = np.average(Y, weights=w)
        Y_centered = Y - Y_mean
        X_mean = np.average(X, weights=w)
        X_centered = X - X_mean
        numerator = np.sum(w * X_centered * Y_centered)
        denominator = np.sum(w * X_centered**2)
        c1 = numerator / denominator if denominator > 0 else 0.0
        if c1 < -1e-8:
            B_fit = -1.0 / c1
        else:
            B_fit = 1.0
        B_fit = min(B_fit, 10.0)
        p_integrate = np.linspace(0.01, 2.0, 1000)
        integrand = p_integrate * np.exp(-(p_integrate**2) / B_fit)
        integral = np.trapz(integrand, p_integrate)
        A_fit = 1.0 / integral if integral > 0 else 1.0
        return centers, y2d, A_fit, B_fit, label_suffix, len(pT_list)

    centers_off, y2d_off, A_off, B_off, _, n_off = fit_gaussian_single_width(pTrel_to_remnant_list, "OFF")
    centers_on, y2d_on, A_on, B_on, _, n_on = fit_gaussian_single_width(pTrel_to_remnant_list_on, "ON")

    if centers_off is None or centers_on is None:
        print("Could not fit one or both datasets.")
        return
    avg_x_off = np.mean(x_list) if len(x_list) > 0 else 0.0
    avg_Q_off = np.mean(Q_list) if len(Q_list) > 0 else 0.0
    avg_x_on = np.mean(x_list_on) if len(x_list_on) > 0 else 0.0
    avg_Q_on = np.mean(Q_list_on) if len(Q_list_on) > 0 else 0.0
    print("Gaussian fit results:")
    print(f"  ISR/FSR OFF: B = {B_off:.6f}, A = {A_off:.6f}, N = {n_off}")
    print(f"  ISR/FSR ON:  B = {B_on:.6f}, A = {A_on:.6f}, N = {n_on}")

    fontsize = 20.25
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['legend.fontsize'] = fontsize
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(centers_off, y2d_off, where='mid', label='PYTHIA: ISR/FSR Off', linewidth=2, alpha=0.8, color='blue')
    ax.step(centers_on, y2d_on, where='mid', label='PYTHIA: ISR/FSR On', linewidth=2, alpha=0.8, color='red')
    p_grid = np.linspace(0.01, 2.0, 200)
    ax.plot(p_grid, A_off * np.exp(-(p_grid**2) / B_off), 'b--', alpha=0.7, linewidth=1.5, label='Fit a')
    ax.plot(p_grid, A_on * np.exp(-(p_grid**2) / B_on), 'r--', alpha=0.7, linewidth=1.5, label='Fit b')
    ax.set_xlabel(r"$P_{h\perp}$", fontsize=fontsize)
    ax.set_ylabel(r"$\dfrac{1}{\sigma}\dfrac{d\sigma}{d P_{h\perp}}$", fontsize=fontsize)
    ax.tick_params(direction='in', labelsize=fontsize)
    ax.legend(loc='best', frameon=False, fontsize=fontsize)
    if avg_x_off > 0 and avg_Q_off > 0:
        text_str = f"$\\langle x \\rangle = {avg_x_off:.4f}$, $\\langle Q \\rangle = {avg_Q_off:.2f}$ GeV"
        ax.text(0.98, 0.45, text_str, transform=ax.transAxes,
                fontsize=fontsize, verticalalignment='top', horizontalalignment='right')
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    plt.savefig(_PROJECT_ROOT / "pTrel_target_pion_wrt_remnant_axis_2D_fit_gaussian_comparison.pdf", format="pdf")
    plt.close(fig)
    print("Saved: pTrel_target_pion_wrt_remnant_axis_2D_fit_gaussian_comparison.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PDF plots or run pTrel from cached shards (diff harness).")
    parser.add_argument("--use_cached_shards", action="store_true", help="Read shards instead of PYTHIA; for OLD vs NEW diff.")
    parser.add_argument("--label", type=str, default=None, help="Label for cached mode (e.g. ISRFSR_ON, ISRFSR_OFF).")
    args = parser.parse_args()

    if args.use_cached_shards and args.label:
        print(f"Running pTrel from cached shards (label={args.label}) ...")
        run_ptrel_from_cached_shards(args.label)
        print("Done.")
    else:
        if args.use_cached_shards or args.label:
            parser.error("--use_cached_shards requires --label (e.g. ISRFSR_ON or ISRFSR_OFF).")
        print("Generating eta_hadron_EIC_hardware_QCD_regions.pdf ...")
        run_eta_analysis_and_plot()
        print("Generating pTrel_target_pion_wrt_remnant_axis_2D_fit_gaussian_comparison.pdf ...")
        run_pTrel_comparison_and_plot()
        print("Done.")
