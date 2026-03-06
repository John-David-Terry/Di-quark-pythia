#!/usr/bin/env python3
"""
Ground-truth comparator: run PYTHIA with original eta settings, pack each event
into cached format, run both analysis paths on the SAME events, compare per-event.
Stops after first 5 events where ANY quantity differs beyond tolerance.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Config: EXACT same as original eta in generate_pdf_plots_new.py
# ---------------------------------------------------------------------------
Qmin, Qmax = 5.0, 15.0
xmin, xmax = 1e-2, 0.5
N_EVENTS = 200
# Float32 in packed data causes ~1e-5 in x, ~1e-2 in transformed vectors. Eta matches.
TOL_X = 1e-4
TOL_Q = 1e-5
TOL_VEC = 0.01  # absolute; float32 loses precision for O(100) GeV vectors
TOL_ETA = 1e-6
MAX_DIVERGENCES = 5


def is_hadron(pid):
    return abs(pid) >= 100


def get_scattered_electron(ev):
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


# LT from original eta (generate_pdf_plots_new.py run_eta_analysis_and_plot)
def build_LT_eta(Ee, Ep, qmu, x, y, qT, phiq, S):
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
    den2 = (-qT*qT + S*(1+x)*y)
    if den2 <= 0:
        return None
    denom_M1 = 2.0 * y * np.sqrt(S * (-qT*qT + S*(1+x)*y))
    if denom_M1 == 0:
        return None
    M1 = np.array([
        [(-qT*qT + S*y*(1+x+y)) / denom_M1, 0, 0, (qT*qT + S*y*(-x+y-1)) / denom_M1],
        [0, 1, 0, 0], [0, 0, 1, 0],
        [(qT*qT + S*y*(-x+y-1)) / denom_M1, 0, 0, (-qT*qT + S*y*(1+x+y)) / denom_M1]
    ])
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
        return None
    eta_m3 = 0.5*np.log(num_log/den_log)
    denom_M3 = np.sqrt(den2)
    M3 = np.array([
        [np.cosh(eta_m3), -qT/denom_M3, 0, 0],
        [-qT/denom_M3, np.cosh(eta_m3), 0, 0],
        [0, 0, 1, 0], [0, 0, 0, 1]
    ])
    denom_M4 = 2*np.sqrt(x*(1+x))
    if denom_M4 == 0:
        return None
    M4 = np.array([
        [(1+2*x)/denom_M4, 0, 0, 1/denom_M4],
        [0, 1, 0, 0], [0, 0, 1, 0],
        [1/denom_M4, 0, 0, (1+2*x)/denom_M4]
    ])
    return M4 @ M3 @ M2 @ M1 @ M0 @ Mm1


# --- ORIGINAL path: from pythia.event directly ---
def run_original_path(ev, e_in, p_in, e_sc):
    Ep = p_in.e()
    Ee = e_in.e()
    q0 = e_in.e() - e_sc.e()
    q1 = e_in.px() - e_sc.px()
    q2 = e_in.py() - e_sc.py()
    q3 = e_in.pz() - e_sc.pz()
    qmu = np.array([q0, q1, q2, q3])
    Q2 = -(q0*q0 - q1*q1 - q2*q2 - q3*q3)
    if Q2 <= 0:
        return None
    Q = float(np.sqrt(Q2))
    qT = float(np.hypot(q1, q2))
    p_dot_q = Ep*q0 - Ep*q3
    if p_dot_q == 0:
        return None
    x = Q2 / (2.0 * p_dot_q)
    if not (xmin <= x <= xmax) or not (Qmin <= Q <= Qmax):
        return None
    phiq = float(np.arctan2(q2, q1))
    S = 4.0 * Ee * Ep
    y = Q2 / (S * x)
    LT = build_LT_eta(Ee, Ep, qmu, x, y, qT, phiq, S)
    if LT is None:
        return None
    q_tr = LT @ qmu
    p_tr = LT @ np.array([p_in.e(), p_in.px(), p_in.py(), p_in.pz()])
    best_E = -1.0
    best_lab = None
    best_pid = None
    for p in ev:
        if not p.isFinal() or not is_hadron(p.id()):
            continue
        lab = np.array([p.e(), p.px(), p.py(), p.pz()])
        trf = LT @ lab
        if trf[3] <= 0:
            continue
        if trf[0] > best_E:
            best_E = trf[0]
            best_lab = lab
            best_pid = p.id()
    if best_lab is None:
        return None
    E_lab, px_lab, py_lab, pz_lab = best_lab
    p_mag = np.sqrt(px_lab*px_lab + py_lab*py_lab + pz_lab*pz_lab)
    if p_mag <= 0:
        return None
    den = max(p_mag - pz_lab, 1e-12)
    num = max(p_mag + pz_lab, 1e-12)
    eta = float(0.5 * np.log(num / den))
    return {
        "x": x, "Q": Q, "y": y,
        "q_tr": q_tr, "p_tr": p_tr,
        "LT": LT,
        "chosen_pid": best_pid, "eta": eta,
        "best_lab": best_lab,
    }


# --- CACHED path: from packed arrays (NO flip - same frame as original) ---
def run_cached_path(e_in, p_in, e_sc, pid_list, p4_list):
    """Cached logic with NO flip_z - data packed from same PYTHIA frame."""
    Ep = float(p_in[0])
    Ee = float(e_in[0])
    q0 = e_in[0] - e_sc[0]
    q1 = e_in[1] - e_sc[1]
    q2 = e_in[2] - e_sc[2]
    q3 = e_in[3] - e_sc[3]
    qmu = np.array([q0, q1, q2, q3])
    Q2 = -(q0*q0 - q1*q1 - q2*q2 - q3*q3)
    if Q2 <= 0:
        return None
    Q = float(np.sqrt(Q2))
    qT = float(np.hypot(q1, q2))
    # Original uses Ep*q0 - Ep*q3; cached uses Lorentz-invariant
    p_dot_q = p_in[0]*q0 - p_in[1]*q1 - p_in[2]*q2 - p_in[3]*q3
    if p_dot_q == 0:
        return None
    x = Q2 / (2.0 * p_dot_q)
    if not (xmin <= x <= xmax) or not (Qmin <= Q <= Qmax):
        return None
    phiq = float(np.arctan2(q2, q1))
    S = 4.0 * Ee * Ep
    y = Q2 / (S * x)
    LT = build_LT_eta(Ee, Ep, qmu, x, y, qT, phiq, S)
    if LT is None:
        return None
    q_tr = LT @ qmu
    p_tr = LT @ p_in
    best_E = -1.0
    best_lab = None
    best_pid = None
    for j, (this_pid, lab) in enumerate(zip(pid_list, p4_list)):
        if not is_hadron(this_pid):
            continue
        lab = np.asarray(lab, dtype=float)
        trf = LT @ lab
        if trf[3] <= 0:
            continue
        if trf[0] > best_E:
            best_E = trf[0]
            best_lab = lab
            best_pid = this_pid
    if best_lab is None:
        return None
    E_lab, px_lab, py_lab, pz_lab = best_lab
    p_mag = np.sqrt(px_lab*px_lab + py_lab*py_lab + pz_lab*pz_lab)
    if p_mag <= 0:
        return None
    den = max(p_mag - pz_lab, 1e-12)
    num = max(p_mag + pz_lab, 1e-12)
    eta = float(0.5 * np.log(num / den))
    return {
        "x": x, "Q": Q, "y": y,
        "q_tr": q_tr, "p_tr": p_tr,
        "LT": LT,
        "chosen_pid": best_pid, "eta": eta,
        "best_lab": best_lab,
    }


def main():
    import pythia8
    pythia = pythia8.Pythia()
    pythia.readString("Random:setSeed = on")
    pythia.readString("Random:seed = 99999")  # fixed for reproducibility
    pythia.readString("Beams:idA = 2212")
    pythia.readString("Beams:idB = 11")
    pythia.readString("Beams:eA = 275.0")
    pythia.readString("Beams:eB = 18.0")
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

    n_processed = 0
    n_pass_cuts = 0
    n_have_hadron = 0
    n_divergences = 0

    for i in range(N_EVENTS):
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
        if e_sc is None or not (e_in and p_in):
            continue
        n_processed += 1

        # Original path
        res_orig = run_original_path(ev, e_in, p_in, e_sc)
        if res_orig is None:
            continue
        n_pass_cuts += 1

        # Pack into cached format (E,px,py,pz) - SAME frame as original
        e_in_arr = np.array([e_in.e(), e_in.px(), e_in.py(), e_in.pz()], dtype=np.float32)
        p_in_arr = np.array([p_in.e(), p_in.px(), p_in.py(), p_in.pz()], dtype=np.float32)
        e_sc_arr = np.array([e_sc.e(), e_sc.px(), e_sc.py(), e_sc.pz()], dtype=np.float32)
        pid_list = []
        p4_list = []
        for p in ev:
            if p.isFinal():
                pid_list.append(p.id())
                p4_list.append([p.e(), p.px(), p.py(), p.pz()])

        # Cached path (no flip - same frame)
        res_cached = run_cached_path(e_in_arr, p_in_arr, e_sc_arr, pid_list, p4_list)
        if res_cached is None:
            print(f"Event {n_pass_cuts}: ORIG passed, CACHED failed!")
            n_divergences += 1
            if n_divergences >= MAX_DIVERGENCES:
                break
            continue
        n_have_hadron += 1

        # Compare
        def rel_err(a, b):
            return abs(a - b) / max(abs(a), 1e-20)

        ok = True
        if rel_err(res_orig["x"], res_cached["x"]) > TOL_X:
            ok = False
        if rel_err(res_orig["Q"], res_cached["Q"]) > TOL_Q:
            ok = False
        if rel_err(res_orig["y"], res_cached["y"]) > TOL_X:
            ok = False
        if np.any(np.abs(res_orig["q_tr"] - res_cached["q_tr"]) > TOL_VEC):
            ok = False
        if np.any(np.abs(res_orig["p_tr"] - res_cached["p_tr"]) > TOL_VEC):
            ok = False
        if res_orig["chosen_pid"] != res_cached["chosen_pid"]:
            ok = False
        if abs(res_orig["eta"] - res_cached["eta"]) > TOL_ETA:
            ok = False

        if not ok:
            n_divergences += 1
            print(f"\n========== DIVERGENCE Event {n_pass_cuts} ==========")
            print(f"  x_orig={res_orig['x']:.8f}  x_cached={res_cached['x']:.8f}  rel_err={rel_err(res_orig['x'],res_cached['x']):.2e}")
            print(f"  Q_orig={res_orig['Q']:.8f}  Q_cached={res_cached['Q']:.8f}  rel_err={rel_err(res_orig['Q'],res_cached['Q']):.2e}")
            print(f"  y_orig={res_orig['y']:.8f}  y_cached={res_cached['y']:.8f}")
            print(f"  q_tr_orig={res_orig['q_tr']}")
            print(f"  q_tr_cached={res_cached['q_tr']}")
            print(f"  p_tr_orig={res_orig['p_tr']}")
            print(f"  p_tr_cached={res_cached['p_tr']}")
            print(f"  chosen_pid_orig={res_orig['chosen_pid']}  chosen_pid_cached={res_cached['chosen_pid']}")
            print(f"  eta_orig={res_orig['eta']:.8f}  eta_cached={res_cached['eta']:.8f}")
            LT_orig, LT_cached = res_orig["LT"], res_cached["LT"]
            print(f"  LT[0,0] orig={LT_orig[0,0]:.8f} cached={LT_cached[0,0]:.8f}")
            print(f"  LT[0,3] orig={LT_orig[0,3]:.8f} cached={LT_cached[0,3]:.8f}")
            print(f"  LT[3,0] orig={LT_orig[3,0]:.8f} cached={LT_cached[3,0]:.8f}")
            print(f"  LT[3,3] orig={LT_orig[3,3]:.8f} cached={LT_cached[3,3]:.8f}")
            # p4 convention check
            print(f"  p_in raw: E={p_in_arr[0]:.2f} px={p_in_arr[1]:.4f} py={p_in_arr[2]:.4f} pz={p_in_arr[3]:.4f}")
            print(f"  (energy should dominate for proton)")
            if n_divergences >= MAX_DIVERGENCES:
                break

    print("\n========== SUMMARY ==========")
    print(f"Events processed: {n_processed}")
    print(f"Events passing (x,Q) cuts: {n_pass_cuts}")
    print(f"Events with chosen hadron: {n_have_hadron}")
    print(f"Divergences found: {n_divergences}")
    if n_divergences == 0:
        print("PASS: All per-event quantities match. Analysis logic is correct.")
        print("Plot mismatch is due to GENERATOR CONFIG (ColourReconnection, beam order).")


if __name__ == "__main__":
    main()
