import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dynamics import TripleLinkPendulum
from src.lqr_controller import LQRController
from src.mpc_controller import MPCController

def simulate_with_morphology_change(controller='lqr',
                                     t_span=(0, 15), dt=0.01,
                                     x0=None,
                                     change_time=5.0,
                                     m_before=[1,1,1], l_before=[1,1,1],
                                     m_after=[1.5,1,0.5], l_after=[1.2,1,0.8]):
    """
    Simulate with a mid-flight morphology change at change_time.

    Before change_time : original configuration
    After change_time  : new mass/length configuration

    LQR : NOT updated after change — shows degradation
    MPC : updates its internal model — shows adaptation
    """
    if x0 is None:
        x0 = np.array([0.2, -0.15, 0.1, 0.0, 0.0, 0.0])

    t_eval = np.arange(t_span[0], t_span[1], dt)
    X = np.zeros((len(t_eval), 6))
    U = np.zeros((len(t_eval), 3))
    X[0] = x0

    # Build both pendulum models
    pend_before = TripleLinkPendulum(m=m_before, l=l_before)
    pend_after  = TripleLinkPendulum(m=m_after,  l=l_after)
    A_before, B_before = pend_before.linearize()
    A_after,  B_after  = pend_after.linearize()

    # Build controller on initial morphology
    if controller == 'lqr':
        ctrl = LQRController(A_before, B_before)
    else:
        ctrl = MPCController(A_before, B_before, dt=dt)
        ctrl.reset(x0)

    morphology_changed = False

    for i in range(len(t_eval) - 1):

        # ── Morphology change trigger ──────────────────────────
        if t_eval[i] >= change_time and not morphology_changed:
            print(f"\n  Morphology change at t={t_eval[i]:.2f}s")
            print(f"  Mass:   {m_before} -> {m_after}")
            print(f"  Length: {l_before} -> {l_after}")

            # Apply disturbance kick simultaneously
            X[i] = X[i] + np.array([0.3, -0.25, 0.2, 0.1, -0.1, 0.05])
            print(f"  Disturbance applied at morphology change.")

            if controller == 'mpc':
                ctrl.update_model(A_after, B_after, current_x=X[i])

            morphology_changed = True

        current_pendulum = pend_after if morphology_changed else pend_before

        tau  = ctrl.compute_torque(X[i])
        U[i] = tau

        sol = solve_ivp(
            fun=lambda t, x: current_pendulum.derivatives(t, x, tau),
            t_span=(t_eval[i], t_eval[i+1]),
            y0=X[i], method='RK45', max_step=dt)
        X[i+1] = sol.y[:, -1]

        # Safety clip if state diverges
        if np.any(np.abs(X[i+1]) > 50):
            print(f"  Warning: diverged at t={t_eval[i+1]:.2f}s — clipping")
            X[i+1] = np.clip(X[i+1], -50, 50)

    U[-1] = ctrl.compute_torque(X[-1])
    print(f"\n{controller.upper()} done. Final state: {np.round(X[-1], 4)}")
    return t_eval, X, U


def compute_metrics(t, X, U, change_time=None, threshold=0.01):
    """Compute settling time, recovery time, overshoot, energy."""
    dt = t[1] - t[0]

    # Settling time
    settled_idx = None
    for i in range(len(t)):
        if np.all(np.abs(X[i, :3]) < threshold):
            if i + 50 < len(t) and np.all(np.abs(X[i:i+50, :3]) < threshold):
                settled_idx = i
                break
    settling_time = t[settled_idx] if settled_idx else t[-1]

    # Recovery time after morphology change
    recovery_time = None
    if change_time is not None:
        change_idx = np.argmin(np.abs(t - change_time))
        for i in range(change_idx, len(t)):
            if np.all(np.abs(X[i, :3]) < threshold):
                if i + 50 < len(t) and np.all(np.abs(X[i:i+50, :3]) < threshold):
                    recovery_time = round(t[i] - change_time, 3)
                    break

    max_overshoot = round(float(np.degrees(np.max(np.abs(X[:, :3])))), 3)
    energy        = round(float(np.sum(U**2) * dt), 3)

    return {
        'Settling Time (s)' : round(settling_time, 3),
        'Recovery Time (s)' : recovery_time if recovery_time else '>15s',
        'Max Overshoot (deg)': max_overshoot,
        'Total Energy (J)'  : energy,
    }


if __name__ == "__main__":
    print("=" * 55)
    print("  LQR vs MPC — Morphology Change Comparison")
    print("=" * 55)

    x0 = np.array([0.2, -0.15, 0.1, 0.0, 0.0, 0.0])

    print("\n[1/2] Running LQR...")
    t_lqr, X_lqr, U_lqr = simulate_with_morphology_change(
        controller='lqr', x0=x0,
        m_before=[1,1,1],  l_before=[1,1,1],
        m_after=[3,0.5,2], l_after=[1.5,0.5,1.5])

    print("\n[2/2] Running MPC...")
    t_mpc, X_mpc, U_mpc = simulate_with_morphology_change(
        controller='mpc', x0=x0,
        m_before=[1,1,1],  l_before=[1,1,1],
        m_after=[3,0.5,2], l_after=[1.5,0.5,1.5])

    m_lqr = compute_metrics(t_lqr, X_lqr, U_lqr, change_time=5.0)
    m_mpc = compute_metrics(t_mpc, X_mpc, U_mpc, change_time=5.0)

    print("\n" + "=" * 55)
    print(f"  {'Metric':<25} {'LQR':>12} {'MPC':>12}")
    print("-" * 55)
    for key in m_lqr:
        print(f"  {key:<25} {str(m_lqr[key]):>12} {str(m_mpc[key]):>12}")
    print("=" * 55)