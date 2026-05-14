import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.aerial_robot_dynamics import Aerial_robotAerialModel
from src.lqr_controller import LQRController
from src.mpc_controller import MPCController


def simulate_aerial_robot_lqr(t_span=(0,10), dt=0.01, x0=None,
                         m=[1,1,1], l=[1,1,1]):
    """Simulate AERIAL_ROBOT aerial model under LQR control."""
    aerial_robot = Aerial_robotAerialModel(m=m, l=l)
    A, B   = aerial_robot.linearize()

    Q = np.diag([1.0, 10.0, 10.0, 10.0, 10.0,
                 0.1,  1.0,  1.0,  1.0,  1.0])
    R = np.eye(6) * 5.0

    lqr = LQRController(A, B, Q=Q, R=R)

    if x0 is None:
        x0 = np.array([0.0, 0.1, 0.2, -0.15, 0.1,
                        0.0, 0.0, 0.0,  0.0,  0.0])

    t_eval = np.arange(t_span[0], t_span[1], dt)
    X = np.zeros((len(t_eval), 10))
    U = np.zeros((len(t_eval), 6))
    X[0] = x0

    u_hover = aerial_robot.hover_control()

    for i in range(len(t_eval) - 1):
        delta_u = lqr.compute_torque(X[i])
        u_total = u_hover + delta_u
        U[i]    = delta_u

        sol = solve_ivp(
            fun=lambda t, x: aerial_robot.derivatives(t, x, u_total),
            t_span=(t_eval[i], t_eval[i+1]),
            y0=X[i], method='RK45', max_step=dt)
        X[i+1] = sol.y[:, -1]

        if np.any(np.abs(X[i+1]) > 50):
            X[i+1] = np.clip(X[i+1], -50, 50)

    U[-1] = lqr.compute_torque(X[-1])
    print(f"LQR aerial done. Final state: {np.round(X[-1], 4)}")
    return t_eval, X, U


def simulate_aerial_robot_morphology(controller='lqr', t_span=(0,15), dt=0.01,
                                x0=None, change_time=5.0,
                                m_before=[1,1,1], l_before=[1,1,1],
                                m_after=[2,0.5,1.5], l_after=[1.3,0.7,1.2]):
    """Simulate aerial model with mid-flight morphology change."""
    drag_before = Aerial_robotAerialModel(m=m_before, l=l_before)
    drag_after  = Aerial_robotAerialModel(m=m_after,  l=l_after)
    A_before, B_before = drag_before.linearize()
    A_after,  B_after  = drag_after.linearize()

    Q = np.diag([1.0, 10.0, 10.0, 10.0, 10.0,
             0.1,  1.0,  1.0,  1.0,  1.0])
    R = np.eye(6) * 5.0   # increased from 0.1

    if x0 is None:
        x0 = np.array([0.0, 0.1, 0.2, -0.15, 0.1,
                        0.0, 0.0, 0.0,  0.0,  0.0])

    if controller == 'lqr':
        ctrl = LQRController(A_before, B_before, Q=Q, R=R)
    else:
        ctrl = MPCController(A_before, B_before, dt=dt, Q=Q, R=R)

    t_eval = np.arange(t_span[0], t_span[1], dt)
    X = np.zeros((len(t_eval), 10))
    U = np.zeros((len(t_eval), 6))
    X[0] = x0

    morphology_changed = False

    for i in range(len(t_eval) - 1):
        if t_eval[i] >= change_time and not morphology_changed:
            print(f"\n  Aerial morphology change at t={t_eval[i]:.2f}s")
            print(f"  Mass:   {m_before} -> {m_after}")
            print(f"  Length: {l_before} -> {l_after}")

            X[i] += np.array([0.0, 0.15, 0.2, -0.15, 0.1,
                               0.0, 0.05, 0.0,  0.0,  0.0])

            if controller == 'mpc':
                ctrl.update_model(A_after, B_after, current_x=X[i])

            morphology_changed = True

        current_aerial_robot = drag_after if morphology_changed else drag_before
        u_hover        = current_aerial_robot.hover_control()

        delta_u = ctrl.compute_torque(X[i])
        u_total = u_hover + delta_u
        U[i]    = delta_u

        sol = solve_ivp(
            fun=lambda t, x: current_aerial_robot.derivatives(t, x, u_total),
            t_span=(t_eval[i], t_eval[i+1]),
            y0=X[i], method='RK45', max_step=dt)
        X[i+1] = sol.y[:, -1]

        if np.any(np.abs(X[i+1]) > 50):
            X[i+1] = np.clip(X[i+1], -50, 50)

    U[-1] = ctrl.compute_torque(X[-1])
    print(f"\n{controller.upper()} aerial done. Final: {np.round(X[-1], 4)}")
    return t_eval, X, U


def compute_aerial_metrics(t, X, U, change_time=None, threshold=0.05):
    """Metrics for aerial model."""
    dt = t[1] - t[0]

    settled_idx = None
    for i in range(len(t)):
        if (abs(X[i,1]) < threshold and
                np.all(np.abs(X[i,2:5]) < threshold)):
            if i + 50 < len(t):
                settled_idx = i
                break

    settling_time = t[settled_idx] if settled_idx else t[-1]

    recovery_time = None
    if change_time is not None:
        change_idx = np.argmin(np.abs(t - change_time))
        for i in range(change_idx, len(t)):
            if (abs(X[i,1]) < threshold and
                    np.all(np.abs(X[i,2:5]) < threshold)):
                if i + 50 < len(t):
                    recovery_time = round(t[i] - change_time, 3)
                    break

    return {
        'Settling Time (s)'  : round(settling_time, 3),
        'Recovery Time (s)'  : recovery_time if recovery_time else '>15s',
        'Max Tilt (deg)'     : round(float(np.degrees(np.max(np.abs(X[:,2:5])))), 3),
        'Max Z Drift (m)'    : round(float(np.max(np.abs(X[:,1]))), 3),
        'Total Energy (J)'   : round(float(np.sum(U**2) * dt), 3),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  AERIAL_ROBOT Aerial Model — LQR vs MPC Comparison")
    print("=" * 60)

    print("\n[1/3] Basic LQR hover stabilization...")
    t, X, U = simulate_aerial_robot_lqr()

    print("\n[2/3] LQR under aerial morphology change...")
    t_lqr, X_lqr, U_lqr = simulate_aerial_robot_morphology(controller='lqr')

    print("\n[3/3] MPC under aerial morphology change...")
    t_mpc, X_mpc, U_mpc = simulate_aerial_robot_morphology(controller='mpc')

    m_lqr = compute_aerial_metrics(t_lqr, X_lqr, U_lqr, change_time=5.0)
    m_mpc = compute_aerial_metrics(t_mpc, X_mpc, U_mpc, change_time=5.0)

    print("\n" + "=" * 65)
    print(f"  {'Metric':<25} {'LQR':>15} {'MPC':>15}")
    print("-" * 65)
    for key in m_lqr:
        print(f"  {key:<25} {str(m_lqr[key]):>15} {str(m_mpc[key]):>15}")
    print("=" * 65)