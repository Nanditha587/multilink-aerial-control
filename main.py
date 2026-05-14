import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.dynamics import TripleLinkPendulum
from src.lqr_controller import LQRController
from src.mpc_controller import MPCController
from src.rl_controller import RLController
from src.visualize import plot_states, plot_control, plot_comparison
from scipy.integrate import solve_ivp


def simulate_controller(ctrl, pendulum, x0, t_span=(0,10), dt=0.01):
    """Generic simulation loop for any controller."""
    t_eval = np.arange(t_span[0], t_span[1], dt)
    X = np.zeros((len(t_eval), 6))
    U = np.zeros((len(t_eval), 3))
    X[0] = x0

    for i in range(len(t_eval) - 1):
        tau  = ctrl.compute_torque(X[i])
        U[i] = tau
        sol  = solve_ivp(
            fun=lambda t, x: pendulum.derivatives(t, x, tau),
            t_span=(t_eval[i], t_eval[i+1]),
            y0=X[i], method='RK45', max_step=dt)
        X[i+1] = sol.y[:, -1]

        if np.any(np.abs(X[i+1]) > 50):
            X[i+1] = np.clip(X[i+1], -50, 50)

    U[-1] = ctrl.compute_torque(X[-1])
    return t_eval, X, U


def compute_metrics(t, X, U, threshold=0.01):
    """Compute settling time, overshoot, energy."""
    dt = t[1] - t[0]

    settled_idx = None
    for i in range(len(t)):
        if np.all(np.abs(X[i, :3]) < threshold):
            if i + 50 < len(t) and np.all(np.abs(X[i:i+50, :3]) < threshold):
                settled_idx = i
                break

    settling_time    = t[settled_idx] if settled_idx else t[-1]
    max_overshoot    = float(np.degrees(np.max(np.abs(X[:, :3]))))
    energy           = float(np.sum(U**2) * dt)

    return {
        'Settling Time (s)'  : round(settling_time, 3),
        'Max Overshoot (deg)': round(max_overshoot, 3),
        'Total Energy (J)'   : round(energy, 3),
    }


def plot_all_three(t_lqr, X_lqr, U_lqr,
                   t_mpc, X_mpc, U_mpc,
                   t_rl,  X_rl,  U_rl,  save=True):
    """Plot all three controllers on same axes."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('LQR vs MPC vs RL — Triple-Link Pendulum Stabilization',
                 fontsize=14, fontweight='bold')

    # Angles (q1 only for clarity)
    axes[0].plot(t_lqr, np.degrees(X_lqr[:, 0]), label='LQR', color='#2563EB', lw=2)
    axes[0].plot(t_mpc, np.degrees(X_mpc[:, 0]), label='MPC', color='#DC2626', lw=2)
    axes[0].plot(t_rl,  np.degrees(X_rl[:, 0]),  label='RL',  color='#16A34A', lw=2)
    axes[0].axhline(0, color='gray', linestyle='--', lw=0.8)
    axes[0].set_ylabel('q1 Angle (degrees)')
    axes[0].set_title('Joint 1 Angle')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Velocities (dq1 only)
    axes[1].plot(t_lqr, X_lqr[:, 3], label='LQR', color='#2563EB', lw=2)
    axes[1].plot(t_mpc, X_mpc[:, 3], label='MPC', color='#DC2626', lw=2)
    axes[1].plot(t_rl,  X_rl[:, 3],  label='RL',  color='#16A34A', lw=2)
    axes[1].axhline(0, color='gray', linestyle='--', lw=0.8)
    axes[1].set_ylabel('dq1 (rad/s)')
    axes[1].set_title('Joint 1 Angular Velocity')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Control effort (tau1 only)
    axes[2].plot(t_lqr, U_lqr[:, 0], label='LQR', color='#2563EB', lw=2)
    axes[2].plot(t_mpc, U_mpc[:, 0], label='MPC', color='#DC2626', lw=2)
    axes[2].plot(t_rl,  U_rl[:, 0],  label='RL',  color='#16A34A', lw=2)
    axes[2].axhline(0, color='gray', linestyle='--', lw=0.8)
    axes[2].set_ylabel('tau1 (N·m)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Joint 1 Control Effort')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('plots/all_three_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved: plots/all_three_comparison.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("  Triple-Link Pendulum — Controller Comparison")
    print("=" * 60)

    x0       = np.array([0.2, -0.15, 0.1, 0.0, 0.0, 0.0])
    pendulum = TripleLinkPendulum()
    A, B     = pendulum.linearize()

    # Build controllers
    print("\n[1/3] Building LQR...")
    lqr = LQRController(A, B)

    print("\n[2/3] Building MPC...")
    mpc = MPCController(A, B, dt=0.01)

    print("\n[3/3] Loading RL...")
    rl  = RLController()

    # Run simulations
    print("\nRunning simulations...")
    t_lqr, X_lqr, U_lqr = simulate_controller(lqr, pendulum, x0)
    print("LQR done.")
    t_mpc, X_mpc, U_mpc = simulate_controller(mpc, pendulum, x0)
    print("MPC done.")
    t_rl,  X_rl,  U_rl  = simulate_controller(rl,  pendulum, x0)
    print("RL  done.")

    # Metrics
    m_lqr = compute_metrics(t_lqr, X_lqr, U_lqr)
    m_mpc = compute_metrics(t_mpc, X_mpc, U_mpc)
    m_rl  = compute_metrics(t_rl,  X_rl,  U_rl)

    print("\n" + "=" * 65)
    print(f"  {'Metric':<25} {'LQR':>12} {'MPC':>12} {'RL (PPO)':>12}")
    print("-" * 65)
    for key in m_lqr:
        print(f"  {key:<25} {str(m_lqr[key]):>12} "
              f"{str(m_mpc[key]):>12} {str(m_rl[key]):>12}")
    print("=" * 65)

    # Plot
    plot_all_three(t_lqr, X_lqr, U_lqr,
                   t_mpc, X_mpc, U_mpc,
                   t_rl,  X_rl,  U_rl)