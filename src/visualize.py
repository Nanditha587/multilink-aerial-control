import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def plot_states(t, X, save=True):
    """Plot joint angles and velocities over time."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    fig.suptitle('LQR Controller — State Trajectories', fontsize=14, fontweight='bold')

    # Angles
    axes[0].plot(t, np.degrees(X[:, 0]), label='q1 (Link 1)', color='#2563EB')
    axes[0].plot(t, np.degrees(X[:, 1]), label='q2 (Link 2)', color='#7C3AED')
    axes[0].plot(t, np.degrees(X[:, 2]), label='q3 (Link 3)', color='#DC2626')
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[0].set_ylabel('Joint Angle (degrees)')
    axes[0].set_title('Joint Angles')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Velocities
    axes[1].plot(t, X[:, 3], label='dq1', color='#2563EB')
    axes[1].plot(t, X[:, 4], label='dq2', color='#7C3AED')
    axes[1].plot(t, X[:, 5], label='dq3', color='#DC2626')
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
    axes[1].set_ylabel('Angular Velocity (rad/s)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Joint Velocities')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('plots/state_trajectories.png', dpi=150, bbox_inches='tight')
        print("Saved: plots/state_trajectories.png")
    plt.show()


def plot_control(t, U, save=True):
    """Plot control torques over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, U[:, 0], label='tau1 (Joint 1)', color='#2563EB')
    ax.plot(t, U[:, 1], label='tau2 (Joint 2)', color='#7C3AED')
    ax.plot(t, U[:, 2], label='tau3 (Joint 3)', color='#DC2626')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Torque (N·m)')
    ax.set_xlabel('Time (s)')
    ax.set_title('LQR Controller — Control Effort', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('plots/control_effort.png', dpi=150, bbox_inches='tight')
        print("Saved: plots/control_effort.png")
    plt.show()


def animate_pendulum(t, X, l=[1,1,1], save=True):
    """Animate the triple-link pendulum stabilizing."""
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_title('Triple-Link Pendulum — LQR Stabilization', fontweight='bold')

    # Pivot at origin
    pivot = np.array([0, 0])
    ax.plot(*pivot, 'ks', markersize=8)

    line,    = ax.plot([], [], 'o-', color='#2563EB', lw=3, markersize=8)
    time_txt = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def get_positions(state):
        q1, q2, q3 = state[:3]
        x1 = l[0] * np.sin(q1); y1 = -l[0] * np.cos(q1)
        x2 = x1 + l[1] * np.sin(q2); y2 = y1 - l[1] * np.cos(q2)
        x3 = x2 + l[2] * np.sin(q3); y3 = y2 - l[2] * np.cos(q3)
        return [0, x1, x2, x3], [0, y1, y2, y3]

    def init():
        line.set_data([], [])
        time_txt.set_text('')
        return line, time_txt

    # Subsample for smooth animation
    step = max(1, len(t) // 200)

    def update(frame):
        idx = frame * step
        xs, ys = get_positions(X[idx])
        line.set_data(xs, ys)
        time_txt.set_text(f't = {t[idx]:.2f}s')
        return line, time_txt

    ani = animation.FuncAnimation(
        fig, update, frames=len(t)//step,
        init_func=init, blit=True, interval=50
    )

    if save:
        ani.save('animations/lqr_pendulum.gif', writer='pillow', fps=20)
        print("Saved: animations/lqr_pendulum.gif")

    plt.show()
    return ani
def plot_comparison(t_lqr, X_lqr, U_lqr,
                    t_mpc, X_mpc, U_mpc,
                    change_time=5.0, save=True):
    """Side-by-side LQR vs MPC comparison plots."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('LQR vs MPC — Morphology Change at t=5s',
                 fontsize=15, fontweight='bold')

    controllers = [('LQR', t_lqr, X_lqr, U_lqr, '#2563EB'),
                   ('MPC', t_mpc, X_mpc, U_mpc, '#DC2626')]

    for col, (name, t, X, U, color) in enumerate(controllers):

        # Angles
        axes[0, col].plot(t, np.degrees(X[:, 0]), label='q1', color=color, alpha=0.9)
        axes[0, col].plot(t, np.degrees(X[:, 1]), label='q2', color=color, alpha=0.6, linestyle='--')
        axes[0, col].plot(t, np.degrees(X[:, 2]), label='q3', color=color, alpha=0.4, linestyle=':')
        axes[0, col].axvline(change_time, color='orange', linestyle='--', lw=1.5, label='Morphology change')
        axes[0, col].set_title(f'{name} — Joint Angles', fontweight='bold')
        axes[0, col].set_ylabel('Angle (degrees)')
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].set_ylim(-100, 100)

        # Velocities
        axes[1, col].plot(t, X[:, 3], label='dq1', color=color, alpha=0.9)
        axes[1, col].plot(t, X[:, 4], label='dq2', color=color, alpha=0.6, linestyle='--')
        axes[1, col].plot(t, X[:, 5], label='dq3', color=color, alpha=0.4, linestyle=':')
        axes[1, col].axvline(change_time, color='orange', linestyle='--', lw=1.5)
        axes[1, col].set_title(f'{name} — Joint Velocities', fontweight='bold')
        axes[1, col].set_ylabel('Velocity (rad/s)')
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)

        # Control effort
        axes[2, col].plot(t, U[:, 0], label='tau1', color=color, alpha=0.9)
        axes[2, col].plot(t, U[:, 1], label='tau2', color=color, alpha=0.6, linestyle='--')
        axes[2, col].plot(t, U[:, 2], label='tau3', color=color, alpha=0.4, linestyle=':')
        axes[2, col].axvline(change_time, color='orange', linestyle='--', lw=1.5)
        axes[2, col].set_title(f'{name} — Control Effort', fontweight='bold')
        axes[2, col].set_ylabel('Torque (N·m)')
        axes[2, col].set_xlabel('Time (s)')
        axes[2, col].legend(fontsize=8)
        axes[2, col].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('plots/lqr_vs_mpc_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved: plots/lqr_vs_mpc_comparison.png")
    plt.show()

def plot_dragon_comparison(t_lqr, X_lqr, U_lqr,
                            t_mpc, X_mpc, U_mpc,
                            change_time=5.0, save=True):
    """Plot LQR vs MPC on DRAGON aerial model."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('DRAGON Aerial Model — LQR vs MPC\nMorphology Change at t=5s',
                 fontsize=14, fontweight='bold')

    controllers = [('LQR', t_lqr, X_lqr, U_lqr, '#2563EB'),
                   ('MPC', t_mpc, X_mpc, U_mpc, '#DC2626')]

    for col, (name, t, X, U, color) in enumerate(controllers):

        # Z position (altitude)
        axes[0, col].plot(t, X[:, 1], color=color, lw=2)
        axes[0, col].axvline(change_time, color='orange', linestyle='--',
                              lw=1.5, label='Morphology change')
        axes[0, col].axhline(0, color='gray', linestyle='--', lw=0.8)
        axes[0, col].set_title(f'{name} — Altitude (z)', fontweight='bold')
        axes[0, col].set_ylabel('z position (m)')
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)

        # Link angles
        axes[1, col].plot(t, np.degrees(X[:, 2]), label='θ1', color=color, alpha=0.9)
        axes[1, col].plot(t, np.degrees(X[:, 3]), label='θ2', color=color,
                          alpha=0.6, linestyle='--')
        axes[1, col].plot(t, np.degrees(X[:, 4]), label='θ3', color=color,
                          alpha=0.4, linestyle=':')
        axes[1, col].axvline(change_time, color='orange', linestyle='--', lw=1.5)
        axes[1, col].axhline(0, color='gray', linestyle='--', lw=0.8)
        axes[1, col].set_title(f'{name} — Link Angles', fontweight='bold')
        axes[1, col].set_ylabel('Angle (degrees)')
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)

        # Thrust inputs
        axes[2, col].plot(t, U[:, 0], label='F1', color=color, alpha=0.9)
        axes[2, col].plot(t, U[:, 1], label='F2', color=color,
                          alpha=0.6, linestyle='--')
        axes[2, col].plot(t, U[:, 2], label='F3', color=color,
                          alpha=0.4, linestyle=':')
        axes[2, col].axvline(change_time, color='orange', linestyle='--', lw=1.5)
        axes[2, col].set_title(f'{name} — Thrust Perturbation', fontweight='bold')
        axes[2, col].set_ylabel('ΔF (N)')
        axes[2, col].set_xlabel('Time (s)')
        axes[2, col].legend(fontsize=8)
        axes[2, col].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('plots/dragon_lqr_vs_mpc.png', dpi=150, bbox_inches='tight')
        print("Saved: plots/dragon_lqr_vs_mpc.png")
    plt.show()    

if __name__ == "__main__":
    from src.simulate import simulate_lqr
    from src.simulate_comparison import simulate_with_morphology_change
    from src.simulate_dragon import simulate_dragon_morphology

    x0 = np.array([0.2, -0.15, 0.1, 0.0, 0.0, 0.0])

    # Basic LQR plots
    t, X, U = simulate_lqr(x0=x0)
    plot_states(t, X)
    plot_control(t, U)
    animate_pendulum(t, X)

    # Pendulum comparison
    t_lqr, X_lqr, U_lqr = simulate_with_morphology_change(
        controller='lqr', x0=x0,
        m_after=[3,0.5,2], l_after=[1.5,0.5,1.5])
    t_mpc, X_mpc, U_mpc = simulate_with_morphology_change(
        controller='mpc', x0=x0,
        m_after=[3,0.5,2], l_after=[1.5,0.5,1.5])
    plot_comparison(t_lqr, X_lqr, U_lqr, t_mpc, X_mpc, U_mpc)

    # Dragon aerial comparison
    t_lqr_d, X_lqr_d, U_lqr_d = simulate_dragon_morphology(controller='lqr')
    t_mpc_d, X_mpc_d, U_mpc_d = simulate_dragon_morphology(controller='mpc')
    plot_dragon_comparison(t_lqr_d, X_lqr_d, U_lqr_d,
                           t_mpc_d, X_mpc_d, U_mpc_d)