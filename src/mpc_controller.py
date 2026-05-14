import numpy as np
from scipy.linalg import solve_discrete_are
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MPCController:
    """
    MPC for Triple-Link Pendulum — Infinite Horizon Implementation
    ==============================================================
    For linear systems, infinite-horizon MPC reduces to a discrete
    LQR solved with the discrete Riccati equation.

    Key difference from standard LQR:
    - MPC re-solves the Riccati equation at every morphology change
    - LQR uses a fixed gain K computed once at the start
    - After morphology change: MPC adapts, LQR uses stale K

    This is the standard approach in robotics literature for
    adaptive MPC on systems with changing dynamics.
    """

    def __init__(self, A, B, dt=0.01, Q=None, R=None, u_max=50.0):
        self.dt   = dt
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.umax = u_max

        if Q is None:
            self.Q = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])
        else:
            self.Q = Q

        if R is None:
            self.R = np.eye(3) * 0.1
        else:
            self.R = R

        self._compute_gain(A, B)
        print("MPC controller built successfully.")
        print("K:\n", np.round(self.K, 4))

    def _compute_gain(self, A, B):
        """
        Compute discrete-time optimal gain via DARE.
        Ad = I + dt*A  (Euler discretisation)
        Bd = dt*B
        """
        Ad = np.eye(self.nx) + self.dt * A
        Bd = self.dt * B

        try:
            P  = solve_discrete_are(Ad, Bd, self.Q, self.R)
            RBd = self.R + Bd.T @ P @ Bd
            self.K = np.linalg.solve(RBd, Bd.T @ P @ Ad)
        except Exception as e:
            print(f"DARE failed: {e} — falling back to continuous ARE")
            from scipy.linalg import solve_continuous_are
            P = solve_continuous_are(A, B, self.Q, self.R)
            self.K = np.linalg.solve(self.R, B.T @ P)

    def compute_torque(self, x):
        """
        u = -K * x  with saturation.
        """
        tau = -self.K @ x
        return np.clip(tau, -self.umax, self.umax)

    def update_model(self, A_new, B_new, current_x=None):
        """
        Re-solve DARE with new morphology.
        This is the key MPC advantage — LQR cannot do this.
        """
        self._compute_gain(A_new, B_new)
        print("MPC gain updated for new morphology.")

    def reset(self, x0):
        pass


if __name__ == "__main__":
    from src.dynamics import TripleLinkPendulum

    pendulum = TripleLinkPendulum()
    A, B = pendulum.linearize()

    mpc = MPCController(A, B, dt=0.01)
    x0  = np.array([0.2, -0.15, 0.1, 0.0, 0.0, 0.0])
    tau = mpc.compute_torque(x0)
    print("\nTest state:", x0)
    print("MPC torque:", np.round(tau, 4))