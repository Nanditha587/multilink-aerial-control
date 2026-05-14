import numpy as np
from scipy.linalg import solve_continuous_are

class LQRController:
    """
    Linear Quadratic Regulator (LQR) Controller
    =============================================
    Computes optimal state feedback gain matrix K such that:
        u = -K * x
    minimises the cost function:
        J = integral( x^T*Q*x + u^T*R*u ) dt

    Q : state cost matrix  — how much we penalise deviation from upright
    R : input cost matrix  — how much we penalise control effort (torque)

    Higher Q → prioritise stability (aggressive control)
    Higher R → prioritise efficiency (gentle control)
    """

    def __init__(self, A, B, Q=None, R=None):
        """
        Parameters
        ----------
        A : ndarray (6x6) — linearised system matrix from dynamics
        B : ndarray (6x3) — linearised input matrix from dynamics
        Q : ndarray (6x6) — state weighting matrix (default: identity)
        R : ndarray (3x3) — input weighting matrix (default: identity)
        """
        self.A = A
        self.B = B

        # Default Q — penalise all states equally
        # Diagonal: [q1, q2, q3, dq1, dq2, dq3]
        if Q is None:
            self.Q = np.diag([10.0, 10.0, 10.0,   # angle penalties
                               1.0,  1.0,  1.0])   # velocity penalties
        else:
            self.Q = Q

        # Default R — penalise all torques equally
        if R is None:
            self.R = np.eye(3) * 0.1
        else:
            self.R = R

        # Compute gain matrix K
        self.K = self._compute_gain()

    def _compute_gain(self):
        """
        Solve the Continuous Algebraic Riccati Equation (CARE):
            A^T*P + P*A - P*B*R^-1*B^T*P + Q = 0

        Then compute optimal gain:
            K = R^-1 * B^T * P

        Returns
        -------
        K : ndarray (3x6) — optimal feedback gain matrix
        """
        # Solve Riccati equation for P
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)

        # Compute K
        K = np.linalg.solve(self.R, self.B.T @ P)

        print("LQR gain matrix K computed successfully.")
        print("K shape:", K.shape)
        print("K:\n", np.round(K, 4))

        return K

    def compute_torque(self, x):
        """
        Compute control torque for current state x.

        u = -K * x

        Negative sign: if pendulum tilts right (positive q),
        controller applies negative torque to push it back.

        Parameters
        ----------
        x : array (6,) — current state [q1,q2,q3,dq1,dq2,dq3]

        Returns
        -------
        tau : ndarray (3,) — control torques [tau1, tau2, tau3]
        """
        return -self.K @ x


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.dynamics import TripleLinkPendulum

    # Build system
    pendulum = TripleLinkPendulum()
    A, B = pendulum.linearize()

    # Build LQR controller
    lqr = LQRController(A, B)

    # Test: small initial displacement
    x_test = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0])
    tau = lqr.compute_torque(x_test)
    print("\nTest state:", x_test)
    print("Control torque:", np.round(tau, 4))