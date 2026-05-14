import numpy as np

class TripleLinkPendulum:
    """
    Triple-Link Pendulum Dynamics Model
    =====================================
    Models three rigid links connected by revolute joints.
    Each link is treated as a point mass at its end (standard simplification).

    State vector x (6x1):
        x = [q1, q2, q3, dq1, dq2, dq3]
        q1, q2, q3   : absolute joint angles from vertical upright (rad)
        dq1, dq2, dq3: angular velocities (rad/s)

    Control input u (3x1):
        u = [tau1, tau2, tau3] : torques applied at each joint (N.m)

    Equation of motion:
        M(q)*ddq + C(q,dq) + G(q) = tau
        ddq = M(q)^-1 * (tau - C(q,dq) - G(q))
    """

    def __init__(self,
                 m=[1.0, 1.0, 1.0],
                 l=[1.0, 1.0, 1.0],
                 g=9.81):
        """
        Parameters
        ----------
        m : list of float — mass of each link [m1, m2, m3] (kg)
        l : list of float — length of each link [l1, l2, l3] (m)
        g : float          — gravitational acceleration (m/s^2)
        """
        self.m = np.array(m, dtype=float)
        self.l = np.array(l, dtype=float)
        self.g = g
        self.n = 3  # number of links

    # ── 1. Mass Matrix M(q) ───────────────────────────────────────────────────
    def mass_matrix(self, q):
        """
        Compute the 3x3 configuration-dependent mass/inertia matrix.

        Diagonal entries: self-inertia of each link
        Off-diagonal entries: inertial coupling between links
        M is always symmetric and positive definite.

        Parameters
        ----------
        q : array (3,) — joint angles [q1, q2, q3]

        Returns
        -------
        M : ndarray (3x3)
        """
        m, l = self.m, self.l
        q1, q2, q3 = q

        M = np.zeros((3, 3))

        # Diagonal — cascading mass pattern
        M[0, 0] = (m[0] + m[1] + m[2]) * l[0]**2
        M[1, 1] = (m[1] + m[2])         * l[1]**2
        M[2, 2] =  m[2]                  * l[2]**2

        # Off-diagonal — coupling terms (symmetric)
        M[0, 1] = (m[1] + m[2]) * l[0] * l[1] * np.cos(q1 - q2)
        M[0, 2] =  m[2]         * l[0] * l[2] * np.cos(q1 - q3)
        M[1, 2] =  m[2]         * l[1] * l[2] * np.cos(q2 - q3)

        # Symmetry
        M[1, 0] = M[0, 1]
        M[2, 0] = M[0, 2]
        M[2, 1] = M[1, 2]

        return M

    # ── 2. Coriolis Vector C(q, dq) ───────────────────────────────────────────
    def coriolis_vector(self, q, dq):
        """
        Compute the 3x1 Coriolis and centripetal force vector.

        Derived from Christoffel symbols of M(q).
        Each term has the form: ±(mass)*(length)*sin(qi-qj)*dqj^2
        C = 0 when dq = 0 (no motion = no Coriolis forces).

        Parameters
        ----------
        q  : array (3,) — joint angles
        dq : array (3,) — joint angular velocities

        Returns
        -------
        C : ndarray (3,)
        """
        m, l = self.m, self.l
        q1, q2, q3 = q
        dq1, dq2, dq3 = dq

        C = np.zeros(3)

        # C1: coupling from link 2 and link 3 onto link 1
        C[0] = (-(m[1] + m[2]) * l[0] * l[1] * np.sin(q1 - q2) * dq2**2
                - m[2]          * l[0] * l[2] * np.sin(q1 - q3) * dq3**2)

        # C2: coupling from link 1 and link 3 onto link 2
        C[1] = ( (m[1] + m[2]) * l[0] * l[1] * np.sin(q1 - q2) * dq1**2
                - m[2]          * l[1] * l[2] * np.sin(q2 - q3) * dq3**2)

        # C3: coupling from link 1 and link 2 onto link 3
        C[2] = ( m[2] * l[0] * l[2] * np.sin(q1 - q3) * dq1**2
                + m[2] * l[1] * l[2] * np.sin(q2 - q3) * dq2**2)

        return C

    # ── 3. Gravity Vector G(q) ────────────────────────────────────────────────
    def gravity_vector(self, q):
        """
        Compute the 3x1 gravity torque vector.

        Derived from dV/dq where V is total potential energy.
        G = 0 at upright equilibrium (sin(0) = 0) — but unstable.
        Grows with sin(qi) — larger tilt = stronger restoring pull downward.

        Parameters
        ----------
        q : array (3,) — joint angles

        Returns
        -------
        G : ndarray (3,)
        """
        m, l, g = self.m, self.l, self.g
        q1, q2, q3 = q

        G = np.zeros(3)

        # Cascading mass pattern — same as M diagonal
        G[0] = -(m[0] + m[1] + m[2]) * g * l[0] * np.sin(q1)
        G[1] = -(m[1] + m[2])         * g * l[1] * np.sin(q2)
        G[2] = -m[2]                   * g * l[2] * np.sin(q3)

        return G

    # ── 4. Full Nonlinear Dynamics ────────────────────────────────────────────
    def derivatives(self, t, state, tau):
        """
        Compute dx/dt for the full nonlinear system.
        Used by scipy ODE solver (solve_ivp).

        state = [q1, q2, q3, dq1, dq2, dq3]
        returns d(state)/dt = [dq1, dq2, dq3, ddq1, ddq2, ddq3]

        Parameters
        ----------
        t     : float      — current time (required by solve_ivp)
        state : array (6,) — current state vector
        tau   : array (3,) — control torques

        Returns
        -------
        dstate : ndarray (6,)
        """
        q  = state[:3]   # angles
        dq = state[3:]   # velocities

        M = self.mass_matrix(q)
        C = self.coriolis_vector(q, dq)
        G = self.gravity_vector(q)

        # ddq = M^-1 * (tau - C - G)
        ddq = np.linalg.solve(M, tau - C - G)

        return np.concatenate([dq, ddq])

    # ── 5. Linearisation Around Upright Equilibrium ───────────────────────────
    def linearize(self):
        """
        Linearise the system around the upright equilibrium:
            q* = [0, 0, 0], dq* = [0, 0, 0], tau* = [0, 0, 0]

        At equilibrium:
            sin(qi) = 0, cos(qi - qj) = 1
            M simplifies to diagonal M0
            C = 0 (no velocity)
            G linearises to Kg * q

        Returns
        -------
        A : ndarray (6x6) — linearised system matrix
        B : ndarray (6x3) — linearised input matrix

        Linear system: dx/dt = A*x + B*u
        """
        m, l, g = self.m, self.l, self.g

        # M at equilibrium — diagonal (cos(qi-qj)=1, but off-diag still present)
        M0 = np.array([
            [(m[0]+m[1]+m[2])*l[0]**2,  (m[1]+m[2])*l[0]*l[1],  m[2]*l[0]*l[2]],
            [(m[1]+m[2])*l[0]*l[1],      (m[1]+m[2])*l[1]**2,    m[2]*l[1]*l[2]],
            [m[2]*l[0]*l[2],              m[2]*l[1]*l[2],          m[2]*l[2]**2  ]
        ])

        # Linearised gravity stiffness: dG/dq at q=0
        # G_i = -(mass_i)*g*l_i*sin(qi) → dGi/dqi = -(mass_i)*g*l_i*cos(0)
        Kg = np.diag([
            (m[0]+m[1]+m[2]) * g * l[0],
            (m[1]+m[2])       * g * l[1],
             m[2]              * g * l[2]
        ])

        # A matrix (6x6)
        # [ 0    I  ]
        # [ M0^-1*Kg  0 ]
        A = np.zeros((6, 6))
        A[:3, 3:] = np.eye(3)
        A[3:, :3] = np.linalg.solve(M0, Kg)

        # B matrix (6x3)
        # [ 0     ]
        # [ M0^-1 ]
        B = np.zeros((6, 3))
        B[3:, :] = np.linalg.inv(M0)

        return A, B


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    pendulum = TripleLinkPendulum()

    # Check M at equilibrium
    q0 = np.zeros(3)
    M0 = pendulum.mass_matrix(q0)
    print("M at equilibrium:\n", np.round(M0, 4))

    # Check G at equilibrium (should be zero)
    G0 = pendulum.gravity_vector(q0)
    print("\nG at equilibrium (should be [0,0,0]):", G0)

    # Check linearisation
    A, B = pendulum.linearize()
    print("\nA matrix:\n", np.round(A, 4))
    print("\nB matrix:\n", np.round(B, 4))