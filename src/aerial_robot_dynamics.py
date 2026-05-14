import numpy as np


class Aerial_robotAerialModel:
    """
    Simplified 2D Aerial Model for Three-Link AERIAL_ROBOT-like Robot
    ============================================================
    Based on Zhao et al. 2018/2022 with gimbal vectoring included.

    Each link has one 2DoF vectorable rotor (simplified to 1D gimbal
    in planar model). Gimbal angle alpha_i tilts thrust from vertical.

    State (10x1):
        [x, z, theta1, theta2, theta3,
         dx, dz, dtheta1, dtheta2, dtheta3]

    Control (6x1):
        [F1, F2, F3,   <- thrust per link (N)
         a1, a2, a3]   <- gimbal angles from vertical (rad)

    Equilibrium (hover):
        All states = 0
        F_i* = m_i * g, a_i* = 0
    """

    def __init__(self, m=[1.0,1.0,1.0], l=[1.0,1.0,1.0], g=9.81):
        self.m       = np.array(m, dtype=float)
        self.l       = np.array(l, dtype=float)
        self.g       = g
        self.n       = 3
        self.nx      = 10
        self.nu      = 6   # now 6 inputs
        self.m_total = np.sum(self.m)
        self.I       = self.m * self.l**2 / 12.0

    def hover_control(self):
        """
        Equilibrium control: hover thrust + zero gimbal.
        F_i* = m_i*g, a_i* = 0
        """
        F_hover = self.m * self.g
        a_hover = np.zeros(3)
        return np.concatenate([F_hover, a_hover])

    def derivatives(self, t, state, u):
        """
        Nonlinear equations of motion with gimbal vectoring.

        Parameters
        ----------
        state : array (10,)
        u     : array (6,) — [F1,F2,F3, a1,a2,a3]
        """
        theta  = state[2:5]
        dx     = state[5]
        dz     = state[6]
        dtheta = state[7:10]

        F = u[:3]   # thrust magnitudes
        a = u[3:]   # gimbal angles

        # Total forces in world frame
        # Each rotor tilted by (theta_i + a_i) from vertical
        Fx_total = np.sum(F * np.sin(theta + a))
        Fz_total = np.sum(F * np.cos(theta + a))

        # Translational dynamics
        ddx = Fx_total / self.m_total
        ddz = (Fz_total - self.m_total * self.g) / self.m_total

        # Rotational dynamics per link
        ddtheta = np.zeros(3)
        for i in range(3):
            tau_thrust  = (F[i] - self.m[i]*self.g) * \
                          (self.l[i]/2) * np.cos(theta[i] + a[i])
            tau_gimbal  =  F[i] * np.sin(a[i]) * (self.l[i]/2)
            ddtheta[i]  = (tau_thrust + tau_gimbal) / self.I[i]

        return np.concatenate([[dx, dz], dtheta, [ddx, ddz], ddtheta])

    def linearize(self):
        """
        Linearize around hover equilibrium.
        State: x* = 0, Control: u* = [m_i*g, 0, 0, 0]

        At equilibrium:
            sin(theta+a) ≈ theta+a, cos(theta+a) ≈ 1
            sin(a) ≈ a, cos(a) ≈ 1

        Returns
        -------
        A : ndarray (10x10)
        B : ndarray (10x6)
        """
        m, l, g = self.m, self.l, self.g
        I       = self.I
        m_total = self.m_total
        F_eq    = m * g   # hover thrust

        A = np.zeros((10, 10))
        B = np.zeros((10, 6))

        # Position kinematics
        A[0, 5] = 1.0
        A[1, 6] = 1.0
        A[2, 7] = 1.0
        A[3, 8] = 1.0
        A[4, 9] = 1.0

        # Translational acceleration
        # ddx: depends on theta_i and a_i (linearised sin ≈ angle)
        for i in range(3):
            A[5, 2+i] = F_eq[i] / m_total   # dFx/dtheta_i
            B[5, i]   = 0.0                  # dFx/dF_i = 0 at a*=0
            B[5, 3+i] = F_eq[i] / m_total   # dFx/da_i

        # ddz: depends on F_i (linearised cos ≈ 1)
        for i in range(3):
            B[6, i] = 1.0 / m_total          # dFz/dF_i

        # Rotational acceleration per link
        for i in range(3):
            # Gravity stiffness from theta
            A[7+i, 2+i] = -m[i] * g * (l[i]/2) / I[i]

            # Effect of thrust perturbation dF_i
            B[7+i, i]   = (l[i]/2) / I[i]

            # Effect of gimbal angle da_i
            # tau_gimbal linearised: F_eq*a_i*(l/2)
            B[7+i, 3+i] = F_eq[i] * (l[i]/2) / I[i]

        return A, B


if __name__ == "__main__":
    aerial_robot = Aerial_robotAerialModel()

    # Check hover
    u0     = aerial_robot.hover_control()
    state0 = np.zeros(10)
    dstate = aerial_robot.derivatives(0, state0, u0)
    print("Derivatives at hover (should be zero):", np.round(dstate, 6))

    A, B = aerial_robot.linearize()
    print("\nA matrix:\n", np.round(A, 4))
    print("\nB matrix:\n", np.round(B, 4))

    # Check controllability
    C = B.copy()
    for i in range(1, 10):
        C = np.hstack([C, np.linalg.matrix_power(A, i) @ B])
    rank = np.linalg.matrix_rank(C)
    print(f"\nControllability rank: {rank}/10")