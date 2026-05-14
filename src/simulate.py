import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dynamics import TripleLinkPendulum
from src.lqr_controller import LQRController

def simulate_lqr(t_span=(0, 10), dt=0.01,
                 x0=None,
                 m=[1,1,1], l=[1,1,1]):
    """
    Simulate triple-link pendulum under LQR control.

    Parameters
    ----------
    t_span : (t_start, t_end) in seconds
    dt     : time step
    x0     : initial state [q1,q2,q3,dq1,dq2,dq3]
    m, l   : link masses and lengths

    Returns
    -------
    t      : time array
    X      : state history (N x 6)
    U      : control input history (N x 3)
    """
    # Default initial condition — small push from upright
    if x0 is None:
        x0 = np.array([0.2, -0.15, 0.1, 0.0, 0.0, 0.0])

    # Build system and controller
    pendulum = TripleLinkPendulum(m=m, l=l)
    A, B     = pendulum.linearize()
    lqr      = LQRController(A, B)

    # Time array
    t_eval = np.arange(t_span[0], t_span[1], dt)

    # Storage
    X = np.zeros((len(t_eval), 6))
    U = np.zeros((len(t_eval), 3))
    X[0] = x0

    # Simulation loop
    for i in range(len(t_eval) - 1):
        tau = lqr.compute_torque(X[i])
        U[i] = tau

        sol = solve_ivp(
            fun=lambda t, x: pendulum.derivatives(t, x, tau),
            t_span=(t_eval[i], t_eval[i+1]),
            y0=X[i],
            method='RK45',
            max_step=dt
        )
        X[i+1] = sol.y[:, -1]

    U[-1] = lqr.compute_torque(X[-1])

    print(f"Simulation complete. Steps: {len(t_eval)}")
    print(f"Final state: {np.round(X[-1], 6)}")

    return t_eval, X, U


if __name__ == "__main__":
    t, X, U = simulate_lqr()