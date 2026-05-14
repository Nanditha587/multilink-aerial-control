import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dynamics import TripleLinkPendulum


class TriplePendulumEnv(gym.Env):
    """
    Gymnasium environment for triple-link pendulum.
    Used to train PPO agent via stable-baselines3.

    Observation: [q1, q2, q3, dq1, dq2, dq3]  (6,)
    Action:      [tau1, tau2, tau3]              (3,) continuous
    Reward:      negative state cost + control penalty
    Episode ends when any angle exceeds 60 degrees or t > 10s
    """

    def __init__(self, m=[1,1,1], l=[1,1,1], dt=0.02):
        super().__init__()
        self.pendulum = TripleLinkPendulum(m=m, l=l)
        self.dt       = dt
        self.max_steps= 500
        self.step_count = 0

        # Action space: torques in [-20, 20] N.m
        self.action_space = spaces.Box(
            low=-20.0, high=20.0, shape=(3,), dtype=np.float32)

        # Observation space
        high = np.array([np.pi, np.pi, np.pi, 10, 10, 10], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random small initial displacement
        self.state = np.random.uniform(
            low=[-0.3, -0.3, -0.3, -0.1, -0.1, -0.1],
            high=[ 0.3,  0.3,  0.3,  0.1,  0.1,  0.1]
        )
        self.step_count = 0
        return self.state.astype(np.float32), {}

    def step(self, action):
        tau = np.clip(action, -20.0, 20.0)

        # Integrate one step
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            fun=lambda t, x: self.pendulum.derivatives(t, x, tau),
            t_span=(0, self.dt),
            y0=self.state,
            method='RK45',
            max_step=self.dt
        )
        self.state = sol.y[:, -1]
        self.step_count += 1

        # Reward: penalise angle error and control effort
        angle_cost   = np.sum(self.state[:3]**2)
        vel_cost     = 0.1 * np.sum(self.state[3:]**2)
        control_cost = 0.01 * np.sum(tau**2)
        reward = -(angle_cost + vel_cost + control_cost)

        # Termination
        terminated = bool(np.any(np.abs(self.state[:3]) > np.pi/3))
        truncated  = bool(self.step_count >= self.max_steps)

        return self.state.astype(np.float32), reward, terminated, truncated, {}

    def render(self):
        pass