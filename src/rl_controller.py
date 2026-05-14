import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from src.pendulum_env import TriplePendulumEnv


def train_rl(total_timesteps=200_000, save_path='results/ppo_pendulum'):
    """
    Train PPO agent on triple-link pendulum.
    200k steps takes ~5-10 minutes on CPU.
    """
    print("Training PPO agent...")
    print(f"Total timesteps: {total_timesteps:,}")

    env = make_vec_env(TriplePendulumEnv, n_envs=4)

    model = PPO(
        'MlpPolicy', env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=None
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    return model


def load_rl(path='results/ppo_pendulum'):
    """Load trained PPO model."""
    model = PPO.load(path)
    print(f"Model loaded from {path}.zip")
    return model


class RLController:
    """
    Wrapper around trained PPO model.
    Provides same interface as LQR and MPC controllers.
    """

    def __init__(self, model_path='results/ppo_pendulum'):
        if os.path.exists(model_path + '.zip'):
            self.model = load_rl(model_path)
        else:
            print("No trained model found. Training now...")
            self.model = train_rl(save_path=model_path)

    def compute_torque(self, x):
        """
        Get action from PPO policy.
        obs must be float32 for stable-baselines3.
        """
        obs   = x.astype(np.float32).reshape(1, -1)
        action, _ = self.model.predict(obs, deterministic=True)
        return action.flatten()


if __name__ == "__main__":
    # Train and test
    model = train_rl(total_timesteps=200_000)

    ctrl = RLController()
    x0   = np.array([0.2, -0.15, 0.1, 0.0, 0.0, 0.0])
    tau  = ctrl.compute_torque(x0)
    print("\nTest state:", x0)
    print("RL torque: ", np.round(tau, 4))