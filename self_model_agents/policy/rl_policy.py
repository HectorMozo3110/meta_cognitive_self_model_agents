# self_model_agents/policy/rl_policy.py

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class RLPolicy:
    """
    RLPolicy wrapper using PPO algorithm from Stable-Baselines3.
    All parameters are controlled externally by the script.
    """

    def __init__(self, env, total_timesteps):
        # Create vectorized environment
        self.env = make_vec_env(lambda: env, n_envs=1)

        # Initialize PPO model
        print("\nUsing RLPolicy (PPO)...\n")
        self.model = PPO("MlpPolicy", self.env, verbose=1)

        # Train the model
        self.model.learn(total_timesteps=total_timesteps)

    def select_action(self, observation):
        # Predict action using PPO model
        action, _ = self.model.predict(observation, deterministic=True)
        return action
