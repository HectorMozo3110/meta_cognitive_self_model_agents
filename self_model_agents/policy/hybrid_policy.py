# self_model_agents/policy/hybrid_policy.py

import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class HybridPolicy:
    def __init__(self, env, hybrid_ratio, total_timesteps):
        """
        Initialize Hybrid Policy.
        
        Args:
            env: Gym environment
            hybrid_ratio: Probability of using PPO policy vs. random action
            total_timesteps: Number of PPO training timesteps
        """
        self.env = env
        self.hybrid_ratio = hybrid_ratio

        # Initialize PPO model
        print("\nUsing HybridPolicy (PPO + Random)...\n")
        self.model = PPO("MlpPolicy", make_vec_env(lambda: env, n_envs=1), verbose=0)
        self.model.learn(total_timesteps=total_timesteps)

    def select_action(self, observation):
        """
        Select an action using PPO policy with probability hybrid_ratio,
        or random action with probability (1 - hybrid_ratio).
        """
        if random.random() < self.hybrid_ratio:
            # Use PPO model
            action, _ = self.model.predict(observation, deterministic=True)
            return action
        else:
            # Use random action (exploration)
            return self.env.action_space.sample()
