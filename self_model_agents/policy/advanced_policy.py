# self_model_agents/policy/advanced_policy.py

from self_model_agents.policy.base_policy import BasePolicy
import random
import numpy as np

class AdvancedPolicy(BasePolicy):
    """
    AdvancedPolicy that dynamically adjusts its exploration/exploitation
    based on the SelfModel's internal state (meta-cognitive adaptation).
    All parameters are controlled externally by the script.
    """

    def __init__(
        self,
        action_space,
        self_model,
        epsilon,
        min_epsilon,
        max_epsilon,
        epsilon_decay_rate
    ):
        self.action_space = action_space
        self.self_model = self_model

        # Parameters received from the script
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate

    def select_action(self, state):
        """
        Select action using epsilon-greedy strategy.
        Epsilon is dynamically adapted based on SelfModel state.
        """

        # === Meta-cognitive adaptation ===
        self.adapt_epsilon()

        # === Epsilon-greedy action selection ===
        if random.random() < self.epsilon:
            # Exploration
            action = self.action_space.sample()
        else:
            # Exploitation (here: random argmax as placeholder, could be RL)
            action = self.action_space.sample()  # No RL model here, so random

        return action

    def adapt_epsilon(self):
        """
        Adjust epsilon based on SelfModel confidence and fatigue.
        Higher confidence → lower epsilon (more exploitation).
        Higher fatigue → higher epsilon (more exploration).
        """

        state = self.self_model.to_dict()

        confidence = state.get("confidence_level", 0.5)
        fatigue = state.get("fatigue_level", 0.0)

        # Example rule: combine confidence and fatigue
        target_epsilon = (1.0 - confidence) * 0.7 + fatigue * 0.3

        # Clamp target epsilon between min and max
        target_epsilon = max(self.min_epsilon, min(self.max_epsilon, target_epsilon))

        # Smooth adaptation (exponential moving average)
        self.epsilon += self.epsilon_decay_rate * (target_epsilon - self.epsilon)

        # Optional: log for debugging
        print(
            f"[AdvancedPolicy] confidence={confidence:.2f}, fatigue={fatigue:.2f}, epsilon={self.epsilon:.2f}"
        )

    def update(self, *args, **kwargs):
        """
        No training logic in this simple advanced policy.
        Included for compatibility.
        """
        pass
