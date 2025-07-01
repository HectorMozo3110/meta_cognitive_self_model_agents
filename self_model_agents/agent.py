# self_model_agents/agent.py

class SelfModelAgent:
    """
    SelfModelAgent that integrates a SelfModel and a Policy to interact with an environment.
    """

    def __init__(self, self_model, policy, env):
        self.self_model = self_model
        self.policy = policy
        self.env = env

        # Reset env and extract only observation
        obs, _ = self.env.reset()
        self.observation = obs
        self.total_steps = 0

    def step(self):
        """
        Perform one interaction step: select action, apply to environment,
        update SelfModel based on reward and done flag.
        """
        # Select action
        action = self.policy.select_action(self.observation)

        # Step in env
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        # Update SelfModel
        self.self_model.update(reward, done)

        # Update observation
        if done:
            next_obs, _ = self.env.reset()
        self.observation = next_obs

        self.total_steps += 1

        return reward, done

    def get_self_model_state(self):
        """
        Return the current state of the SelfModel as a dictionary.
        """
        return self.self_model.to_dict()

    def reset(self):
        """
        Reset the agent (SelfModel and environment).
        """
        self.self_model.reset()
        obs, _ = self.env.reset()
        self.observation = obs
        self.total_steps = 0