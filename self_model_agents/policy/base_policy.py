# self_model_agents/policy/base_policy.py

class BasePolicy:
    """
    Base class for all policies.
    Provides a consistent interface.
    """

    def select_action(self, observation):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        pass
