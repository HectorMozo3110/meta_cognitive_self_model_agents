# self_model/base_self_model.py

from abc import ABC, abstractmethod

class BaseSelfModel(ABC):
    """
    Abstract base class for SelfModel implementations.
    All SelfModels must implement this interface.
    """

    @abstractmethod
    def update(self, reward, done):
        """
        Update the SelfModel based on new experience.
        
        Args:
            reward (float): The reward received from the environment.
            done (bool): Whether the episode has ended.
        """
        pass

    @abstractmethod
    def to_dict(self):
        """
        Return the current state of the SelfModel as a dictionary.
        
        Returns:
            dict: The current state of the SelfModel.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the SelfModel to its initial state.
        """
        pass
