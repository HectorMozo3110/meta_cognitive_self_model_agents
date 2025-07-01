import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from self_model_agents.self_model.base_self_model import BaseSelfModel

# Simple feedforward NN for SelfModel
class SimpleSelfModelNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=2):
        super(SimpleSelfModelNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Outputs in [0,1] for confidence and fatigue
        )
    
    def forward(self, x):
        return self.model(x)

class SimpleSelfModel(BaseSelfModel):
    """
    SimpleSelfModel (intelligent version with random architecture).
    Learns confidence_level and fatigue_level using a neural network.
    """
    def __init__(self, params, log_dir=None):  # added log_dir
        # Initialize internal state
        self.confidence_level = 0.5
        self.fatigue_level = 0.0
        self.current_mode = "exploration"

        # Hyperparameters
        self.confidence_threshold = params["CONFIDENCE_THRESHOLD"]
        self.fatigue_threshold = params["FATIGUE_THRESHOLD"]

        # Last step info
        self.last_reward = 0.0
        self.last_action_type = 0.0  # 0 = exploration, 1 = exploitation

        # Random architecture parameters
        self.hidden_dim = random.choice([8, 16, 32, 64])
        self.learning_rate = random.uniform(0.0001, 0.01)

        # Neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn = SimpleSelfModelNN(hidden_dim=self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.nn.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Logging
        self.log_dir = log_dir or "outputs/self_model_logs/simple"
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = open(os.path.join(self.log_dir, "log_training_simple.csv"), "w")
        self.log_file.write("step,loss,confidence_level,fatigue_level\n")
        self.training_step = 0

        # Save architecture info (for reproducibility)
        with open(os.path.join(self.log_dir, "architecture_info.txt"), "w") as f:
            f.write(f"Hidden_dim: {self.hidden_dim}\n")
            f.write(f"Learning_rate: {self.learning_rate:.6f}\n")

    def update(self, reward, done=None):
        # Prepare input tensor
        input_tensor = torch.tensor([
            self.last_reward,
            self.confidence_level,
            self.fatigue_level,
            self.last_action_type
        ], dtype=torch.float32).unsqueeze(0).to(self.device)

        # Target values for confidence and fatigue
        target_confidence = self.confidence_level + (0.05 if reward > 0 else -0.05)
        target_fatigue = self.fatigue_level + (0.02 if self.current_mode == "exploitation" else -0.02)

        target_confidence = np.clip(target_confidence, 0.0, 1.0)
        target_fatigue = np.clip(target_fatigue, 0.0, 1.0)

        target_tensor = torch.tensor([target_confidence, target_fatigue], dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward pass
        output = self.nn(input_tensor)
        loss = self.criterion(output, target_tensor)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update internal state with NN output
        output_np = output.detach().cpu().numpy()[0]
        self.confidence_level = output_np[0]
        self.fatigue_level = output_np[1]

        # Update current mode based on thresholds
        if self.confidence_level >= self.confidence_threshold and self.fatigue_level < self.fatigue_threshold:
            self.current_mode = "exploitation"
            self.last_action_type = 1.0
        else:
            self.current_mode = "exploration"
            self.last_action_type = 0.0

        # Save reward for next step
        self.last_reward = reward

        # Log this training step
        self.training_step += 1
        self.log_training_step(loss.item())

    def log_training_step(self, loss):
        self.log_file.write(f"{self.training_step},{loss:.6f},{self.confidence_level:.4f},{self.fatigue_level:.4f}\n")
        self.log_file.flush()

    def save_weights(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.nn.state_dict(), path)

    def load_weights(self, path):
        self.nn.load_state_dict(torch.load(path, map_location=self.device))

    def get_state(self):
        return {
            "confidence_level": self.confidence_level,
            "fatigue_level": self.fatigue_level,
            "current_mode": self.current_mode,
            "predicted_confidence": 0.0,  # Not used in SimpleSelfModel
            "confidence_error_history": 0.0,  # Not used in SimpleSelfModel
            "mode_error_history": 0.0  # Not used in SimpleSelfModel
        }

    def to_dict(self):
        return self.get_state()

    def reset(self):
        self.confidence_level = 0.5
        self.fatigue_level = 0.0
        self.current_mode = "exploration"
        self.last_reward = 0.0
        self.last_action_type = 0.0
