import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from self_model_agents.self_model.base_self_model import BaseSelfModel

# NN for confidence and fatigue
class ConfidenceFatigueNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=2):
        super(ConfidenceFatigueNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Outputs in [0,1]
        )
    def forward(self, x):
        return self.model(x)

# NN for predicted_confidence
class PredictedConfidenceNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=1):
        super(PredictedConfidenceNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # [0,1]
        )
    def forward(self, x):
        return self.model(x)

# NN for predicted_mode (classification → 2 classes)
class PredictedModeNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=2):
        super(PredictedModeNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.model(x)

class AdvancedSelfModel(BaseSelfModel):
    """
    AdvancedSelfModel (intelligent version with full randomization).
    Learns confidence_level, fatigue_level, predicted_confidence, and predicted_mode.
    """
    def __init__(
        self,
        initial_confidence,
        initial_fatigue,
        confidence_increase,
        confidence_decrease,
        fatigue_increase,
        fatigue_decrease,
        confidence_threshold,
        fatigue_threshold,
        confidence_learning_rate,
        log_dir=None  # ADDED PARAMETER
    ):
        # Internal state
        self.confidence_level = initial_confidence
        self.fatigue_level = initial_fatigue
        self.current_mode = "exploration"

        self.confidence_threshold = confidence_threshold
        self.fatigue_threshold = fatigue_threshold

        # Meta-cognitive state
        self.predicted_confidence = initial_confidence
        self.confidence_error_history = []

        self.predicted_mode = "exploration"
        self.mode_error_history = []

        # Last step info
        self.last_reward = 0.0
        self.last_action_type = 0.0  # 0 = exploration, 1 = exploitation

        # Parameters for state update (from params)
        self.confidence_increase = confidence_increase
        self.confidence_decrease = confidence_decrease
        self.fatigue_increase = fatigue_increase
        self.fatigue_decrease = fatigue_decrease

        # Random architecture parameters
        self.hidden_dim_conf_fat = random.choice([8, 16, 32, 64])
        self.hidden_dim_pred_conf = random.choice([8, 16, 32, 64])
        self.hidden_dim_pred_mode = random.choice([8, 16, 32, 64])

        self.lr_conf_fat = random.uniform(0.0001, 0.01)
        self.lr_pred_conf = random.uniform(0.0001, 0.01)
        self.lr_pred_mode = random.uniform(0.0001, 0.01)

        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.nn_conf_fat = ConfidenceFatigueNN(hidden_dim=self.hidden_dim_conf_fat).to(self.device)
        self.optimizer_conf_fat = optim.Adam(self.nn_conf_fat.parameters(), lr=self.lr_conf_fat)

        self.nn_pred_conf = PredictedConfidenceNN(hidden_dim=self.hidden_dim_pred_conf).to(self.device)
        self.optimizer_pred_conf = optim.Adam(self.nn_pred_conf.parameters(), lr=self.lr_pred_conf)

        self.nn_pred_mode = PredictedModeNN(hidden_dim=self.hidden_dim_pred_mode).to(self.device)
        self.optimizer_pred_mode = optim.Adam(self.nn_pred_mode.parameters(), lr=self.lr_pred_mode)

        self.criterion_reg = nn.MSELoss()
        self.criterion_class = nn.CrossEntropyLoss()

        # Logging (modified with log_dir)
        self.log_dir = log_dir or "outputs/self_model_logs/advanced"
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = open(os.path.join(self.log_dir, "log_training_advanced.csv"), "w")
        self.log_file.write("step,loss_conf_fat,loss_pred_conf,loss_pred_mode,confidence_level,fatigue_level,predicted_confidence,mode_error\n")
        self.training_step = 0

        # Save architecture info
        with open(os.path.join(self.log_dir, "architecture_info.txt"), "w") as f:
            f.write(f"Hidden_dim_conf_fat: {self.hidden_dim_conf_fat}\n")
            f.write(f"Hidden_dim_pred_conf: {self.hidden_dim_pred_conf}\n")
            f.write(f"Hidden_dim_pred_mode: {self.hidden_dim_pred_mode}\n")
            f.write(f"LR_conf_fat: {self.lr_conf_fat:.6f}\n")
            f.write(f"LR_pred_conf: {self.lr_pred_conf:.6f}\n")
            f.write(f"LR_pred_mode: {self.lr_pred_mode:.6f}\n")
            f.write(f"Confidence_increase: {self.confidence_increase}\n")
            f.write(f"Confidence_decrease: {self.confidence_decrease}\n")
            f.write(f"Fatigue_increase: {self.fatigue_increase}\n")
            f.write(f"Fatigue_decrease: {self.fatigue_decrease}\n")

    def update(self, reward, done=None):
        input_tensor = torch.tensor([
            self.last_reward,
            self.confidence_level,
            self.fatigue_level,
            self.last_action_type
        ], dtype=torch.float32).unsqueeze(0).to(self.device)

        target_confidence = self.confidence_level + (self.confidence_increase if reward > 0 else -self.confidence_decrease)
        target_fatigue = self.fatigue_level + (self.fatigue_increase if self.current_mode == "exploitation" else -self.fatigue_decrease)

        target_confidence = np.clip(target_confidence, 0.0, 1.0)
        target_fatigue = np.clip(target_fatigue, 0.0, 1.0)

        target_conf_fat = torch.tensor([target_confidence, target_fatigue], dtype=torch.float32).unsqueeze(0).to(self.device)

        target_pred_conf = torch.tensor([self.confidence_level], dtype=torch.float32).unsqueeze(0).to(self.device)

        current_mode_numeric = 1 if self.current_mode == "exploitation" else 0
        target_pred_mode = torch.tensor([current_mode_numeric], dtype=torch.long).to(self.device)

        output_conf_fat = self.nn_conf_fat(input_tensor)
        loss_conf_fat = self.criterion_reg(output_conf_fat, target_conf_fat)
        self.optimizer_conf_fat.zero_grad()
        loss_conf_fat.backward()
        self.optimizer_conf_fat.step()

        output_pred_conf = self.nn_pred_conf(input_tensor)
        loss_pred_conf = self.criterion_reg(output_pred_conf, target_pred_conf)
        self.optimizer_pred_conf.zero_grad()
        loss_pred_conf.backward()
        self.optimizer_pred_conf.step()

        output_pred_mode = self.nn_pred_mode(input_tensor)
        loss_pred_mode = self.criterion_class(output_pred_mode, target_pred_mode)
        self.optimizer_pred_mode.zero_grad()
        loss_pred_mode.backward()
        self.optimizer_pred_mode.step()

        output_np = output_conf_fat.detach().cpu().numpy()[0]
        self.confidence_level = output_np[0]
        self.fatigue_level = output_np[1]

        if self.confidence_level >= self.confidence_threshold and self.fatigue_level < self.fatigue_threshold:
            self.current_mode = "exploitation"
            self.last_action_type = 1.0
        else:
            self.current_mode = "exploration"
            self.last_action_type = 0.0

        self.predicted_confidence = output_pred_conf.detach().cpu().numpy()[0][0]

        pred_mode_class = output_pred_mode.argmax(dim=1).item()
        predicted_mode_str = "exploitation" if pred_mode_class == 1 else "exploration"

        mode_error = current_mode_numeric - pred_mode_class
        self.confidence_error_history.append(self.confidence_level - self.predicted_confidence)
        self.mode_error_history.append(mode_error)

        if len(self.confidence_error_history) > 100:
            self.confidence_error_history.pop(0)
        if len(self.mode_error_history) > 100:
            self.mode_error_history.pop(0)

        self.predicted_mode = predicted_mode_str

        self.last_reward = reward

        self.training_step += 1
        self.log_training_step(loss_conf_fat.item(), loss_pred_conf.item(), loss_pred_mode.item(), mode_error)

    def log_training_step(self, loss_conf_fat, loss_pred_conf, loss_pred_mode, mode_error):
        self.log_file.write(f"{self.training_step},{loss_conf_fat:.6f},{loss_pred_conf:.6f},{loss_pred_mode:.6f},"
                            f"{self.confidence_level:.4f},{self.fatigue_level:.4f},{self.predicted_confidence:.4f},{mode_error}\n")
        self.log_file.flush()

    def save_weights(self, path_base):
        os.makedirs(os.path.dirname(path_base), exist_ok=True)
        torch.save(self.nn_conf_fat.state_dict(), path_base + "_conf_fat.pth")
        torch.save(self.nn_pred_conf.state_dict(), path_base + "_pred_conf.pth")
        torch.save(self.nn_pred_mode.state_dict(), path_base + "_pred_mode.pth")

    def load_weights(self, path_base):
        self.nn_conf_fat.load_state_dict(torch.load(path_base + "_conf_fat.pth", map_location=self.device))
        self.nn_pred_conf.load_state_dict(torch.load(path_base + "_pred_conf.pth", map_location=self.device))
        self.nn_pred_mode.load_state_dict(torch.load(path_base + "_pred_mode.pth", map_location=self.device))

    def get_state(self):
        return {
            "confidence_level": self.confidence_level,
            "fatigue_level": self.fatigue_level,
            "current_mode": self.current_mode,
            "predicted_confidence": self.predicted_confidence,
            "confidence_error_history": self.confidence_error_history[-1] if self.confidence_error_history else 0.0,
            "predicted_mode": self.predicted_mode,
            "mode_error_history": self.mode_error_history[-1] if self.mode_error_history else 0.0
        }

    def to_dict(self):
        return self.get_state()

    def reset(self):
        self.confidence_level = 0.5
        self.fatigue_level = 0.0
        self.current_mode = "exploration"
        self.predicted_confidence = 0.5
        self.confidence_error_history = []
        self.predicted_mode = "exploration"
        self.mode_error_history = []
        self.last_reward = 0.0
        self.last_action_type = 0.0
