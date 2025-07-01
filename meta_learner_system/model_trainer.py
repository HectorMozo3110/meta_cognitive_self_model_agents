# meta_learner_system/model_trainer.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from meta_learner_system.feature_engineering import FeatureEngineer

class DeepMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

class ModelTrainer:
    def __init__(self, memory_manager, model=None):
        self.memory_manager = memory_manager
        self.feature_engineer = FeatureEngineer()
        self.scaler_X = StandardScaler()  # NEW: feature scaler
        self.model = model or self._build_model()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=0.5)

    def _build_model(self):
        return DeepMLP(input_dim=9, output_dim=5)

    def train(self, metrics_data, scientific_data, epochs=15, batch_size=64):
        X_list = []
        y_list = []

        def _safe_mean(x):
            if isinstance(x, list):
                return np.mean(x) if len(x) > 0 else 0.0
            return x

        # === Prepare training data ===
        for df in metrics_data:
            try:
                for col in ["predicted_confidence", "confidence_error_history", "mode_error_history"]:
                    if col in df.columns:
                        df[col] = df[col].apply(_safe_mean)

                X_feat = self.feature_engineer.transform(df).values

                switch_rate = scientific_data[-1].get("switching_rate", [0])[0]
                avg_conf = scientific_data[-1].get("avg_confidence", [0])[0]
                avg_fatigue = scientific_data[-1].get("avg_fatigue", [0])[0]

                target_rows = df.shape[0]
                y_target = np.tile([switch_rate, avg_conf, avg_fatigue], (target_rows, 1))
                y_full = np.hstack((
                    df[["confidence_level", "fatigue_level"]].fillna(0).values,
                    y_target
                ))

                X_list.append(X_feat)
                y_list.append(y_full)

            except Exception as e:
                print(f"⚠️ Skipping file due to error: {e}")

        if len(X_list) == 0:
            print("❌ No valid data to train on.")
            return

        X_all = np.vstack(X_list)
        y_all = np.vstack(y_list)

        # === Scale X ===
        X_all_scaled = self.scaler_X.fit_transform(X_all)

        # === Save scaler to memory ===
        scaler_path = os.path.join(self.memory_manager.memory_folder, "scaler_X.pkl")
        joblib.dump(self.scaler_X, scaler_path)
        print(f"✅ Feature scaler saved to {scaler_path}")

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_all_scaled, dtype=torch.float32),
            torch.tensor(y_all, dtype=torch.float32)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float("inf")

        # Optional: initialize training log
        training_log = []

        for epoch in range(epochs):
            running_loss = 0.0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

            self.scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.memory_manager.save_model(self.model)
                print(f"✅ New best model saved at epoch {epoch + 1}!")

            # Save checkpoint
            self.memory_manager.save_checkpoint({
                'epoch': epoch + 1,
                'loss': avg_loss
            })

            # Save to training log
            training_log.append({"epoch": epoch + 1, "loss": avg_loss})

        # === Save training log CSV ===
        training_log_df = pd.DataFrame(training_log)
        log_path = os.path.join(self.memory_manager.memory_folder, "training_log.csv")
        training_log_df.to_csv(log_path, index=False)
        print(f"✅ Training log saved to {log_path}")

    def get_model(self):
        return self.model
