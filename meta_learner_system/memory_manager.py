# meta_learner_system/memory_manager.py

import os
import torch
import json
import joblib  # NEW: to handle saving scaler

class MemoryManager:
    def __init__(self, memory_folder):
        self.memory_folder = memory_folder
        self.model_path = os.path.join(memory_folder, "meta_learner_model.pt")
        self.scaler_path = os.path.join(memory_folder, "scaler_X.pkl")  # NEW

    def save_model(self, model):
        torch.save(model.state_dict(), self.model_path)
        print(f"✅ Model saved to {self.model_path}")

    def load_model(self):
        if os.path.exists(self.model_path):
            from meta_learner_system.model_trainer import ModelTrainer  # delayed import
            trainer = ModelTrainer(memory_manager=self)  # Pass self
            trainer.model.load_state_dict(torch.load(self.model_path))
            print(f"✅ Loaded model from {self.model_path}")

            # Optionally load scaler if exists
            if os.path.exists(self.scaler_path):
                trainer.scaler_X = joblib.load(self.scaler_path)
                print(f"✅ Loaded scaler from {self.scaler_path}")
            else:
                print(f"⚠️ No scaler found at {self.scaler_path}")

            return trainer.model
        else:
            return None

    def save_scaler(self, scaler):
        joblib.dump(scaler, self.scaler_path)
        print(f"✅ Scaler saved to {self.scaler_path}")

    def load_scaler(self):
        if os.path.exists(self.scaler_path):
            scaler = joblib.load(self.scaler_path)
            print(f"✅ Scaler loaded from {self.scaler_path}")
            return scaler
        else:
            print(f"⚠️ No scaler found at {self.scaler_path}")
            return None

    def save_checkpoint(self, checkpoint_dict):
        epoch_num = checkpoint_dict.get('epoch', 'unknown')
        filename = os.path.join(self.memory_folder, f'checkpoint_epoch_{epoch_num}.json')
        with open(filename, 'w') as f:
            json.dump(checkpoint_dict, f, indent=4)
        print(f"💾 Saved checkpoint: checkpoint_epoch_{epoch_num}.json")
