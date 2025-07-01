# meta_learner_system/data_loader.py

import os
import pandas as pd
import numpy as np
import ast

class DataLoader:
    def __init__(self):
        self.metrics_folder = "outputs/metrics"
        self.scientific_folder = "outputs/scientific_metrics"

    def load_all_metrics(self):
        print("🔍 Scanning CSVs in metrics & scientific_metrics folders...")

        metrics_data = []
        scientific_data = []

        # === Load all metrics CSVs ===
        for root, _, files in os.walk(self.metrics_folder):
            for file in files:
                if file.endswith(".csv"):
                    path = os.path.join(root, file)
                    df = pd.read_csv(path)
                    print(f"✅ Loaded metrics: {file} ({len(df)} rows)")
                    metrics_data.append(df)

        # === Load all scientific_metrics CSVs ===
        for root, _, files in os.walk(self.scientific_folder):
            for file in files:
                if file.endswith(".csv"):
                    path = os.path.join(root, file)
                    df = pd.read_csv(path)
                    print(f"✅ Loaded scientific metrics: {file}")
                    scientific_data.append(df)

        return metrics_data, scientific_data

    def _safe_mean(self, val):
        try:
            # If already numeric, return as float
            if isinstance(val, (int, float)):
                return float(val)
            
            # If string that looks like a list
            if isinstance(val, str):
                val = val.strip()
                if val.startswith("[") and val.endswith("]"):
                    # Try parsing safely
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        clean_values = []
                        for x in parsed:
                            try:
                                # Attempt to convert element to float
                                clean_values.append(float(x))
                            except:
                                # Skip elements that cannot be converted
                                continue
                        if len(clean_values) == 0:
                            # No valid numbers, return 0.0
                            return 0.0
                        else:
                            # Return mean of valid numbers
                            return float(np.mean(clean_values))
            
            # If val is a string that looks like a single number
            try:
                return float(val)
            except:
                pass
            
            # If all else fails
            return 0.0
        except:
            return 0.0
