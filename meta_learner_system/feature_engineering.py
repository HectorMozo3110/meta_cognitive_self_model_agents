# meta_learner_system/feature_engineering.py

import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, rolling_window=50):
        self.rolling_window = rolling_window

    def transform(self, df):
        df = df.copy()

        # Encode current_mode (string) → numeric
        if df["current_mode"].dtype == object:
            df["mode_numeric"] = df["current_mode"].apply(lambda x: 1 if x == "exploitation" else 0)
        else:
            df["mode_numeric"] = df["current_mode"]

        # Rolling mean and std dev
        df["conf_roll_mean"] = df["confidence_level"].rolling(self.rolling_window, min_periods=1).mean()
        df["conf_roll_std"] = df["confidence_level"].rolling(self.rolling_window, min_periods=1).std().fillna(0)

        df["fatigue_roll_mean"] = df["fatigue_level"].rolling(self.rolling_window, min_periods=1).mean()
        df["fatigue_roll_std"] = df["fatigue_level"].rolling(self.rolling_window, min_periods=1).std().fillna(0)

        # Deltas
        df["conf_delta"] = df["confidence_level"].diff().fillna(0)
        df["fatigue_delta"] = df["fatigue_level"].diff().fillna(0)
        df["reward_delta"] = df["reward"].diff().fillna(0)

        # Normalize reward
        df["reward_norm"] = (df["reward"] - df["reward"].mean()) / (df["reward"].std() + 1e-6)

        # Select final features
        features = df[[
            "reward_norm",
            "conf_roll_mean",
            "conf_roll_std",
            "conf_delta",
            "fatigue_roll_mean",
            "fatigue_roll_std",
            "fatigue_delta",
            "mode_numeric",
            "reward_delta"
        ]].fillna(0)

        return features
