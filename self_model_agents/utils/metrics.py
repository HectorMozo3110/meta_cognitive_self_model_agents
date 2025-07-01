# === Scientific metrics for SelfModel experiments ===

import pandas as pd

def compute_scientific_metrics(metrics_csv_path):
    df = pd.read_csv(metrics_csv_path)

    # Compute switching rate
    mode_changes = df["current_mode"].shift(1) != df["current_mode"]
    num_switches = mode_changes.sum() - 1  # discount first row
    switching_rate = num_switches / (len(df) - 1)

    # Compute averages
    avg_confidence = df["confidence_level"].mean()
    avg_fatigue = df["fatigue_level"].mean()

    # Compute time spent in each mode
    num_exploitation = (df["current_mode"] == "exploitation").sum()
    num_exploration = (df["current_mode"] == "exploration").sum()
    total_steps = len(df)

    time_in_exploitation = num_exploitation / total_steps
    time_in_exploration = num_exploration / total_steps

    # Prepare results dictionary
    results = {
        "switching_rate": switching_rate,
        "avg_confidence": avg_confidence,
        "avg_fatigue": avg_fatigue,
        "time_in_exploitation": time_in_exploitation,
        "time_in_exploration": time_in_exploration
    }

    # Return results dataframe (the caller will save it)
    df_results = pd.DataFrame([results])
    return df_results




