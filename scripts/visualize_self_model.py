# visualize_self_model.py

import os
import pandas as pd
import matplotlib.pyplot as plt

# Main function to visualize self-model metrics
def visualize_self_model(policy_name=None, self_model_name=None):

    # Ask user input if not provided
    valid_policies = ["dummy", "rl", "hybrid", "advanced"]
    valid_self_models = ["simple", "advanced"]

    if policy_name is None:
        print("Which POLICY visualization do you want to generate? (dummy / rl / hybrid / advanced): ", end="")
        policy_name = input().strip().lower()

    if self_model_name is None:
        print("Which SELF MODEL was used? (simple / advanced): ", end="")
        self_model_name = input().strip().lower()

    # Validate inputs
    if policy_name not in valid_policies:
        raise ValueError(f"\n❌ Invalid policy: {policy_name}. Please choose one of: dummy / rl / hybrid / advanced.\n")

    if self_model_name not in valid_self_models:
        raise ValueError(f"\n❌ Invalid SelfModel: {self_model_name}. Please choose one of: simple / advanced.\n")

    # Construct metrics file path
    metrics_path = f"outputs/metrics/gridworld_metrics_{policy_name}_{self_model_name}.csv"

    # Check if file exists
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"\n❌ Metrics file not found: {metrics_path}\nRun training first for this policy + self model.\n")

    # Create visualization output folder if it doesn't exist
    os.makedirs("outputs/visualizations", exist_ok=True)

    # Load metrics from CSV
    df = pd.read_csv(metrics_path)
    print(f"\n✅ Loaded metrics from: {metrics_path}")
    print(f"🎨 Generating visualization for POLICY: {policy_name.upper()} + SELF MODEL: {self_model_name.upper()}\n")

    # Load scientific metrics CSV
    scientific_metrics_path = f"outputs/scientific_metrics/scientific_metrics_{policy_name}_{self_model_name}.csv"
    if os.path.isfile(scientific_metrics_path):
        scientific_df = pd.read_csv(scientific_metrics_path)
        scientific_text = ""
        for col in scientific_df.columns:
            scientific_text += f"{col}: {scientific_df[col].values[0]:.4f}\n"

        print(f"\n✅ Loaded scientific metrics from: {scientific_metrics_path}\n")
    else:
        scientific_text = "No scientific metrics found."
        print(f"\n⚠️ No scientific metrics file found at: {scientific_metrics_path}\n")

    # Plot configuration
    plt.style.use("seaborn-v0_8")
    fig, axs = plt.subplots(7, 1, figsize=(14, 24), sharex=True, constrained_layout=True)

    ylabel_fontsize = 9  # smaller font size for Y-axis labels

    # Plot reward
    axs[0].plot(df["step"], df["reward"], color="red")
    axs[0].set_ylabel("Reward", fontsize=ylabel_fontsize)
    axs[0].set_title(f"Reward over Time ({policy_name.upper()} + {self_model_name.upper()})")

    # Plot confidence level
    axs[1].plot(df["step"], df["confidence_level"], color="blue")
    axs[1].set_ylabel("Confidence", fontsize=ylabel_fontsize)
    axs[1].set_title(f"Evolution of SelfModel Confidence ({policy_name.upper()} + {self_model_name.upper()})")

    # Plot fatigue level
    axs[2].plot(df["step"], df["fatigue_level"], color="orange")
    axs[2].set_ylabel("Fatigue", fontsize=ylabel_fontsize)
    axs[2].set_title(f"Evolution of Agent Fatigue ({policy_name.upper()} + {self_model_name.upper()})")

    # Plot current mode
    if df["current_mode"].dtype == object:
        axs[3].plot(df["step"], df["current_mode"].apply(lambda x: 1 if x == "exploitation" else 0), color="green")
        axs[3].set_ylabel("Mode\n(1=explt, 0=explr)", fontsize=ylabel_fontsize)
    else:
        axs[3].plot(df["step"], df["current_mode"], color="green")
        axs[3].set_ylabel("Mode", fontsize=ylabel_fontsize)

    axs[3].set_xlabel("Step")
    axs[3].set_title(f"Agent Mode over Time ({policy_name.upper()} + {self_model_name.upper()})")

    # Plot predicted confidence vs actual confidence
    if "predicted_confidence" in df.columns:
        axs[4].plot(df["step"], df["confidence_level"], label="Actual (SelfModel real confidence)", color="blue")
        axs[4].plot(df["step"], df["predicted_confidence"], label="Predicted (Internal model estimation)", color="red", linestyle="--")
        axs[4].set_ylabel("Conf. (Act vs Pred)", fontsize=ylabel_fontsize)
        axs[4].set_title(f"Confidence Prediction vs Actual ({policy_name.upper()} + {self_model_name.upper()})")
        axs[4].legend(fontsize=9)

    # Plot confidence error history
    if "confidence_error_history" in df.columns:
        axs[5].plot(df["step"], df["confidence_error_history"], color="purple")
        axs[5].set_ylabel("Conf. Pred Err", fontsize=ylabel_fontsize)
        axs[5].set_title(f"Confidence Prediction Error History ({policy_name.upper()} + {self_model_name.upper()})")

    # Plot mode error history
    if "mode_error_history" in df.columns:
        axs[6].plot(df["step"], df["mode_error_history"], color="brown")
        axs[6].set_ylabel("Mode Pred Err", fontsize=ylabel_fontsize)
        axs[6].set_title(f"Mode Prediction Error History ({policy_name.upper()} + {self_model_name.upper()})")
        axs[6].set_xlabel("Step")

    # Adjust left margin
    fig.subplots_adjust(left=0.15)

    # Add scientific metrics as a text box in top right corner
    fig.text(
        0.75, 0.85,
        scientific_text,
        fontsize=11,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    # Save figure
    save_name = f"outputs/visualizations/self_model_evolution_{policy_name}_{self_model_name}.png"
    plt.savefig(save_name)
    print(f"\n✅ Visualization saved to: {save_name}\n")

    # Show plot
    plt.show()

# Entry point for terminal execution
if __name__ == "__main__":
    visualize_self_model()
