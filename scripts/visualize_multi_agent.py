# visualize_multi_agent.py

import os
import pandas as pd
import matplotlib.pyplot as plt

# Main function to visualize multi-agent metrics
def visualize_multi_agent(num_agents=None):
    # Ask user input if not provided
    if num_agents is None:
        num_agents = int(input("How many agents do you want to visualize?: ").strip())

    # Create visualization output folder if it doesn't exist
    os.makedirs("outputs/visualizations", exist_ok=True)

    # Plot configuration
    plt.style.use("seaborn-v0_8")
    fig, axs = plt.subplots(2, 1, figsize=(14, 16), sharex=True, constrained_layout=True)

    # Colors for agents (10 distinct colors)
    colors = [
        'blue', 'orange', 'green', 'red', 'purple',
        'brown', 'pink', 'gray', 'olive', 'cyan'
    ]

    # Plot Confidence and Fatigue (shared plots)
    for agent_id in range(num_agents):
        metrics_path = f"outputs/metrics/multi_agent_metrics/multi_agent_metrics_agent_{agent_id + 1}.csv"

        if not os.path.isfile(metrics_path):
            print(f"\n⚠️ Metrics file not found for agent {agent_id}: {metrics_path}\n")
            continue

        df = pd.read_csv(metrics_path)

        # Plot confidence
        axs[0].plot(
            df["step"],
            df["confidence_level"],
            label=f"Agent {agent_id + 1}",
            color=colors[agent_id % len(colors)],
            linewidth=2
        )

        # Plot fatigue
        axs[1].plot(
            df["step"],
            df["fatigue_level"],
            label=f"Agent {agent_id + 1}",
            color=colors[agent_id % len(colors)],
            linewidth=2
        )

    # Titles and labels for shared plots
    axs[0].set_ylabel("Confidence Level")
    axs[0].set_title("Multi-Agent: SelfModel Confidence Evolution")
    axs[0].legend()

    axs[1].set_ylabel("Fatigue Level")
    axs[1].set_title("Multi-Agent: Agent Fatigue Evolution")
    axs[1].legend()

    # New Figure for Agent Mode Over Time (subplots per agent)
    fig_mode, axs_mode = plt.subplots(num_agents, 1, figsize=(14, 3 * num_agents), sharex=True, constrained_layout=True)

    if num_agents == 1:
        axs_mode = [axs_mode]  # Ensure it's iterable

    for agent_id in range(num_agents):
        metrics_path = f"outputs/metrics/multi_agent_metrics/multi_agent_metrics_agent_{agent_id + 1}.csv"

        if not os.path.isfile(metrics_path):
            print(f"\n⚠️ Metrics file not found for agent {agent_id}: {metrics_path}\n")
            continue

        df = pd.read_csv(metrics_path)

        # Convert mode to numeric if needed
        if df["current_mode"].dtype == object:
            mode_series = df["current_mode"].apply(lambda x: 1 if x == "exploitation" else 0)
        else:
            mode_series = df["current_mode"]

        # Plot Agent Mode in its own subplot
        axs_mode[agent_id].plot(
            df["step"],
            mode_series,
            color=colors[agent_id % len(colors)],
            linewidth=2
        )

        # Cleaner ylabel → only "Agent X\nMode"
        axs_mode[agent_id].set_ylabel(f"Agent {agent_id + 1}\nMode")
        axs_mode[agent_id].set_ylim(-0.1, 1.1)
        axs_mode[agent_id].grid(True)

    axs_mode[-1].set_xlabel("Step")
    fig_mode.suptitle("Multi-Agent: Agent Mode Over Time (1 = Exploitation, 0 = Exploration)", fontsize=16)

    # Add scientific metrics box for each agent
    initial_y_position = 0.85
    box_vertical_spacing = 0.16  # spacing to avoid overlap

    for agent_id in range(num_agents):
        scientific_metrics_path = f"outputs/scientific_metrics/multi_agent_scientific_metrics/multi_agent_scientific_metrics_agent_{agent_id + 1}.csv"

        if os.path.isfile(scientific_metrics_path):
            scientific_df = pd.read_csv(scientific_metrics_path)

            scientific_text = f"Agent {agent_id + 1}:\n"
            for col in scientific_df.columns:
                scientific_text += f"{col}: {scientific_df[col].values[0]:.4f}\n"

            y_position = initial_y_position - agent_id * box_vertical_spacing
            y_position = max(y_position, 0.05)

            fig.text(
                0.78,
                y_position,
                scientific_text,
                fontsize=11,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
            )
        else:
            print(f"⚠️ No scientific metrics file found for agent {agent_id + 1}.")

    # Adjust left margin
    fig.subplots_adjust(left=0.15)

    # Save figures
    save_name_main = "outputs/visualizations/multi_agent_self_model_evolution.png"
    save_name_mode = "outputs/visualizations/multi_agent_agent_mode_over_time.png"

    fig.savefig(save_name_main)
    fig_mode.savefig(save_name_mode)

    print(f"\n✅ Visualization saved to: {save_name_main}")
    print(f"✅ Mode visualization saved to: {save_name_mode}\n")

    # Show plots
    plt.show()

# Entry point for terminal execution
if __name__ == "__main__":
    visualize_multi_agent()
