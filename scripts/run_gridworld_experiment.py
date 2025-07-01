import gymnasium as gym
import yaml
import os
import csv
import sys
import pandas as pd
import random

from self_model_agents.agent import SelfModelAgent
from self_model_agents.self_model.simple_self_model import SimpleSelfModel
from self_model_agents.self_model.advanced_self_model import AdvancedSelfModel
from self_model_agents.policy.rl_policy import RLPolicy
from self_model_agents.policy.hybrid_policy import HybridPolicy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from self_model_agents.utils.metrics import compute_scientific_metrics

# DummyPolicy for initial testing
class DummyPolicy:
    def select_action(self, observation):
        return env.action_space.sample()

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create output directories
os.makedirs("outputs/logs", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/visualizations", exist_ok=True)
os.makedirs("outputs/scientific_metrics", exist_ok=True)

# Initialize environment
env = gym.make(config["environment"])

# Function to generate random parameters
def get_random_parameters():
    params = {}
    params['TOTAL_TIMESTEPS'] = random.randint(5000, 100000)
    params['HYBRID_RATIO'] = random.uniform(0.0, 1.0)
    params['EPSILON'] = random.uniform(0.0, 1.0)
    params['MIN_EPSILON'] = random.uniform(0.0, 0.1)
    params['MAX_EPSILON'] = random.uniform(0.8, 1.0)
    params['EPSILON_DECAY_RATE'] = random.uniform(0.0001, 0.1)
    params['CONFIDENCE_INCREASE'] = random.uniform(0.001, 0.05)
    params['CONFIDENCE_DECREASE'] = random.uniform(0.001, 0.05)
    params['FATIGUE_INCREASE'] = random.uniform(0.001, 0.02)
    params['FATIGUE_DECREASE'] = random.uniform(0.001, 0.02)
    params['CONFIDENCE_THRESHOLD'] = random.uniform(0.6, 0.9)
    params['FATIGUE_THRESHOLD'] = random.uniform(0.6, 0.9)
    params['CONFIDENCE_LEARNING_RATE'] = random.uniform(0.001, 0.1)
    params['INITIAL_CONFIDENCE'] = random.uniform(0.0, 1.0)
    params['INITIAL_FATIGUE'] = random.uniform(0.0, 1.0)
    params['HISTORY_SIZE'] = random.randint(1, 100)
    params['EMOTION_FACTOR'] = random.uniform(0.0, 1.0)
    return params

# Main function to run the experiment
def run_gridworld_experiment(self_model_choice=None, policy_choice=None, params=None):

    if params is None:
        params = get_random_parameters()

    print("\nParameters for this run:")
    for k, v in params.items():
        print(f"{k}: {v}")

    # Select SelfModel
    if self_model_choice is None:
        self_model_choice = input("Which SelfModel do you want to run? (simple / advanced): ").strip().lower()

    if self_model_choice == "simple":
        self_model = SimpleSelfModel(params)
    elif self_model_choice == "advanced":
        self_model = AdvancedSelfModel(
            initial_confidence=params['INITIAL_CONFIDENCE'],
            initial_fatigue=params['INITIAL_FATIGUE'],
            confidence_increase=params['CONFIDENCE_INCREASE'],
            confidence_decrease=params['CONFIDENCE_DECREASE'],
            fatigue_increase=params['FATIGUE_INCREASE'],
            fatigue_decrease=params['FATIGUE_DECREASE'],
            confidence_threshold=params['CONFIDENCE_THRESHOLD'],
            fatigue_threshold=params['FATIGUE_THRESHOLD'],
            confidence_learning_rate=params['CONFIDENCE_LEARNING_RATE']
        )
    else:
        raise ValueError("Invalid SelfModel selected. Choose 'simple' or 'advanced'.")

    # Select Policy
    if policy_choice is None:
        policy_choice = input("Which Policy do you want to run? (dummy / rl / hybrid / advanced): ").strip().lower()

    if policy_choice == "dummy":
        policy = DummyPolicy()
    elif policy_choice == "rl":
        print("\nUsing RLPolicy (PPO)...\n")
        policy = RLPolicy(env, total_timesteps=params['TOTAL_TIMESTEPS'])
    elif policy_choice == "hybrid":
        print("\nUsing HybridPolicy (RL + SelfModel)...\n")
        policy = HybridPolicy(
            env,
            hybrid_ratio=params['HYBRID_RATIO'],
            total_timesteps=params['TOTAL_TIMESTEPS']
        )
    elif policy_choice == "advanced":
        print("\nUsing AdvancedPolicy (Self-Adaptive)...\n")
        from self_model_agents.policy.advanced_policy import AdvancedPolicy
        policy = AdvancedPolicy(
            env.action_space,
            self_model,
            epsilon=params['EPSILON'],
            min_epsilon=params['MIN_EPSILON'],
            max_epsilon=params['MAX_EPSILON'],
            epsilon_decay_rate=params['EPSILON_DECAY_RATE']
        )
    else:
        raise ValueError("Invalid Policy selected. Choose 'dummy', 'rl', 'hybrid' or 'advanced'.")

    # Initialize agent
    agent = SelfModelAgent(self_model, policy, env)

    # Training loop parameters
    num_steps = config["num_steps"]
    log_interval = config["log_interval"]

    # Prepare CSV file for metrics
    metrics_path = f"outputs/metrics/gridworld_metrics_{policy_choice}_{self_model_choice}.csv"
    metrics_file = open(metrics_path, mode="w", newline="")
    csv_writer = csv.writer(metrics_file)

    # Write CSV header
    csv_writer.writerow([
        "step",
        "reward",
        "confidence_level",
        "fatigue_level",
        "current_mode",
        "predicted_confidence",
        "confidence_error_history",
        "mode_error_history"
    ])

    print(f"\n✅ Running experiment with {policy_choice.upper()} policy and {self_model_choice.upper()} SelfModel.\n")
    print("Starting training loop...")

    # Main training loop
    for step in range(1, num_steps + 1):
        reward, done = agent.step()
        state = agent.get_self_model_state()

        csv_writer.writerow([
            step,
            reward,
            state["confidence_level"],
            state["fatigue_level"],
            state["current_mode"],
            state.get("predicted_confidence", 0.0),
            state.get("confidence_error_history", 0.0),
            state.get("mode_error_history", 0.0)
        ])

        if step % log_interval == 0:
            print(f"Step {step} | Reward: {reward:.2f} | Confidence: {state['confidence_level']:.2f} | Fatigue: {state['fatigue_level']:.2f} | Mode: {state['current_mode']}")

    # Close CSV file
    metrics_file.close()

    print(f"\n✅ Training complete. Metrics saved to: {metrics_path}\n")

    # Compute and save scientific metrics
    scientific_metrics = compute_scientific_metrics(metrics_path)
    scientific_metrics_path = f"outputs/scientific_metrics/scientific_metrics_{policy_choice}_{self_model_choice}.csv"
    scientific_metrics.to_csv(scientific_metrics_path, index=False)

    print(f"\n✅ Scientific metrics saved to: {scientific_metrics_path}\n")

    # === Save model weights ===
    weights_path_base = (
        f"outputs/self_model_weights/advanced/advanced_model"
        if self_model_choice == "advanced"
        else f"outputs/self_model_weights/simple/simple_model"
    )
    os.makedirs(os.path.dirname(weights_path_base), exist_ok=True)
    self_model.save_weights(weights_path_base)
    print(f"\n✅ Weights saved to: {weights_path_base}_*.pth\n")

# Entry point for terminal execution
if __name__ == "__main__":
    run_gridworld_experiment()
