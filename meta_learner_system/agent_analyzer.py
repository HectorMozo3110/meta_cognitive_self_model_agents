# meta_learner_system/agent_analyzer.py

import pandas as pd
import os

class AgentAnalyzer:
    def __init__(self, reports_folder):
        self.reports_folder = reports_folder

    def analyze(self, scientific_data):
        print("🧠 Analyzing agents...")

        # Build DataFrame with all agents' scientific metrics
        summary_list = []
        for i, df in enumerate(scientific_data):
            try:
                row = {col: df[col].values[0] for col in df.columns}
                row["Agent"] = i + 1
                summary_list.append(row)
            except Exception as e:
                print(f"⚠️ Skipping agent {i+1} due to error: {e}")

        df_summary = pd.DataFrame(summary_list)

        if df_summary.empty:
            print("❌ No valid scientific data to analyze.")
            return

        # Basic analysis
        print("\n=== Agent Insights ===")
        best_conf_agent = df_summary.loc[df_summary['avg_confidence'].idxmax()]["Agent"]
        best_fatigue_agent = df_summary.loc[df_summary['avg_fatigue'].idxmin()]["Agent"]
        best_switching_agent = df_summary.loc[df_summary['switching_rate'].idxmax()]["Agent"]

        print(f"🔹 Agent with highest avg_confidence: Agent {best_conf_agent}")
        print(f"🔹 Agent with lowest avg_fatigue: Agent {best_fatigue_agent}")
        print(f"🔹 Agent with highest switching_rate: Agent {best_switching_agent}")

        # Save insights to text file
        insights_path = os.path.join(self.reports_folder, "meta_learner_insights.txt")
        with open(insights_path, "w") as f:
            f.write("=== Agent Insights ===\n")
            f.write(f"Agent with highest avg_confidence: Agent {best_conf_agent}\n")
            f.write(f"Agent with lowest avg_fatigue: Agent {best_fatigue_agent}\n")
            f.write(f"Agent with highest switching_rate: Agent {best_switching_agent}\n")

        print(f"\n✅ Insights saved to {insights_path}")
