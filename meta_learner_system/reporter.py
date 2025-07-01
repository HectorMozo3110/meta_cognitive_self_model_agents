# meta_learner_system/reporter.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

class Reporter:
    def __init__(self, reports_folder):
        self.reports_folder = reports_folder

    def generate(self, metrics_data, scientific_data, trainer_model=None):
        # 1️⃣ Combined plot for last agent
        print("📊 Generating combined plot for last agent...")
        df_last = metrics_data[-1] if len(metrics_data) > 0 else None
        if df_last is not None:
            self._plot_last_agent(df_last)

        # 2️⃣ Global agent trends (keep your original)
        print("📊 Generating agent comparison plot...")
        self._plot_agent_comparison(metrics_data)

        # 3️⃣ Scientific summary CSV
        print("📊 Generating scientific summary CSV...")
        self._generate_scientific_summary(scientific_data)

        # 4️⃣ Training Loss plot
        print("📊 Generating training loss plot...")
        self._plot_training_loss()

        # 5️⃣ Faceted agent plots (new)
        print("📊 Generating faceted agent plot...")
        self._plot_agent_faceted(metrics_data)

        # 6️⃣ Mean ± Std plot (new)
        print("📊 Generating mean ± std plot...")
        self._plot_mean_std_across_agents(metrics_data)

        # 7️⃣ PDF report
        print("📊 Generating PDF report...")
        self._generate_pdf_report()

    def _plot_last_agent(self, df_last):
        plt.figure(figsize=(12, 6))
        plt.title("Last Agent Metrics Overview")
        plt.plot(df_last["step"], df_last["reward"], label="Reward")
        plt.plot(df_last["step"], df_last["confidence_level"], label="Confidence")
        plt.plot(df_last["step"], df_last["fatigue_level"], label="Fatigue")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(alpha=0.3)

        save_path = os.path.join(self.reports_folder, "meta_learner_last_agent_metrics.png")
        plt.savefig(save_path)
        print(f"✅ Report saved: {save_path}")
        plt.close()

    def _plot_agent_comparison(self, metrics_data):
        plt.figure(figsize=(14, 8))

        for i, df in enumerate(metrics_data):
            label = f"Agent {i+1}"
            plt.plot(df["step"], df["confidence_level"], label=f"{label} - Confidence", linewidth=1.5)
            plt.plot(df["step"], df["fatigue_level"], label=f"{label} - Fatigue", linestyle='--', linewidth=1.2)

        plt.title("Multi-Agent Confidence & Fatigue Evolution")
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend(loc="upper right")
        plt.grid(alpha=0.3)

        save_path = os.path.join(self.reports_folder, "meta_learner_agents_comparison.png")
        plt.savefig(save_path)
        print(f"✅ Agent comparison plot saved: {save_path}")
        plt.close()

    def _generate_scientific_summary(self, scientific_data):
        summary_list = []

        for i, df in enumerate(scientific_data):
            try:
                row = {col: df[col].values[0] for col in df.columns}
                row["Agent"] = i + 1
                summary_list.append(row)
            except Exception as e:
                print(f"⚠️ Skipping scientific data for Agent {i+1} due to error: {e}")

        df_summary = pd.DataFrame(summary_list)
        save_path = os.path.join(self.reports_folder, "meta_learner_summary.csv")
        df_summary.to_csv(save_path, index=False)
        print(f"✅ Summary CSV saved: {save_path}")

    def _plot_training_loss(self):
        training_log_path = os.path.join("meta_learner_memory", "training_log.csv")
        if os.path.exists(training_log_path):
            df_log = pd.read_csv(training_log_path)

            plt.figure(figsize=(10, 6))
            plt.plot(df_log["epoch"], df_log["loss"], marker='o', color='purple')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss vs Epoch")
            plt.grid(alpha=0.3)
            plt.tight_layout()

            save_path = os.path.join(self.reports_folder, "meta_learner_training_loss.png")
            plt.savefig(save_path)
            print(f"✅ Training loss plot saved: {save_path}")
            plt.close()
        else:
            print("⚠️ No training_log.csv found → skipping Training Loss plot.")

    def _plot_agent_faceted(self, metrics_data):
        num_agents = len(metrics_data)
        cols = 4
        rows = (num_agents + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        axes = axes.flatten()

        for i, df in enumerate(metrics_data):
            ax = axes[i]
            ax.plot(df["step"], df["confidence_level"], label="Confidence", color='blue')
            ax.plot(df["step"], df["fatigue_level"], label="Fatigue", color='orange')
            ax.set_title(f"Agent {i+1}")
            ax.legend()
            ax.grid(alpha=0.3)

        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        save_path = os.path.join(self.reports_folder, "meta_learner_agents_faceted.png")
        plt.savefig(save_path)
        print(f"✅ Faceted agent plot saved: {save_path}")
        plt.close()

    def _plot_mean_std_across_agents(self, metrics_data):
        min_steps = min([df.shape[0] for df in metrics_data])
        conf_matrix = np.stack([df["confidence_level"].values[:min_steps] for df in metrics_data])
        fatigue_matrix = np.stack([df["fatigue_level"].values[:min_steps] for df in metrics_data])
        steps = np.arange(min_steps)

        plt.figure(figsize=(12, 6))
        plt.title("Mean ± Std Confidence & Fatigue across Agents")

        # Confidence
        conf_mean = conf_matrix.mean(axis=0)
        conf_std = conf_matrix.std(axis=0)
        plt.plot(steps, conf_mean, label="Confidence Mean", color='blue')
        plt.fill_between(steps, conf_mean - conf_std, conf_mean + conf_std, color='blue', alpha=0.2)

        # Fatigue
        fatigue_mean = fatigue_matrix.mean(axis=0)
        fatigue_std = fatigue_matrix.std(axis=0)
        plt.plot(steps, fatigue_mean, label="Fatigue Mean", color='orange')
        plt.fill_between(steps, fatigue_mean - fatigue_std, fatigue_mean + fatigue_std, color='orange', alpha=0.2)

        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(alpha=0.3)

        save_path = os.path.join(self.reports_folder, "meta_learner_agents_mean_std.png")
        plt.savefig(save_path)
        print(f"✅ Mean ± Std plot saved: {save_path}")
        plt.close()

    def _generate_pdf_report(self):
        pdf_path = os.path.join(self.reports_folder, "meta_learner_report.pdf")
        pp = PdfPages(pdf_path)

        image_files = [
            "meta_learner_last_agent_metrics.png",
            "meta_learner_agents_comparison.png",
            "meta_learner_training_loss.png",
            "meta_learner_agents_faceted.png",
            "meta_learner_agents_mean_std.png"
        ]

        for img_file in image_files:
            img_path = os.path.join(self.reports_folder, img_file)
            if os.path.exists(img_path):
                try:
                    fig = plt.figure()
                    img = plt.imread(img_path)
                    plt.imshow(img)
                    plt.axis('off')
                    pp.savefig(fig)
                    plt.close()
                except Exception as e:
                    print(f"⚠️ Skipping image {img_path} in PDF due to error: {e}")
            else:
                print(f"⚠️ Image {img_path} not found, skipping in PDF.")

        pp.close()
        print(f"✅ PDF report saved: {pdf_path}")
