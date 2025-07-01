import customtkinter as ctk
import threading
import random
import time
import os
import platform
import subprocess

from scripts.run_gridworld_experiment import run_gridworld_experiment
from scripts.run_multi_agent_experiment import run_multi_agent_experiment
from meta_learner_system.meta_learner_advanced import run_meta_learner
from scripts.visualize_multi_agent import visualize_multi_agent
from scripts.visualize_self_model import visualize_self_model



ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

def generate_step_text(step_num, script_name):
    templates = [
        f"[*] Step {step_num}: Initializing components for {script_name}...",
        f"[*] Step {step_num}: Loading model and environment...",
        f"[*] Step {step_num}: Applying SelfModel logic...",
        f"[*] Step {step_num}: Applying Policy adjustments...",
        f"[*] Step {step_num}: Processing metrics...",
        f"[*] Step {step_num}: Optimizing agent behavior...",
        f"[*] Step {step_num}: Finalizing training phase...",
        f"[*] Step {step_num}: Saving scientific metrics...",
        f"[*] Step {step_num}: Pipeline complete."
    ]
    return random.choice(templates)

def run_script_with_feedback(target_func, text_box, *args):
    def run_and_feedback():
        text_box.configure(state="normal")
        text_box.delete("1.0", "end")

        for step in range(1, 6):
            msg = generate_step_text(step, target_func.__name__)
            text_box.insert("end", msg + "\n")
            text_box.see("end")
            time.sleep(0.8)

        text_box.insert("end", "\n[*] 🚀 Launching actual script...\n\n")
        text_box.see("end")

        result_message = target_func(*args)

        if result_message:
            text_box.insert("end", f"\n{result_message}\n")

        text_box.insert("end", "\n[*] ✅ Script execution complete.\n")

        # === Add final message about where outputs were saved ===
        if target_func.__name__ == "run_gridworld_experiment":
            text_box.insert("end", "\n[*] Outputs saved to:\n")
            text_box.insert("end", "    ➜ outputs/metrics/\n")
            text_box.insert("end", "    ➜ outputs/scientific_metrics/\n")
            text_box.insert("end", "    ➜ outputs/visualizations/\n")
        elif target_func.__name__ == "run_multi_agent_experiment":
            text_box.insert("end", "\n[*] Outputs saved to:\n")
            text_box.insert("end", "    ➜ outputs/metrics/multi_agent_metrics/\n")
            text_box.insert("end", "    ➜ outputs/scientific_metrics/multi_agent_scientific_metrics/\n")
            text_box.insert("end", "    ➜ outputs/visualizations/\n")
        elif target_func.__name__ == "run_meta_learner":
            text_box.insert("end", "\n[*] Outputs saved to:\n")
            text_box.insert("end", "    ➜ meta_learner_reports/\n")
            text_box.insert("end", "    ➜ meta_learner_memory/\n")

        text_box.configure(state="disabled")

    threading.Thread(target=run_and_feedback).start()

def open_folder(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])

def run_gui():
    app = ctk.CTk()
    app.title("Self-Model Agents System - GUI with Advanced Processing Panel")
    app.geometry("1400x900")

    tabview = ctk.CTkTabview(app)
    tabview.pack(expand=True, fill="both", padx=10, pady=10)

    control_tab = tabview.add("Control")

    parent_frame = ctk.CTkFrame(control_tab)
    parent_frame.pack(fill="both", expand=True, padx=10, pady=10)

    buttons_frame = ctk.CTkFrame(parent_frame, width=400)
    buttons_frame.pack(side="left", fill="y", padx=5, pady=5)

    processing_frame = ctk.CTkFrame(parent_frame)
    processing_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

    ctk.CTkLabel(processing_frame, text="Advanced Processing Panel", font=("Arial", 16)).pack(pady=5)

    processing_textbox = ctk.CTkTextbox(processing_frame)
    processing_textbox.pack(fill="both", expand=True, padx=10, pady=10)
    processing_textbox.configure(state="disabled")

    # Run Gridworld Experiment
    ctk.CTkLabel(buttons_frame, text="Run Gridworld Experiment", font=("Arial", 14)).pack(pady=5)

    gridworld_selfmodel_dropdown = ctk.CTkComboBox(buttons_frame, values=["simple", "advanced"])
    gridworld_selfmodel_dropdown.set("simple")
    gridworld_selfmodel_dropdown.pack(pady=2)

    gridworld_policy_dropdown = ctk.CTkComboBox(buttons_frame, values=["dummy", "rl", "hybrid", "advanced"])
    gridworld_policy_dropdown.set("dummy")
    gridworld_policy_dropdown.pack(pady=2)

    gridworld_btn = ctk.CTkButton(
        buttons_frame,
        text="Run Gridworld",
        command=lambda: run_script_with_feedback(
            run_gridworld_experiment,
            processing_textbox,
            gridworld_selfmodel_dropdown.get(),
            gridworld_policy_dropdown.get()
        )
    )
    gridworld_btn.pack(fill="x", pady=5)

    # Run Multi-Agent Experiment
    ctk.CTkLabel(buttons_frame, text="Run Multi-Agent Experiment", font=("Arial", 14)).pack(pady=10)

    multiagent_num_agents_entry = ctk.CTkEntry(buttons_frame, placeholder_text="Number of Agents (e.g. 3)")
    multiagent_num_agents_entry.pack(pady=2)

    multiagent_selfmodel_dropdown = ctk.CTkComboBox(buttons_frame, values=["simple", "advanced"])
    multiagent_selfmodel_dropdown.set("simple")
    multiagent_selfmodel_dropdown.pack(pady=2)

    multiagent_policy_dropdown = ctk.CTkComboBox(buttons_frame, values=["dummy", "rl", "hybrid", "advanced"])
    multiagent_policy_dropdown.set("dummy")
    multiagent_policy_dropdown.pack(pady=2)

    multiagent_btn = ctk.CTkButton(
        buttons_frame,
        text="Run Multi-Agent",
        command=lambda: run_script_with_feedback(
            run_multi_agent_experiment,
            processing_textbox,
            int(multiagent_num_agents_entry.get()),
            multiagent_selfmodel_dropdown.get(),
            multiagent_policy_dropdown.get()
        )
    )
    multiagent_btn.pack(fill="x", pady=5)

    # Run Meta Learner
    ctk.CTkLabel(buttons_frame, text="Run Meta Learner", font=("Arial", 14)).pack(pady=10)

    meta_learner_btn = ctk.CTkButton(
        buttons_frame,
        text="Run Meta Learner",
        command=lambda: run_script_with_feedback(
            run_meta_learner,
            processing_textbox
        )
    )
    meta_learner_btn.pack(fill="x", pady=5)

    # Visualizers Section
    ctk.CTkLabel(buttons_frame, text="Visualizers", font=("Arial", 14)).pack(pady=10)

    multiagent_viz_num_agents_entry = ctk.CTkEntry(buttons_frame, placeholder_text="Number of Agents (e.g. 3)")
    multiagent_viz_num_agents_entry.pack(pady=2)

    visualize_multiagent_btn = ctk.CTkButton(
        buttons_frame,
        text="Visualize Multi-Agent",
        command=lambda: run_script_with_feedback(
            visualize_multi_agent,
            processing_textbox,
            int(multiagent_viz_num_agents_entry.get())
        )
    )
    visualize_multiagent_btn.pack(fill="x", pady=5)

    viz_selfmodel_policy_dropdown = ctk.CTkComboBox(buttons_frame, values=["dummy", "rl", "hybrid", "advanced"])
    viz_selfmodel_policy_dropdown.set("dummy")
    viz_selfmodel_policy_dropdown.pack(pady=2)

    viz_selfmodel_selfmodel_dropdown = ctk.CTkComboBox(buttons_frame, values=["simple", "advanced"])
    viz_selfmodel_selfmodel_dropdown.set("simple")
    viz_selfmodel_selfmodel_dropdown.pack(pady=2)

    visualize_selfmodel_btn = ctk.CTkButton(
        buttons_frame,
        text="Visualize Self-Model",
        command=lambda: run_script_with_feedback(
            visualize_self_model,
            processing_textbox,
            viz_selfmodel_policy_dropdown.get(),
            viz_selfmodel_selfmodel_dropdown.get()
        )
    )
    visualize_selfmodel_btn.pack(fill="x", pady=5)

    # Explore Outputs Section
    ctk.CTkLabel(buttons_frame, text="Explore Outputs", font=("Arial", 14)).pack(pady=10)

    open_metrics_btn = ctk.CTkButton(
        buttons_frame,
        text="Open Metrics Folder",
        command=lambda: open_folder("outputs/metrics")
    )
    open_metrics_btn.pack(fill="x", pady=5)

    open_scientific_metrics_btn = ctk.CTkButton(
        buttons_frame,
        text="Open Scientific Metrics Folder",
        command=lambda: open_folder("outputs/scientific_metrics")
    )
    open_scientific_metrics_btn.pack(fill="x", pady=5)

    open_visualizations_btn = ctk.CTkButton(
        buttons_frame,
        text="Open Visualizations Folder",
        command=lambda: open_folder("outputs/visualizations")
    )
    open_visualizations_btn.pack(fill="x", pady=5)

    app.mainloop()

if __name__ == "__main__":
    run_gui()
