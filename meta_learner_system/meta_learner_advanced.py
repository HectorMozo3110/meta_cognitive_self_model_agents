# meta_learner_advanced.py

import os
from meta_learner_system.data_loader import DataLoader
from meta_learner_system.model_trainer import ModelTrainer
from meta_learner_system.memory_manager import MemoryManager
from meta_learner_system.reporter import Reporter
from meta_learner_system.agent_analyzer import AgentAnalyzer

# Main function to run the Meta-Learner pipeline
def run_meta_learner():

    # Paths
    MEMORY_FOLDER = "meta_learner_memory"
    REPORTS_FOLDER = "meta_learner_reports"
    os.makedirs(MEMORY_FOLDER, exist_ok=True)
    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    # Step 1: Load data
    print("🔍 Loading data...")
    loader = DataLoader()
    metrics_data, scientific_data = loader.load_all_metrics()

    # Step 2: Load or initialize model
    memory_manager = MemoryManager(MEMORY_FOLDER)
    model = memory_manager.load_model()

    if model is None:
        print("🧠 No existing model found. Initializing new model...")
        trainer = ModelTrainer(memory_manager)
    else:
        print("🧠 Loaded existing model from memory.")
        trainer = ModelTrainer(memory_manager, model=model)

    # Step 3: Train model
    print("🚀 Training model on loaded data...")
    trainer.train(metrics_data, scientific_data)

    # Step 4: Save model
    print("💾 Saving trained model...")
    memory_manager.save_model(trainer.get_model())

    # Step 5: Generate reports
    print("📊 Generating reports...")
    reporter = Reporter(REPORTS_FOLDER)
    reporter.generate(metrics_data, scientific_data, trainer)

    # Step 6: Analyze agents
    print("🔍 Running agent analyzer...")
    analyzer = AgentAnalyzer(REPORTS_FOLDER)
    analyzer.analyze(scientific_data)

    # Pipeline complete
    print("\n✅ Meta-Learner pipeline complete.\n")

# Entry point for terminal execution
if __name__ == "__main__":
    run_meta_learner()
