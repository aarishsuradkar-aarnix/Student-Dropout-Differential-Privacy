"""
run_experiments.py
Master execution pipeline that trains the baseline model and differentially private models
(across ε in {1, 3, 5, 10}), computes all evaluation metrics, produces comparison tables,
and saves all 7 required visualization charts.
"""

import os
import sys
import io
import json
from pathlib import Path

# Fix Windows console encoding for Unicode if needed
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import pandas as pd
import torch

# Ensure local imports work regardless of execution directory
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATASET_PATH, RESULTS_DIR, CLASS_NAMES, TARGET_EPSILONS,
    TARGET_DELTA, EPOCHS, LEARNING_RATE_BASELINE, LEARNING_RATE_DP,
    RANDOM_SEED
)
from dataset_loader import load_dataset, get_dataset_summary
from preprocessing import get_preprocessed_pipeline
from baseline_model import StudentDropoutMLP, train_baseline_model, set_seed
from dp_model import train_dp_model_with_target_epsilon
from evaluation import evaluate_model, compile_comparison_table, print_evaluation_summary
from visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_accuracy_comparison,
    plot_f1_comparison,
    plot_privacy_vs_accuracy,
    plot_training_losses
)


def run_all_experiments():
    print("=" * 70)
    print("STUDENT DROPOUT PREDICTION USING DIFFERENTIAL PRIVACY (IA-1)")
    print("=" * 70)

    # 1. Load Dataset
    print("\n[Step 1/5] Loading and verifying dataset...")
    df = load_dataset(filepath=DATASET_PATH)
    summary = get_dataset_summary(df)
    print(f"Loaded {summary['total_records']} records with {summary['total_features']} features.")
    print(f"Missing values: {summary['missing_values_total']}, Duplicates: {summary['duplicate_rows']}")

    # Plot 1: Class Distribution
    plot_class_distribution(df, save_path=RESULTS_DIR / "01_class_distribution.png")

    # 2. Preprocess Data
    print("\n[Step 2/5] Running preprocessing and stratified train/test split...")
    pipeline = get_preprocessed_pipeline(df=df)
    train_loader = pipeline["train_loader"]
    test_loader = pipeline["test_loader"]
    print(f"Training set: {len(pipeline['y_train'])} samples | Test set: {len(pipeline['y_test'])} samples")

    experiments_log = {}
    loss_curves = {}

    # 3. Train Baseline Model
    print("\n[Step 3/5] Training Baseline (Non-Private) Model...")
    set_seed(RANDOM_SEED)
    baseline_model = StudentDropoutMLP()
    baseline_model, baseline_losses, baseline_time = train_baseline_model(
        baseline_model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE_BASELINE, verbose=True
    )
    baseline_metrics = evaluate_model(baseline_model, test_loader)
    print_evaluation_summary(baseline_metrics, "Baseline MLP")

    experiments_log["Baseline (Non-Private)"] = {
        "metrics": baseline_metrics,
        "target_epsilon": "N/A (Non-private)",
        "achieved_epsilon": "Non-private (inf)",
        "achieved_delta": "0.0",
        "noise_multiplier": "0.0",
        "training_time": baseline_time
    }
    loss_curves["Baseline (Non-Private)"] = baseline_losses

    # Plot 2: Baseline Confusion Matrix
    plot_confusion_matrix(
        baseline_metrics["confusion_matrix"],
        title="Baseline (Non-Private) Confusion Matrix",
        save_path=RESULTS_DIR / "02_baseline_confusion_matrix.png"
    )

    # 4. Train Differentially Private Models across Epsilon Budgets
    print("\n[Step 4/5] Training Differentially Private Models (DP-SGD)...")
    dp_metrics_dict = {}

    for eps in TARGET_EPSILONS:
        print(f"\n--- DP Training for Target eps = {eps} ---")
        set_seed(RANDOM_SEED)
        dp_nn = StudentDropoutMLP()
        trained_dp, achieved_eps, achieved_delta, dp_losses, dp_time, noise_mult = train_dp_model_with_target_epsilon(
            model=dp_nn,
            train_loader=train_loader,
            target_epsilon=eps,
            target_delta=TARGET_DELTA,
            epochs=EPOCHS,
            lr=LEARNING_RATE_DP,
            verbose=True
        )

        metrics = evaluate_model(trained_dp, test_loader)
        print_evaluation_summary(metrics, f"DP-SGD (Target eps={eps}, Achieved eps={achieved_eps:.2f})")

        label = f"DP (eps={eps:.1f})"
        experiments_log[label] = {
            "metrics": metrics,
            "target_epsilon": eps,
            "achieved_epsilon": round(achieved_eps, 2),
            "achieved_delta": achieved_delta,
            "noise_multiplier": round(noise_mult, 4),
            "training_time": dp_time
        }
        loss_curves[label] = dp_losses
        dp_metrics_dict[eps] = (metrics, achieved_eps)

    # Plot 3: DP Confusion Matrix (for ε = 3.0 representative moderate budget)
    if 3.0 in dp_metrics_dict:
        rep_metrics, rep_eps = dp_metrics_dict[3.0]
        plot_confusion_matrix(
            rep_metrics["confusion_matrix"],
            title=f"Differentially Private (ε={rep_eps:.2f}) Confusion Matrix",
            save_path=RESULTS_DIR / "03_dp_confusion_matrix_eps3.png"
        )
    else:
        first_eps = list(dp_metrics_dict.keys())[0]
        rep_metrics, rep_eps = dp_metrics_dict[first_eps]
        plot_confusion_matrix(
            rep_metrics["confusion_matrix"],
            title=f"Differentially Private (ε={rep_eps:.2f}) Confusion Matrix",
            save_path=RESULTS_DIR / "03_dp_confusion_matrix.png"
        )

    # 5. Compile Comparison Table and Remaining Figures
    print("\n[Step 5/5] Compiling results, generating comparison plots, and saving artifacts...")
    df_comparison = compile_comparison_table(experiments_log)
    print("\n" + "=" * 80)
    print("EXPERIMENTAL RESULTS COMPARISON TABLE")
    print("=" * 80)
    print(df_comparison.to_string(index=False))

    # Save summary table to CSV and Markdown
    csv_path = RESULTS_DIR / "metrics_summary.csv"
    md_path = RESULTS_DIR / "metrics_summary.md"
    df_comparison.to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Model Performance and Privacy Trade-Off Summary\n\n")
        f.write(df_comparison.to_markdown(index=False))
    print(f"\n[Results] Saved comparison summary to: {csv_path}")

    # Plot 4: Accuracy Comparison
    plot_accuracy_comparison(df_comparison, save_path=RESULTS_DIR / "04_accuracy_comparison.png")

    # Plot 5: F1-Score Comparison
    plot_f1_comparison(df_comparison, save_path=RESULTS_DIR / "05_f1_score_comparison.png")

    # Plot 6: Privacy Budget vs Accuracy Curve
    plot_privacy_vs_accuracy(df_comparison, save_path=RESULTS_DIR / "06_privacy_vs_accuracy.png")

    # Plot 7: Training Loss Curves
    plot_training_losses(loss_curves, save_path=RESULTS_DIR / "07_training_loss_curves.png")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS AND VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return df_comparison, experiments_log


if __name__ == "__main__":
    run_all_experiments()
