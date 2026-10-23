"""
evaluation.py
Comprehensive evaluation module for Student Dropout Prediction models.
Calculates:
  - Accuracy
  - Macro and Weighted Precision
  - Macro and Weighted Recall
  - Macro and Weighted F1-Score
  - Multi-class Confusion Matrix
  - Per-class classification reports
  - Actual achieved (ε, δ) privacy parameters
  - Training execution timings
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

try:
    from .config import CLASS_NAMES
except ImportError:
    from config import CLASS_NAMES


def evaluate_model(model, data_loader, class_names=CLASS_NAMES, device="cpu"):
    """
    Evaluates a trained PyTorch model (baseline or differentially private) on a DataLoader.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    data_loader : DataLoader
        PyTorch DataLoader (e.g. test_loader).
    class_names : list of str, optional
        Human-readable class labels ['Dropout', 'Enrolled', 'Graduate'].
    device : str, optional
        Device to run inference on ('cpu' or 'cuda').

    Returns
    -------
    dict
        Dictionary containing all evaluation metrics, confusion matrix,
        and predictions.
    """
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            total_samples += X_batch.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    avg_loss = total_loss / max(1, total_samples)

    # Core Academic Metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )

    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_mac),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_wt),
        "confusion_matrix": cm,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs
    }
    return metrics


def compile_comparison_table(experiments_dict):
    """
    Compiles a comprehensive summary DataFrame from a dictionary of model evaluation outputs.

    Parameters
    ----------
    experiments_dict : dict
        Keys represent experiment labels (e.g. 'Baseline', 'DP (ε=1.0)', etc.)
        Values are dictionaries containing 'metrics', 'achieved_epsilon', 'achieved_delta', 'training_time'.

    Returns
    -------
    pd.DataFrame
        Formatted comparison table.
    """
    records = []
    for model_name, data in experiments_dict.items():
        m = data["metrics"]
        record = {
            "Model Name": model_name,
            "Target Epsilon": data.get("target_epsilon", "N/A (Non-private)"),
            "Achieved Epsilon": data.get("achieved_epsilon", "Non-private (inf)"),
            "Delta": data.get("achieved_delta", "0.0"),
            "Noise Multiplier (sigma)": data.get("noise_multiplier", "0.0"),
            "Accuracy (%)": round(m["accuracy"] * 100, 2),
            "F1 Macro (%)": round(m["f1_macro"] * 100, 2),
            "F1 Weighted (%)": round(m["f1_weighted"] * 100, 2),
            "Precision Macro (%)": round(m["precision_macro"] * 100, 2),
            "Recall Macro (%)": round(m["recall_macro"] * 100, 2),
            "Test Loss": round(m["loss"], 4),
            "Training Time (s)": round(data.get("training_time", 0.0), 2)
        }
        records.append(record)

    df_comparison = pd.DataFrame(records)
    return df_comparison


def print_evaluation_summary(metrics, model_name="Model"):
    """Prints a formatted evaluation report for quick inspection in console / notebook."""
    print("\n" + "="*50)
    print(f"Evaluation Summary: {model_name}")
    print("="*50)
    print(f"Accuracy:            {metrics['accuracy']*100:.2f}%")
    print(f"Macro Precision:     {metrics['precision_macro']*100:.2f}%")
    print(f"Macro Recall:        {metrics['recall_macro']*100:.2f}%")
    print(f"Macro F1-Score:      {metrics['f1_macro']*100:.2f}%")
    print(f"Weighted F1-Score:   {metrics['f1_weighted']*100:.2f}%")
    print(f"Cross-Entropy Loss:  {metrics['loss']:.4f}")
    print("\nDetailed Classification Report:")
    print(metrics["classification_report_text"])
