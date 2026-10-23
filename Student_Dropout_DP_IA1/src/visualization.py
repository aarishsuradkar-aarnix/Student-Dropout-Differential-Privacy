"""
visualization.py
Visualization module for the Student Dropout Prediction Differential Privacy project.
Generates publication-quality figures:
  1. Dataset class distribution graph
  2. Baseline model confusion matrix
  3. Differentially private model confusion matrix (at ε = 3.0 or specified budget)
  4. Baseline vs DP model accuracy comparison bar chart
  5. Baseline vs DP model F1-score comparison bar chart
  6. Privacy budget (ε) vs Model accuracy trade-off curve
  7. Training loss convergence curves across all models
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from .config import CLASS_NAMES, RESULTS_DIR
except ImportError:
    from config import CLASS_NAMES, RESULTS_DIR


# Use clean, publication-ready style defaults
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def plot_class_distribution(y_or_df, class_names=CLASS_NAMES, save_path=None):
    """
    Plots the distribution of student academic outcomes (Dropout, Enrolled, Graduate).
    """
    if isinstance(y_or_df, pd.DataFrame):
        target_counts = y_or_df["Target"].value_counts()
        labels = target_counts.index.tolist()
        counts = target_counts.values.tolist()
    elif isinstance(y_or_df, (list, np.ndarray, pd.Series)):
        unique, counts_arr = np.unique(y_or_df, return_counts=True)
        if isinstance(unique[0], (int, np.integer)):
            labels = [class_names[i] for i in unique]
        else:
            labels = [str(x) for x in unique]
        counts = counts_arr.tolist()
    else:
        labels = class_names
        counts = list(y_or_df.values())

    total = sum(counts)
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=1.1, width=0.55)

    for bar, count in zip(bars, counts):
        pct = (count / total) * 100
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(counts) * 0.015,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontweight="bold", fontsize=10
        )

    ax.set_title("UCI Dataset: Student Outcome Class Distribution", pad=15, fontweight="bold")
    ax.set_xlabel("Academic Outcome Category", labelpad=10)
    ax.set_ylabel("Number of Students", labelpad=10)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved class distribution plot to: {save_path}")

    return fig, ax


def plot_confusion_matrix(cm, class_names=CLASS_NAMES, title="Confusion Matrix", save_path=None):
    """
    Visualizes a multi-class confusion matrix with raw counts and row-wise recall percentages.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        row_sum = np.sum(cm[i, :])
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / row_sum * 100) if row_sum > 0 else 0
            text_color = "white" if val > thresh else "black"
            ax.text(
                j, i, f"{val}\n({pct:.1f}%)",
                ha="center", va="center",
                color=text_color, fontweight="bold", fontsize=9
            )

    ax.set_ylim(cm.shape[0] - 0.5, -0.5)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved confusion matrix to: {save_path}")

    return fig, ax


def plot_accuracy_comparison(comparison_df, save_path=None):
    """
    Bar chart comparing Baseline vs Differentially Private models.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    models = comparison_df["Model Name"].tolist()
    accuracies = comparison_df["Accuracy (%)"].tolist()

    # Distinguish baseline from DP models with colors
    colors = ["#2b5c8f"] + ["#e67e22" if "ε=1.0" in m else "#3498db" for m in models[1:]]

    bars = ax.bar(models, accuracies, color=colors, edgecolor="black", linewidth=1.1, width=0.55)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.0,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontweight="bold", fontsize=10
        )

    ax.set_title("Test Accuracy: Baseline vs Differentially Private Models", pad=15, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)", labelpad=10)
    ax.set_ylim(0, 100)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved accuracy comparison plot to: {save_path}")

    return fig, ax


def plot_f1_comparison(comparison_df, save_path=None):
    """
    Grouped bar chart comparing Macro and Weighted F1-scores across all evaluated models.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    models = comparison_df["Model Name"].tolist()
    macro_f1 = comparison_df["F1 Macro (%)"].tolist()
    weighted_f1 = comparison_df["F1 Weighted (%)"].tolist()

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width/2, macro_f1, width, label="Macro F1-Score", color="#2980b9", edgecolor="black")
    rects2 = ax.bar(x + width/2, weighted_f1, width, label="Weighted F1-Score", color="#27ae60", edgecolor="black")

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 1.0, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_title("F1-Score Comparison: Baseline vs Differentially Private Models", pad=15, fontweight="bold")
    ax.set_ylabel("F1-Score (%)", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved F1-score comparison plot to: {save_path}")

    return fig, ax


def plot_privacy_vs_accuracy(comparison_df, save_path=None):
    """
    Plots the empirical Privacy Budget (ε) vs Test Accuracy trade-off curve.
    """
    # Filter for DP models with numerical epsilons
    dp_df = comparison_df[comparison_df["Target Epsilon"] != "N/A (Non-private)"].copy()
    if dp_df.empty:
        return None, None

    dp_eps = [float(x) for x in dp_df["Target Epsilon"].values]
    dp_acc = dp_df["Accuracy (%)"].astype(float).values

    # Baseline reference
    base_df = comparison_df[comparison_df["Target Epsilon"] == "N/A (Non-private)"]
    baseline_acc = float(base_df["Accuracy (%)"].values[0]) if not base_df.empty else None

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(dp_eps, dp_acc, marker="o", color="#d35400", linewidth=2.5, markersize=8, label="Differentially Private Models (DP-SGD)")

    for x, y in zip(dp_eps, dp_acc):
        ax.annotate(f"{y:.2f}% (ε={x})", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold", fontsize=9.5)

    if baseline_acc is not None:
        ax.axhline(baseline_acc, color="#2c3e50", linestyle="--", linewidth=1.8, label=f"Baseline Non-Private ({baseline_acc:.2f}%)")

    ax.set_title("Privacy–Utility Trade-Off: Privacy Budget (ε) vs Accuracy", pad=15, fontweight="bold")
    ax.set_xlabel("Privacy Budget Epsilon (ε) [Log-Scale / Strictness decreases to the right]", labelpad=10)
    ax.set_ylabel("Test Accuracy (%)", labelpad=10)
    ax.set_ylim(min(dp_acc) - 8, (baseline_acc or max(dp_acc)) + 8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved privacy-utility curve to: {save_path}")

    return fig, ax


def plot_training_losses(loss_dict, save_path=None):
    """
    Plots training loss per epoch across Baseline and DP models.
    """
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = ["#2c3e50", "#e74c3c", "#f39c12", "#3498db", "#27ae60"]

    for i, (name, losses) in enumerate(loss_dict.items()):
        color = colors[i % len(colors)]
        epochs_arr = np.arange(1, len(losses) + 1)
        ax.plot(epochs_arr, losses, label=name, color=color, linewidth=2)

    ax.set_title("Training Loss Convergence Across Epochs", pad=15, fontweight="bold")
    ax.set_xlabel("Epoch", labelpad=10)
    ax.set_ylabel("Cross-Entropy Loss", labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"[Visualization] Saved training loss plot to: {save_path}")

    return fig, ax
