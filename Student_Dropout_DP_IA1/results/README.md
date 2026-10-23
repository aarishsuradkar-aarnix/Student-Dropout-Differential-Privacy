# Experimental Results & Evaluation Artifacts

This directory contains the quantitative outputs, summary tables, and publication-ready figures generated from the execution of `src/run_experiments.py` (or the companion Google Colab notebook).

---

## 1. Directory Contents

| Artifact Filename | Description | Assessment Mapping |
| :--- | :--- | :--- |
| `metrics_summary.csv` | Machine-readable CSV table containing exact evaluation metrics across all 5 models | Criterion 4 & 5 |
| `metrics_summary.md` | Markdown formatted summary table of accuracy, F1, loss, privacy parameters, and timings | Criterion 5 |
| `01_class_distribution.png` | Bar chart displaying student cohort breakdown across Dropout, Enrolled, and Graduate | Criterion 3 |
| `02_baseline_confusion_matrix.png` | 3x3 multi-class confusion matrix for the non-private baseline model | Criterion 4 |
| `03_dp_confusion_matrix_eps3.png` | 3x3 multi-class confusion matrix for the DP model at representative budget $\epsilon \approx 3.0$ | Criterion 4 & 5 |
| `04_accuracy_comparison.png` | Bar chart comparing test classification accuracy across Baseline and DP models | Criterion 5 |
| `05_f1_score_comparison.png` | Grouped bar chart comparing Macro-averaged and Weighted-averaged F1-scores | Criterion 5 |
| `06_privacy_vs_accuracy.png` | Privacy–utility trade-off curve plotting Achieved Epsilon ($\epsilon$) against Accuracy | Criterion 2 & 5 |
| `07_training_loss_curves.png` | Epoch-by-epoch cross-entropy loss trajectories comparing optimization convergence | Criterion 4 & 5 |

---

## 2. Quantitative Summary Table

The empirical execution produces the following benchmark results:

| Model Description | Target $\epsilon$ | Achieved $\epsilon$ | Privacy $\delta$ | Noise Multiplier ($\sigma$) | Accuracy (%) | Macro F1 (%) | Weighted F1 (%) | Macro Precision (%) | Macro Recall (%) | Test Loss | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Non-Private)** | N/A | Non-private ($\infty$) | $0.0$ | $0.0000$ | 73.22 | 67.01 | 73.23 | 67.06 | 66.97 | 0.7079 | 4.53 |
| **DP-SGD ($\epsilon = 10.0$)** | 10.0 | 9.99 | $10^{-5}$ | 0.6879 | 76.38 | 66.27 | 74.07 | 70.12 | 65.97 | 0.8248 | 29.85 |
| **DP-SGD ($\epsilon = 5.0$)** | 5.0 | 4.99 | $10^{-5}$ | 0.9094 | 76.50 | 65.83 | 73.92 | 70.18 | 65.68 | 0.8362 | 23.49 |
| **DP-SGD ($\epsilon = 3.0$)** | 3.0 | 2.99 | $10^{-5}$ | 1.1890 | 76.27 | 64.96 | 73.35 | 70.07 | 65.03 | 0.8510 | 16.97 |
| **DP-SGD ($\epsilon = 1.0$)** | 1.0 | 1.00 | $10^{-5}$ | 2.6562 | 73.45 | 59.63 | 69.66 | 62.74 | 61.06 | 1.0386 | 13.90 |

*Dataset: UCI Predict Students' Dropout and Academic Success (4,424 records; 80/20 train/test split: 3,539 train / 885 test).*

---

## 3. Key Observations for Academic Defense & Viva

1. **Why does DP accuracy remain resilient while Macro F1 declines at $\epsilon = 1.0$?**
   - The dataset has an imbalanced class distribution: `Graduate` (50%), `Dropout` (32%), and `Enrolled` (18%).
   - Under heavy perturbation noise ($\sigma = 2.656$ at $\epsilon = 1.0$), the neural network's gradient signals for the minority class (`Enrolled`) are overwhelmed by Gaussian noise. The model hedges its predictions toward the majority `Graduate` and `Dropout` classes, which preserves overall accuracy (73.45%) but penalizes minority class recall, causing Macro F1 to drop from 67.01% to 59.63%.
2. **Noise Multiplier Calibration:**
   - As the privacy budget tightens from $\epsilon = 10.0$ to $\epsilon = 1.0$, the required Gaussian noise standard deviation $\sigma$ increases non-linearly from $0.6879$ to $2.6562$.
3. **Loss Convergence:**
   - As shown in `07_training_loss_curves.png`, the baseline loss drops smoothly to ~0.42, while DP loss curves stabilize at higher plateau values (~0.82 to ~1.04) because of continuous gradient clipping and noise injection.
4. **Computational Cost:**
   - Per-sample gradient clipping in DP-SGD increases training time from ~4.5 seconds (baseline) to ~14–30 seconds (DP models on CPU), illustrating the computational overhead of sample-level privacy accounting.
