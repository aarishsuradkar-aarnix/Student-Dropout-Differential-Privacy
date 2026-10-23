# Student Dropout Prediction Using Differential Privacy (IA-1)

**Course:** Security and Privacy for Machine Learning  
**Assessment:** IA-1 Individual Project (20 Marks)  
**Academic Level:** Third Year B.Tech — Artificial Intelligence & Data Science  
**Module Alignment:** Module 2 — Differential Privacy for Machine Learning (with conceptual coverage of Modules 1, 3, 4, 5)

---

## 1. Project Overview

Higher education institutions collect extensive student records to predict academic attrition and proactively support struggling students. However, raw educational databases contain sensitive indicators—such as tuition debt status, scholarship dependency, parental occupation, and semester failure counts. Exposing predictive models trained on these records leaves students vulnerable to **Membership Inference Attacks (MIA)** and **Attribute Disclosure**.

This project provides a complete, mathematically verified implementation of **Differentially Private Stochastic Gradient Descent (DP-SGD)** using PyTorch and Opacus. It benchmark-evaluates a non-private baseline against differentially private neural network classifiers across varying privacy budgets ($\epsilon \in \{1.0, 3.0, 5.0, 10.0\}$ at fixed failure probability $\delta = 10^{-5}$), documenting the resulting privacy–utility trade-off.

---

## 2. Directory and File Structure

```
Student_Dropout_DP_IA1/
│
├── notebooks/
│   └── Student_Dropout_Differential_Privacy_IA1.ipynb  # Self-contained Google Colab notebook
│
├── src/
│   ├── config.py             # Hyperparameters, random seed, paths, and constants
│   ├── dataset_loader.py     # UCI archive downloader, missing-value and duplicate audits
│   ├── preprocessing.py      # Stratified train/test split, zero-leakage StandardScaler, DataLoader
│   ├── baseline_model.py     # Non-private PyTorch MLP architecture and training loop
│   ├── dp_model.py           # Opacus DP-SGD implementation, gradient clipping, RDP accounting
│   ├── evaluation.py         # Multi-class accuracy, precision, recall, macro/weighted F1, confusion matrices
│   ├── visualization.py      # Publication-quality plotting routines (all 7 required figures)
│   └── run_experiments.py    # Master script running the entire experimental pipeline
│
├── data/
│   ├── dataset_source.txt    # UCI repository attribution, metadata, and citation details
│   └── data.csv              # Sanitized benchmark dataset (4,424 records, 36 features, 3 classes)
│
├── results/
│   ├── README.md             # Guide to generated outputs and evaluation artifacts
│   ├── metrics_summary.csv   # Consolidated quantitative evaluation table across all models
│   ├── metrics_summary.md    # Markdown formatted comparison table
│   ├── 01_class_distribution.png
│   ├── 02_baseline_confusion_matrix.png
│   ├── 03_dp_confusion_matrix_eps3.png
│   ├── 04_accuracy_comparison.png
│   ├── 05_f1_score_comparison.png
│   ├── 06_privacy_vs_accuracy.png
│   └── 07_training_loss_curves.png
│
├── docs/
│   ├── IA1_Report_Draft.md   # Comprehensive academic report draft matching marking rubric
│   ├── syllabus_mapping.md   # Rigorous alignment to all five syllabus modules
│   ├── dataset_information.md# Full provenance, ethical considerations, and schema details
│   └── evaluation_plan.md    # Metrics definition, privacy trade-off analysis, and IA-2 roadmap
│
├── requirements.txt          # Python dependencies with compatible versions
└── README.md                 # Primary project guide and instructions
```

---

## 3. Dataset Information

- **Dataset Name:** Predict Students' Dropout and Academic Success
- **Repository:** UCI Machine Learning Repository (ID: 697)
- **DOI:** [10.24432/C5MC89](https://doi.org/10.24432/C5MC89)
- **Creators:** Valentim Realinho, Jorge Machado, Baptista, Mónica V. Martins (Instituto Politécnico de Portalegre)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Cohort Dimensions:** 4,424 student records, 36 input features, 1 target outcome (`Target`: `Graduate` 49.93%, `Dropout` 32.12%, `Enrolled` 17.95%).
- **Missing Values:** 0 null values; 0 duplicates.

---

## 4. How to Run in Google Colab (Zero-Setup Instructions)

The Google Colab notebook is completely self-contained and requires **no API keys, no paid subscription, and no GPU**:

1. Open [Google Colab](https://colab.research.google.com).
2. Click **File** $\to$ **Upload notebook**, and upload:
   `notebooks/Student_Dropout_Differential_Privacy_IA1.ipynb`
3. Ensure runtime is set to **CPU** (Runtime $\to$ Change runtime type $\to$ CPU).
4. Select **Runtime** $\to$ **Run all** (or execute cells sequentially with `Shift + Enter`).
5. **Expected Execution Time:** Approximately 1.5 to 2.5 minutes on standard Colab CPU.
6. The notebook automatically downloads the dataset, trains baseline and DP models, and displays all 7 figures inline.

---

## 5. How to Run Locally

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- pip package manager

### Step-by-Step Local Setup

1. **Clone or extract the project directory:**
   ```bash
   cd Student_Dropout_DP_IA1
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute the complete experimental pipeline:**
   ```bash
   python src/run_experiments.py
   ```
   *This trains the baseline model and DP models across $\epsilon \in \{1, 3, 5, 10\}$, logs per-epoch loss and privacy budget consumption, saves the 7 visualization plots in `results/`, and prints the final comparison table.*

---

## 6. Expected Outputs

Upon running `src/run_experiments.py` or the Colab notebook:

### Quantitative Metric Summary
- **Baseline Accuracy:** ~73.2% (Macro F1: ~67.0%, Weighted F1: ~73.2%)
- **DP ($\epsilon = 10.0$):** ~76.4% (Noise $\sigma \approx 0.69$, Macro F1: ~66.3%)
- **DP ($\epsilon = 5.0$):** ~76.5% (Noise $\sigma \approx 0.91$, Macro F1: ~65.8%)
- **DP ($\epsilon = 3.0$):** ~76.3% (Noise $\sigma \approx 1.19$, Macro F1: ~65.0%)
- **DP ($\epsilon = 1.0$):** ~73.5% (Noise $\sigma \approx 2.66$, Macro F1: ~59.6%)

*(Note: In DP models, noise addition and gradient clipping provide an implicit $L_2$ regularization effect that prevents overfitting on majority classes, while strict privacy $\epsilon = 1.0$ shows the expected drop in minority class recall).*

### Generated Visual Artifacts (`results/`)
1. `01_class_distribution.png`: Bar chart of student outcomes.
2. `02_baseline_confusion_matrix.png`: Normalized baseline confusion matrix.
3. `03_dp_confusion_matrix_eps3.png`: Differentially private confusion matrix at $\epsilon \approx 3.0$.
4. `04_accuracy_comparison.png`: Test accuracy comparison bar chart.
5. `05_f1_score_comparison.png`: Macro vs Weighted F1 grouped bar chart.
6. `06_privacy_vs_accuracy.png`: Privacy–utility trade-off curve across $\epsilon$.
7. `07_training_loss_curves.png`: Multi-epoch loss convergence curves.

---

## 7. Troubleshooting & Common Issues

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `UnicodeEncodeError: 'charmap'` on Windows | Windows console CP-1252 codepage unable to print Greek character $\epsilon$ | Fixed in `run_experiments.py` and `dp_model.py` via UTF-8 reconfigure and ASCII fallbacks. |
| `UserWarning: Secure RNG turned off` | Opacus uses pseudo-RNG during CPU experimentation | Expected during experimentation; does not affect mathematical privacy bounds or execution. |
| `UserWarning: Full backward hook is firing` | PyTorch backward hook notification with Opacus | Standard PyTorch internal diagnostic for Opacus per-sample gradient capturing. Harmless. |
| `ModuleNotFoundError: No module named 'opacus'` | Opacus not installed in active environment | Run `pip install opacus` or execute the installation cell at top of notebook. |
| Slow training on CPU | Inefficient batch size | Default batch size 64 completes 25 epochs in ~15-30s per model on CPU. Do not reduce batch size below 32. |
