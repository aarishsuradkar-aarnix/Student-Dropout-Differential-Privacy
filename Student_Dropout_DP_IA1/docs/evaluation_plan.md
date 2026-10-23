# Evaluation Plan & IA-2 Roadmap

**Project Title:** Student Dropout Prediction Using Differential Privacy  
**Course:** Security and Privacy for Machine Learning  
**Assessment Target:** IA-1 Preliminary Evaluation & Transition to IA-2  

---

## 1. Overview of Evaluation Framework

The evaluation methodology measures the trade-off between **privacy preservation** and **predictive utility** for student academic outcome classification (Dropout vs Enrolled vs Graduate).

To evaluate differential privacy mechanisms fairly and rigorously, every model variant shares:
- Identical underlying neural network architecture (`StudentDropoutMLP`: 36 -> 64 -> 32 -> 3)
- Identical stratified train/test split (80% train, 20% test; random seed 42)
- Identical pre-processing and feature standardization parameters ($\mu_{train}, \sigma_{train}$)
- Consistent batch size ($B = 64$) and training epoch budgets ($E = 25$)

---

## 2. Evaluation Metrics Definition

### Primary Utility Metrics

1. **Overall Classification Accuracy:**
   $$\text{Accuracy} = \frac{1}{N_{test}} \sum_{i=1}^{N_{test}} \mathbb{I}(\hat{y}_i = y_i)$$
   Measures global proportion of correct student outcome predictions.

2. **Macro-Averaged F1-Score:**
   $$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^K \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$
   Critical metric for academic retention because classes are imbalanced (Graduate: ~50%, Dropout: ~32%, Enrolled: ~18%). Macro F1 treats each class with equal importance, preventing the model from merely predicting the majority class.

3. **Weighted-Averaged F1-Score:**
   $$\text{Weighted F1} = \sum_{k=1}^K \left(\frac{N_k}{N_{total}}\right) \text{F1}_k$$
   Weights per-class performance by class prevalence.

4. **Multi-Class Confusion Matrix:**
   Visualizes misclassification flows, specifically examining whether DP noise disproportionately increases misclassifications between Dropout and Enrolled students.

5. **Cross-Entropy Loss:**
   Evaluates calibrated probability confidence on held-out test data.

### Privacy Accounting Metrics

1. **Achieved Epsilon ($\epsilon$):**
   Computed using Rényi Differential Privacy (RDP) composition via Opacus `privacy_engine.get_epsilon(delta=1e-5)`.
   Represents the upper bound on worst-case log-likelihood ratio of model outputs between neighboring datasets.

2. **Privacy Parameter Delta ($\delta$):**
   Fixed at $\delta = 1 \times 10^{-5}$.
   Given $N_{train} = 3539$, the theoretical condition $\delta \ll 1/N$ ($10^{-5} \ll 2.82 \times 10^{-4}$) is satisfied.

3. **Noise Multiplier ($\sigma$):**
   The standard deviation ratio of Gaussian perturbation added to clipped gradients:
   $$\sigma = \frac{\sigma_{noise}}{C}$$
   Where $C = 1.0$ is the $L_2$ gradient clipping threshold.

4. **Computational Overhead / Training Time:**
   Measured in wall-clock seconds to assess the computational cost of per-sample gradient clipping in DP-SGD.

---

## 3. Experimental Privacy Budget Spectrum

| Setting | Privacy Regime | Target $\epsilon$ | Target $\delta$ | Expected Empirical Behavior |
| :--- | :--- | :---: | :---: | :--- |
| **Baseline** | Non-Private | $\infty$ | $0.0$ | Upper utility bound; no noise added. |
| **DP-1** | Strict Privacy | $1.0$ | $10^{-5}$ | Strongest theoretical guarantee; higher noise $\sigma$; noticeable utility penalty. |
| **DP-2** | Moderate Privacy | $3.0$ | $10^{-5}$ | Balanced regime frequently targeted in academic and commercial literature. |
| **DP-3** | Permissive Privacy | $5.0$ | $10^{-5}$ | Modest noise; approaches near-baseline utility. |
| **DP-4** | Relaxed Privacy | $10.0$ | $10^{-5}$ | Minimal noise perturbation; minimal drop in accuracy. |

---

## 4. Required Visual Artifacts Generated

1. `01_class_distribution.png`: Bar chart of dataset class frequencies and percentages.
2. `02_baseline_confusion_matrix.png`: Normalized confusion matrix for non-private baseline.
3. `03_dp_confusion_matrix_eps3.png`: Normalized confusion matrix for DP model ($\epsilon = 3.0$).
4. `04_accuracy_comparison.png`: Bar chart of test accuracy across all models.
5. `05_f1_score_comparison.png`: Grouped bar chart comparing Macro vs Weighted F1.
6. `06_privacy_vs_accuracy.png`: Privacy–utility curve depicting accuracy as a function of $\epsilon$.
7. `07_training_loss_curves.png`: Loss trajectories across training epochs showing convergence dynamics.

---

## 5. Transition Plan & Scope Expansion for IA-2

While IA-1 focuses on establishing the baseline and verifying DP-SGD implementation under Opacus, **IA-2 will expand the study into empirical attack validation and algorithmic refinements:**

### Phase 1: Empirical Privacy Leakage Quantification (Membership Inference Attacks)
- Implement a black-box Membership Inference Attack (MIA) using shadow models (Shokri et al. / Salem et al.).
- Compare attack accuracy and Area Under the ROC Curve (AUC-ROC) between the Non-Private Baseline and the Differentially Private models.
- Demonstrate that as $\epsilon \to 1.0$, MIA attack accuracy drops to ~50% (random guessing), empirically proving the privacy guarantee.

### Phase 2: Local Differential Privacy (LDP) Data Ingestion
- Implement a simulated student data ingestion pipeline using Warner's Randomized Response and the Piecewise Mechanism for sensitive continuous features.
- Compare an LDP-perturbed Naive Bayes classifier against Central DP-SGD.

### Phase 3: Fairness and Disparate Impact Under DP Noise
- Investigate whether the addition of DP noise disproportionately degrades accuracy for under-represented student subgroups (e.g. students with educational special needs or international students).

### Phase 4: Privacy-Utility Optimization
- Experiment with cosine learning rate schedules and gradient clipping adaptation to optimize utility under low privacy budgets ($\epsilon \le 2.0$).
