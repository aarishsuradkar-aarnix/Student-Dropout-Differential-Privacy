# Syllabus Mapping and Theoretical Foundations

**Course:** Security and Privacy for Machine Learning  
**Project Title:** Student Dropout Prediction Using Differential Privacy  
**Target Assessment:** IA-1 (Individual Academic Assessment, 20 Marks)  
**Academic Level:** Third Year B.Tech Artificial Intelligence & Data Science  

---

## 1. Executive Summary & Taxonomy of Methods

This academic project systematically aligns with the five modules of the *Security and Privacy for Machine Learning* curriculum. The primary practical implementation is concentrated in **Module 2 (Differential Privacy for Machine Learning)**, while Modules 1, 3, 4, and 5 provide the broader conceptual framework, threat modeling, alternative privacy architectures, and comparative context.

### Methodological Classification
To ensure academic integrity, all concepts and techniques in this project are strictly categorized into four operational states:

1. **Implemented:** Complete, executable source code running locally and in Google Colab (PyTorch + Opacus).
2. **Demonstrated:** Validated with empirical benchmarks, privacy accounting curves, confusion matrices, and loss trajectories.
3. **Compared:** Statistically evaluated against a non-private baseline across multiple privacy budgets ($\epsilon \in \{1.0, 3.0, 5.0, 10.0\}$).
4. **Analytically Discussed:** Rigorously explained from first principles, mathematical definitions, relevance to educational data mining, and technical limitations.

---

## 2. Comprehensive Module-by-Module Mapping

```
                               SYLLABUS ARCHITECTURE
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │ Module 1: Privacy Considerations in ML                                          │
 │ (Threat Model, Membership Inference Attacks, Attribute Leakage, PPML vs PPDM)  │
 └──────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ Sets Threat Landscape
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │ Module 2: Differential Privacy for ML  [PRIMARY PRACTICAL IMPLEMENTATION]      │
 │ (DP-SGD, Per-Sample Gradient Clipping, Gaussian Mechanism, RDP Accounting)      │
 └──────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ Contrasted Against
         ┌──────────────────────────────┴──────────────────────────────┐
         ▼                                                             ▼
 ┌─────────────────────────────────────────┐  ┌────────────────────────────────────────┐
 │ Module 3: Local Differential Privacy    │  │ Module 4: Privacy-Preserving Data      │
 │ (Warner Randomized Response, RAPPOR,    │  │ Mining (PPDP, k-Anonymity limitations, │
 │ Piecewise, LDP Naive Bayes)             │  │ Input/Output Perturbation)             │
 └─────────────────────────────────────────┘  └────────────────────────────────────────┘
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │ Module 5: Compressive Privacy for ML                                            │
 │ (Information Bottleneck, Privacy-Constrained Projections, Utility Trade-off)    │
 └─────────────────────────────────────────────────────────────────────────────────┘
```

---

### Module 1: Privacy Considerations in Machine Learning

#### Syllabus Topics Covered
- Privacy complications in the modern AI era
- Threats and attacks against machine learning systems:
  - *Membership Inference Attacks (MIA)*
  - *Model Inversion / Reconstruction Attacks*
  - *Attribute Inference Attacks*
  - *Model Stealing / Extraction*
- Privacy-Preserving Machine Learning (PPML)
- Privacy-Preserving Data Mining (PPDM)

#### Technical Conceptual Explanation
In modern machine learning, deep neural networks possess massive parameter capacity and tend to memorize training instances—especially vulnerable outlier records. This phenomenon renders deployed models susceptible to adversarial attacks:
1. **Membership Inference Attacks (Shokri et al., 2017):** An attacker with black-box API access evaluates the model's output prediction entropy and confidence distribution to determine whether a specific individual's record was part of the training corpus.
2. **Model Inversion Attacks (Fredrikson et al., 2015):** Gradient descent optimization is performed over input space to reconstruct canonical feature representations of training classes from model weights or confidence scores.
3. **Attribute Inference Attacks:** An adversary exploits learned correlation patterns to infer sensitive unobserved attributes (e.g. household income or debt status) given partial non-sensitive attributes.

#### Relevance to Student Dropout Prediction
Educational records are legally protected under student privacy statutes (such as FERPA in the US and GDPR in Europe). A model trained on academic data contains sensitive indicators:
- Parental qualifications and occupation (`Mother's occupation`, `Father's qualification`)
- Financial distress indicators (`Debtor`, `Tuition fees up to date`, `Scholarship holder`)
- Academic underachievement (`Curricular units approved`, `Curricular units without evaluations`)

If an academic institution deploys a dropout prediction model without privacy guarantees, a predatory lender, employer, or malicious actor could perform a membership inference query. Discovering that a student was part of the "at-risk dropout" training set leaks their private academic hardship and financial status.

#### Operational Status
- **Analytically Discussed & Formalized:** Threat model defined, motivation established, risk vectors identified.

---

### Module 2: Differential Privacy for Machine Learning [MAIN IMPLEMENTATION]

#### Syllabus Topics Covered
- Mathematical definitions and axiomatic properties of Differential Privacy ($\epsilon$-DP, $(\epsilon, \delta)$-DP)
- Differential privacy properties: *Composability* (Sequential and Advanced), *Post-Processing Invariance*, *Subsampling Amplification*
- Applying Differential Privacy in Machine Learning: Differentially Private Stochastic Gradient Descent (DP-SGD)
- Differentially private supervised learning: Classification with PyTorch and Opacus
- Differentially private unsupervised learning: Conceptual formulation (DP-PCA, DP-KMeans)

#### Mathematical Definition
A randomized algorithm $\mathcal{M}$ guarantees $(\epsilon, \delta)$-Differential Privacy if for all neighboring datasets $D, D'$ differing by at most one individual record ($||D - D'||_1 = 1$), and for all measurable subsets of outputs $\mathcal{S} \subseteq \text{Range}(\mathcal{M})$:

$$\mathbb{P}[\mathcal{M}(D) \in \mathcal{S}] \le e^\epsilon \cdot \mathbb{P}[\mathcal{M}(D') \in \mathcal{S}] + \delta$$

Where:
- $\epsilon > 0$ denotes the **privacy budget**. Smaller $\epsilon$ enforces stronger indistinguishability.
- $\delta \ge 0$ represents the **failure probability**. In practice, $\delta \ll 1/|D|$ (here $\delta = 10^{-5} \ll 1/3539 \approx 2.8 \times 10^{-4}$).

#### Core DP-SGD Implementation Steps (Abadi et al., 2016)
In standard SGD, the mini-batch gradient is:
$$g_B = \frac{1}{|B|} \sum_{i \in B} \nabla_\theta \mathcal{L}(f(x_i; \theta), y_i)$$

In DP-SGD, privacy is enforced via three mandatory steps:
1. **Per-Sample Gradient Computation:**
   Individual gradients $g_i(\theta) = \nabla_\theta \mathcal{L}(f(x_i; \theta), y_i)$ are evaluated separately for each sample $i \in B$.
2. **Per-Sample Gradient Clipping (Norm Bounding):**
   To bound the global sensitivity to $\Delta_2 = C$, each gradient is scaled:
   $$\bar{g}_i(\theta) = \frac{g_i(\theta)}{\max\left(1, \frac{\|g_i(\theta)\|_2}{C}\right)}$$
   This guarantees $\|\bar{g}_i(\theta)\|_2 \le C$.
3. **Calibrated Gaussian Noise Addition & Perturbed Step:**
   $$\tilde{g}_B(\theta) = \frac{1}{|B|} \left( \sum_{i \in B} \bar{g}_i(\theta) + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right) \right)$$
   $$\theta_{t+1} = \theta_t - \eta \tilde{g}_B(\theta)$$
4. **Rényi Differential Privacy (RDP) Accounting:**
   Accumulates moment generating functions across Poisson-subsampled mini-batches to compute tight privacy bounds $(\epsilon, \delta)$ via Opacus.

#### Operational Status
- **Implemented:** Full PyTorch neural network (`StudentDropoutMLP`) trained with `opacus.PrivacyEngine`.
- **Demonstrated:** Gradient clipping with $C = 1.0$, Gaussian noise addition $\sigma$, and RDP accounting at $\delta = 10^{-5}$.
- **Compared:** Rigorous benchmark against non-private baseline across $\epsilon \in \{1.0, 3.0, 5.0, 10.0\}$.

---

### Module 3: Local Differential Privacy for Machine Learning

#### Syllabus Topics Covered
- Local Differential Privacy (LDP) vs Central (Global) Differential Privacy
- Classic LDP mechanisms: Warner's Randomized Response (1965), RAPPOR (Erlingsson et al., Google)
- Advanced LDP mechanisms: Duchi's mechanism, Piecewise Mechanism, Square Wave Mechanism
- LDP with Naive Bayes classification

#### Technical Conceptual Explanation
In Central DP (Module 2), raw data is gathered by a trusted curator (the central university server), which then perturbs the aggregate gradients during model training.

In **Local Differential Privacy (LDP)**, the data collector is **untrusted**. Each individual user perturbs their own data record on their local device before transmitting it over the network:
$$\mathbb{P}[\mathcal{R}(v) = y] \le e^\epsilon \cdot \mathbb{P}[\mathcal{R}(v') = y] \quad \forall v, v' \in \mathcal{V}, y \in \mathcal{Y}$$

#### Classic LDP Mechanism: Randomized Response (Warner, 1965)
For a sensitive binary survey question (e.g. *"Are you currently facing severe tuition fee debts?"* $X \in \{0, 1\}$):
1. The student flips an unbiased coin.
2. If **Heads**, the student answers truthfully ($Y = X$).
3. If **Tails**, the student flips a second coin and answers **Yes** ($Y = 1$) if Heads, **No** ($Y = 0$) if Tails.

The perturbation probability is:
$$p = \mathbb{P}[Y = 1 \mid X = 1] = \frac{3}{4}, \quad q = \mathbb{P}[Y = 1 \mid X = 0] = \frac{1}{4}$$
The privacy budget is:
$$\epsilon = \ln\left(\frac{p}{q}\right) = \ln(3) \approx 1.098$$
The central collector estimates the true proportion $\pi$ without ever learning any student's true financial status:
$$\hat{\pi} = 2 \cdot \left(\frac{N_{yes}}{N}\right) - 0.5$$

#### Advanced LDP for Continuous Attributes: The Piecewise Mechanism
For continuous features (e.g. `Curricular units 1st sem (grade)` normalized to $[-1, 1]$), the Piecewise Mechanism (Wang et al., 2018) maps input $x \in [-1, 1]$ to a probability density function defined over $[-C, C]$:
$$f(y \mid x) = \begin{cases} \frac{e^\epsilon}{2(e^\epsilon + 1)}, & y \in [\ell(x), r(x)] \\ \frac{1}{2(e^\epsilon + 1)}, & \text{otherwise} \end{cases}$$
Where $[\ell(x), r(x)]$ is a bounded interval centered around $x$.

#### Application to Student Dropout Prediction
LDP is ideal for decentralized student data collection: students can submit stigmatizing data (mental health indicators, family debt, special needs) directly through a student portal mobile app using Randomized Response. The university trains an LDP-Naive Bayes classifier:
$$P(Y = c \mid X) \propto P(Y = c) \prod_{j=1}^d \hat{P}_{LDP}(X_j \mid Y = c)$$
Where feature conditional distributions are estimated from perturbed local reports.

#### Operational Status
- **Analytically Discussed & Formalized:** Mathematical formulations provided; contrasted with Central DP.

---

### Module 4: Privacy-Preserving Data Mining Techniques

#### Syllabus Topics Covered
- Privacy protection in data processing workflows
- Input modification: Random noise addition, data swapping, microaggregation
- Privacy-Preserving Data Publishing (PPDP): $k$-Anonymity, $l$-Diversity, $t$-Closeness
- Protection beyond $k$-anonymity: Why syntactic privacy fails and necessitates Differential Privacy
- Output modification: Query auditing, output perturbation

#### Technical Conceptual Explanation & Comparison
Prior to differential privacy, tabular data privacy relied on **syntactic partition-based models**:

1. **$k$-Anonymity (Sweeney, 2002):**
   Each combination of quasi-identifiers (e.g., Age, Gender, Nationality) must appear in at least $k$ records in the published table.
   *Vulnerability:* Susceptible to **Homogeneity Attacks**. If all $k$ records in an equivalence class share the same sensitive attribute (e.g., all 5 international students aged 28 in Informatics dropped out), the attacker infers the attribute with 100% confidence.

2. **$l$-Diversity (Machanavajjhala et al., 2007):**
   Requires that each equivalence class contains at least $l$ "well-represented" distinct values for each sensitive attribute.
   *Vulnerability:* Susceptible to **Skewness and Similarity Attacks**. If the distribution of grades or debt in an equivalence class is semantically clustered, sensitive information still leaks.

3. **$t$-Closeness (Li et al., 2007):**
   Requires the distance (e.g. Earth Mover's Distance) between the marginal distribution of a sensitive attribute in any equivalence class and its global distribution across the entire dataset to be no greater than $t$.

#### Why Modern PPML Demands Differential Privacy Over PPDP
Syntactic models make dangerous assumptions about the adversary's background knowledge. If an attacker acquires auxiliary data (e.g. linked social media or campus registry info), $k$-anonymity guarantees collapse.
In contrast, **Differential Privacy is provably robust against arbitrary auxiliary background knowledge** and independent of the attacker's computational power.

```
+--------------------+-----------------------------+-------------------------------+
| Dimension          | Syntactic PPDP (k-Anonymity)| Differential Privacy (DP-SGD) |
+--------------------+-----------------------------+-------------------------------+
| Privacy Primitive  | Syntactic indistinguishability| Probabilistic bound on output |
| Auxiliary Knowledge| Assumes bounded knowledge   | Robust to ANY background info |
| Curse of Dim.     | Fails for high-D data (>10) | Scales gracefully with norm C |
| Post-Processing    | No formal guarantee         | Strictly invariant            |
| Composition        | No mathematical composition | Bounded via RDP / Moments     |
+--------------------+-----------------------------+-------------------------------+
```

#### Operational Status
- **Analytically Discussed:** Syntactic mechanisms contrasted against DP; justification for DP-SGD provided.

---

### Module 5: Compressive Privacy for Machine Learning

#### Syllabus Topics Covered
- Foundations of Compressive Privacy
- Information-theoretic privacy formulations (Mutual Information & Directed Information)
- Mechanisms of Compressive Privacy: Privacy-constrained dimensionality reduction, adversarial representation learning
- Machine learning applications and Utility-Task Accuracy trade-offs

#### Technical Conceptual Explanation
Compressive Privacy is an information-theoretic framework where data $X$ is compressed into a compact latent representation $Z$ using a lossy mapping $q(Z \mid X)$ designed to optimize the trade-off between:
1. **Utility:** Maximizing mutual information $I(Z; Y)$ with respect to the primary prediction target $Y$ (e.g. student dropout status).
2. **Privacy:** Minimizing mutual information $I(Z; S)$ with respect to a private/sensitive attribute $S$ (e.g. student financial indebtedness or parental income).

The objective is formalized via the **Privacy-Funnel / Information Bottleneck Optimization**:
$$\min_{q(Z \mid X)} \quad - I(Z; Y) + \beta \cdot I(Z; S)$$
Subject to compression constraint $I(X; Z) \le R$.

#### Critical Distinction: Dimensionality Reduction $\neq$ Compressive Privacy
> **Important Academic Clarification:** Standard dimensionality reduction techniques (such as Principal Component Analysis (PCA), t-SNE, or vanilla Autoencoders) **DO NOT automatically guarantee compressive privacy**.
> Standard PCA selects projection axes that maximize variance in $X$. If the sensitive attribute $S$ (e.g., student debt or age) accounts for significant variance in the feature space, standard PCA will enthusiastically preserve and amplify that sensitive signal in the top principal components.
> A projection achieves *compressive privacy* **only when explicitly constrained or regularized** to decorrelate, nullify, or minimize mutual information with $S$ (e.g. through adversarial minimax training or null-space projection).

#### Application to Student Retention
An academic institution can compute a private latent student representation $Z$ such that university advisors can predict dropout risk $Y$ with high accuracy, while ensuring that an adversary with access to $Z$ cannot reconstruct the student's family debt status or demographic background $S$.

#### Operational Status
- **Analytically Discussed:** Formalized mathematically, clearly distinguished from standard PCA, and mapped to future IA-2 scope.
