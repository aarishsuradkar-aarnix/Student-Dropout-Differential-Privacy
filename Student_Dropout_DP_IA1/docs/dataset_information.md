# Dataset Information: Predict Students' Dropout and Academic Success

## 1. Overview and Identification

- **Dataset Title:** Predict Students' Dropout and Academic Success
- **Repository:** UCI Machine Learning Repository
- **Dataset ID:** 697
- **Persistent DOI:** [https://doi.org/10.24432/C5MC89](https://doi.org/10.24432/C5MC89)
- **Repository URL:** [https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Direct Download Archive:** `https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip`

---

## 2. Creators, Institutions, and Publication Date

- **Creators / Authors:**
  - **Valentim Realinho** (Polytechnic Institute of Portalegre, Portugal)
  - **Jorge Machado** (Polytechnic Institute of Portalegre, Portugal)
  - **Baptista** (Polytechnic Institute of Portalegre, Portugal)
  - **Mónica V. Martins** (Polytechnic Institute of Portalegre, Portugal)
- **Publishing Institution:** Instituto Politécnico de Portalegre & UCI Machine Learning Repository
- **Date Published / Donated to UCI:** December 13, 2021
- **Funding / Research Project:** Supported by national funding through SATDAP (*Sistema de Apoio à Transformação Digital da Administração Pública*) under the project "Fazer Rir e Aprender" / academic persistence analytics.

---

## 3. License and Permitted Usage

- **License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**
  - *Permissions:* The dataset may be freely shared, adapted, modified, and redistributed for academic, research, and educational purposes.
  - *Conditions:* Appropriate credit/attribution must be provided, along with a link to the license and notification of any modifications made.
- **Redistribution Status:** Bundled as `data/data.csv` in accordance with CC BY 4.0 attribution terms.

---

## 4. Dataset Size, Dimensions, and Characteristics

| Attribute | Specification |
| :--- | :--- |
| **Total Records (Instances)** | 4,424 students |
| **Input Features** | 36 predictor attributes |
| **Target Variable** | 1 (`Target`: multi-class outcome) |
| **Total Columns** | 37 columns |
| **Delimiter** | Semicolon (`;`) |
| **Missing Values** | **0** (Complete dataset without null entries) |
| **Duplicate Rows** | **0** (All records are distinct) |

### Class Label Distribution

The target variable represents student academic status at the end of the normal curriculum duration:

| Class Label | Encoded Class | Count | Proportion (%) | Status Description |
| :--- | :---: | :---: | :---: | :--- |
| **Graduate** | 2 | 2,209 | 49.93% | Successfully completed degree requirements |
| **Dropout** | 0 | 1,421 | 32.12% | Left institution without graduating |
| **Enrolled** | 1 | 794 | 17.95% | Still actively pursuing coursework past standard timeframe |
| **Total** | — | **4,424** | **100.0%** | Comprehensive cohort size |

---

## 5. Feature Categorization

The 36 predictor features span four distinct domains:

### A. Demographic Data
1. `Marital status` (Categorical: 1-Single, 2-Married, 3-Widower, 4-Divorced, 5-Facto Union, 6-Legally Separated)
2. `Nacionality` (Categorical: country codes)
3. `Displaced` (Binary: 1-Yes, 0-No; student living away from hometown)
4. `Gender` (Binary: 1-Male, 0-Female)
5. `Age at enrollment` (Continuous: student age in years)
6. `International` (Binary: 1-Yes, 0-No)

### B. Socio-Economic Background
7. `Mother's qualification` (Categorical education levels)
8. `Father's qualification` (Categorical education levels)
9. `Mother's occupation` (Categorical standard occupational classifications)
10. `Father's occupation` (Categorical standard occupational classifications)
11. `Educational special needs` (Binary: 1-Yes, 0-No)
12. `Debtor` (Binary: 1-Yes, 0-No; outstanding tuition fee debts)
13. `Tuition fees up to date` (Binary: 1-Yes, 0-No)
14. `Scholarship holder` (Binary: 1-Yes, 0-No)

### C. Academic Path at Admission
15. `Application mode` (Categorical: general contingent, mature student 23+, transfer, etc.)
16. `Application order` (Ordinal: 0 to 9 preference priority)
17. `Course` (Categorical: degree programme code, e.g. Informatics, Agronomy, Nursing)
18. `Daytime/evening attendance` (Binary: 1-Daytime, 0-Evening)
19. `Previous qualification` (Categorical)
20. `Previous qualification (grade)` (Continuous: 0 to 200 scale)
21. `Admission grade` (Continuous: 0 to 200 scale)

### D. Curricular Units (Semesters 1 & 2 Performance)
22. `Curricular units 1st sem (credited)` (Integer)
23. `Curricular units 1st sem (enrolled)` (Integer)
24. `Curricular units 1st sem (evaluations)` (Integer)
25. `Curricular units 1st sem (approved)` (Integer)
26. `Curricular units 1st sem (grade)` (Continuous: 0 to 20 scale)
27. `Curricular units 1st sem (without evaluations)` (Integer)
28. `Curricular units 2nd sem (credited)` (Integer)
29. `Curricular units 2nd sem (enrolled)` (Integer)
30. `Curricular units 2nd sem (evaluations)` (Integer)
31. `Curricular units 2nd sem (approved)` (Integer)
32. `Curricular units 2nd sem (grade)` (Continuous: 0 to 20 scale)
33. `Curricular units 2nd sem (without evaluations)` (Integer)

### E. Macro-Economic Context Indicators
34. `Unemployment rate` (Continuous: regional economic percentage)
35. `Inflation rate` (Continuous: annual inflation index)
36. `GDP` (Continuous: national GDP growth indicator)

---

## 6. Preprocessing Methodology & Data Leakage Prevention

1. **Header Sanitization:** Strip extraneous escape characters (`\t`) in column headers such as `Daytime/evening attendance\t`.
2. **Missing & Duplicate Audit:** Verified 0 null values and 0 identical records.
3. **Target Label Encoding:** Mapped string classes `{'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}`.
4. **Stratified Splitting:** 80% training (3,539 instances) and 20% testing (885 instances) preserving class balance.
5. **Strict Data Leakage Prevention:** `StandardScaler` is fitted **exclusively on the training fold** ($X_{train}$), and the fitted statistical moments ($\mu_{train}, \sigma_{train}$) are applied to transform $X_{test}$ without retraining.

---

## 7. Privacy and Ethical Considerations

1. **Inherent Sensitivity:** The records contain financial distress indicators (`Debtor`, `Tuition fees up to date`, `Scholarship holder`), parental occupations, and academic underperformance.
2. **Re-identification & Linkage Attacks:** Although direct identifiers (names, student IDs) were expunged by the source institution, quasi-identifiers (such as combining age, exact admission grade, nationality, and parental qualification) can be joined against public municipal graduation lists or institutional bursary rosters to re-identify students.
3. **Membership Inference Attacks (MIA):** Without differential privacy, an adversary possessing a student's partial academic record can query a deployed neural network and determine with high confidence whether that specific student was in the training cohort, inferring their dropout status and debt situation.
4. **Differential Privacy Guarantee:** Training with DP-SGD bounds the maximum influence of any single student's record on model parameters, ensuring rigorous $(\epsilon, \delta)$-indistinguishability.

---

## 8. Dataset Citation

### BibTeX
```bibtex
@misc{misc_predict_students_dropout_and_academic_success_697,
  author       = {Realinho, Valentim and Machado, Jorge and Baptista, and Martins, Mónica V.},
  title        = {{Predict Students' Dropout and Academic Success}},
  year         = {2021},
  howpublished = {UCI Machine Learning Repository},
  doi          = {10.24432/C5MC89},
  note         = {{DOI}: 10.24432/C5MC89}
}
```

### Text Citation
> Realinho, V., Machado, J., Baptista, & Martins, M. V. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89.
