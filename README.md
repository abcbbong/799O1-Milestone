# CDSDX 799 O1 — Module C Capstone
**Boston University MSDS | Credit Card Fraud Detection**

Binary classification of fraudulent credit card transactions (0.58% fraud rate).  
Three datasets: Fraud / Credit Approval / Loan Default.  
Primary metric: AUC-ROC.

---

## Datasets

| Dataset | Source |
|---------|--------|
| Credit Card Fraud Detection | [Kaggle — tusharbhadouria](https://www.kaggle.com/datasets/tusharbhadouria/credit-card-fraud-detection) |
| Credit Card Application Data | [Kaggle — caesarmario](https://www.kaggle.com/datasets/caesarmario/application-data) |
| Loan Default Prediction | [Kaggle — marcbuji](https://www.kaggle.com/datasets/marcbuji/loan-default-prediction) |

---

## Notebooks

| File | Topic | Method | Fraud AUC | Credit AUC | Loan AUC |
|------|-------|--------|-----------|------------|----------|
| `Week1.ipynb` | Feature Engineering | Polynomial terms + VIF filtering + Logistic Regression | 0.9820 | 0.9990 | 0.6816 |
| `Week2.ipynb` | Regularization | Lasso / Ridge / Elastic Net | 0.9817 | 0.9993 | 0.6832 |
| `Week3.ipynb` | Dimension Reduction | Forward/Backward Selection, PCR, PLSR | 0.9378 | 0.9874 | 0.6810 |
| `Week4.ipynb` | Logistic Regression | Feature scaling, stratified sampling | 0.9651 | 0.9998 | 0.6864 |
| `Week5.ipynb` | SVM | Kernel trick (RBF/Linear), C/γ tuning | 0.9457 | 0.9999 | 0.6039 |
| `Week6.ipynb` | Tree Ensembles | Decision Tree (ccp_alpha) + Random Forest | 0.9802 | 0.9995 | 0.6943 |
| `Week8.ipynb` | KNN | Euclidean / Manhattan / Chebyshev, CV k-selection | 0.8791 | 0.9056 | 0.6254 |
| `Week9.ipynb` | Gradient Boosting | Staged early stopping, η sensitivity | 0.9825 | 0.9999 | 0.7030 |
| `Week10.ipynb` | K-Means | Elbow + Silhouette, K-Means++ | Sil 0.4058 (k=13) | Sil 0.2063 (k=26) | Sil 0.1449 (k=23) |
| `Week11.ipynb` | DBSCAN + HAC | Density clustering, 4-linkage HAC | Noise 7.66× lift | 407 clusters | Noise 0.87× lift |

---

## Key Findings
- **Best supervised model**: Week 9 GBM — Fraud **0.9825** / Credit **0.9999** / Loan **0.7030**
- **Best unsupervised signal**: Week 11 DBSCAN — Fraud noise 7.66× lift (74.1% recall, zero labels used)
- **Credit paradox**: GBM AUC 0.9999 yet HAC Ward Silhouette 0.1711 — hyperplane boundary ≠ spherical clusters
- **Loan Default ceiling**: AUC ~0.70 persists across all methods (feature geometry constraint, not algorithm limitation)
- **Noise lift reversal**: Fraud noise = anomaly signal (7.66×) vs Loan noise = below baseline (0.87×) — domain-dependent interpretation
