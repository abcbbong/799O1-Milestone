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

| File | Topic | Key Method | Best Fraud AUC |
|------|-------|-----------|----------------|
| `Week1.ipynb` | Feature Engineering | Polynomial & interaction terms, VIF filtering | 0.9820 |
| `Week2.ipynb` | Regularization | Lasso, Ridge, Elastic Net | 0.9817 |
| `Week3.ipynb` | Dimension Reduction | Forward/Backward selection, PCR, PLSR | 0.9378 |
| `Week4.ipynb` | Logistic Regression | Feature scaling, stratified sampling | 0.9651 |
| `Week5.ipynb` | SVM | Kernel trick (RBF/Poly), C/γ tuning | 0.9457 |
| `Week6.ipynb` | Tree Ensembles | Decision Tree (ccp_alpha), Random Forest | 0.9802 |
| `Week8.ipynb` | KNN | Euclidean / Manhattan / Chebyshev distance | 0.8791 |
| `Week9.ipynb` | Gradient Boosting | Staged early stopping, η sensitivity | 0.9825 |
| `Week10.ipynb` | K-Means | Elbow + Silhouette, K-Means++ | Sil 0.4058 |
| `Week11.ipynb` | DBSCAN + HAC | Density clustering, 4-linkage comparison | Noise 7.66× lift |

---

## Key Findings
- **Best supervised model**: Week 9 GBM — Fraud AUC **0.9825**
- **Best unsupervised signal**: Week 11 DBSCAN noise — **74.1% fraud recall** with zero labels
- **Loan Default ceiling**: AUC ~0.70 across all methods (feature geometry constraint)
