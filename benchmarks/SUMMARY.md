# Model Evaluation & Analytics Benchmark

- **Dataset:** 2000 enterprise accounts (Churn rate: 38.6%)
- **Best Model:** **Random Forest** (saved to `artifacts/best_churn_model.joblib`)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.852 | 0.7847 | 0.8497 | 0.8159 | 0.9481 |
| Gradient Boosting | 0.85 | 0.781 | 0.8497 | 0.8139 | 0.9467 |
| Logistic Regression | 0.824 | 0.7376 | 0.8446 | 0.7874 | 0.8833 |