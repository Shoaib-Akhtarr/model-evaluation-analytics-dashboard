import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, Any, Tuple

class MLModelPipeline:
    """
    Automated feature engineering, model training, and evaluation pipeline.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
        }

    def generate_synthetic_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generates realistic enterprise tabular benchmark dataset."""
        np.random.seed(42)
        data = {
            "account_age_months": np.random.randint(1, 60, n_samples),
            "monthly_usage_gb": np.random.exponential(50, n_samples),
            "support_tickets_filed": np.random.poisson(2, n_samples),
            "avg_latency_ms": np.random.normal(120, 25, n_samples),
            "billing_amount_usd": np.random.normal(250, 75, n_samples)
        }
        df = pd.DataFrame(data)
        # Target: probability of churn or upgrade
        churn_prob = (
            (df["support_tickets_filed"] > 3).astype(int) * 0.4 +
            (df["avg_latency_ms"] > 140).astype(int) * 0.3 +
            (df["account_age_months"] < 6).astype(int) * 0.2
        )
        df["target_churn"] = (churn_prob + np.random.normal(0, 0.1, n_samples) > 0.45).astype(int)
        return df

    def evaluate_all(self, df: pd.DataFrame, target_col: str = "target_churn") -> Dict[str, Dict[str, Any]]:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else preds

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            roc = roc_auc_score(y_test, probs)
            cm = confusion_matrix(y_test, preds).tolist()

            results[name] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc, 4),
                "confusion_matrix": cm
            }

        return results
