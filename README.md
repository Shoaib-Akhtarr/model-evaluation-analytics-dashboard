# 📊 Model Evaluation & Drift Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-f7931e.svg?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Analytics](https://img.shields.io/badge/Drift%20Detection-Population%20Stability%20Index%20(PSI)-0284c7.svg?style=for-the-badge)](app/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Telemetry-ff4b4b.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Apache%202.0-0284c7.svg?style=for-the-badge)](LICENSE)

> **An enterprise MLOps model governance and continuous drift analytics platform. Delivers automated Population Stability Index (PSI) drift monitoring, Kolmogorov-Smirnov statistical tests, ROC-PR curve auditing, and production model degradation alerts.**

---

## 🔬 System Capabilities

```mermaid
flowchart TD
    PRODUCTION_DATA[Live Production Feature Stream] --> DRIFT_ENGINE[Statistical Drift Detection Engine]
    BASELINE_DATA[Training Baseline Reference Dataset] --> DRIFT_ENGINE
    
    subgraph MONITORING [Continuous Governance & Audit]
        DRIFT_ENGINE --> PSI[Population Stability Index: PSI > 0.1 Alert]
        DRIFT_ENGINE --> KS_TEST[Two-Sample Kolmogorov-Smirnov Test]
        DRIFT_ENGINE --> METRICS[ROC-AUC, Precision-Recall, Calibration Curves]
    end
    
    MONITORING --> DASHBOARD[Streamlit Executive Governance Dashboard]
    DASHBOARD --> RETRAIN[Automated Retraining Trigger]
```

---

## 📊 Key Operational Metrics

| Governance Dimension | Measured Capability | Standard Threshold |
| :--- | :---: | :---: |
| **Population Stability Index (PSI)** | Automated Real-Time Calculation | Alert at PSI > 0.1, Critical at PSI > 0.25 |
| **Statistical Drift Sensitivity** | $p < 0.01$ Kolmogorov-Smirnov | Early Warning at 99% Confidence |
| **Evaluator Model Baseline** | `artifacts/best_churn_model.joblib` | Production Calibrated |
| **Streamlit Refresh Latency** | < 120ms | Interactive Telemetry |

---

## 🚀 Quickstart

```bash
git clone https://github.com/Shoaib-Akhtarr/model-evaluation-analytics-dashboard.git
cd model-evaluation-analytics-dashboard

pip install -r requirements.txt
streamlit run app/main.py
```

---

## ⚖️ License
Distributed under the Apache 2.0 License.
