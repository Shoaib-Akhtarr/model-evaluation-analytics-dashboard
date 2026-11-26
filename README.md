# 📊 Interactive ML Model Evaluation & Predictive Analytics Dashboard

> **Status: Experimental ML Baseline (Secondary Repo)**  
> For core Forward-Deployed AI Systems & Reliability flagships, see:  
> 1. [`corrective-rag-agent`](https://github.com/Shoaib-Akhtarr/corrective-rag-agent) — Temporal Claim-Diff & Neural NLI RAG  
> 2. [`rag-evaluation-benchmark-harness`](https://github.com/Shoaib-Akhtarr/rag-evaluation-benchmark-harness) — Adversarial Prompt-Injection Red-Team & SQLite Registry  
> 3. [`unified-enterprise-rag-platform`](https://github.com/Shoaib-Akhtarr/unified-enterprise-rag-platform) — Full-Stack Next.js + FastAPI + Qdrant FDE Control Plane

Interactive Scikit-Learn classification benchmark (Random Forest, Gradient Boosting, Logistic Regression) with Streamlit diagnostics over a synthetic 2,000-row SaaS churn benchmark.

```bash
python training/train_and_export.py
pytest -v tests/
streamlit run app/dashboard.py
```
