import json
import os
import sys
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.pipeline import MLModelPipeline

ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
BENCH_DIR = os.path.join(ROOT, "benchmarks")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(BENCH_DIR, exist_ok=True)


def main():
    pipe = MLModelPipeline()
    df = pipe.generate_synthetic_dataset(n_samples=2000)
    results = pipe.evaluate_all(df)

    # Identify best model by ROC-AUC and persist it + scaler
    best_name = max(results.keys(), key=lambda k: results[k]["roc_auc"])
    best_model = pipe.models[best_name]
    joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, "best_churn_model.joblib"))
    joblib.dump(pipe.scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))

    report = {
        "dataset_samples": len(df),
        "positive_churn_rate": round(float(df["target_churn"].mean()), 4),
        "best_model": best_name,
        "best_model_artifact": "artifacts/best_churn_model.joblib",
        "models": results,
    }
    with open(os.path.join(BENCH_DIR, "benchmark_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = (
        "# Model Evaluation & Analytics Benchmark\n\n"
        f"- **Dataset:** {len(df)} enterprise accounts (Churn rate: {report['positive_churn_rate']*100:.1f}%)\n"
        f"- **Best Model:** **{best_name}** (saved to `artifacts/best_churn_model.joblib`)\n\n"
        "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {m} | {v['accuracy']} | {v['precision']} | {v['recall']} | {v['f1_score']} | {v['roc_auc']} |"
            for m, v in results.items()
        )
    )
    with open(os.path.join(BENCH_DIR, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
