import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from app.pipeline import MLModelPipeline

st.set_page_config(page_title="Model Evaluation & Analytics Studio", page_icon="📊", layout="wide")

st.title("📊 Enterprise ML Model Evaluation & Predictive Analytics Studio")
st.markdown("Automated tabular data feature engineering, multi-model benchmark comparisons, and diagnostic telemetry.")

pipeline = MLModelPipeline()

st.sidebar.header("🔧 Configuration")
n_samples = st.sidebar.slider("Dataset Size (Rows)", min_value=200, max_value=5000, value=1000, step=100)

df = pipeline.generate_synthetic_dataset(n_samples=n_samples)

tab1, tab2, tab3 = st.tabs(["📈 Model Benchmark Table", "🔍 Confusion Matrices", "📁 Dataset Explorer"])

results = pipeline.evaluate_all(df)

with tab1:
    st.subheader("Model Performance Evaluation")
    metrics_list = []
    for model_name, metrics in results.items():
        metrics_list.append({
            "Model": model_name,
            "Accuracy": f"{metrics['accuracy']*100:.1f}%",
            "Precision": f"{metrics['precision']*100:.1f}%",
            "Recall": f"{metrics['recall']*100:.1f}%",
            "F1-Score": f"{metrics['f1_score']:.4f}",
            "ROC-AUC": f"{metrics['roc_auc']:.4f}"
        })
    metrics_df = pd.DataFrame(metrics_list)
    st.dataframe(metrics_df, use_container_width=True)

with tab2:
    st.subheader("Diagnostic Confusion Matrices")
    cols = st.columns(3)
    for idx, (model_name, metrics) in enumerate(results.items()):
        with cols[idx]:
            st.write(f"**{model_name}**")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

with tab3:
    st.subheader("Ingested Feature Distribution")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Total Rows: {len(df)} | Target Balance: {df['target_churn'].value_counts().to_dict()}")
