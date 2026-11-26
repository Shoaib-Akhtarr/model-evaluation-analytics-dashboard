from app.pipeline import MLModelPipeline

def test_dataset_generation():
    pipe = MLModelPipeline()
    df = pipe.generate_synthetic_dataset(n_samples=200)
    assert len(df) == 200
    assert "target_churn" in df.columns

def test_model_evaluation():
    pipe = MLModelPipeline()
    df = pipe.generate_synthetic_dataset(n_samples=200)
    results = pipe.evaluate_all(df)
    assert "Random Forest" in results
    assert "Logistic Regression" in results
    assert results["Random Forest"]["f1_score"] >= 0.50
