import numpy as np

from src.evaluate import confidence_from_proba, evaluate_binary_classifier


def test_evaluate_binary_classifier_perfect_scores():
    y = [0, 1, 0, 1]
    metrics = evaluate_binary_classifier(y, y, [0.1, 0.9, 0.2, 0.8])
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_confidence_is_max_probability():
    assert confidence_from_proba(np.array([0.3, 0.7])) == 0.7


def test_training_pipeline_end_to_end(tmp_path, monkeypatch):
    """Train on a small synthetic dataset to check the full pipeline wiring."""
    import pandas as pd
    import yaml

    from src import train as train_mod

    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "age": rng.integers(30, 75, n), "sex": rng.integers(0, 2, n), "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(100, 180, n), "chol": rng.integers(150, 350, n),
        "fbs": rng.integers(0, 2, n), "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(90, 200, n), "exang": rng.integers(0, 2, n),
        "oldpeak": rng.random(n) * 4, "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
    })
    df["target"] = ((df["oldpeak"] > 2) | (df["exang"] == 1)).astype(int)
    (tmp_path / "data/raw").mkdir(parents=True)
    df.to_csv(tmp_path / "data/raw/heart.csv", index=False)
    cfg = {
        "random_state": 0, "test_size": 0.25, "target_column": "target",
        "tracking_uri": f"file:{tmp_path / 'mlruns'}", "experiment_name": "unit-test",
        "artifacts_dir": str(tmp_path / "artifacts"), "cv_folds": 3,
        "models": {"logistic_regression": {"C": [1.0], "max_iter": [500]},
                   "random_forest": {"n_estimators": [20], "max_depth": [3]}},
    }
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    result = train_mod.train(str(tmp_path / "cfg.yaml"))
    assert result["best_model"] in ("logistic_regression", "random_forest")
    assert (tmp_path / "artifacts/model_pipeline.joblib").exists()
    assert 0.0 <= result["summary"]["random_forest"]["holdout"]["roc_auc"] <= 1.0
