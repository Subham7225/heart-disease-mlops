from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.config import load_config
from src.evaluate import evaluate_binary_classifier
from src.preprocess import build_preprocessor, load_dataset

def get_models(cfg_models: dict) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            C=cfg_models["logistic_regression"]["C"],
            max_iter=cfg_models["logistic_regression"]["max_iter"],
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=cfg_models["random_forest"]["n_estimators"],
            max_depth=cfg_models["random_forest"]["max_depth"],
            min_samples_split=cfg_models["random_forest"]["min_samples_split"],
            random_state=42,
        ),
    }

def train(config_path: str) -> None:
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    data = load_dataset("data/raw/heart.csv", cfg.target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        data.X,
        data.y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=data.y,
    )

    artifacts_dir = Path(cfg.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_name = None
    best_score = -1.0
    best_pipeline = None
    summary = {}

    for name, estimator in get_models(cfg.models).items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ])
        with mlflow.start_run(run_name=name):
            cv = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=5,
                scoring=["accuracy", "precision", "recall", "roc_auc"],
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate_binary_classifier(y_test, y_pred, y_proba)

            mlflow.log_params({"model_name": name, **cfg.models[name]})
            mlflow.log_metrics(metrics)
            mlflow.log_metrics({f"cv_mean_{k.replace('test_', '')}": float(np.mean(v)) for k, v in cv.items() if k.startswith('test_')})

            run_metrics = {
                "holdout": metrics,
                "cross_validation": {k: float(np.mean(v)) for k, v in cv.items() if k.startswith("test_")},
            }
            summary[name] = run_metrics

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_name = name
                best_pipeline = pipeline

    assert best_pipeline is not None
    joblib.dump(best_pipeline, artifacts_dir / "model_pipeline.joblib")
    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "summary": summary}, f, indent=2)
    with open(artifacts_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"0": "low_risk", "1": "high_risk"}, f, indent=2)
    print(f"Saved best model: {best_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
