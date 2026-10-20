"""Train, tune and compare classifiers with MLflow tracking.

For every model family we
1. run GridSearchCV (stratified k-fold, ROC-AUC) on the training split,
2. log every grid candidate as a nested MLflow run (params + CV scores),
3. evaluate the best candidate on the hold-out split,
4. log params, metrics, plots (ROC, confusion matrix, feature importance)
   and the fitted sklearn pipeline to MLflow.
The best family (by hold-out ROC-AUC) is saved to artifacts/ for serving.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.config import load_config  # noqa: E402
from src.evaluate import evaluate_binary_classifier  # noqa: E402
from src.preprocess import build_preprocessor, load_dataset  # noqa: E402

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def get_estimators(random_state: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(),
        "random_forest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
    }


def _save_fig(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_plots(name: str, pipeline: Pipeline, X_test, y_test, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_title(f"ROC curve - {name}")
    paths.append(_save_fig(fig, out_dir / f"roc_{name}.png"))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test, ax=ax, cmap="Blues", display_labels=["no disease", "disease"]
    )
    ax.set_title(f"Confusion matrix - {name}")
    paths.append(_save_fig(fig, out_dir / f"confusion_{name}.png"))

    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        label = "Gini importance"
    else:
        values = np.abs(model.coef_[0])
        label = "|coefficient|"
    imp = pd.Series(values, index=[n.split("__")[-1] for n in names]).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.plot.barh(ax=ax, color="#2a6f97")
    ax.set_xlabel(label)
    ax.set_title(f"Top features - {name}")
    paths.append(_save_fig(fig, out_dir / f"feature_importance_{name}.png"))
    return paths


def train(config_path: str) -> dict:
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    data = load_dataset("data/raw/heart.csv", cfg.target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=data.y
    )
    artifacts_dir = Path(cfg.artifacts_dir)
    plots_dir = artifacts_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)

    summary, best_name, best_score, best_pipeline = {}, None, -1.0, None
    for name, estimator in get_estimators(cfg.random_state).items():
        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        grid = {f"model__{k}": v for k, v in cfg.models[name].items()}
        search = GridSearchCV(
            pipeline, grid, scoring=SCORING, refit=cfg.selection_metric, cv=cv, n_jobs=-1
        )
        with mlflow.start_run(run_name=name) as parent:
            mlflow.set_tags({"model_family": name, "stage": "tuning+evaluation"})
            search.fit(X_train, y_train)
            res = search.cv_results_
            for i, params in enumerate(res["params"]):
                with mlflow.start_run(run_name=f"{name}_cand{i:02d}", nested=True):
                    mlflow.log_params({k.replace("model__", ""): v for k, v in params.items()})
                    mlflow.log_metrics({f"cv_{m}": float(res[f"mean_test_{m}"][i]) for m in SCORING})

            best = search.best_estimator_
            y_pred = best.predict(X_test)
            y_proba = best.predict_proba(X_test)[:, 1]
            holdout = evaluate_binary_classifier(y_test, y_pred, y_proba)
            cv_best = {m: float(res[f"mean_test_{m}"][search.best_index_]) for m in SCORING}
            cv_std = {m: float(res[f"std_test_{m}"][search.best_index_]) for m in SCORING}

            mlflow.log_params({k.replace("model__", "best_"): v for k, v in search.best_params_.items()})
            mlflow.log_params({"model_name": name, "cv_folds": cfg.cv_folds,
                               "n_candidates": len(res["params"]), "train_rows": len(X_train),
                               "test_rows": len(X_test)})
            mlflow.log_metrics({f"test_{k}": v for k, v in holdout.items()})
            mlflow.log_metrics({f"cv_mean_{k}": v for k, v in cv_best.items()})
            mlflow.log_metrics({f"cv_std_{k}": v for k, v in cv_std.items()})
            for p in make_plots(name, best, X_test, y_test, plots_dir):
                mlflow.log_artifact(str(p), artifact_path="plots")
            pd.DataFrame(res).drop(columns=["params"]).to_csv(plots_dir / f"cv_results_{name}.csv", index=False)
            mlflow.log_artifact(str(plots_dir / f"cv_results_{name}.csv"), artifact_path="tuning")
            mlflow.sklearn.log_model(best, name="model", input_example=X_test.head(3).astype(float))

            summary[name] = {
                "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
                "cross_validation": cv_best,
                "cross_validation_std": cv_std,
                "holdout": holdout,
                "mlflow_run_id": parent.info.run_id,
            }
            print(f"{name}: best={search.best_params_} cv_auc={cv_best['roc_auc']:.3f} "
                  f"test_auc={holdout['roc_auc']:.3f} test_acc={holdout['accuracy']:.3f}")
            if holdout["roc_auc"] > best_score:
                best_score, best_name, best_pipeline = holdout["roc_auc"], name, best

    assert best_pipeline is not None
    joblib.dump(best_pipeline, artifacts_dir / "model_pipeline.joblib")
    result = {"best_model": best_name, "selection_metric": "holdout roc_auc", "summary": summary}
    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(artifacts_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump({"0": "low_risk", "1": "high_risk"}, f, indent=2)
    print(f"Saved best model: {best_name} -> {artifacts_dir / 'model_pipeline.joblib'}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
