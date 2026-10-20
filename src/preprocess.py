"""Data loading and the reusable preprocessing pipeline.

Feature groups were chosen from the UCI documentation and the EDA notebook:
- continuous vitals are median-imputed and standardised,
- binary flags are kept as 0/1 (mode-imputed),
- multi-level clinical codes are one-hot encoded so the model does not
  assume an ordering between, e.g., chest-pain types or thal results.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]
NUMERIC_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
ALL_FEATURES = NUMERIC_FEATURES


@dataclass
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series


def load_dataset(path: str | Path, target_column: str = "target") -> DatasetBundle:
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    df = df.drop_duplicates()
    return DatasetBundle(X=df[ALL_FEATURES].copy(), y=df[target_column].astype(int).copy())


def build_preprocessor() -> ColumnTransformer:
    continuous = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    binary = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[
            ("continuous", continuous, CONTINUOUS_FEATURES),
            ("binary", binary, BINARY_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
