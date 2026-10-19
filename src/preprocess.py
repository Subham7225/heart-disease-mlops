from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

@dataclass
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series

def load_dataset(path: str | Path, target_column: str = "target") -> DatasetBundle:
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    missing = [c for c in NUMERIC_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    X = df[NUMERIC_FEATURES].copy()
    y = df[target_column].copy()
    return DatasetBundle(X=X, y=y)

def build_preprocessor() -> ColumnTransformer:
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", num_pipeline, NUMERIC_FEATURES)],
        remainder="drop",
    )
