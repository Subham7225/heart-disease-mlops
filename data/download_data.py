"""Download the UCI Heart Disease (Cleveland) dataset.

Source: https://archive.ics.uci.edu/dataset/45/heart+disease
The raw file uses '?' for missing values and a 0-4 target (num).
We convert the target to binary: 0 = no disease, 1 = disease present.

Outputs
- data/raw/heart.csv            : binary target, missing values kept (the
                                  training pipeline imputes them itself)
- data/processed/heart_clean.csv: de-duplicated, missing values imputed
                                  (median / mode) - the committed cleaned copy
"""

from pathlib import Path

import pandas as pd

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "raw" / "heart.csv"
CLEAN_FILE = ROOT / "processed" / "heart_clean.csv"
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates().copy()
    for col in out.columns:
        if out[col].isna().any():
            fill = out[col].mode()[0] if col in CATEGORICAL else out[col].median()
            out[col] = out[col].fillna(fill)
    for col in CATEGORICAL + ["target"]:
        out[col] = out[col].astype(int)
    return out


def main() -> None:
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    df.to_csv(RAW_FILE, index=False)
    print(f"Saved {len(df)} raw rows to {RAW_FILE}")
    print(f"Missing values per column:\n{df.isna().sum()[df.isna().sum() > 0]}")
    cleaned = clean(df)
    cleaned.to_csv(CLEAN_FILE, index=False)
    print(f"Saved {len(cleaned)} cleaned rows to {CLEAN_FILE}")


if __name__ == "__main__":
    main()
