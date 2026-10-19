"""Download the UCI Heart Disease (Cleveland) dataset and save a clean CSV.

Source: https://archive.ics.uci.edu/dataset/45/heart+disease
The raw file uses '?' for missing values and a 0-4 target (num).
We convert the target to binary: 0 = no disease, 1 = disease present.
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
OUT_DIR = Path(__file__).resolve().parent / "raw"
OUT_FILE = OUT_DIR / "heart.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    df.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(df)} rows to {OUT_FILE}")
    print(f"Missing values per column:\n{df.isna().sum()[df.isna().sum() > 0]}")


if __name__ == "__main__":
    main()
