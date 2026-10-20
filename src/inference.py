from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.evaluate import confidence_from_proba
from src.preprocess import NUMERIC_FEATURES

ARTIFACT_PATH = Path("artifacts/model_pipeline.joblib")

def load_model(path: Path = ARTIFACT_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    return joblib.load(path)

def predict_payload(model, payload: dict) -> dict:
    row = {feature: payload[feature] for feature in NUMERIC_FEATURES}
    df = pd.DataFrame([row])
    proba = model.predict_proba(df)[0]
    pred = int(model.predict(df)[0])
    return {
        "prediction": pred,
        "label": "high_risk" if pred == 1 else "low_risk",
        "confidence": confidence_from_proba(proba),
        "probabilities": {"low_risk": float(proba[0]), "high_risk": float(proba[1])},
    }
