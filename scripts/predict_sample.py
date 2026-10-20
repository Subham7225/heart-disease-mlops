"""Offline inference: load the saved pipeline and score a sample patient.

Usage: python scripts/predict_sample.py [--json '{"age": 63, ...}']
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference import load_model, predict_payload  # noqa: E402

SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
          "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    payload = json.loads(args.json) if args.json else SAMPLE
    print(json.dumps(predict_payload(load_model(), payload), indent=2))
