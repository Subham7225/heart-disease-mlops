from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLE = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}
MODEL_EXISTS = Path("artifacts/model_pipeline.joblib").exists()


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code in (200, 503)
    assert "model_loaded" in response.json()


def test_metrics_endpoint_exposes_prometheus_counters():
    with TestClient(app) as client:
        client.get("/health")
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text


def test_predict_rejects_invalid_payload():
    with TestClient(app) as client:
        response = client.post("/predict", json={"age": 50})
    assert response.status_code == 422


@pytest.mark.skipif(not MODEL_EXISTS, reason="model artifact not trained yet")
def test_predict_returns_label_and_confidence():
    with TestClient(app) as client:
        response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("low_risk", "high_risk")
    assert 0.5 <= body["confidence"] <= 1.0
