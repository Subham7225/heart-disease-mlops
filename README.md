# Heart Disease Risk Prediction - End-to-End MLOps

[![ci](https://github.com/Subham7225/heart-disease-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Subham7225/heart-disease-mlops/actions/workflows/ci.yml)

MLOps (AIMLCZG523) Assignment 01. This project predicts heart-disease risk from 13 clinical features of the
[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), then tracks experiments with MLflow, runs CI on GitHub Actions,
serves predictions with FastAPI in Docker, deploys to Kubernetes behind a LoadBalancer, and monitors the API with Prometheus and Grafana.

**Full report:** [`reports/final_report.docx`](reports/final_report.docx) (PDF copy: [`reports/final_report.pdf`](reports/final_report.pdf)) - evidence in [`screenshots/`](screenshots/).

## Results (stratified 20% hold-out, seed 42)

| Model (tuned with 5-fold GridSearchCV) | CV ROC-AUC | Test accuracy | Test precision | Test recall | Test ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (C=1.0) - served** | 0.904 ± 0.014 | 0.885 | 0.839 | 0.929 | **0.968** |
| Random Forest (400 trees, depth 4) | 0.897 ± 0.022 | 0.885 | 0.839 | 0.929 | 0.950 |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/download_data.py                        # data/raw/heart.csv + data/processed/heart_clean.csv
python -m src.train --config configs/default.yaml   # tuning + MLflow + artifacts/
ruff check . && pytest -v                           # lint + 11 tests
mlflow ui --backend-store-uri ./mlruns --port 5001  # experiment browser (port 5000 is used by AirPlay on macOS)
python scripts/predict_sample.py                    # offline inference
```

## Serve

```bash
# Docker
docker build -t heart-disease-api:1.0.0 .
docker run -p 8000:8000 heart-disease-api:1.0.0
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'
# -> {"prediction":1,"label":"high_risk","confidence":0.983,...}

# Kubernetes (Docker Desktop with Kubernetes enabled, minikube or kind)
./deployment/deploy_local.sh            # build, apply manifests, verify via LoadBalancer
curl http://localhost:8080/health       # Swagger UI at http://localhost:8080/docs
kubectl get all -n mlops

# Monitoring: API :8000, Prometheus :9090, Grafana :3000 (dashboard "Heart Disease API", admin/admin)
docker compose -f monitoring/docker-compose.yml up -d --build
```

## Repository layout

| Path | Contents |
|---|---|
| `data/` | download script, raw and cleaned dataset |
| `notebooks/01_eda.ipynb` | EDA (figures in `reports/figures/`) |
| `src/` | preprocessing pipeline, training/tuning, evaluation, inference |
| `app/main.py` | FastAPI: `/predict`, `/health`, `/metrics`, `/docs`, JSON request logging |
| `tests/` | Pytest suite (preprocessing, training, API) |
| `.github/workflows/ci.yml` | lint, train, test, artifacts, Docker build + smoke test, push to GHCR |
| `deployment/` | namespace, ConfigMap, Deployment (2 replicas, probes, limits), LoadBalancer Service, optional Ingress |
| `monitoring/` | Prometheus config, Grafana provisioning and dashboard |
| `artifacts/` | saved pipeline (`model_pipeline.joblib`), `metrics.json`, evaluation plots, CV tables |
| `reports/` | final report, architecture diagram, video walkthrough script |

## CI/CD

Every push or pull request to `main` runs **build-test-train** (ruff, data download, training, pytest with JUnit XML, artifact upload). Only if that job passes does **docker-build-and-smoke-test** run (build the image, call `/health` and `/predict` in the container, push `ghcr.io/subham7225/heart-disease-mlops:latest`). Any failing step stops the pipeline.
