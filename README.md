# Heart Disease MLOps Project

End-to-end MLOps assignment project for predicting heart disease risk using the UCI Heart Disease dataset.

## Features

- Reproducible data download and preprocessing
- EDA notebook and generated visualizations
- Multiple model training with MLflow tracking
- Packaged preprocessing + model pipeline
- FastAPI inference service with `/health`, `/predict`, `/metrics`
- Pytest unit tests
- GitHub Actions CI pipeline
- Docker containerization
- Kubernetes deployment manifests
- Prometheus scrape configuration and request logging
- Report template and architecture diagram

## Repository Structure

```text
heart-disease-mlops/
├── app/
├── data/
├── deployment/
├── monitoring/
├── notebooks/
├── reports/
├── screenshots/
├── src/
├── tests/
├── .github/workflows/
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python data/download_data.py
python -m src.train --config configs/default.yaml
pytest -q
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Results (holdout set, 61 patients)

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.869 | 0.813 | 0.929 | 0.951 |
| Random Forest (selected) | 0.918 | 0.871 | 0.964 | 0.952 |

Dataset: [UCI Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease), 303 rows, target binarised (`num > 0` = disease).

## API

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
curl http://localhost:8000/metrics
```

Interactive docs: http://localhost:8000/docs

## Deployment

### Docker

```bash
docker build -t heart-disease-api:latest .
docker run -p 8000:8000 heart-disease-api:latest
```

### Kubernetes (Docker Desktop / minikube / kind)

```bash
./deployment/deploy_local.sh      # builds image, applies manifests, port-forwards to :8080
kubectl get pods,svc -n mlops
curl http://localhost:8080/health
```

### Monitoring (Prometheus + Grafana)

```bash
docker compose -f monitoring/docker-compose.yml up --build
# API :8000, Prometheus :9090, Grafana :3000 (admin/admin)
```

### Public URL (Render, free tier)

`render.yaml` is a Render Blueprint. On https://render.com: New → Blueprint → select this repo → Apply. Render builds the Dockerfile and serves the API at `https://heart-disease-api-xxxx.onrender.com`.

### CI/CD

`.github/workflows/ci.yml` runs on every push: lint → download data → train → test → upload artifacts → build Docker image → smoke-test the container → push the image to `ghcr.io/subham7225/heart-disease-mlops:latest`.
