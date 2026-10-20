# Heart Disease Risk Prediction - MLOps Assignment 01 (summary)

The full 14-page report with all screenshots is in [`final_report.docx`](final_report.docx) ([PDF](final_report.pdf)). This page is a short Markdown summary.

- Repository: https://github.com/Subham7225/heart-disease-mlops
- Dataset: UCI Heart Disease (Cleveland), 303 patients, 13 features, binary target (164 no disease / 139 disease)

## Pipeline
1. `data/download_data.py` downloads the data and writes raw + cleaned CSVs (only `ca` and `thal` have missing values, 6 cells in total).
2. EDA in `notebooks/01_eda.ipynb` produces 6 figures in `reports/figures/`.
3. Preprocessing is a single ColumnTransformer: median impute + scale for 5 continuous features, mode impute for 3 binary features, mode impute + one-hot for 5 categorical features (25 model features).
4. `src/train.py` runs GridSearchCV (stratified 5-fold, ROC-AUC) for Logistic Regression (4 candidates) and Random Forest (24 candidates). All 30 runs are logged to MLflow with params, metrics, plots and the model.
5. The best model (by hold-out ROC-AUC) is saved to `artifacts/model_pipeline.joblib`.
6. CI runs ruff, training, 11 pytest tests, artifact upload, Docker build + smoke test, and a GHCR push.
7. Serving: FastAPI in Docker, deployed to Docker Desktop Kubernetes (2 replicas, LoadBalancer on localhost:8080).
8. Monitoring: JSON request logs, Prometheus `/metrics`, and a provisioned Grafana dashboard.

## Results (hold-out, n=61)
| Model | Best params | CV ROC-AUC | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression (selected) | C=1.0 | 0.904 | 0.885 | 0.839 | 0.929 | 0.881 | 0.968 |
| Random Forest | n_estimators=400, max_depth=4 | 0.897 | 0.885 | 0.839 | 0.929 | 0.881 | 0.950 |

## Evidence
See `screenshots/`: MLflow (`mlflow_*.png`), CI (`github_actions_*.png`, `02_pytest.png`), Docker (`03_docker_build_run.png`), Kubernetes (`04_k8s_deploy_script.png`, `05_k8s_status.png`, `06_k8s_traffic_logs_metrics.png`), monitoring (`grafana_dashboard.png`, `prometheus_*.png`, `api_metrics_endpoint.png`).

## Local access instructions
```bash
./deployment/deploy_local.sh
curl http://localhost:8080/health
open http://localhost:8080/docs
```
