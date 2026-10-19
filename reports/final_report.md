# MLOps Assignment Report

## 1. Introduction
This report presents an end-to-end MLOps workflow for heart disease risk prediction using the UCI Heart Disease dataset. The solution covers data ingestion, EDA, feature engineering, model training, experiment tracking, reproducible packaging, API serving, CI/CD, containerization, deployment, and monitoring.

## 2. Problem Statement
The objective is to classify whether a patient is at risk of heart disease based on clinical features such as age, chest pain type, resting blood pressure, cholesterol, ECG results, and exercise-induced angina.

## 3. Dataset and Acquisition
- Dataset: UCI Heart Disease dataset
- Acquisition method: `python data/download_data.py`
- Target variable: `target`
- Input features: 13 structured clinical variables

## 4. EDA Summary
Include the following screenshots and observations:
- Class balance chart
- Histograms for key numeric variables
- Correlation heatmap
- Missing value summary and data quality checks

## 5. Preprocessing and Feature Engineering
Describe:
- Median imputation for numerical fields
- Standard scaling
- Unified preprocessing pipeline with scikit-learn `ColumnTransformer`
- Reasons for keeping the feature set reproducible end to end

## 6. Model Development
Models trained:
- Logistic Regression
- Random Forest

Evaluation methodology:
- Stratified train/test split
- 5-fold cross-validation
- Metrics: accuracy, precision, recall, F1, ROC-AUC

Include final metrics table copied from `artifacts/metrics.json`.

## 7. Experiment Tracking
Document MLflow usage:
- Parameters per run
- Metrics per run
- Model comparison
- Saved artifacts and screenshots from MLflow UI

## 8. Packaging and Reproducibility
Include:
- `requirements.txt`
- `Dockerfile`
- saved model pipeline in `artifacts/model_pipeline.joblib`
- exact commands for training and inference

## 9. API and Containerization
Endpoints:
- `GET /health`
- `POST /predict`
- `GET /metrics`

Include sample request/response and local Docker run proof.

## 10. CI/CD
Summarize GitHub Actions workflow:
- dependency installation
- linting
- dataset download
- unit tests
- training job
- artifact upload

Add screenshots of successful runs.

## 11. Kubernetes Deployment
Document:
- namespace, deployment, service, ingress
- replica count
- health probes
- exposure method
- verification commands and screenshots

## 12. Monitoring and Logging
Show:
- structured JSON request logs
- Prometheus scrape configuration
- sample metrics from `/metrics`

## 13. Architecture
Embed or convert the Mermaid architecture diagram from `reports/architecture.mmd`.

## 14. Conclusion
This project demonstrates a production-oriented MLOps implementation with reproducibility, experiment tracking, automated validation, containerized serving, Kubernetes deployment, and operational observability.
