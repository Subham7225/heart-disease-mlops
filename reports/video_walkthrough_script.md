# Video walkthrough script (about 5-7 minutes)

Record the screen (macOS: Cmd+Shift+5, "Record Entire Screen") and narrate in your own words.

1. **Intro (30 s)** - problem statement; open the GitHub repo and walk through the folder layout.
2. **Data + EDA (60 s)** - `python data/download_data.py`; open `notebooks/01_eda.ipynb` and explain class balance, histograms, heatmap and why cp/thal/slope/ca are one-hot encoded.
3. **Training + MLflow (75 s)** - `python -m src.train --config configs/default.yaml`; `mlflow ui --backend-store-uri ./mlruns --port 5001`; show the two parent runs, the nested grid candidates, the chart comparison and the plots in Artifacts.
4. **Tests + CI (45 s)** - `ruff check . && pytest -v`; open the Actions tab and show both green jobs and the uploaded artifact.
5. **Docker (45 s)** - `docker build -t heart-disease-api:1.0.0 .`, `docker run -p 8000:8000 heart-disease-api:1.0.0`, open `http://localhost:8000/docs` and run a prediction.
6. **Kubernetes (60 s)** - `./deployment/deploy_local.sh`, then `kubectl get all -n mlops`, `curl localhost:8080/health`, and `kubectl logs -n mlops deploy/heart-disease-api --tail=5`.
7. **Monitoring (45 s)** - `docker compose -f monitoring/docker-compose.yml up -d --build`, send a few requests, then open Grafana at `http://localhost:3000` (Heart Disease API dashboard) and Prometheus targets.
8. **Wrap-up (20 s)** - results (ROC-AUC 0.968, accuracy 0.885), limitations and next steps.
