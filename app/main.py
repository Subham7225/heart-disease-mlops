from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from pythonjsonlogger.json import JsonFormatter

from src.inference import load_model, predict_payload

logger = logging.getLogger("heart_disease_api")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "API latency", ["method", "endpoint"])
model = None

class PredictionRequest(BaseModel):
    age: float = Field(..., ge=0)
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = load_model()
        logger.info({"event": "model_loaded"})
    except FileNotFoundError as exc:
        model = None
        logger.warning({"event": "model_missing", "detail": str(exc)})
    yield

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0", lifespan=lifespan)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, request.url.path).observe(latency)
    logger.info({
        "event": "request",
        "path": request.url.path,
        "method": request.method,
        "status_code": response.status_code,
        "latency_seconds": round(latency, 4),
    })
    return response

@app.get("/")
def root() -> dict:
    return {"service": "heart-disease-api", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health(response: Response) -> dict:
    if model is None:
        response.status_code = 503
        return {"status": "degraded", "model_loaded": False}
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predict_payload(model, payload.model_dump())

@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
