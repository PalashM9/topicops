import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

import joblib
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator

from app import store
from app.common import MODELS, read_json, sha256

REQUESTS = Counter("router_predictions_total", "Completed predictions", ["label", "version"])
LATENCY = Histogram("router_prediction_seconds", "Prediction and logging latency")
LOG_FAILURES = Counter("router_logging_failures_total", "Failed prediction log writes")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application):
    pointer = read_json(MODELS / "production.json")
    folder = MODELS / pointer["version"]
    artifact = folder / "model.joblib"
    if sha256(artifact) != pointer["model_sha256"]:
        raise RuntimeError("Model integrity verification failed")
    application.state.model = joblib.load(artifact)
    application.state.version = pointer["version"]
    store.initialize()
    yield


app = FastAPI(title="Topic Router", version="1.0.0", lifespan=lifespan)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=3, max_length=10000)

    @field_validator("text")
    @classmethod
    def not_blank(cls, value):
        if len(value.strip()) < 3:
            raise ValueError("Provide at least three non-whitespace characters")
        return value


class PredictionResponse(BaseModel):
    request_id: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    version: str
    logging_ok: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if not hasattr(app.state, "model"):
        raise HTTPException(503, "Model unavailable")
    return {"status": "ready", "version": app.state.version}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    start = time.perf_counter()
    model = app.state.model
    version = app.state.version
    probabilities = model.predict_proba([payload.text])[0]
    index = int(probabilities.argmax())
    label = str(model.classes_[index])
    confidence = float(probabilities[index])
    request_id = str(uuid4())
    empty = int(model.named_steps["tfidf"].transform([payload.text]).nnz == 0)
    logging_ok = True
    try:
        store.record((request_id, datetime.now(timezone.utc).isoformat(), version,
                      label, confidence, len(payload.text), empty,
                      (time.perf_counter() - start) * 1000))
    except Exception:
        logging_ok = False
        LOG_FAILURES.inc()
        logger.exception("Prediction logging failed")
    REQUESTS.labels(label=label, version=version).inc()
    LATENCY.observe(time.perf_counter() - start)
    return PredictionResponse(
        request_id=request_id, label=label, confidence=confidence, version=version,
        probabilities={str(key): float(value) for key, value in zip(model.classes_, probabilities)},
        logging_ok=logging_ok,
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
