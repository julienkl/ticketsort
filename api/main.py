"""C9 - API REST exposant le modèle de classification de tickets.

Lancer : uvicorn api.main:app --reload     Docs : http://localhost:8000/docs
Authentification : en-tête `X-API-Key` (clés dans la variable API_KEYS, séparées par des virgules).
Monitoring (C11) : /metrics au format Prometheus + journal des prédictions pour la dérive.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.responses import Response
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator

from src.data.prepare import normalize_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
PRED_LOG = Path(os.getenv("PRED_LOG", ROOT / "monitoring" / "predictions.jsonl"))
LOW_CONFIDENCE = float(os.getenv("LOW_CONFIDENCE", "0.5"))

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

# ---------- métriques Prometheus ----------
REQUESTS = Counter("ticketsort_requests_total", "Requêtes", ["endpoint", "status"])
LATENCY = Histogram("ticketsort_latency_seconds", "Latence des requêtes", ["endpoint"])
PREDICTIONS = Counter("ticketsort_predictions_total", "Prédictions par catégorie", ["categorie"])
LOW_CONF = Counter("ticketsort_low_confidence_total", "Prédictions sous le seuil de confiance")

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


def load_model() -> None:
    if not MODEL_PATH.exists():
        log.error("modèle introuvable : %s (lancer python -m src.model.train)", MODEL_PATH)
        state.clear()
        return
    bundle = joblib.load(MODEL_PATH)
    state["pipeline"], state["meta"] = bundle["pipeline"], bundle["meta"]
    log.info("modèle chargé : version %s", state["meta"]["version"])


app = FastAPI(title="TicketSort API", version="0.1.0", lifespan=lifespan,
              description="Classification automatique de tickets de support. Authentification par en-tête `X-API-Key`.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: Annotated[str | None, Security(api_key_header)]) -> str:
    valid = [k for k in os.getenv("API_KEYS", "").split(",") if k]
    if not key or not any(secrets.compare_digest(key, v) for v in valid):
        REQUESTS.labels("auth", "401").inc()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Clé d'API absente ou invalide")
    return key


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    LATENCY.labels(request.url.path).observe(time.perf_counter() - t0)
    REQUESTS.labels(request.url.path, str(response.status_code)).inc()
    return response


# ---------- schémas ----------
class TicketIn(BaseModel):
    texte: str = Field(..., min_length=5, max_length=5000, examples=["Mon PC portable ne s'allume plus"])

    @field_validator("texte")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("le texte ne peut pas être vide")
        return v


class BatchIn(BaseModel):
    tickets: list[TicketIn] = Field(..., min_length=1, max_length=100)


class Prediction(BaseModel):
    categorie: str
    confiance: float
    a_verifier: bool = Field(description="vrai si la confiance est sous le seuil : à router vers un humain")
    probabilites: dict[str, float]
    version_modele: str


# ---------- logique ----------
def _model():
    if "pipeline" not in state:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Modèle non chargé")
    return state["pipeline"]


def _predict(texts: list[str]) -> list[Prediction]:
    pipe = _model()
    clean = [normalize_text(t) for t in texts]
    probas = pipe.predict_proba(clean)
    out = []
    for text, p in zip(clean, probas):
        idx = int(p.argmax())
        cat, conf = str(pipe.classes_[idx]), float(p[idx])
        PREDICTIONS.labels(cat).inc()
        if conf < LOW_CONFIDENCE:
            LOW_CONF.inc()
        out.append(Prediction(categorie=cat, confiance=round(conf, 4), a_verifier=conf < LOW_CONFIDENCE,
                              probabilites={str(c): round(float(x), 4) for c, x in zip(pipe.classes_, p)},
                              version_modele=state["meta"]["version"]))
        _log_prediction(text, cat, conf)
    return out


def _log_prediction(text: str, cat: str, conf: float) -> None:
    """Journal pour la dérive : on stocke le texte normalisé (emails/numéros déjà masqués), pas d'identifiant."""
    PRED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PRED_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "texte": text,
                            "categorie": cat, "confiance": round(conf, 4)}, ensure_ascii=False) + "\n")


# ---------- routes ----------
@app.get("/health", tags=["système"])
def health() -> dict:
    return {"status": "ok" if "pipeline" in state else "degraded", "modele_charge": "pipeline" in state}


@app.get("/metrics", tags=["système"], include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", tags=["modèle"], dependencies=[Depends(require_key)])
def model_info() -> dict:
    _model()
    return state["meta"]


@app.post("/predict", response_model=Prediction, tags=["modèle"], dependencies=[Depends(require_key)])
def predict(ticket: TicketIn) -> Prediction:
    return _predict([ticket.texte])[0]


@app.post("/predict/batch", response_model=list[Prediction], tags=["modèle"], dependencies=[Depends(require_key)])
def predict_batch(batch: BatchIn) -> list[Prediction]:
    return _predict([t.texte for t in batch.tickets])
