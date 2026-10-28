"""Entraînement et évaluation du modèle retenu (TF-IDF + régression logistique).

Usage :
    python -m src.model.train                         # paramètres par défaut (modèle retenu)
    python -m src.model.train --C 1 --ngram-max 1     # essai d'autres hyperparamètres (tracé dans MLflow)
Sorties :
    models/model.joblib         modèle versionné (pipeline complet + métadonnées)
    reports/metrics.json        métriques d'évaluation (lues par la CI et par /model/info)
    reports/classification_report.txt
Si MLflow est installé et MLFLOW_TRACKING_URI défini, le run y est aussi enregistré (C8).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

PARAMS = {"ngram_range": (1, 2), "min_df": 1, "sublinear_tf": True, "C": 5.0, "max_iter": 1000}


def build_pipeline(params: dict = PARAMS) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=params["ngram_range"], min_df=params["min_df"],
                                  sublinear_tf=params["sublinear_tf"])),
        ("clf", LogisticRegression(C=params["C"], max_iter=params["max_iter"], class_weight="balanced")),
    ])


def evaluate(model: Pipeline, test: pd.DataFrame) -> dict:
    t0 = time.perf_counter()
    pred = model.predict(test["texte"])
    latency_ms = (time.perf_counter() - t0) / max(len(test), 1) * 1000
    return {
        "f1_macro": round(f1_score(test["categorie"], pred, average="macro"), 4),
        "accuracy": round(accuracy_score(test["categorie"], pred), 4),
        "latence_ms_par_ticket": round(latency_ms, 4),
        "n_test": int(len(test)),
    }


def _data_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:12]


def train(params: dict | None = None) -> dict:
    params = {**PARAMS, **(params or {})}
    train_df = pd.read_csv(PROCESSED / "train.csv")
    test_df = pd.read_csv(PROCESSED / "test.csv")

    t0 = time.perf_counter()
    model = build_pipeline(params)
    model.fit(train_df["texte"], train_df["categorie"])
    duree_entrainement = time.perf_counter() - t0
    metrics = evaluate(model, test_df)
    metrics["duree_entrainement_s"] = round(duree_entrainement, 2)

    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    meta = {
        "version": version,
        "algo": "tfidf+logreg",
        "params": {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()},
        "classes": sorted(model.classes_.tolist()),
        "train_data_hash": _data_hash(train_df),
        "n_train": int(len(train_df)),
        "metrics": metrics,
    }
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump({"pipeline": model, "meta": meta}, MODELS / "model.joblib")
    # Statistiques de référence pour la détection de dérive (C11)
    ref = {"longueur_moy": float(train_df["texte"].str.len().mean()),
           "distribution_classes": train_df["categorie"].value_counts(normalize=True).round(4).to_dict(),
           "vocabulaire": sorted(model.named_steps["tfidf"].vocabulary_)[:20000]}
    (MODELS / "reference_stats.json").write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
    (REPORTS / "metrics.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS / "classification_report.txt").write_text(
        classification_report(test_df["categorie"], model.predict(test_df["texte"]), digits=3), encoding="utf-8")

    _log_mlflow(meta)
    print(json.dumps(meta["metrics"], ensure_ascii=False))
    return meta


def _log_mlflow(meta: dict) -> None:
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_experiment("ticketsort")
    with mlflow.start_run(run_name=f"tfidf-logreg-C{meta['params']['C']}-ngram{meta['params']['ngram_range'][1]}"):
        mlflow.set_tags({"algo": meta["algo"], "version_modele": meta["version"],
                         "train_data_hash": meta["train_data_hash"], "n_train": meta["n_train"]})
        mlflow.log_params({k: str(v) for k, v in meta["params"].items()})
        mlflow.log_metrics({k: v for k, v in meta["metrics"].items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(MODELS / "model.joblib"))
        mlflow.log_artifact(str(REPORTS / "classification_report.txt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entraînement TicketSort")
    ap.add_argument("--C", type=float, default=PARAMS["C"], help="inverse de la régularisation")
    ap.add_argument("--ngram-max", type=int, default=PARAMS["ngram_range"][1], help="1 = mots seuls, 2 = + paires de mots")
    ap.add_argument("--min-df", type=int, default=PARAMS["min_df"], help="fréquence minimale d'un mot")
    a = ap.parse_args()
    train({"C": a.C, "ngram_range": (1, a.ngram_max), "min_df": a.min_df})
