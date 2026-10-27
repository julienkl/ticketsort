"""C7 - Benchmark des services IA candidats, sur le même jeu de test et les mêmes critères.

Usage : python -m src.benchmark.run_benchmark [--max-llm 50]

Candidats :
    baseline_nb        TF-IDF + Naive Bayes (local)
    tfidf_logreg       TF-IDF + régression logistique (local)  <- candidat retenu par défaut
    mistral_zero_shot  API Mistral, classification sans entraînement (si MISTRAL_API_KEY défini)
    hf_zero_shot       modèle Hugging Face local zero-shot (si `transformers` installé)

Critères mesurés : F1 macro, latence par ticket, coût estimé / 1000 tickets, émissions CO2 (CodeCarbon si installé),
localisation des données. Résultats : reports/benchmark.json et reports/benchmark.md
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.model.train import build_pipeline

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# Hypothèses de coût à mettre à jour lors de la veille (C6) : tarifs publics du fournisseur au moment du benchmark.
MISTRAL_EUR_PER_1K_TICKETS = float(os.getenv("MISTRAL_EUR_PER_1K_TICKETS", "0.05"))


@contextmanager
def emissions():
    """Mesure CO2 avec CodeCarbon si disponible, sinon None."""
    result = {"kg_co2": None}
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code="FRA", log_level="error", save_to_file=False)
        tracker.start()
        yield result
        result["kg_co2"] = tracker.stop()
    except ImportError:
        yield result


def _local(name: str, pipe: Pipeline, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    with emissions() as em:
        pipe.fit(train["texte"], train["categorie"])
        t0 = time.perf_counter()
        pred = pipe.predict(test["texte"])
        lat = (time.perf_counter() - t0) / len(test) * 1000
    return {"candidat": name, "f1_macro": round(f1_score(test["categorie"], pred, average="macro"), 4),
            "latence_ms": round(lat, 3), "cout_eur_1k": 0.0, "kg_co2": em["kg_co2"],
            "donnees": "restent en local", "entrainement": "requis"}


def mistral_zero_shot(test: pd.DataFrame, labels: list[str], max_n: int) -> dict | None:
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        return None
    sample = test.head(max_n)
    preds, t0 = [], time.perf_counter()
    for text in sample["texte"]:
        prompt = (f"Classe ce ticket de support informatique dans UNE catégorie parmi {labels}. "
                  f"Réponds uniquement par le nom de la catégorie.\nTicket : {text}")
        r = requests.post("https://api.mistral.ai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
                                "messages": [{"role": "user", "content": prompt}], "temperature": 0},
                          timeout=30)
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip().lower()
        preds.append(next((lab for lab in labels if lab in answer), "inconnu"))
    lat = (time.perf_counter() - t0) / len(sample) * 1000
    return {"candidat": "mistral_zero_shot", "f1_macro": round(f1_score(sample["categorie"], preds, average="macro"), 4),
            "latence_ms": round(lat, 1), "cout_eur_1k": MISTRAL_EUR_PER_1K_TICKETS, "kg_co2": None,
            "donnees": "envoyées au fournisseur (UE)", "entrainement": "aucun", "n_evalues": len(sample)}


def hf_zero_shot(test: pd.DataFrame, labels: list[str], max_n: int) -> dict | None:
    try:
        from transformers import pipeline
    except ImportError:
        return None
    clf = pipeline("zero-shot-classification", model=os.getenv("HF_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"))
    sample = test.head(max_n)
    with emissions() as em:
        t0 = time.perf_counter()
        preds = [clf(t, candidate_labels=labels)["labels"][0] for t in sample["texte"]]
        lat = (time.perf_counter() - t0) / len(sample) * 1000
    return {"candidat": "hf_zero_shot", "f1_macro": round(f1_score(sample["categorie"], preds, average="macro"), 4),
            "latence_ms": round(lat, 1), "cout_eur_1k": 0.0, "kg_co2": em["kg_co2"],
            "donnees": "restent en local", "entrainement": "aucun", "n_evalues": len(sample)}


def to_markdown(rows: list[dict]) -> str:
    cols = ["candidat", "f1_macro", "latence_ms", "cout_eur_1k", "kg_co2", "donnees", "entrainement"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("—" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None) -> list[dict]:
    p = argparse.ArgumentParser()
    p.add_argument("--max-llm", type=int, default=50, help="nb de tickets envoyés aux modèles zero-shot")
    a = p.parse_args(argv)
    train = pd.read_csv(PROCESSED / "train.csv")
    test = pd.read_csv(PROCESSED / "test.csv")
    labels = sorted(train["categorie"].unique())

    rows = [
        _local("baseline_nb", Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())]), train, test),
        _local("tfidf_logreg", build_pipeline(), train, test),
    ]
    for fn in (mistral_zero_shot, hf_zero_shot):
        try:
            if (r := fn(test, labels, a.max_llm)) is not None:
                rows.append(r)
        except Exception as exc:  # un candidat en échec est noté, pas bloquant
            rows.append({"candidat": fn.__name__, "erreur": str(exc)})

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "benchmark.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS / "benchmark.md").write_text(to_markdown(rows) + "\n", encoding="utf-8")
    print(to_markdown(rows))
    return rows


if __name__ == "__main__":
    main()
