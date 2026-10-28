"""C7 - Benchmark des services IA candidats, sur le même jeu de test et les mêmes critères.

Usage : python -m src.benchmark.run_benchmark [--max-llm 50]

Candidats :
    baseline_nb        TF-IDF + Naive Bayes (local)
    tfidf_logreg       TF-IDF + régression logistique (local)  <- candidat retenu par défaut
    claude_zero_shot   API Claude (Anthropic), classification sans entraînement (si ANTHROPIC_API_KEY défini)
    mistral_zero_shot  API Mistral, classification sans entraînement (si MISTRAL_API_KEY défini)
    hf_zero_shot       modèle Hugging Face local zero-shot (si `transformers` installé)

Les candidats LLM sont évalués sur un ÉCHANTILLON STRATIFIÉ (même nombre de tickets par catégorie) pour limiter
le coût ; le modèle local est réévalué sur ce même échantillon (ligne « tfidf_logreg_meme_echantillon »)
afin que la comparaison soit équitable.

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
# Tarif public Claude Haiku 5.5 (docs Anthropic, octobre 2026) : 0,10 $ / M tokens en entrée, 0,50 $ / M en sortie
# (prompts < 100 000 tokens). Le coût est calculé à partir des tokens réellement consommés.
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
CLAUDE_USD_PER_MTOK_IN = float(os.getenv("CLAUDE_USD_PER_MTOK_IN", "0.10"))
CLAUDE_USD_PER_MTOK_OUT = float(os.getenv("CLAUDE_USD_PER_MTOK_OUT", "0.50"))
EUR_PER_USD = float(os.getenv("EUR_PER_USD", "0.92"))  # taux indicatif, à mettre à jour


def echantillon_stratifie(test: pd.DataFrame, n_total: int, seed: int = 42) -> pd.DataFrame:
    """Même nombre de tickets par catégorie : sinon les petites catégories seraient absentes de l'échantillon."""
    par_classe = max(1, n_total // test["categorie"].nunique())
    parts = [g.sample(min(len(g), par_classe), random_state=seed) for _, g in test.groupby("categorie")]
    return pd.concat(parts).reset_index(drop=True)


def _label_depuis_reponse(reponse: str, labels: list[str]) -> str:
    rep = reponse.strip().strip(".\"'` ").lower()
    if rep in labels:
        return rep
    # sinon : libellé le plus long contenu dans la réponse (évite qu'un libellé court en masque un plus long)
    return next((lab for lab in sorted(labels, key=len, reverse=True) if lab in rep), "inconnu")


def claude_zero_shot(sample: pd.DataFrame, labels: list[str]) -> dict | None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    consignes = ("You classify IT support tickets. Answer with exactly one category name from this list, "
                 f"and nothing else: {', '.join(labels)}.")
    preds, tok_in, tok_out, t0 = [], 0, 0, time.perf_counter()
    for text in sample["texte"]:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": CLAUDE_MODEL, "max_tokens": 1024,
                  "output_config": {"effort": "low"},  # tâche simple : réflexion minimale, coût et latence réduits
                  "system": consignes,
                  "messages": [{"role": "user", "content": f"Ticket: {text}"}]},
            timeout=60)
        r.raise_for_status()
        body = r.json()
        # on lit les blocs de type « text » (le premier bloc peut être un bloc de réflexion)
        reponse = " ".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        preds.append(_label_depuis_reponse(reponse, labels))
        tok_in += body.get("usage", {}).get("input_tokens", 0)
        tok_out += body.get("usage", {}).get("output_tokens", 0)
    n = len(sample)
    lat = (time.perf_counter() - t0) / n * 1000
    cout_usd_1k = (tok_in * CLAUDE_USD_PER_MTOK_IN + tok_out * CLAUDE_USD_PER_MTOK_OUT) / 1e6 / n * 1000
    return {"candidat": f"claude_zero_shot ({CLAUDE_MODEL})",
            "f1_macro": round(f1_score(sample["categorie"], preds, average="macro"), 4),
            "latence_ms": round(lat, 1), "cout_eur_1k": round(cout_usd_1k * EUR_PER_USD, 4), "kg_co2": None,
            "donnees": "envoyées à un fournisseur externe (Anthropic)", "entrainement": "aucun", "n_evalues": n,
            "tokens_moyens_entree": round(tok_in / n, 1), "tokens_moyens_sortie": round(tok_out / n, 1),
            "reponses_non_reconnues": preds.count("inconnu")}


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
            "donnees": "restent en local", "entrainement": "requis", "n_evalues": int(len(test))}


def mistral_zero_shot(sample: pd.DataFrame, labels: list[str]) -> dict | None:
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        return None
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
        preds.append(_label_depuis_reponse(answer, labels))
    lat = (time.perf_counter() - t0) / len(sample) * 1000
    return {"candidat": "mistral_zero_shot", "f1_macro": round(f1_score(sample["categorie"], preds, average="macro"), 4),
            "latence_ms": round(lat, 1), "cout_eur_1k": MISTRAL_EUR_PER_1K_TICKETS, "kg_co2": None,
            "donnees": "envoyées au fournisseur (UE)", "entrainement": "aucun", "n_evalues": len(sample)}


def hf_zero_shot(sample: pd.DataFrame, labels: list[str]) -> dict | None:
    try:
        from transformers import pipeline
    except ImportError:
        return None
    clf = pipeline("zero-shot-classification", model=os.getenv("HF_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"))
    with emissions() as em:
        t0 = time.perf_counter()
        preds = [clf(t, candidate_labels=labels)["labels"][0] for t in sample["texte"]]
        lat = (time.perf_counter() - t0) / len(sample) * 1000
    return {"candidat": "hf_zero_shot", "f1_macro": round(f1_score(sample["categorie"], preds, average="macro"), 4),
            "latence_ms": round(lat, 1), "cout_eur_1k": 0.0, "kg_co2": em["kg_co2"],
            "donnees": "restent en local", "entrainement": "aucun", "n_evalues": len(sample)}


def to_markdown(rows: list[dict]) -> str:
    cols = ["candidat", "n_evalues", "f1_macro", "latence_ms", "cout_eur_1k", "kg_co2", "donnees", "entrainement"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("—" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None) -> list[dict]:
    p = argparse.ArgumentParser()
    p.add_argument("--max-llm", type=int, default=80,
                   help="nb de tickets envoyés aux modèles zero-shot (échantillon stratifié)")
    a = p.parse_args(argv)
    train = pd.read_csv(PROCESSED / "train.csv")
    test = pd.read_csv(PROCESSED / "test.csv")
    labels = sorted(train["categorie"].unique())

    retenu = build_pipeline()
    rows = [
        _local("baseline_nb", Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())]), train, test),
        _local("tfidf_logreg", retenu, train, test),
    ]

    sample = echantillon_stratifie(test, a.max_llm)
    llm_rows = []
    for fn in (claude_zero_shot, mistral_zero_shot, hf_zero_shot):
        try:
            if (r := fn(sample, labels)) is not None:
                llm_rows.append(r)
        except Exception as exc:  # un candidat en échec est noté, pas bloquant
            llm_rows.append({"candidat": fn.__name__, "erreur": str(exc)})
    if llm_rows:
        # comparaison équitable : le modèle retenu (déjà entraîné) évalué sur le même échantillon que les LLM
        pred = retenu.predict(sample["texte"])
        rows.append({"candidat": "tfidf_logreg_meme_echantillon", "n_evalues": len(sample),
                     "f1_macro": round(f1_score(sample["categorie"], pred, average="macro"), 4),
                     "latence_ms": rows[1]["latence_ms"], "cout_eur_1k": 0.0, "kg_co2": None,
                     "donnees": "restent en local", "entrainement": "requis"})
        rows.extend(llm_rows)

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "benchmark.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (REPORTS / "benchmark.md").write_text(to_markdown(rows) + "\n", encoding="utf-8")
    print(to_markdown(rows))
    for r in rows:
        if "erreur" in r:
            print(f"ERREUR {r['candidat']} : {r['erreur']}")
    return rows


if __name__ == "__main__":
    main()
