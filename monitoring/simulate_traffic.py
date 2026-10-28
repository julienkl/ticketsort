"""C11 - Simulateur de trafic pour démontrer le monitoring et la détection de dérive.

Usage :
    python -m monitoring.simulate_traffic --phase normal --n 200    # tickets du jeu de test (même distribution qu'à l'entraînement)
    python -m monitoring.simulate_traffic --phase derive --n 100    # tickets « nouveaux » : autre langue, nouveaux sujets

La clé d'API est lue dans la variable APP_API_KEY (ou --key). Chaque appel alimente :
    - les métriques Prometheus de l'API (/metrics) -> tableau de bord Grafana et alertes
    - le journal des prédictions (monitoring/predictions.jsonl) -> python -m monitoring.drift
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]

# Scénario de dérive : l'entreprise ouvre le support à une filiale francophone et à de nouveaux outils.
# Ces tickets diffèrent de l'entraînement (langue, vocabulaire, sujets) : le modèle devrait être peu confiant.
TICKETS_DERIVE = [
    "Bonjour, mon badge ne fonctionne plus pour entrer dans le bâtiment B",
    "Copilot ne répond plus dans Teams depuis la mise à jour de ce matin",
    "La licence de l'outil de visioconférence a expiré, merci de la renouveler",
    "Le robot RPA de facturation s'est arrêté pendant la nuit",
    "Impossible de signer électroniquement le contrat fournisseur",
    "Mon téléphone portable professionnel ne synchronise plus le calendrier",
    "Le tableau de bord Power BI affiche des chiffres du mois dernier",
    "Demande de création d'un espace SharePoint pour le projet logistique",
    "Le scanner du service courrier n'envoie plus les documents par mail",
    "L'authentification à deux facteurs ne m'envoie plus de code SMS",
]


def envoyer(url: str, key: str, texte: str) -> dict | None:
    try:
        r = requests.post(f"{url}/predict", json={"texte": texte}, headers={"X-API-Key": key}, timeout=5)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["normal", "derive"], default="normal")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--url", default=os.getenv("API_URL", "http://localhost:8000"))
    ap.add_argument("--key", default=os.getenv("APP_API_KEY", ""))
    ap.add_argument("--pause", type=float, default=0.05, help="secondes entre deux appels")
    a = ap.parse_args()
    if not a.key:
        raise SystemExit("Clé d'API manquante : charger le .env (set -a; source .env; set +a) ou passer --key")

    if a.phase == "normal":
        test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
        textes = test["texte"].sample(a.n, random_state=random.randint(0, 10_000)).tolist()
    else:
        textes = [random.choice(TICKETS_DERIVE) for _ in range(a.n)]

    ok = faible = 0
    for i, t in enumerate(textes, 1):
        res = envoyer(a.url, a.key, t)
        if res:
            ok += 1
            faible += res["a_verifier"]
        if i % 50 == 0:
            print(f"{i}/{len(textes)} envoyés")
        time.sleep(a.pause)
    print(f"Phase {a.phase} : {ok}/{len(textes)} prédictions, dont {faible} à faible confiance "
          f"({faible / max(ok, 1):.0%}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
