"""C10 - Client de l'API, séparé de l'interface pour pouvoir être testé.

Toutes les erreurs réseau/HTTP sont converties en ApiError avec un message lisible par l'utilisateur.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("APP_API_KEY", "")
TIMEOUT = float(os.getenv("API_TIMEOUT", "5"))


class ApiError(Exception):
    pass


@dataclass
class Result:
    categorie: str
    confiance: float
    a_verifier: bool
    probabilites: dict[str, float]
    version_modele: str


def classify(texte: str, url: str = API_URL, key: str = API_KEY, session=requests) -> Result:
    try:
        r = session.post(f"{url}/predict", json={"texte": texte}, headers={"X-API-Key": key}, timeout=TIMEOUT)
    except requests.Timeout:
        raise ApiError("Le service de classification ne répond pas (délai dépassé). Réessayez dans un instant.")
    except requests.ConnectionError:
        raise ApiError("Impossible de joindre le service de classification.")
    if r.status_code == 401:
        raise ApiError("Accès refusé : clé d'API invalide.")
    if r.status_code == 422:
        raise ApiError("Le texte du ticket est trop court ou invalide (5 caractères minimum).")
    if r.status_code == 503:
        raise ApiError("Le modèle n'est pas disponible pour le moment.")
    if r.status_code >= 400:
        raise ApiError(f"Erreur du service ({r.status_code}).")
    return Result(**r.json())
