"""C10/C11 - Gestion des erreurs du client applicatif et détection de dérive."""
from unittest.mock import MagicMock

import pytest
import requests

from app.client import ApiError, classify
from monitoring.drift import compute, psi

REF = {"distribution_classes": {"materiel": 0.5, "reseau": 0.5}, "longueur_moy": 30.0,
       "vocabulaire": ["mon", "pc", "wifi", "coupe", "ne", "marche", "plus"]}


def _session(status=200, json=None, exc=None):
    s = MagicMock()
    if exc:
        s.post.side_effect = exc
    else:
        s.post.return_value = MagicMock(status_code=status, json=lambda: json)
    return s


def test_client_succes():
    payload = {"categorie": "reseau", "confiance": 0.9, "a_verifier": False,
               "probabilites": {"reseau": 0.9, "materiel": 0.1}, "version_modele": "1"}
    assert classify("le wifi coupe", session=_session(json=payload)).categorie == "reseau"


@pytest.mark.parametrize("status,msg", [(401, "clé"), (422, "trop court"), (503, "pas disponible"), (500, "500")])
def test_client_erreurs_http(status, msg):
    with pytest.raises(ApiError, match=msg):
        classify("texte", session=_session(status=status))


@pytest.mark.parametrize("exc,msg", [(requests.Timeout(), "délai"), (requests.ConnectionError(), "joindre")])
def test_client_erreurs_reseau(exc, msg):
    with pytest.raises(ApiError, match=msg):
        classify("texte", session=_session(exc=exc))


def test_psi_nul_si_distribution_identique():
    assert psi({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0


def test_pas_de_derive_sur_trafic_similaire():
    recs = [{"texte": "mon pc ne marche plus", "categorie": c, "confiance": 0.9} for c in ["materiel", "reseau"] * 10]
    assert compute(recs, REF)["alertes"] == []


def test_derive_detectee():
    recs = [{"texte": "imprimante 3d sharepoint copilot licence", "categorie": "materiel", "confiance": 0.3}] * 20
    alertes = compute(recs, REF)["alertes"]
    assert {"psi_classes", "taux_hors_vocabulaire", "taux_faible_confiance"} <= set(alertes)


def test_journal_vide():
    assert compute([], REF)["n"] == 0
