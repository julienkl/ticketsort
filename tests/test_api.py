"""C9/C12 - Tests des endpoints : authentification, validation des entrées, prédictions, monitoring."""
import pytest
from fastapi.testclient import TestClient

import api.main as api_main

KEY = {"X-API-Key": "test-key-123"}


@pytest.fixture
def client(trained_model, tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "MODEL_PATH", trained_model["model_path"])
    monkeypatch.setattr(api_main, "PRED_LOG", tmp_path / "predictions.jsonl")
    with TestClient(api_main.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "modele_charge": True}


def test_sans_cle_401(client):
    assert client.post("/predict", json={"texte": "mon pc est en panne"}).status_code == 401


def test_mauvaise_cle_401(client):
    r = client.post("/predict", json={"texte": "mon pc est en panne"}, headers={"X-API-Key": "faux"})
    assert r.status_code == 401


@pytest.mark.parametrize("payload", [{}, {"texte": ""}, {"texte": "abc"}, {"texte": "     "}, {"texte": 42}])
def test_entrees_invalides_422(client, payload):
    assert client.post("/predict", json=payload, headers=KEY).status_code == 422


def test_predict(client):
    r = client.post("/predict", json={"texte": "Le VPN ne se connecte pas depuis chez moi"}, headers=KEY)
    body = r.json()
    assert r.status_code == 200
    assert body["categorie"] == "reseau"
    assert 0 <= body["confiance"] <= 1
    assert abs(sum(body["probabilites"].values()) - 1) < 0.01
    assert body["version_modele"]


def test_predict_batch(client):
    tickets = [{"texte": "écran noir sur mon laptop"}, {"texte": "ma boîte mail est pleine"}]
    r = client.post("/predict/batch", json={"tickets": tickets}, headers=KEY)
    assert r.status_code == 200 and len(r.json()) == 2


def test_batch_limite(client):
    r = client.post("/predict/batch", json={"tickets": [{"texte": "mon pc est en panne"}] * 101}, headers=KEY)
    assert r.status_code == 422


def test_model_info(client):
    assert "f1_macro" in client.get("/model/info", headers=KEY).json()["metrics"]


def test_journal_predictions_ecrit(client):
    client.post("/predict", json={"texte": "Contact: a.b@corp.fr mon pc plante"}, headers=KEY)
    content = api_main.PRED_LOG.read_text(encoding="utf-8")
    assert "<email>" in content and "a.b@corp.fr" not in content


def test_metrics_prometheus(client):
    client.post("/predict", json={"texte": "mon pc ne démarre plus"}, headers=KEY)
    text = client.get("/metrics").text
    assert "ticketsort_predictions_total" in text and "ticketsort_latency_seconds" in text


def test_modele_absent_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main, "MODEL_PATH", tmp_path / "absent.joblib")
    with TestClient(api_main.app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.post("/predict", json={"texte": "mon pc est en panne"}, headers=KEY).status_code == 503


def test_openapi(client):
    assert {"/predict", "/predict/batch", "/model/info"} <= set(client.get("/openapi.json").json()["paths"])
