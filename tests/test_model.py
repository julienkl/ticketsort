"""C12 - Tests de validation du modèle (performance minimale, sorties, reproductibilité)."""
import joblib
import numpy as np

from src.model.quality_gate import check

MIN_F1 = 0.80


def test_performance_au_dessus_du_seuil(trained_model):
    assert trained_model["meta"]["metrics"]["f1_macro"] >= MIN_F1


def test_bat_la_baseline_aleatoire(trained_model):
    n_classes = len(trained_model["meta"]["classes"])
    assert trained_model["meta"]["metrics"]["accuracy"] > 2 / n_classes


def test_probabilites_valides(trained_model):
    pipe = joblib.load(trained_model["model_path"])["pipeline"]
    proba = pipe.predict_proba(["mon ordinateur ne s'allume plus", "le wifi coupe"])
    assert proba.shape == (2, len(pipe.classes_))
    assert np.allclose(proba.sum(axis=1), 1)


def test_cas_metier_evidents(trained_model):
    pipe = joblib.load(trained_model["model_path"])["pipeline"]
    assert pipe.predict(["mot de passe expiré je ne peux plus me connecter"])[0] == "acces"
    assert pipe.predict(["je ne reçois plus d'emails dans outlook"])[0] == "messagerie"


def test_metadonnees_de_version(trained_model):
    meta = joblib.load(trained_model["model_path"])["meta"]
    assert {"version", "algo", "params", "classes", "train_data_hash", "metrics"} <= set(meta)


def test_quality_gate():
    assert check({"f1_macro": 0.85}, 0.80)
    assert not check({"f1_macro": 0.70}, 0.80)
