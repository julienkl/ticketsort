"""C12 - Tests sur le format des données et le preprocessing."""
import pandas as pd
import pytest

from src.data.make_sample import make
from src.data.prepare import SchemaError, normalize_text, split, validate


def test_normalisation_masque_les_donnees_personnelles():
    out = normalize_text("Contactez jean.dupont@corp.fr  ou https://intranet/x  TICKET 12345678")
    assert "@" not in out and "http" not in out
    assert "<email>" in out and "<url>" in out and "<num>" in out


def test_normalisation_minuscules_et_espaces():
    assert normalize_text("  Mon   PC  ") == "mon pc"


def test_validation_colonnes_manquantes():
    with pytest.raises(SchemaError):
        validate(pd.DataFrame({"texte": ["a"]}), "texte", "categorie")


def test_validation_une_seule_classe_refusee():
    df = pd.DataFrame({"texte": ["mon pc est en panne"] * 3, "categorie": ["materiel"] * 3})
    with pytest.raises(SchemaError):
        validate(df, "texte", "categorie")


def test_validation_supprime_vides_courts_et_doublons():
    df = pd.DataFrame({
        "texte": ["mon pc ne marche plus", "mon pc ne marche plus", "ok", None, "le vpn coupe"],
        "categorie": ["materiel", "materiel", "reseau", "reseau", "reseau"],
    })
    out = validate(df, "texte", "categorie")
    assert len(out) == 2


def test_mapping_de_colonnes_dataset_externe():
    df = pd.DataFrame({"Document": ["vpn ne marche pas", "écran noir"], "Topic_group": ["Network", "Hardware"]})
    out = validate(df, "Document", "Topic_group")
    assert list(out.columns) == ["texte", "categorie"]
    assert set(out["categorie"]) == {"network", "hardware"}


def test_split_stratifie_sans_fuite():
    df = validate(make(400), "texte", "categorie")
    train, test = split(df)
    assert set(train.index).isdisjoint(test.index)
    assert set(test["categorie"]) == set(train["categorie"])
