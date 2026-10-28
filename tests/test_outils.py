"""C12 - Tests des outils de la chaîne MLOps : barrière qualité (C13), analyse de dérive et simulateur
de trafic (C11), fonctions du benchmark (C7). Aucun appel réseau réel : les API externes sont simulées."""
import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

import monitoring.drift as drift
import monitoring.simulate_traffic as sim
import src.benchmark.run_benchmark as bench
from src.model import quality_gate

# --- Barrière qualité (C13) ---------------------------------------------------------------------------


@pytest.mark.parametrize("seuil,code", [("0.80", 0), ("0.99", 1)])
def test_quality_gate_bloque_sous_le_seuil(tmp_path, capsys, seuil, code):
    m = tmp_path / "metrics.json"
    m.write_text(json.dumps({"version": "v1", "metrics": {"f1_macro": 0.866}}), encoding="utf-8")
    assert quality_gate.main(["--min-f1", seuil, "--metrics", str(m)]) == code
    assert ("livraison bloquée" in capsys.readouterr().out) is (code == 1)


# --- Analyse de dérive (C11) --------------------------------------------------------------------------

REF = {"distribution_classes": {"hardware": 0.5, "access": 0.5}, "longueur_moy": 20.0,
       "vocabulaire": ["my", "laptop", "screen", "broken", "password", "reset"]}


def _drift_env(tmp_path, monkeypatch, records):
    (tmp_path / "models").mkdir()
    (tmp_path / "monitoring").mkdir()
    (tmp_path / "models" / "reference_stats.json").write_text(json.dumps(REF), encoding="utf-8")
    log = tmp_path / "monitoring" / "predictions.jsonl"
    log.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    monkeypatch.setattr(drift, "ROOT", tmp_path)
    return log


def test_drift_trafic_normal_code_0(tmp_path, monkeypatch):
    recs = [{"texte": "my laptop screen broken", "categorie": "hardware", "confiance": 0.9},
            {"texte": "password reset my laptop", "categorie": "access", "confiance": 0.8}] * 10
    log = _drift_env(tmp_path, monkeypatch, recs)
    assert drift.main(["--log", str(log)]) == 0
    rapport = json.loads((tmp_path / "monitoring" / "drift_report.json").read_text(encoding="utf-8"))
    assert rapport["n"] == 20 and rapport["alertes"] == []


def test_drift_trafic_derive_code_2(tmp_path, monkeypatch):
    recs = [{"texte": "mon badge ne fonctionne plus pour entrer dans le bâtiment depuis ce matin",
             "categorie": "hardware", "confiance": 0.3}] * 20
    log = _drift_env(tmp_path, monkeypatch, recs)
    assert drift.main(["--log", str(log)]) == 2
    rapport = json.loads((tmp_path / "monitoring" / "drift_report.json").read_text(encoding="utf-8"))
    assert {"psi_classes", "taux_hors_vocabulaire", "ecart_longueur_relatif", "taux_faible_confiance"} \
        <= set(rapport["alertes"])


def test_drift_journal_absent(tmp_path, monkeypatch):
    _drift_env(tmp_path, monkeypatch, [])
    assert drift.main(["--log", str(tmp_path / "absent.jsonl")]) == 0


# --- Simulateur de trafic (C11) -----------------------------------------------------------------------


def test_simulateur_phase_derive(monkeypatch, capsys):
    reponse = MagicMock(status_code=200, json=lambda: {"a_verifier": True})
    post = MagicMock(return_value=reponse)
    monkeypatch.setattr(sim.requests, "post", post)
    monkeypatch.setattr("sys.argv", ["sim", "--phase", "derive", "--n", "4", "--key", "k", "--pause", "0"])
    assert sim.main() == 0
    assert post.call_count == 4
    assert post.call_args.kwargs["headers"] == {"X-API-Key": "k"}
    assert "4/4 prédictions, dont 4 à faible confiance" in capsys.readouterr().out


def test_simulateur_phase_normale_lit_le_jeu_de_test(tmp_path, monkeypatch, capsys):
    (tmp_path / "data" / "processed").mkdir(parents=True)
    pd.DataFrame({"texte": [f"ticket {i}" for i in range(10)], "categorie": ["hardware"] * 10}) \
        .to_csv(tmp_path / "data" / "processed" / "test.csv", index=False)
    monkeypatch.setattr(sim, "ROOT", tmp_path)
    monkeypatch.setattr(sim.requests, "post", MagicMock(return_value=MagicMock(status_code=401)))
    monkeypatch.setattr("sys.argv", ["sim", "--n", "3", "--key", "mauvaise", "--pause", "0"])
    sim.main()
    assert "0/3 prédictions" in capsys.readouterr().out  # les refus (401) ne sont pas comptés


def test_simulateur_api_injoignable():
    assert sim.envoyer("http://localhost:1", "k", "texte") is None


def test_simulateur_exige_une_cle(monkeypatch):
    monkeypatch.delenv("APP_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["sim"])
    with pytest.raises(SystemExit, match="Clé d'API manquante"):
        sim.main()


# --- Benchmark (C7) -----------------------------------------------------------------------------------

LABELS = ["access", "administrative rights", "hardware", "storage"]


@pytest.mark.parametrize("reponse,attendu", [
    ("hardware", "hardware"),
    ("Hardware.", "hardware"),
    ("The category is: administrative rights", "administrative rights"),
    ("I don't know", "inconnu"),
])
def test_lecture_de_la_reponse_du_llm(reponse, attendu):
    assert bench._label_depuis_reponse(reponse, LABELS) == attendu


def test_echantillon_stratifie_equilibre():
    test = pd.DataFrame({"texte": ["t"] * 100, "categorie": ["a"] * 90 + ["b"] * 10})
    ech = bench.echantillon_stratifie(test, 10)
    assert ech["categorie"].value_counts().to_dict() == {"a": 5, "b": 5}


def test_claude_ignore_sans_cle(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert bench.claude_zero_shot(pd.DataFrame({"texte": ["x"], "categorie": ["a"]}), LABELS) is None


def test_claude_appel_simule(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "cle-de-test")
    corps = {"content": [{"type": "thinking", "thinking": "…"}, {"type": "text", "text": "storage"}],
             "usage": {"input_tokens": 100, "output_tokens": 5}}
    post = MagicMock(return_value=MagicMock(json=lambda: corps))
    monkeypatch.setattr(bench.requests, "post", post)
    sample = pd.DataFrame({"texte": ["disk full", "quota exceeded"], "categorie": ["storage", "storage"]})
    r = bench.claude_zero_shot(sample, LABELS)
    assert r["f1_macro"] == 1.0 and r["n_evalues"] == 2 and r["reponses_non_reconnues"] == 0
    assert r["cout_eur_1k"] > 0
    assert post.call_args.kwargs["headers"]["anthropic-version"] == "2023-06-01"


def test_benchmark_complet_sur_echantillon(trained_model, tmp_path, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(bench, "PROCESSED", trained_model["dir"] / "processed")
    monkeypatch.setattr(bench, "REPORTS", tmp_path)
    rows = bench.main(["--max-llm", "16"])
    noms = [r["candidat"] for r in rows]
    assert noms[:2] == ["baseline_nb", "tfidf_logreg"]
    assert (tmp_path / "benchmark.md").read_text(encoding="utf-8").startswith("| candidat")
    retenu = next(r for r in rows if r["candidat"] == "tfidf_logreg")
    assert retenu["cout_eur_1k"] == 0.0 and retenu["donnees"] == "restent en local"
