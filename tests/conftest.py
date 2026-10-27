import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["API_KEYS"] = "test-key-123,autre-cle"


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory):
    """Entraîne un modèle sur l'échantillon dans un dossier temporaire (isolé du dépôt)."""
    import src.data.prepare as prepare
    import src.model.train as train
    from src.data.make_sample import make

    tmp = tmp_path_factory.mktemp("run")
    df = prepare.validate(make(800), "texte", "categorie")
    tr, te = prepare.split(df)
    (tmp / "processed").mkdir()
    tr.to_csv(tmp / "processed" / "train.csv", index=False)
    te.to_csv(tmp / "processed" / "test.csv", index=False)

    train.PROCESSED, train.MODELS, train.REPORTS = tmp / "processed", tmp / "models", tmp / "reports"
    meta = train.train()
    return {"dir": tmp, "meta": meta, "model_path": tmp / "models" / "model.joblib", "test": te}
