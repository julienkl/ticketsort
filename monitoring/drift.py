"""C11 - Détection de dérive à partir du journal des prédictions de l'API.

Usage : python -m monitoring.drift [--window 500]
Compare les N dernières prédictions aux statistiques de référence calculées à l'entraînement :
    - PSI sur la distribution des catégories prédites
    - taux de mots hors vocabulaire (nouveaux sujets de tickets ?)
    - évolution de la longueur moyenne des tickets
    - taux de prédictions à faible confiance
Écrit monitoring/drift_report.json ; code retour 2 si une alerte est levée (exploitable par cron / CI).
Pour un rapport visuel, Evidently peut être branché sur le même journal (voir README).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = {"psi_classes": 0.2, "taux_hors_vocabulaire": 0.30, "ecart_longueur_relatif": 0.5, "taux_faible_confiance": 0.25}


def psi(expected: dict[str, float], actual: dict[str, float], eps: float = 1e-4) -> float:
    keys = set(expected) | set(actual)
    total = 0.0
    for k in keys:
        e, a = max(expected.get(k, 0), eps), max(actual.get(k, 0), eps)
        total += (a - e) * math.log(a / e)
    return round(total, 4)


def compute(records: list[dict], ref: dict, low_conf: float = 0.5) -> dict:
    n = len(records)
    if n == 0:
        return {"n": 0, "alertes": [], "message": "aucune prédiction à analyser"}
    vocab = set(ref["vocabulaire"])
    counts: dict[str, int] = {}
    oov = tokens = 0
    for r in records:
        counts[r["categorie"]] = counts.get(r["categorie"], 0) + 1
        for w in r["texte"].split():
            tokens += 1
            oov += w not in vocab
    dist = {k: v / n for k, v in counts.items()}
    longueur = sum(len(r["texte"]) for r in records) / n
    metrics = {
        "psi_classes": psi(ref["distribution_classes"], dist),
        "taux_hors_vocabulaire": round(oov / max(tokens, 1), 4),
        "ecart_longueur_relatif": round(abs(longueur - ref["longueur_moy"]) / ref["longueur_moy"], 4),
        "taux_faible_confiance": round(sum(r["confiance"] < low_conf for r in records) / n, 4),
    }
    alertes = [k for k, v in metrics.items() if v > THRESHOLDS[k]]
    return {"n": n, "distribution_predite": {k: round(v, 4) for k, v in dist.items()},
            "metriques": metrics, "seuils": THRESHOLDS, "alertes": alertes}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--log", default=str(ROOT / "monitoring" / "predictions.jsonl"))
    a = p.parse_args(argv)
    ref = json.loads((ROOT / "models" / "reference_stats.json").read_text(encoding="utf-8"))
    path = Path(a.log)
    lines = path.read_text(encoding="utf-8").splitlines()[-a.window:] if path.exists() else []
    report = compute([json.loads(line) for line in lines if line.strip()], ref)
    (ROOT / "monitoring" / "drift_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 2 if report["alertes"] else 0


if __name__ == "__main__":
    sys.exit(main())
