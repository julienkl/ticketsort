"""Barrière qualité utilisée par la CI (C13) : échoue si le modèle est sous le seuil.

Usage : python -m src.model.quality_gate --min-f1 0.80
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def check(metrics: dict, min_f1: float) -> bool:
    return metrics["f1_macro"] >= min_f1


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--min-f1", type=float, default=0.80)
    p.add_argument("--metrics", default=str(ROOT / "reports" / "metrics.json"))
    a = p.parse_args(argv)
    meta = json.loads(Path(a.metrics).read_text(encoding="utf-8"))
    f1 = meta["metrics"]["f1_macro"]
    if check(meta["metrics"], a.min_f1):
        print(f"OK : F1 macro {f1} >= seuil {a.min_f1} (modèle {meta['version']})")
        return 0
    print(f"ÉCHEC : F1 macro {f1} < seuil {a.min_f1} — livraison bloquée")
    return 1


if __name__ == "__main__":
    sys.exit(main())
