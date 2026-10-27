"""Préparation des données : validation du schéma, nettoyage du texte, découpage train/test.

Usage :
    python -m src.data.prepare --input data/samples/tickets_sample.csv
    python -m src.data.prepare --input data/raw/all_tickets.csv --text-col Document --label-col Topic_group
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MIN_CHARS = 5


class SchemaError(ValueError):
    pass


def normalize_text(text: str) -> str:
    """Minuscules, accents conservés mais normalisés (NFC), espaces et ponctuation répétée compactés."""
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = re.sub(r"https?://\S+", " <url> ", text)
    text = re.sub(r"\S+@\S+", " <email> ", text)  # on ne garde pas d'adresses email dans les données d'entraînement
    text = re.sub(r"\d{6,}", " <num> ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise SchemaError(f"colonnes manquantes : {sorted(missing)}")
    df = df.rename(columns={text_col: "texte", label_col: "categorie"})[["texte", "categorie"]]
    df = df.dropna()
    df["texte"] = df["texte"].map(normalize_text)
    df["categorie"] = df["categorie"].astype(str).str.strip().str.lower()
    df = df[df["texte"].str.len() >= MIN_CHARS].drop_duplicates()
    if df["categorie"].nunique() < 2:
        raise SchemaError("au moins deux catégories sont nécessaires")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df["categorie"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(ROOT / "data" / "samples" / "tickets_sample.csv"))
    p.add_argument("--text-col", default="texte")
    p.add_argument("--label-col", default="categorie")
    a = p.parse_args(argv)

    df = validate(pd.read_csv(a.input), a.text_col, a.label_col)
    train, test = split(df)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / "train.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)
    stats = {"n_total": len(df), "n_train": len(train), "n_test": len(test),
             "classes": df["categorie"].value_counts().to_dict()}
    (PROCESSED / "data_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
