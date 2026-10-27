"""Génère un petit jeu de tickets synthétiques (FR) pour les tests et la CI.

Le vrai entraînement se fait sur un dataset public (voir README). Ce fichier permet
au pipeline CI de tourner sans télécharger de données ni exposer de données réelles.

Usage : python -m src.data.make_sample --n 600
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TEMPLATES = {
    "materiel": [
        "Mon {device} ne s'allume plus depuis ce matin",
        "L'écran de mon {device} reste noir",
        "Le clavier du {device} ne répond plus",
        "Mon {device} fait un bruit étrange et chauffe",
        "Besoin d'un nouveau chargeur pour mon {device}",
        "L'imprimante du {place} affiche un bourrage papier",
    ],
    "logiciel": [
        "{app} plante à l'ouverture",
        "Impossible d'installer {app} sur mon poste",
        "{app} affiche un message d'erreur au démarrage",
        "La mise à jour de {app} a échoué",
        "{app} est très lent depuis la dernière version",
        "Demande d'installation de {app}",
    ],
    "reseau": [
        "Pas de connexion internet au {place}",
        "Le wifi du {place} coupe sans arrêt",
        "Le VPN ne se connecte pas depuis chez moi",
        "Le réseau est très lent au {place}",
        "Impossible d'accéder au lecteur réseau partagé",
        "Câble réseau défectueux au {place}",
    ],
    "acces": [
        "Mot de passe expiré, je ne peux plus me connecter",
        "Compte bloqué après plusieurs tentatives",
        "Demande d'accès au dossier partagé {app}",
        "Je n'ai pas les droits pour ouvrir {app}",
        "Réinitialisation de mon mot de passe svp",
        "Création de compte pour un nouvel arrivant",
    ],
    "messagerie": [
        "Je ne reçois plus d'emails dans Outlook",
        "Ma boîte mail est pleine",
        "Impossible d'envoyer un mail avec une pièce jointe",
        "Les invitations du calendrier n'arrivent pas",
        "Je reçois beaucoup de spams depuis hier",
        "Signature email à mettre à jour",
    ],
}
FILL = {
    "device": ["PC portable", "ordinateur", "poste fixe", "laptop", "écran"],
    "app": ["Excel", "Teams", "SAP", "Power BI", "Chrome", "l'ERP", "Word"],
    "place": ["bureau", "3e étage", "site de Lyon", "open space", "siège"],
}
NOISE = ["", " merci", " urgent", " svp", " depuis hier", " cordialement", " c'est bloquant",
         " ça marche plus", " pouvez-vous regarder", " (3e fois cette semaine)", " je suis en télétravail"]
# Tickets ambigus réalistes : plusieurs catégories plausibles.
AMBIGUOUS = [
    "Teams ne se connecte pas, je pense que c'est le wifi",
    "Je n'arrive plus à me connecter à Outlook depuis mon nouveau PC",
    "Mon mot de passe ne fonctionne plus sur le VPN",
    "L'imprimante réseau n'apparaît plus dans Word",
    "Je n'ai plus accès à ma boîte mail partagée",
    "Mon PC est lent quand j'ouvre Excel",
]
LABEL_NOISE = 0.06  # part d'étiquettes erronées, comme dans un vrai outil ITSM


def make(n: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    cats = list(TEMPLATES)
    for i in range(n):
        cat = cats[i % len(cats)]
        text = rng.choice(TEMPLATES[cat])
        for key, values in FILL.items():
            text = text.replace("{" + key + "}", rng.choice(values))
        if rng.random() < 0.08:
            text = rng.choice(AMBIGUOUS)
        if rng.random() < LABEL_NOISE:
            cat = rng.choice([c for c in cats if c != cat])
        rows.append({"ticket_id": f"T{i:05d}", "texte": text + rng.choice(NOISE) + rng.choice(NOISE), "categorie": cat})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--out", default=str(ROOT / "data" / "samples" / "tickets_sample.csv"))
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    make(a.n).to_csv(a.out, index=False)
    print(f"{a.n} tickets écrits dans {a.out}")
