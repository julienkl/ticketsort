# C7 — Benchmark des services IA

## Besoin et contraintes (point de départ du benchmark)

| Contrainte | Valeur cible |
|---|---|
| Qualité | F1 macro ≥ 0,80 sur le jeu de test |
| Latence | < 200 ms par ticket (usage interactif dans le formulaire) |
| Volume | ~2 000 tickets / mois (hypothèse à justifier) |
| Coût | Proche de zéro, pas de budget dédié |
| Données | Les tickets peuvent contenir des données personnelles → préférence pour un traitement local |
| Éco-responsabilité | Consommation mesurée, modèle proportionné au besoin |
| Intégration | Exposable en API REST, conteneurisable |

## Candidats

| Candidat | Type | Statut |
|---|---|---|
| TF-IDF + Naive Bayes | Local, classique | Évalué (baseline) |
| TF-IDF + régression logistique | Local, classique | Évalué |
| Mistral (API) en zero-shot | Service LLM hébergé | Évalué si clé disponible |
| mDeBERTa zero-shot (Hugging Face) | Modèle local pré-entraîné | Évalué si `transformers` installé |
| Fine-tuning CamemBERT | Local, deep learning | **Écarté** : coût d'entraînement et de GPU disproportionné pour un gain attendu faible sur des textes courts |
| Service AutoML cloud | Service hébergé | **Écarté** : coût, données hors de la maîtrise du projet |

## Résultats

Générés par `python -m src.benchmark.run_benchmark` (copier ici le contenu de `reports/benchmark.md`) :

| candidat | f1_macro | latence_ms | cout_eur_1k | kg_co2 | donnees | entrainement |
|---|---|---|---|---|---|---|
| *à remplir* | | | | | | |

> Les résultats sur `data/samples/` ne servent qu'à valider la chaîne. Les chiffres présentés au jury doivent venir du dataset réel.

## Décision

*Exemple de formulation, à ajuster selon tes vrais résultats :*
La régression logistique sur TF-IDF est retenue : elle atteint l'objectif de qualité, répond en moins d'une milliseconde, ne coûte rien à l'usage, garde les données en local et consomme très peu d'énergie. Le LLM zero-shot reste une alternative intéressante sans entraînement, mais il envoie les tickets à un tiers et facture chaque appel. Il pourrait servir plus tard à pré-étiqueter de nouvelles catégories.
