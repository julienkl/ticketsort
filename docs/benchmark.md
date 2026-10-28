# C7 — Benchmark des services IA

## Besoin et contraintes (point de départ du benchmark)

| Contrainte | Valeur cible |
|---|---|
| Qualité | F1 macro ≥ 0,80 (toutes les catégories comptent, y compris les petites) |
| Latence | < 200 ms par ticket (usage interactif dans le formulaire de saisie) |
| Volume | quelques milliers de tickets par mois (hypothèse d'un service support de taille moyenne) |
| Coût | Proche de zéro, pas de budget dédié |
| Données | Les tickets réels peuvent contenir des données personnelles → préférence pour un traitement local |
| Éco-responsabilité | Consommation mesurée, modèle proportionné au besoin |
| Intégration | Exposable en API REST, conteneurisable |

## Protocole

- **Données :** dataset « IT Service Ticket Classification » (Kaggle, licence CC0), 47 837 tickets, 8 catégories. Découpage stratifié 80/20 : 38 269 tickets d'entraînement, 9 568 de test.
- **Métrique principale :** F1 macro (moyenne non pondérée des F1 par catégorie), pour ne pas masquer les erreurs sur les petites catégories.
- **Modèles locaux :** évalués sur les 9 568 tickets de test.
- **Service LLM :** évalué sur un **échantillon stratifié de 80 tickets** (10 par catégorie) pour limiter le coût. Pour une comparaison équitable, le modèle local est **réévalué sur ce même échantillon**.
- **Mesures :** F1 macro, latence moyenne par ticket, coût pour 1 000 tickets (tokens réellement consommés × tarif public), émissions de CO2 (CodeCarbon, entraînement + prédiction, pour les modèles locaux), localisation des données.
- **Reproductible :** `python -m src.benchmark.run_benchmark` (résultats bruts dans `reports/benchmark.json`).

## Candidats

| Candidat | Type | Statut |
|---|---|---|
| TF-IDF + Naive Bayes | Local, classique | Évalué (baseline) |
| TF-IDF (mots et paires de mots) + régression logistique pondérée | Local, classique | Évalué |
| Claude Haiku 5.5 (API Anthropic) en zero-shot | Service LLM hébergé | Évalué |
| Fine-tuning d'un modèle Transformer (CamemBERT, DistilBERT) | Local, deep learning | **Écarté** : nécessite un GPU, coût d'entraînement disproportionné pour des tickets courts déjà bien classés par un modèle linéaire |
| Service AutoML cloud | Service hébergé | **Écarté** : coût, données hors de la maîtrise du projet |

## Résultats (28/10/2026)

| Candidat | Tickets évalués | F1 macro | Latence / ticket | Coût / 1 000 tickets | CO2 | Données | Entraînement |
|---|---|---|---|---|---|---|---|
| Naive Bayes (baseline) | 9 568 | 0,671 | 0,03 ms | 0 € | ≈ 0,0003 g | restent en local | requis |
| **Régression logistique (retenu)** | 9 568 | **0,866** | 0,06 ms | 0 € | ≈ 0,03 g | restent en local | requis |
| Régression logistique, même échantillon | 80 | 0,925 | 0,06 ms | 0 € | — | restent en local | requis |
| Claude Haiku 5.5 zero-shot | 80 | 0,475 | ≈ 1 050 ms | ≈ 0,016 € | non mesuré | envoyées à un fournisseur externe | aucun |

## Analyse

- **Qualité :** sur les mêmes 80 tickets, le modèle local obtient 0,925 contre 0,475 pour le LLM. Le LLM ne connaît pas les conventions propres à ce jeu de données : des catégories comme « miscellaneous », « internal project » ou « hr support » sont définies par l'organisation qui a étiqueté les tickets, et les textes sont déjà prétraités (moins naturels). Un modèle entraîné sur 38 000 exemples apprend ces conventions ; un modèle généraliste sans exemple ne peut que les deviner.
- **Latence :** environ 1 seconde par ticket pour le LLM, contre moins d'une milliseconde en local (≈ 18 000 fois plus rapide). Seul le modèle local respecte l'objectif de 200 ms ; le LLM le dépasse largement.
- **Coût :** le LLM coûte environ 1,6 centime pour 1 000 tickets (Claude Haiku 5.5, tarif public) : faible, mais non nul, alors que le modèle local est gratuit à l'usage.
- **Données :** le LLM implique d'envoyer chaque ticket à un prestataire externe, ce qui demande une analyse RGPD (sous-traitance, transferts, conservation) ; le modèle local garde tout en interne.
- **Environnement :** l'entraînement complet du modèle local émet environ 0,03 g de CO2. L'empreinte du LLM n'est pas mesurable de notre côté (calcul dans les centres de données du fournisseur).
- **Baseline :** Naive Bayes est plus rapide encore mais perd 19,5 points de F1 : le gain de la régression logistique justifie largement son surcoût négligeable.

## Limites du benchmark

- L'échantillon LLM (80 tickets) est petit : le score du LLM a une incertitude de plusieurs points. L'écart observé (0,925 contre 0,475) reste toutefois très supérieur à cette incertitude.
- Le LLM est évalué en zero-shot, avec la seule liste des catégories. Des définitions de catégories ou quelques exemples dans la consigne (« few-shot ») amélioreraient probablement son score, au prix de plus de tokens, donc d'un coût et d'une latence supérieurs.
- L'écart entre 0,866 (test complet) et 0,925 (échantillon) s'explique par la composition de l'échantillon : 10 tickets par catégorie, contre une répartition déséquilibrée dans le test complet.

## Décision

La **régression logistique sur TF-IDF** est retenue. Elle atteint l'objectif de qualité (0,866 de F1 macro sur 9 568 tickets), répond en moins d'une milliseconde, ne coûte rien à l'usage, garde les données en interne et émet une quantité négligeable de CO2. Le LLM en zero-shot est nettement moins précis sur ce jeu de données, plus lent, payant et implique un transfert de données à un tiers.

**Perspective :** le LLM pourrait servir en complément, uniquement pour les tickets que le modèle local classe avec une faible confiance (déjà signalés « à vérifier » par l'API), ou pour pré-étiqueter une nouvelle catégorie avant de réentraîner le modèle.