# C6 — Veille technique et réglementaire

> Gabarit à remplir au fil des semaines. Le jury attend une **question claire**, des **sources qualifiées**, une **synthèse** et un **planning**.

## Question de veille

« Pour classer automatiquement des tickets de support en français, faut-il un modèle entraîné localement ou un service LLM, compte tenu des performances, des coûts, de l'impact environnemental et des obligations réglementaires (RGPD, AI Act) ? »

## Organisation

| Élément | Choix |
|---|---|
| Outil de collecte | *ex. Feedly ou Inoreader (flux RSS), alertes Google Scholar* |
| Rythme | 1 h par semaine, le lundi ; synthèse mensuelle |
| Stockage | Ce fichier + dossier `docs/veille/` (fiches datées) |
| Diffusion | Synthèse mensuelle partagée au tuteur |

## Sources retenues et qualification

| Source | Type | Pourquoi fiable | Thème |
|---|---|---|---|
| CNIL (cnil.fr, rubrique IA) | Autorité publique | Régulateur français | RGPD et IA |
| Texte officiel de l'AI Act (EUR-Lex) | Texte réglementaire | Source primaire | Classification des risques |
| Documentation scikit-learn | Documentation officielle | Mainteneurs du projet | Modèles classiques |
| Hugging Face Blog / model cards | Éditeur | Fiches des modèles | Modèles open source |
| Documentation Mistral AI | Fournisseur | Tarifs et conditions officiels | Service LLM |
| Papers with Code / arXiv | Recherche | Publications évaluées ou prépublications traçables | État de l'art classification |
| CodeCarbon / Green Algorithms | Outils de mesure | Méthodologie publiée | Impact environnemental |
| *à compléter* | | | |

Sources écartées : blogs sans auteur identifié, contenus sponsorisés, comparatifs sans méthodologie.

## Fiches de veille (une par semaine)

### Semaine du JJ/MM/2026
- **Source :**
- **Ce que j'en retiens (3 lignes) :**
- **Impact sur le projet :**

## Synthèse réglementaire (à vérifier et compléter à partir des sources primaires)

- **RGPD** : les tickets peuvent contenir des données personnelles (noms, emails). Mesures prises : masquage des emails, URL et numéros longs dès le preprocessing (`normalize_text`), aucun identifiant dans le journal de prédictions, dataset d'entraînement public.
- **AI Act** : un outil de tri de tickets informatiques n'apparaît pas a priori parmi les usages à haut risque. À confirmer en citant l'annexe concernée, et noter les obligations de transparence applicables.
- **Envoi à un service externe** : si un LLM hébergé est utilisé, vérifier la localisation des données, la réutilisation pour l'entraînement et la durée de conservation chez le fournisseur.

## Conclusion de la veille

*Quelle option la veille oriente-t-elle ? Renvoi vers `docs/benchmark.md`.*
