# C6 — Veille technique et réglementaire

## Question de veille

« Pour classer automatiquement des tickets de support informatique, faut-il un modèle entraîné localement ou un service LLM, compte tenu des performances, des coûts, de l'impact environnemental et des obligations réglementaires (RGPD, AI Act) ? »

## Organisation

| Élément | Choix |
|---|---|
| Outil de collecte | *[Feedly ou Inoreader — capture de l'abonnement aux flux à joindre]* |
| Rythme | 1 h par semaine, le lundi ; synthèse mensuelle |
| Stockage | Ce fichier (une fiche datée par semaine) |
| Diffusion | Synthèse mensuelle partagée au tuteur |

## Sources retenues et qualification

| Source | Type | Pourquoi fiable | Thème |
|---|---|---|---|
| CNIL — rubrique Intelligence artificielle (cnil.fr) | Autorité publique | Régulateur français des données personnelles | RGPD et IA |
| Texte officiel de l'AI Act (EUR-Lex, règlement UE 2024/1689) | Texte réglementaire | Source primaire | Niveaux de risque, calendrier |
| Parlement européen / Commission européenne (communiqués) | Institutions | Sources primaires sur l'évolution du texte | Calendrier (Digital Omnibus) |
| Documentation scikit-learn | Documentation officielle | Mainteneurs du projet | Modèles classiques |
| Hugging Face (blog, fiches de modèles) | Éditeur | Fiches des modèles publiées par leurs auteurs | Modèles open source |
| Documentation Mistral AI | Fournisseur | Tarifs et conditions officiels | Service LLM |
| Papers with Code / arXiv | Recherche | Publications évaluées ou prépublications traçables | État de l'art classification de texte |
| CodeCarbon | Outil open source | Méthodologie publiée | Impact environnemental |

**Sources écartées :** blogs sans auteur identifié, contenus sponsorisés, comparatifs sans méthodologie. Les articles de cabinets de conseil sont utilisés seulement comme signal, puis vérifiés sur la source primaire.

---

## Fiches de veille

### Fiche 1 — 28/10/2026 — Choix et licence des données d'entraînement

- **Source :** page Kaggle « IT Service Ticket Classification Dataset » (Adison Goh) ; projet public de référence SanellaPalangic2/IT-ticket-classifier (GitHub).
- **Ce que j'en retiens :** 47 837 tickets réels, 8 catégories déséquilibrées (Hardware 28 % … Administrative rights 4 %), textes anglais déjà prétraités. Licence **CC0 : Public Domain** (vérifiée sur la page Kaggle). Un projet indépendant obtient un F1 macro de 0,865 avec TF-IDF + régression logistique.
- **Impact sur le projet :** dataset retenu pour l'entraînement ; usage autorisé sans condition. Mon modèle obtient 0,866 : le résultat est cohérent avec la référence, donc crédible. Les données d'exemple synthétiques restent réservées aux tests et à la CI.

### Fiche 2 — 28/10/2026 — RGPD : recommandations de la CNIL sur le développement des systèmes d'IA

- **Source :** CNIL, « IA : la CNIL finalise ses recommandations sur le développement des systèmes d'IA » (22/07/2025).
- **Ce que j'en retiens :** le RGPD s'applique dans de nombreux cas aux modèles entraînés sur des données personnelles, car un modèle peut les mémoriser. La CNIL recommande de documenter cette analyse et de placer des filtres autour du modèle. Deux fiches pratiques : annotation des données et sécurité du développement.
- **Impact sur le projet :** le dataset d'entraînement ne contient pas de données personnelles identifiantes (tickets anonymisés et prétraités). En production, des tickets pourraient en contenir : masquage des emails, URL et numéros longs dès le prétraitement (`normalize_text`), journal des prédictions sans identifiant, traitement local (aucune donnée envoyée à un tiers avec le modèle retenu).

### Fiche 3 — 28/10/2026 — AI Act : calendrier et niveau de risque

- **Sources :** texte de l'AI Act (règlement UE 2024/1689) ; analyse DPO Consulting « Report de l'AI Act : ce que les entreprises doivent vraiment retenir » (2026).
- **Ce que j'en retiens :** les pratiques interdites s'appliquent depuis le 2 février 2025, les obligations des modèles d'IA à usage général depuis le 2 août 2025. Les obligations des systèmes à haut risque étaient prévues au 2 août 2026 ; le « Digital Omnibus » propose de les reporter (position du Parlement en mars 2026 : décembre 2027 pour l'annexe III). **À vérifier sur la source primaire :** l'état d'adoption définitive du report à la date de la soutenance.
- **Impact sur le projet :** un outil de tri de tickets informatiques ne relève pas a priori des usages à haut risque de l'annexe III (à confirmer en citant l'annexe). Il reste soumis aux principes de transparence : l'application indique que la catégorie est proposée par une IA et laisse l'utilisateur la corriger ; les prédictions à faible confiance sont envoyées en revue humaine.

### Fiche 4 — JJ/MM/2026 — *[à rédiger la semaine prochaine]*
- **Source :**
- **Ce que j'en retiens (3 lignes) :**
- **Impact sur le projet :**

---

## Synthèse (à mettre à jour chaque mois)

| Question | État de la veille |
|---|---|
| Performance | Un modèle classique atteint 0,866 de F1 macro sur des données réelles, au niveau d'un projet de référence indépendant |
| Coût | Modèle local : coût d'usage nul ; service LLM : facturé à l'appel (à chiffrer au benchmark, C7) |
| Environnement | Entraînement en quelques secondes sur CPU ; mesure CodeCarbon au benchmark |
| RGPD | Traitement local, masquage des données personnelles, journal sans identifiant |
| AI Act | Usage a priori hors haut risque ; transparence et supervision humaine prévues ; calendrier à surveiller |

## Conclusion provisoire

La veille oriente vers un modèle **entraîné localement** : performance suffisante, coût et empreinte faibles, aucune donnée transmise à un tiers. Le LLM reste à évaluer au benchmark (C7) comme alternative sans entraînement. Voir `docs/benchmark.md`.
