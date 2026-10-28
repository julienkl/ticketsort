# Matrice compétences → preuves (Bloc 2)

Projet : **TicketSort**, classification automatique des tickets de support informatique
(dataset « IT Service Ticket Classification », Kaggle, licence CC0, 47 837 tickets, 8 catégories).
Modèle retenu : TF-IDF (mots et paires de mots) + régression logistique, **F1 macro 0,866** sur 9 568 tickets de test.

Les captures sont rangées dans `reports/captures/`, avec un préfixe par compétence (`c9_swagger.png`…).

| Comp. | Preuve technique | Où dans le dépôt | Captures | Statut |
|---|---|---|---|---|
| **C6** Veille | Question de veille, sources qualifiées, fiches hebdomadaires, outil d'agrégation (Inoreader), synthèse RGPD / AI Act | `docs/veille.md` | `c6_inoreader.png`, tableau des sources | ☑ |
| **C7** Benchmark | Script reproductible, critères alignés sur le besoin (F1, latence, coût, CO2, localisation des données), 3 candidats évalués dont un LLM (Claude Haiku 5.5 en zero-shot : 0,475 contre 0,925 sur le même échantillon), 2 écartés, décision argumentée | `src/benchmark/run_benchmark.py`, `docs/benchmark.md` | `c7_benchmark.png`, `c7_benchmark_claude.png` | ☑ |
| **C8** Paramétrage du service | Dépendances, secrets dans `.env` (jamais commité), clés d'accès, suivi des expériences MLflow, conteneurs | `requirements*.txt`, `.env.example`, `src/model/train.py` (MLflow), `docker-compose*.yml`, `Dockerfile` | `c8_mlflow_runs.png`, `c8_mlflow_comparaison.png`, `c8_docker_ps.png` | ☑ |
| **C9** API du modèle | `/predict`, `/predict/batch`, `/model/info`, `/health`, `/metrics`, clé d'API, validation Pydantic, documentation OpenAPI, tests | `api/main.py`, `tests/test_api.py` | `c9_swagger.png`, `c9_200.png`, `c9_401.png`, `c9_422.png` | ☑ |
| **C10** Intégration applicative | Application Streamlit qui appelle l'API, cas « confiance faible » (relecture humaine), mode dégradé si l'API est coupée, accessibilité | `app/streamlit_app.py`, `app/client.py`, `tests/test_app_and_drift.py` | `c10_app_ok.png`, `c10_app_faible_confiance.png`, `c10_app_degrade.png` | ☑ |
| **C11** Monitoring du modèle | Métriques Prometheus, tableau de bord Grafana, 4 règles d'alerte, détection de dérive testée (trafic normal : 0 alerte ; dérive : 3 alertes, code retour 2) | `monitoring/`, `docs/monitoring.md` | `c11_targets.png`, `c11_alerts.png`, `c11_grafana.png`, `c11_drift_normal.png`, `c11_drift_alerte.png` | ☑ |
| **C12** Tests automatisés | 57 tests : données, préparation, modèle (seuil F1, cas métier), endpoints, client, dérive, simulateur, barrière qualité, benchmark (API externes simulées) ; couverture 82 %, la CI échoue sous 75 % | `tests/`, `.github/workflows/mlops.yml` | `c12_pytest_cov.png` | ☑ |
| **C13** Livraison continue | Tests (couverture ≥ 75 %) → préparation + entraînement/évaluation → benchmark → barrière qualité → rapport → image Docker → test de fumée → livraison de l'image en artefact | `.github/workflows/mlops.yml`, `Dockerfile`, `src/model/quality_gate.py` | `c13_run_vert.png` (run #11), `c13_run_rouge.png` + `c13_barriere.png` (run #12, seuil 0,99) | ☑ |

Règle : aucune ligne ne reste « on l'a fait » sans fichier et capture associés.

## Points à savoir expliquer au jury

- **Pourquoi pas un LLM ?** Claude Haiku 5.5 en zero-shot obtient 0,475 de F1 contre 0,925 pour le modèle
  classique sur les mêmes 80 tickets, répond en environ 1 s au lieu de moins d'1 ms, coûte à chaque ticket et envoie les
  données à un prestataire externe. Il reste une piste en complément, pour les tickets à faible confiance.
- **Limites du modèle :** catégories faibles (`administrative rights` 0,77, `miscellaneous` 0,83) et surconfiance
  (exemple : « shared folder » → storage à 0,9999). La calibration des probabilités est une piste d'amélioration.
- **La confiance ne suffit pas pour détecter la dérive :** sur des tickets en français, le modèle se trompe avec
  assurance (14,5 % de faible confiance seulement). Ce sont les indicateurs sur les entrées et le PSI qui alertent.
- **CI et données :** la CI entraîne sur l'échantillon versionné (`data/samples/`) pour vérifier la chaîne ;
  le modèle de production est entraîné sur le jeu complet et tracé dans MLflow.
- **Image Docker :** la publication sur GHCR a été refusée (droits du jeton), l'image est donc livrée en artefact
  du run (`docker load` puis `docker run`).
