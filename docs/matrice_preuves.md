# Matrice compétences → preuves (Bloc 2)

| Comp. | Preuve technique | Où dans le dépôt | Capture à faire pour la slide | Statut |
|---|---|---|---|---|
| **C6** Veille | Question, sources qualifiées, fiches hebdomadaires, synthèse RGPD / AI Act | `docs/veille.md` | Tableau des sources + capture de l'outil de veille | ☐ |
| **C7** Benchmark | Script reproductible, critères alignés sur le besoin, candidats retenus et écartés, décision | `src/benchmark/run_benchmark.py`, `docs/benchmark.md`, `reports/benchmark.md` | Tableau comparatif + phrase de décision | ☐ |
| **C8** Paramétrage du service | Installation, dépendances, secrets en `.env`, clés d'accès, suivi MLflow, monitoring disponible | `requirements*.txt`, `.env.example`, `src/model/train.py` (MLflow), `docker-compose.yml` | Interface MLflow avec un run + `docker compose ps` | ☐ |
| **C9** API du modèle | `/predict`, `/predict/batch`, `/model/info`, clé d'API, validation Pydantic, OpenAPI, tests | `api/main.py`, `tests/test_api.py` | Swagger + requête/réponse JSON + 401 sans clé + pytest | ☐ |
| **C10** Intégration applicative | Application qui appelle réellement l'API, erreurs gérées, cas « confiance faible », accessibilité | `app/streamlit_app.py`, `app/client.py`, `tests/test_app_and_drift.py` | App avec résultat + app avec API coupée (message d'erreur) | ☐ |
| **C11** Monitoring du modèle | Métriques Prometheus, dashboard Grafana, règles d'alerte, rapport de dérive | `monitoring/` | Dashboard Grafana + page Alerts + `drift_report.json` avec une alerte | ☐ |
| **C12** Tests automatisés | Tests données, preprocessing, modèle (seuil F1), endpoints, client, dérive ; couverture | `tests/` | `pytest --cov` | ☐ |
| **C13** Livraison continue | Tests → entraînement/évaluation → barrière qualité → rapport → image Docker → test de fumée → GHCR | `.github/workflows/mlops.yml`, `Dockerfile` | Run vert + run rouge (seuil relevé à 0.99 via « Run workflow ») + image dans Packages | ☐ |

Règle : aucune ligne ne reste « on l'a fait » sans fichier et capture associés.
