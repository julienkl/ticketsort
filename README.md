# TicketSort

Classification automatique de tickets de support informatique, exposée en API et intégrée dans une application de saisie.
Projet de validation du **Bloc 2 (C6 à C13)** du titre RNCP Développeur en Intelligence Artificielle (Simplon).

```
dataset → préparation → entraînement/évaluation → barrière qualité → modèle versionné
       → API FastAPI (clé d'API) → application Streamlit
       → Prometheus/Grafana + journal → rapport de dérive
CI/CD : tests → entraînement → rapport → image Docker → GHCR
```

## Démarrage rapide (sans Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # requirements-extra.txt pour Streamlit, MLflow, CodeCarbon
cp .env.example .env

python -m src.data.prepare                 # données exemple de data/samples/
python -m src.model.train                  # modèle + reports/metrics.json
python -m src.model.quality_gate --min-f1 0.80
python -m src.benchmark.run_benchmark      # reports/benchmark.md

API_KEYS=ma-cle uvicorn api.main:app --reload      # http://localhost:8000/docs
API_URL=http://localhost:8000 APP_API_KEY=ma-cle streamlit run app/streamlit_app.py

python -m monitoring.drift                 # après quelques appels à l'API
```

## Tout en Docker

```bash
docker compose up --build
# API : http://localhost:8000/docs   App : http://localhost:8501
# Prometheus : http://localhost:9090  Grafana : http://localhost:3000 (dashboard « TicketSort »)
```

## Données

`data/samples/tickets_sample.csv` est **synthétique** (`python -m src.data.make_sample`). Il sert aux tests et à la CI.
Pour les résultats présentés au jury, utiliser un dataset public de tickets (par exemple « IT Service Ticket Classification » sur Kaggle) :

```bash
python -m src.data.prepare --input data/raw/all_tickets.csv --text-col Document --label-col Topic_group
```

Ne jamais utiliser de tickets réels d'une entreprise : ils contiennent des données personnelles et confidentielles.

## Tests

```bash
pytest -v --cov=src --cov=api --cov=app --cov=monitoring
```

## Structure

```
docs/veille.md, docs/benchmark.md     C6, C7
src/benchmark/                        C7  benchmark reproductible
src/model/train.py (+ MLflow)         C8  entraînement, versionnement, suivi
api/main.py                           C9  API du modèle
app/                                  C10 application connectée à l'API
monitoring/                           C11 Prometheus, alertes, Grafana, dérive
tests/                                C12 tests automatisés
.github/workflows/mlops.yml, Dockerfile  C13 livraison continue
docs/matrice_preuves.md               matrice compétences → preuves
```

## À faire pour finaliser

- [ ] Télécharger le dataset public, relancer préparation, entraînement et benchmark
- [ ] Remplir `docs/veille.md` semaine après semaine
- [ ] Copier les résultats réels dans `docs/benchmark.md` et rédiger la décision
- [ ] Lancer MLflow (`mlflow ui`) avec `MLFLOW_TRACKING_URI=http://localhost:5000` pour la capture C8
- [ ] Provoquer un run CI rouge (seuil à 0.99) pour la capture C13
- [ ] Faire les captures listées dans `docs/matrice_preuves.md`
