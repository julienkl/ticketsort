# Monitoring du modèle en service (C11)

## Ce qui est surveillé

| Famille | Indicateur | Outil | Seuil / alerte |
|---|---|---|---|
| Service | Disponibilité de l'API | Prometheus (`up`) | `ApiInjoignable` : cible injoignable pendant 1 min |
| Service | Latence p95 de `/predict` | Prometheus + Grafana | `LatenceElevee` : > 500 ms pendant 5 min |
| Service | Erreurs serveur (5xx) | Prometheus + Grafana | `ErreursServeur` : au moins une 5xx pendant 2 min |
| Modèle | Part de prédictions à faible confiance (< 0,5) | Prometheus + Grafana | `TauxFaibleConfianceEleve` : > 25 % sur 15 min, pendant 10 min |
| Modèle | Répartition des catégories prédites | Grafana + `monitoring/drift.py` | PSI > 0,2 |
| Données | Part de mots inconnus du modèle | `monitoring/drift.py` | > 30 % |
| Données | Longueur moyenne des tickets | `monitoring/drift.py` | écart > 50 % |

Les références (répartition des catégories, vocabulaire, longueur moyenne) sont calculées à chaque entraînement
et enregistrées dans `models/reference_stats.json` avec le modèle : la surveillance compare toujours le trafic
au modèle réellement en service.

## Chaîne technique

```
API FastAPI ──/metrics──► Prometheus (relevé toutes les 15 s, règles monitoring/alerts.yml) ──► Grafana
     │
     └── monitoring/predictions.jsonl ──► python -m monitoring.drift ──► drift_report.json + code retour (0 / 2)
```

Le code retour 2 permet de brancher l'analyse de dérive sur une tâche planifiée ou sur la CI
(par exemple : déclencher un réentraînement ou prévenir l'équipe).

## Procédure de démonstration (GitHub Codespaces)

```bash
set -a; source .env; set +a
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000                      # terminal 1
docker compose -f docker-compose.codespaces.yml up -d prometheus grafana       # terminal 2
rm -f monitoring/predictions.jsonl
python -m monitoring.simulate_traffic --phase normal --n 500 --pause 0.2
python -m monitoring.drift                                                     # attendu : aucune alerte, code 0
python -m monitoring.simulate_traffic --phase derive --n 200 --pause 0.2
python -m monitoring.drift --window 200                                        # attendu : alertes, code 2
```

Ports : 9090 Prometheus (Status → Targets, Alerts), 3000 Grafana (tableau de bord « TicketSort — modèle et service »).

## Résultats obtenus (28/10/2026, modèle 20261028102655)

| Indicateur | Trafic normal (500 tickets) | Trafic en dérive (200 tickets) | Seuil |
|---|---|---|---|
| PSI des catégories | 0,0245 | **2,969** | 0,2 |
| Mots inconnus | 0,2 % | **57,4 %** | 30 % |
| Écart de longueur | 0,6 % | **77,5 %** | 50 % |
| Faible confiance | 12,8 % | 14,5 % | 25 % |
| Alertes | aucune (code 0) | `psi_classes`, `taux_hors_vocabulaire`, `ecart_longueur_relatif` (code 2) | |

Scénario de dérive : ouverture du support à une filiale francophone et à de nouveaux outils (badge, Copilot, RPA,
SharePoint, double authentification). En dérive, le modèle classe 59 % des tickets en `hardware`, contre 29 % en temps normal.

## Enseignement principal

**La confiance du modèle ne suffit pas à détecter la dérive.** Sur des tickets en français qu'il n'a jamais vus,
le modèle reste « sûr de lui » (14,5 % de faible confiance, sous le seuil de 25 %) alors qu'il se trompe
massivement. Seule la surveillance des **entrées** (mots inconnus, longueur) et de la **répartition des sorties**
(PSI) révèle le problème. C'est pourquoi le monitoring combine les deux, et pourquoi la calibration des
probabilités fait partie des pistes d'amélioration.

## Actions prévues en cas d'alerte

| Alerte | Action |
|---|---|
| Dérive des données (mots inconnus, longueur) | Collecter et annoter un échantillon des nouveaux tickets, réentraîner, comparer dans MLflow, livrer via la CI si le F1 dépasse le seuil |
| PSI des catégories seul | Vérifier s'il s'agit d'un changement réel de l'activité (campagne, panne) avant de réentraîner |
| Faible confiance élevée | Les tickets concernés partent déjà en relecture humaine ; analyser les cas et enrichir le jeu d'entraînement |
| Latence, 5xx, API injoignable | Consulter les journaux de l'API, redémarrer le conteneur, revenir à la version d'image précédente si besoin |

## Correctifs apportés pendant la mise en place

- Le vocabulaire de référence était tronqué et le découpage en mots différait de celui du modèle : faux taux de mots inconnus. Aligné sur le `token_pattern` de scikit-learn.
- La règle `TauxFaibleConfianceEleve` divisait une série étiquetée par une série agrégée : le calcul ne renvoyait rien et l'alerte ne pouvait jamais se déclencher. Corrigé avec `sum()` des deux côtés.
- Prometheus dépendait du service `api` dans Docker Compose, ce qui démarrait une seconde API sur le port 8000. Dépendance supprimée.
