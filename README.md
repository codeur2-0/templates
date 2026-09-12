# Templates Data, AI & MLOps

Une collection de projets pédagogiques, autonomes et exécutables pour démarrer proprement un produit data/AI. Chaque exemple privilégie une architecture en couches, la programmation orientée objet, la validation des données avec **Pandera**, et une configuration reproductible avec **Hydra**.

## Parcours disponibles

| Domaine | Exemples | Cas d'usage |
| --- | --- | --- |
| `data-science` | `classification/with-sklearn`, `with-pytorch`, `with-spacy`, `with-tensorflow`, `with-keras` | churn, fraude, routage de tickets, risque de crédit, sentiment |
| `data-eng` | `batch-etl/with-pandas`, `data-quality/with-pandera`, `analytics/with-duckdb` | ETL incrémental, contrats de données, couche analytique |
| `ai-eng` | `rag/with-chroma`, `api/with-fastapi` | recherche augmentée, exposition d'un modèle |
| `mlops` | `training/with-mlflow`, `serving/with-fastapi` | suivi d'expériences, service de prédiction |

Chaque projet contient au minimum :

- un `README.md` avec le cas d'utilisation, les hypothèses et les commandes ;
- `src/<package>/` avec des classes séparées pour données, modèle et orchestration ;
- `configs/config.yaml` piloté par Hydra, sans chemin absolu ni secret ;
- des données d'exemple anonymes dans `data/raw/` ;
- des tests unitaires et de contrat ;
- un notebook documenté dans `notebooks/` (exploration, validation ou démonstration).

## Démarrer un exemple

Chaque dossier est un projet indépendant. Exemple avec sklearn :

```bash
cd data-science/classification/with-sklearn
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m churn_sklearn.cli
jupyter lab notebooks/01_exploration.ipynb
```

Pour les projets qui utilisent un framework lourd, installer l'extra indiqué dans son README (`[dev]`, `[gpu]`, etc.). Les exemples de modèles ont volontairement de petits jeux de données locaux afin de rester déterministes et faciles à remplacer par un stockage objet ou une table.

## Règles d'architecture

1. **Raw immutable** : `data/raw` est la source d'entrée ; les données nettoyées sont produites dans `data/processed` et ne sont pas versionnées par défaut.
2. **Contrat avant calcul** : chaque entrée tabulaire est validée par un `DataFrameModel` Pandera, avec `lazy=True` dans les pipelines.
3. **Configuration déclarative** : les hyperparamètres et chemins vivent dans `configs/`; le code reçoit un `DictConfig` et ne lit pas de variables cachées.
4. **Séparation des responsabilités** : `DataLoader`/`DataValidator`, `FeatureBuilder`, `Model`, `Trainer` et `Pipeline` sont testables isolément.
5. **Reproductibilité** : seed, version du schéma, métriques et artefacts doivent être journalisés ; aucun PII ou secret dans le dépôt.
6. **Promotion prudente** : un modèle ne devient pas production sans tests, validation du contrat, métriques de référence et rollback.

Voir `docs/architecture.md` pour les choix détaillés et `docs/notebooks.md` pour le rôle attendu des notebooks.
