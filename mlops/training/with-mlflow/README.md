# Training MLOps avec MLflow

## Cas d'utilisation

Un run d'entraînement de churn doit conserver paramètres, métriques, schéma de données et modèle. `ExperimentRunner` ouvre un run MLflow, valide l'entrée avec Pandera, entraîne une baseline et logue un modèle sklearn. Le tracking local va dans `data/mlruns/`.

```bash
pip install -e ".[dev]"
pytest
python -m churn_mlflow.cli
mlflow ui --backend-store-uri data/mlruns
```

En production, remplacer le store local par un backend sécurisé, enregistrer l'image de l'environnement, gérer les permissions et promouvoir par alias (`candidate`, `champion`) après les checks de qualité.
