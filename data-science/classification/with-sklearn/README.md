# Classification avec scikit-learn — churn client

## Cas d'utilisation

Une équipe Customer Success veut prioriser les clients susceptibles de résilier leur abonnement dans les 30 prochains jours. Le jeu d'exemple est synthétique et ne contient aucune donnée personnelle. La métrique principale est le **recall de la classe churn** : manquer un client à risque coûte davantage qu'un contact commercial inutile.

## Lancer

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m churn_sklearn.cli
python -m churn_sklearn.cli training.test_size=0.25
```

Le modèle et les métriques sont écrits dans `data/artifacts/` (ignoré par Git). Le pipeline valide le CSV avant le split, apprend les features sur le train uniquement et sauvegarde un pipeline sklearn complet.

## Structure

- `src/churn_sklearn/schemas.py` : contrat Pandera ;
- `data.py` : lecture et validation ;
- `model.py` : `ChurnClassifier`, encapsulation du pipeline sklearn ;
- `pipeline.py` : orchestration et métriques ;
- `configs/config.yaml` : configuration Hydra ;
- `notebooks/01_exploration.ipynb` : exploration reproductible.

## Limites et suite

Les données sont trop petites pour décider en production. Ajouter calibration, validation temporelle, coût métier, suivi de dérive, fairness par segment et approbation du modèle avant toute action client.
