# Architecture de référence

Les exemples suivent un flux commun :

```text
raw -> DataLoader -> Pandera contract -> FeatureBuilder -> Model/Trainer -> metrics/artifact
                                      \-> audit log / lineage
```

## Couche données

`DataLoader` ne fait que lire une source et appliquer le schéma d'entrée. `DataFrameModel` décrit les colonnes, types, valeurs admissibles et contraintes métier. La validation est faite avant le split pour détecter un fichier cassé le plus tôt possible. Les transformations apprises (imputation, vocabulaire, scaler) sont ensuite fit uniquement sur le train.

## Couche modèle

Une classe de modèle porte le framework et expose un petit contrat (`fit`, `predict`, `save`). Le pipeline ne connaît pas les détails de sklearn, PyTorch, TensorFlow ou spaCy. Cela facilite le remplacement du framework et les tests avec un faux modèle.

## Configuration

Hydra compose `configs/config.yaml` et des overrides CLI :

```bash
python -m <package>.cli training.epochs=5 data.path=data/raw/example.csv
```

Les chemins sont relatifs au projet et résolus avec `Path`. Les secrets passent par l'environnement ou un secret manager, jamais par YAML.

## Qualité et production

- tests unitaires sur les classes et tests de contrat sur les schémas ;
- seuils de qualité et métriques dans la CI ;
- logs structurés avec `run_id`, version du dataset et commit ;
- artefacts dans un registry (MLflow, S3, registry interne) plutôt que dans Git ;
- observabilité : latence, taux d'erreur, dérive et feedback métier ;
- séparation entraînement / inférence et stratégie de rollback explicite.
