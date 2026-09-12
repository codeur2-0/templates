# Batch ETL avec pandas — commandes quotidiennes

## Cas d'utilisation

Un back-office exporte les commandes chaque jour. Le pipeline déduplique les identifiants, normalise les dates et montants, calcule le montant net et produit un fichier analytique idempotent. Une seconde exécution avec la même source donne le même résultat.

```bash
pip install -e ".[dev]"
pytest
python -m orders_etl.cli
python -m orders_etl.cli data.path=data/raw/orders.csv
```

La source est validée par Pandera avant transformation. En production, remplacer les CSV par un object store ou une table, ajouter un watermark, un catalogue de données et une alerte sur les volumes.
