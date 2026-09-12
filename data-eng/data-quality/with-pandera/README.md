# Data quality avec Pandera — ventes

## Cas d'utilisation

Avant de charger les ventes dans un entrepôt, l'équipe data exécute un contrat qualité : unicité de la vente, dates valides, montants positifs, pays connus et catégories autorisées. Le CLI produit un rapport JSON exploitable par une CI ou un orchestrateur.

```bash
pip install -e ".[dev]"
pytest
python -m sales_quality.cli
python -m sales_quality.cli data.path=data/raw/sales.csv
```

Le mode strict rejette les colonnes inattendues. En production, séparer les lignes valides et invalides dans une quarantine, versionner le schéma et notifier l'équipe productrice avec l'extrait d'erreur, pas avec des PII.
