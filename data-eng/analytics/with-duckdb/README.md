# Analytique avec DuckDB — chiffre d'affaires par pays

## Cas d'utilisation

Un analyste veut interroger un export de ventes local sans déployer un entrepôt. Le pipeline valide le fichier avec Pandera, l'enregistre dans DuckDB en mémoire et expose une requête versionnée qui agrège le revenu par pays et par mois.

```bash
pip install -e ".[dev]"
pytest
python -m sales_duckdb.cli
```

DuckDB est adapté à la démonstration et aux traitements analytiques locaux. En production, conserver un format colonnaire partitionné, gérer les accès et tester les plans de requêtes.
