# Data Engineering

Pipelines batch, qualité et analytique. La priorité est donnée à l'idempotence, au partitionnement, à la validation du contrat, à la traçabilité et à l'absence d'effets de bord cachés.

- `batch-etl/with-pandas` : pipeline incrémental idempotent ;
- `data-quality/with-pandera` : validation multi-règles et rapport d'incidents ;
- `analytics/with-duckdb` : modèle analytique local et requêtes reproductibles.
