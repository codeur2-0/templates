# RAG avec Chroma — assistant de documentation

## Cas d'utilisation

Un assistant répond à des questions sur une mini documentation produit. Les documents sont ingérés dans Chroma, retrouvés par similarité puis passés à un générateur. Le générateur de démonstration est déterministe : il montre le contrat de contexte sans appeler un fournisseur externe ni exposer une clé.

```bash
pip install -e ".[dev]"
pytest
python -m docs_rag.cli
python -m docs_rag.cli query="How long is the trial?"
```

Pour passer en production, implémenter `AnswerGenerator` avec un LLM approuvé, filtrer les documents par tenant, journaliser les citations, définir un seuil de refus et évaluer faithfulness, context precision et latence. Ne jamais envoyer de données confidentielles à un provider non approuvé.
