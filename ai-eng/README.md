# AI Engineering

Exemples d'applications autour de modèles : retrieval augmented generation et API d'inférence. La logique applicative est séparée des adaptateurs d'infrastructure pour pouvoir remplacer le vector store ou le fournisseur de modèle.

- `rag/with-chroma` : ingestion, retrieval et génération avec une interface de provider ;
- `api/with-fastapi` : API prédictive typée, validation d'entrée et health check.
