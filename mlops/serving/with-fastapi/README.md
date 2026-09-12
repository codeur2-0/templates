# Serving MLOps avec FastAPI — modèle versionné

## Cas d'utilisation

Ce service représente la dernière étape après promotion d'un modèle : il charge une version déclarée, vérifie chaque requête, expose `/health` et ajoute un `request_id` dans la réponse. Le score est une baseline locale pour que le projet soit autonome ; remplacer `RuleBasedModel` par un artefact du registry en conservant l'interface.

```bash
pip install -e ".[dev]"
pytest
uvicorn churn_service.main:app --host 0.0.0.0 --port 8000
```

Un payload de démonstration est fourni dans `data/raw/example_request.json`. Checklist production : readiness séparée de liveness, authentification, timeout, limites de payload, métriques Prometheus, traces, canary, rollback et tests de compatibilité du schéma.
