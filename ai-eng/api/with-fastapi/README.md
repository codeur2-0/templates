# API AI avec FastAPI — scoring de risque

## Cas d'utilisation

Un service interne expose un score de risque explicable à d'autres applications. L'exemple est volontairement un scorer déterministe : il démontre le contrat HTTP, la validation Pandera en batch, le health check et une séparation `ModelService` / transport FastAPI.

```bash
pip install -e ".[dev]"
pytest
uvicorn risk_api.main:app --host 0.0.0.0 --port 8000
# POST /v1/predict avec {"income": 50000, "loan_amount": 8000, "credit_score": 700}
```

Un exemple de payload est fourni dans `data/raw/example_request.json`. Les requêtes ne contiennent pas d'identifiant client. En production, ajouter authN/authZ, rate limiting, correlation ID, modèle versionné, monitoring et validation de fairness.
