# Classification avec TensorFlow — risque de défaut

## Cas d'utilisation

Une équipe risque estime si une demande de prêt est susceptible de faire défaut. Les données d'exemple sont synthétiques et la cible `defaulted` sert uniquement à montrer le cycle TensorFlow : contrat, normalisation, réseau dense, callbacks et export.

```bash
pip install -e ".[dev]"
pytest
python -m loan_tensorflow.cli
```

Ne pas utiliser ce modèle pour une décision de crédit réelle sans analyse de biais, explicabilité, validation réglementaire, seuils validés et revue humaine.
