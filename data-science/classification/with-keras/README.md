# Classification avec Keras — sentiment de verbatims

## Cas d'utilisation

Une équipe produit classe des verbatims clients en `positive`, `neutral` ou `negative` pour détecter les irritants. Ce projet utilise l'API Keras avec une vectorisation intégrée au modèle, ce qui simplifie le packaging de l'inférence.

```bash
pip install -e ".[dev]"
pytest
python -m sentiment_keras.cli
```

Le corpus est synthétique et en anglais afin de rester compact. Pour un usage réel : split par utilisateur et par temps, analyse des langues, tests de robustesse et validation humaine des labels.
