# Classification avec spaCy — routage de tickets

## Cas d'utilisation

Le support reçoit des messages courts et veut les router vers `billing`, `technical` ou `account`. L'exemple utilise le composant `textcat` de spaCy. Il illustre le pré-traitement texte, le contrat de données et l'entraînement d'un composant NLP sans mélanger logique métier et framework.

```bash
pip install -e ".[dev]"
pytest
python -m ticket_spacy.cli
```

Le dataset est synthétique. Pour la production, mesurer macro-F1 par équipe, gérer les tickets multilingues, analyser les erreurs et prévoir une file de revue humaine pour les prédictions peu confiantes.
