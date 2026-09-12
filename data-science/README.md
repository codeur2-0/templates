# Data Science

Projets de modélisation supervisée, de la donnée au modèle évalué. Les exemples utilisent le même contrat d'entrée : validation Pandera, split reproductible, pipeline de features et métriques métier. Les variantes montrent comment garder le même design avec des frameworks différents.

## Classification

- **sklearn** : churn client tabulaire, baseline de production ;
- **pytorch** : détection de fraude, boucle d'entraînement explicite ;
- **spacy** : routage de tickets support, texte et NER-friendly ;
- **tensorflow** : risque de défaut, modèle Keras via TensorFlow ;
- **keras** : sentiment de verbatims, API Keras moderne.

Chaque implémentation est volontairement petite : remplacez `data/raw` par votre source, conservez le schéma et faites évoluer les tests avant les features.
