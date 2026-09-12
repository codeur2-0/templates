# Classification avec PyTorch — fraude transactionnelle

## Cas d'utilisation

Une plateforme veut filtrer des transactions au moment du paiement. Le dataset synthétique décrit montant, vitesse des transactions et risque du pays. Le coût d'un faux négatif est prioritaire ; le seuil d'alerte doit donc être calibré avec l'équipe fraude.

## Lancer

```bash
pip install -e ".[dev]"                 # CPU
pytest
python -m fraud_pytorch.cli
python -m fraud_pytorch.cli training.epochs=20 training.batch_size=4
```

Le code sépare `TransactionDataLoader`, `FraudMLP`, `TorchTrainer` et `FraudTrainingPipeline`. Pandera valide le contrat en entrée, Hydra pilote les hyperparamètres et le seed est fixé pour les runs comparables.

## Limites

Le jeu est pédagogique : pas de feature temporelle, de calibration, d'imbalance strategy ni de protection contre le concept drift. En production, compléter par un split temporel, une file d'alerte et une décision human-in-the-loop.
