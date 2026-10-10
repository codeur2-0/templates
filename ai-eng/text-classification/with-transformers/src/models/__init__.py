"""Couche modèles de la stack **Hugging Face Transformers (entraîné sur le corpus)**.

Surface publique, importée partout ailleurs dans le projet::

    from src.models import build_model, load_model        # fabrique + rechargement
    from src.models.contract import BaseTextClassifier    # contrat et objets de retour
    from src.models.factory import available_algorithms   # architectures servies par la stack

Rien en dehors de ce paquet n'importe PyTorch ni `transformers` : le framework reste un détail
d'implémentation derrière :class:`~src.models.contract.BaseTextClassifier` (inversion de
dépendance), ce qui permet de comparer un encodeur contextuel et un modèle lexical sur le même
corpus sans toucher au pipeline.
"""

from src.models.base import FitResult, ModelCard
from src.models.classifier import TransformerTextClassifier
from src.models.contract import BaseTextClassifier, TextPrediction, softmax
from src.models.factory import (
    ALGORITHMS,
    AlgorithmSpec,
    available_algorithms,
    build_model,
    describe_algorithm,
    load_model,
    supported_tasks,
)

__all__ = [
    "ALGORITHMS",
    "AlgorithmSpec",
    "BaseTextClassifier",
    "FitResult",
    "ModelCard",
    "TextPrediction",
    "TransformerTextClassifier",
    "available_algorithms",
    "build_model",
    "describe_algorithm",
    "load_model",
    "softmax",
    "supported_tasks",
]
