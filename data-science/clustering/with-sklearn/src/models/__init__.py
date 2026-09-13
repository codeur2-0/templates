"""Model layer of the **scikit-learn** stack.

Public surface, imported everywhere else in the project::

    from src.models import build_model, load_model          # fabrique + rechargement
    from src.models.base import BaseModel, FitResult        # contrat et résultat d'entraînement
    from src.models.factory import available_algorithms     # algorithmes servis par la stack

Nothing outside this package imports a scikit-learn class: the framework stays an implementation
detail behind :class:`~src.models.base.BaseModel` (dependency inversion).
"""

from src.models.base import PROBABILITY_TASKS, SUPERVISED_TASKS, BaseModel, FitResult, ModelCard
from src.models.factory import (
    available_algorithms,
    build_model,
    describe_algorithm,
    load_model,
)
from src.models.model import ESTIMATORS, SklearnModel, resolve_algorithm

__all__ = [
    "ESTIMATORS",
    "PROBABILITY_TASKS",
    "SUPERVISED_TASKS",
    "BaseModel",
    "FitResult",
    "ModelCard",
    "SklearnModel",
    "available_algorithms",
    "build_model",
    "describe_algorithm",
    "load_model",
    "resolve_algorithm",
]
