"""Model factory of the **scikit-learn** stack.

Le gradient boosting par histogrammes est le meilleur rapport précision/coût de scikit-learn
pour scorer des couples (utilisateur, candidat) : il apprend sans transformation les
**interactions** qui font la pertinence (affinité de catégorie x niveau de prix x intention
récente), il est **natif valeurs manquantes** (un nouvel article sans historique de notation ne
casse pas la publication), il traite les features catégorielles du catalogue sans one-hot massif
— un encodage one-hot de 900 références exploserait la dimension et noierait le signal
d'affinité — et il reste déterministe à graine fixée, ce qui est une exigence pour un moteur
dont on audite les listes publiées. Ses limites sont documentées ici plutôt que cachées : il
apprend un score **ponctuel** par couple et non une préférence relative, si bien qu'il
n'optimise pas directement le NDCG (une perte de classement comme LambdaMART serait plus
alignée) ; il ne construit aucune représentation latente partagée, donc il ne généralise pas à
un utilisateur ou à un article jamais vu en entraînement autrement que par leurs attributs ; et
il est sensible au déséquilibre de classes, la pertinence ne concernant qu'environ 14 % des
candidats. La comparaison au notebook 04 avec une régression logistique (linéaire, lisible, mais
aveugle aux interactions), une forêt aléatoire et un k-plus-proches voisins sur les embeddings
de catégorie montre ce que l'on gagne et ce que l'on perd.

The factory is the single place where the configuration becomes an object:

* :func:`build_model` reads the ``model`` node (algorithm, hyper-parameters, seed) and returns a
  :class:`~src.models.model.SklearnModel` respecting the :class:`~src.models.base.BaseModel`
  contract,
* :func:`load_model` reloads a persisted artefact whatever its format,
* :func:`available_algorithms` lists the algorithms this stack can serve for a task — the notebooks
  use it to compare candidates without hard-coding a name.

Algorithms are declared in :data:`~src.models.model.ESTIMATORS`; this module only selects and
configures them, so adding an algorithm is a one-line registry change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.models.base import BaseModel, FitResult, ModelCard
from src.models.model import (
    ESTIMATORS,
    AlgorithmSpec,
    SklearnModel,
    available_for_task,
    resolve_algorithm,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Tâche apprise par ce projet (injectée depuis le manifeste).
PROJECT_TASK = "ranking"

#: Algorithme retenu dans `conf/model/default.yaml`.
CONFIGURED_ALGORITHM = "hist_gradient_boosting"

#: Classe de modèle de cette stack (documentée dans le README et la configuration).
MODEL_CLASS = "SklearnModel"

#: Artefact écrit par cette stack (injecté depuis `registry/stacks.yaml`).
MODEL_FILE = "model.joblib"

#: Extensions reconnues par :func:`load_model`.
SUPPORTED_SUFFIXES: tuple[str, ...] = (
    ".joblib",
    ".joblib",
    ".pkl",
    ".pickle",
)


def build_model(
    config: Mapping[str, Any] | Any,
    *,
    feature_names: Sequence[str] | None = None,
    algorithm: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> BaseModel:
    """Build the model declared by the configuration.

    ``params`` has three meanings, on purpose:

    * ``None`` — use the hyper-parameters of ``conf/model/default.yaml`` (production path),
    * ``{}`` — use the estimator defaults (loyal comparison between algorithms in notebook 04),
    * a mapping — use it as-is (hyper-parameter grid).

    Args:
        config: Application configuration (mapping or :class:`~src.schemas.config.AppConfig`).
        feature_names: Features expected at inference time.
        algorithm: Algorithm override (defaults to ``model.algorithm``).
        params: Hyper-parameter override.

    Returns:
        An unfitted :class:`BaseModel`.

    Raises:
        ValueError: When the configuration has no ``model`` node, or the algorithm is unknown.
    """
    node = _model_node(config)
    root = _as_mapping(config)
    selected = str(algorithm or node.get("algorithm") or CONFIGURED_ALGORITHM)
    task = str(node.get("task") or dict(root.get("metrics") or {}).get("task") or PROJECT_TASK)
    resolve_algorithm(selected, task)  # échoue tôt, avec un message actionnable

    effective = dict(node.get("params") or {}) if params is None else dict(params)
    seed = node.get("random_state", None)
    model = SklearnModel(
        algorithm=selected,
        params=effective,
        task=task,
        feature_names=[str(name) for name in (feature_names or [])],
        target_name=_target_name(root),
        random_state=int(seed if seed is not None else root.get("seed", 42)),
        config=root,
        name=str(node.get("name") or selected),
    )
    logger.debug(
        "Model built | {} | params={}",
        model.summary(),
        {key: model.params[key] for key in list(model.params)[:6]},
    )
    return model


def load_model(path: str | Path, *, config: Mapping[str, Any] | None = None) -> BaseModel:
    """Reload a model artefact written by :meth:`BaseModel.save`.

    Args:
        path: Artefact path (``.joblib`` for this stack).
        config: Optional configuration refreshed into the reloaded model.

    Returns:
        The reloaded model, ready to predict.

    Raises:
        FileNotFoundError: When the artefact does not exist.
        ValueError: When the file extension is not handled by this stack.
        TypeError: When the artefact does not contain a model of this contract.
    """
    source = Path(path)
    if not source.exists():
        msg = f"Model artefact not found: {source}"
        raise FileNotFoundError(msg)
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        msg = (
            f"Unsupported model artefact '{source.name}' for the scikit-learn stack "
            f"(expected one of {sorted(set(SUPPORTED_SUFFIXES))})"
        )
        raise ValueError(msg)
    return SklearnModel.load(source, config=config)


def available_algorithms(task: str | None = None) -> list[str]:
    """List the algorithms this stack can serve for a task.

    Args:
        task: Learning task; ``None`` returns every algorithm of the registry.

    Returns:
        The algorithm identifiers, in registry order.
    """
    return available_for_task(str(task) if task else "")


def describe_algorithm(name: str) -> dict[str, Any]:
    """Return the documentation of one algorithm (used by the reports and notebooks).

    Args:
        name: Algorithm identifier.

    Returns:
        A mapping with ``name``, ``display_name``, ``tasks``, ``rationale`` and ``defaults``.

    Raises:
        ValueError: When the algorithm is unknown.
    """
    spec: AlgorithmSpec = ESTIMATORS.get(str(name)) or _unknown(name)
    return {
        "name": spec.name,
        "display_name": spec.display_name,
        "tasks": sorted(spec.tasks),
        "rationale": spec.rationale,
        "defaults": dict(spec.defaults),
        "iterative": spec.iterative,
    }


def _unknown(name: str) -> AlgorithmSpec:
    """Raise a helpful error for an unknown algorithm name."""
    msg = f"Unknown algorithm '{name}'. Available: {sorted(ESTIMATORS)}"
    raise ValueError(msg)


def _as_mapping(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Normalise a configuration object (pydantic model, OmegaConf or mapping) to a dict."""
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    if hasattr(config, "to_container"):  # OmegaConf DictConfig
        return dict(config.to_container(resolve=True))
    return dict(config or {})


def _model_node(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Extract the ``model`` node of a configuration.

    Args:
        config: Application configuration.

    Returns:
        The model node.

    Raises:
        ValueError: When no ``model`` node is present.
    """
    root = _as_mapping(config)
    node = root.get("model")
    if not isinstance(node, Mapping) or not node:
        msg = (
            "No 'model' node in the configuration: build_model needs at least "
            "model.algorithm (see conf/model/default.yaml)."
        )
        raise ValueError(msg)
    return dict(node)


def _target_name(config: Mapping[str, Any]) -> str | None:
    """Read the business target column (``None`` for an unsupervised task)."""
    node = dict(config.get("data") or {})
    target = node.get("target")
    return str(target) if target else None


__all__ = [
    "CONFIGURED_ALGORITHM",
    "MODEL_CLASS",
    "MODEL_FILE",
    "PROJECT_TASK",
    "FitResult",
    "ModelCard",
    "available_algorithms",
    "build_model",
    "describe_algorithm",
    "load_model",
]
