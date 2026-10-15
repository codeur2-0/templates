"""Model factory of the **Index vectoriel dense (hachage + SVD)** stack.

Le hachage de n-grammes projette un texte dans un espace de dimension fixe sans apprendre de
vocabulaire — un document jamais vu pendant l'entraînement reste représentable — et la SVD
tronquée, apprise sur les passages d'entraînement uniquement, compresse cet espace en 128
dimensions denses qui captent la cooccurrence. Les vecteurs sont normalisés en norme L2 : la
similarité cosinus devient un produit matriciel (donc un débit mesurable), l'index tient en
quelques centaines de kilo-octets et le score remappé dans ``[0, 1]`` rend le seuil d'abstention
calibrable — c'est ce qui permet de refuser 4 des 7 questions hors corpus du split de test. Ses
propriétés sont mesurées : la fidélité de projection vaut 1,0000 à 128 dimensions (l'ordre des
similarités est celui de l'espace de hachage brut, sur les paires testées) pour un index trente-
deux fois plus petit, et la fiche de modèle publie le chiffre. Ses limites sont mesurées de la
même façon : un hachage de n-grammes reste un index de mots — sur ce corpus, le dense gagne le
classement (MRR 0,96 contre 0,91) mais l'index lexical garde la précision des citations (0,97
contre 0,94) et refuse 3 des 7 questions hors corpus : aucun des deux n'est « meilleur » dans
l'absolu.

The factory is the single place where the configuration becomes an object:

* :func:`build_model` reads the ``model`` node (algorithm, embedding parameters, seed) and returns
  an :class:`~src.models.model.EmbeddingModel` respecting the
  :class:`~src.models.base.BaseModel` contract,
* :func:`load_model` reloads a persisted vector index whatever the caller passes (file or
  directory),
* :func:`available_algorithms` lists the two index shapes this stack can serve, which is what
  notebook 04 iterates over instead of hard-coding a name.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.base import BaseModel
from src.models.model import EmbeddingModel
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Task learned by this project (injected from the manifest).
PROJECT_TASK = "retrieval"

#: Algorithm selected in `conf/model/default.yaml`.
CONFIGURED_ALGORITHM = "hashing_svd"

#: Model class of this stack (documented in the README and the configuration).
MODEL_CLASS = "EmbeddingModel"

#: Artefact written by this stack (injected from `registry/stacks.yaml`).
MODEL_FILE = "vector_index.joblib"

#: Extensions understood by :func:`load_model`.
SUPPORTED_SUFFIXES: tuple[str, ...] = (".joblib", ".pkl", ".pickle")


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """Declaration of one index shape available in this stack.

    Attributes:
        name: Stable identifier used in ``conf/model/default.yaml`` and the notebooks.
        display_name: Human readable name (reports, notebooks).
        n_components: Number of dimensions kept by the truncated SVD (``0`` keeps the hashed space
            untouched, which is exact but wide).
        default_params: Embedding parameters applied when the configuration is silent.
        rationale: Why (and when) to pick this shape.
    """

    name: str
    display_name: str
    n_components: int
    default_params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


#: Registry of the algorithms served by this stack.
ALGORITHMS: dict[str, AlgorithmSpec] = {
    "hashing_svd": AlgorithmSpec(
        name="hashing_svd",
        display_name="Hachage 4096 -> SVD 128 dimensions",
        n_components=128,
        default_params={"n_features": 4096, "ngram_range": (1, 2)},
        rationale=(
            "Compression linéaire apprise sur le corpus d'entraînement : l'index tient en quelques "
            "centaines de kilo-octets et la similarité reste un produit matriciel."
        ),
    ),
    "hashing_raw": AlgorithmSpec(
        name="hashing_raw",
        display_name="Hachage 4096 dimensions (sans projection)",
        n_components=0,
        default_params={"n_features": 4096, "ngram_range": (1, 2)},
        rationale=(
            "Espace de hachage conservé tel quel : fidélité de 1,0 par construction, mais un index "
            "large et un voisinage noyé par les collisions de hachage. C'est la référence qui "
            "justifie (ou non) la projection."
        ),
    ),
}


def _as_mapping(config: Any) -> dict[str, Any]:
    """Return a plain mapping whatever the caller passed (mapping or pydantic model)."""
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return {str(key): value for key, value in config.items()}
    dump = getattr(config, "model_dump", None)
    if callable(dump):
        return {str(key): value for key, value in dump().items()}
    msg = f"Unsupported configuration type: {type(config).__name__}"
    raise TypeError(msg)


def _model_node(config: Any) -> dict[str, Any]:
    """Return the ``model`` node of the configuration as a plain mapping.

    Args:
        config: Application configuration (mapping or pydantic model).

    Returns:
        The ``model`` node.

    Raises:
        ValueError: When the configuration has no ``model`` node.
    """
    node = _as_mapping(config).get("model")
    if not isinstance(node, Mapping):
        msg = "The configuration must declare a 'model' node"
        raise ValueError(msg)
    return {str(key): value for key, value in node.items()}


def _algorithm_spec(name: str) -> AlgorithmSpec:
    """Return the specification of an algorithm, with a readable error when unknown."""
    spec = ALGORITHMS.get(str(name))
    if spec is None:
        msg = (
            f"Unknown algorithm '{name}' for the Index vectoriel dense (hachage + SVD) stack "
            f"(expected one of {sorted(ALGORITHMS)})"
        )
        raise ValueError(msg)
    return spec


def build_model(
    config: Mapping[str, Any] | Any,
    *,
    algorithm: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> BaseModel:
    """Build the model declared by the configuration.

    ``params`` has three meanings, on purpose:

    * ``None`` — use the parameters of ``conf/model/default.yaml`` (production path),
    * ``{}`` — use the registry defaults (loyal comparison between algorithms in notebook 04),
    * a mapping — use it as-is (parameter grid).

    Args:
        config: Application configuration (mapping or pydantic model).
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
    spec = _algorithm_spec(selected)
    task = str(node.get("task") or dict(root.get("metrics") or {}).get("task") or PROJECT_TASK)

    configured = dict(node.get("params") or {})
    effective = dict(configured) if params is None else dict(params)
    embedding = dict(spec.default_params)
    embedding.update(dict(configured.get("embedding") or {}))
    embedding.update(dict(effective.get("embedding") or {}))
    embedding["n_components"] = spec.n_components
    effective["embedding"] = embedding
    effective.setdefault("model_file", MODEL_FILE)
    seed = int(node.get("random_state") or root.get("seed", 42))

    model = EmbeddingModel(
        algorithm=selected,
        params=effective,
        task=task,
        random_state=seed,
        config=root,
        name=str(node.get("name") or spec.display_name),
    )
    logger.debug("Model built | {} | embedding={}", model.summary(), embedding)
    return model


def load_model(path: str | Path, *, config: Mapping[str, Any] | None = None) -> BaseModel:
    """Reload a model artefact written by :meth:`BaseModel.save`.

    Args:
        path: Artefact path (a directory receives the default file name).
        config: Optional configuration refreshed into the reloaded model.

    Returns:
        The reloaded model, ready to retrieve.

    Raises:
        FileNotFoundError: When the artefact does not exist.
        ValueError: When the file extension is not handled by this stack.
    """
    source = Path(path)
    if source.is_dir() or not source.suffix:
        source = source / MODEL_FILE
    if not source.exists():
        msg = f"Model artefact not found: {source}"
        raise FileNotFoundError(msg)
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        msg = (
            f"Unsupported model artefact '{source.name}' for the "
            f"Index vectoriel dense (hachage + SVD) stack "
            f"(expected one of {sorted(SUPPORTED_SUFFIXES)})"
        )
        raise ValueError(msg)
    return EmbeddingModel.load(source, config=config)


def available_algorithms(task: str | None = None) -> list[str]:
    """List the algorithms this stack can serve.

    Args:
        task: Learning task; only ``retrieval`` (and its aliases) is served by this stack.

    Returns:
        The algorithm identifiers, in registry order.
    """
    if task and task not in {"retrieval", "generation", PROJECT_TASK}:
        return []
    return list(ALGORITHMS)


def describe_algorithm(name: str) -> dict[str, Any]:
    """Return the documentation of one algorithm (used by the reports and notebooks)."""
    spec = _algorithm_spec(name)
    return {
        "name": spec.name,
        "display_name": spec.display_name,
        "n_components": spec.n_components,
        "default_params": dict(spec.default_params),
        "rationale": spec.rationale,
    }


def supported_tasks(task: str | None = None) -> bool:
    """Whether the stack serves a task (kept as a function for consistency across stacks)."""
    return not task or task in {"retrieval", "generation", PROJECT_TASK}


__all__ = [
    "ALGORITHMS",
    "CONFIGURED_ALGORITHM",
    "MODEL_CLASS",
    "MODEL_FILE",
    "AlgorithmSpec",
    "available_algorithms",
    "build_model",
    "describe_algorithm",
    "load_model",
    "supported_tasks",
]
