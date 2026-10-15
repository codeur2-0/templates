"""Model factory of the **scikit-learn (TF-IDF / BM25)** stack.

BM25 est le score de référence de la recherche documentaire : rareté du terme, saturation de sa
fréquence dans le passage et normalisation par la longueur. Sur un corpus d'entreprise structuré
— un vocabulaire métier stable, des formulations répétées d'une édition à l'autre — il est très
difficile à battre, et il offre trois propriétés qu'un modèle dense n'offre pas gratuitement :
il est instantané, il s'explique terme par terme, et il n'a besoin d'aucun entraînement ni
d'aucun téléchargement. Ses limites sont mesurées plutôt que cachées : le segment « paraphrase »
de l'évaluation chiffre exactement ce qu'il ne sait pas faire, et c'est cette mesure qui
justifie (ou non) de payer un embedding dense dans la variante LangChain.

The factory is the single place where the configuration becomes an object:

* :func:`build_model` reads the ``model`` node (algorithm, hyper-parameters, seed) and returns a
  :class:`~src.models.model.TfidfModel` respecting the :class:`~src.models.base.BaseModel`
  contract,
* :func:`load_model` reloads a persisted index whatever the caller passes (file or directory),
* :func:`available_algorithms` lists the scorings this stack can serve, which is what notebook 04
  iterates over instead of hard-coding a name.

Algorithms are declared in :data:`~src.models.model` and selected here: adding a scoring is a
registry entry, not a code change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.base import BaseModel, FitResult, ModelCard
from src.models.model import TfidfModel
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Task learned by this project (injected from the manifest).
PROJECT_TASK = "retrieval"

#: Algorithm selected in `conf/model/default.yaml`.
CONFIGURED_ALGORITHM = "bm25"

#: Model class of this stack (documented in the README and the configuration).
MODEL_CLASS = "TfidfModel"

#: Artefact written by this stack (injected from `registry/stacks.yaml`).
MODEL_FILE = "lexical_index.joblib"

#: Extensions understood by :func:`load_model`.
SUPPORTED_SUFFIXES: tuple[str, ...] = (".joblib", ".pkl", ".pickle")


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """Declaration of one lexical scoring available in this stack.

    Attributes:
        name: Stable identifier used in ``conf/model/default.yaml`` and the notebooks.
        display_name: Human readable name (reports, notebooks).
        scoring: Value handed to :class:`src.preprocessing.transformers.LexicalVectorizer`.
        default_params: Lexical parameters applied when the configuration is silent.
        rationale: Why (and when) to pick this scoring.
    """

    name: str
    display_name: str
    scoring: str
    default_params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


#: Registry of the algorithms served by this stack.
ALGORITHMS: dict[str, AlgorithmSpec] = {
    "bm25": AlgorithmSpec(
        name="bm25",
        display_name="BM25 (Okapi)",
        scoring="bm25",
        default_params={"k1": 1.5, "b": 0.75, "ngram_range": (1, 1), "min_df": 1},
        rationale=(
            "Saturation de la fréquence des termes et normalisation par la longueur du passage : "
            "le meilleur compromis sur des passages de tailles inégales."
        ),
    ),
    "tfidf_cosine": AlgorithmSpec(
        name="tfidf_cosine",
        display_name="TF-IDF + similarité cosinus",
        scoring="tfidf",
        default_params={"ngram_range": (1, 1), "min_df": 1, "sublinear_tf": True},
        rationale=(
            "Similarité cosinus sur des vecteurs TF-IDF normalisés : plus simple à expliquer, "
            "mais pénalise les passages longs qui contiennent davantage de termes de la question."
        ),
    ),
}


def build_model(
    config: Mapping[str, Any] | Any,
    *,
    algorithm: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> BaseModel:
    """Build the model declared by the configuration.

    ``params`` has three meanings, on purpose:

    * ``None`` — use the hyper-parameters of ``conf/model/default.yaml`` (production path),
    * ``{}`` — use the registry defaults (loyal comparison between algorithms in notebook 04),
    * a mapping — use it as-is (hyper-parameter grid).

    Args:
        config: Application configuration (mapping or :class:`~src.schemas.config.AppConfig`).
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
    lexical = dict(spec.default_params)
    lexical.update(dict(configured.get("lexical") or {}))
    lexical.update(dict(effective.get("lexical") or {}))
    lexical["mode"] = spec.scoring
    effective["lexical"] = lexical
    effective.setdefault("model_file", MODEL_FILE)
    seed = int(node.get("random_state") or root.get("seed", 42))

    model = TfidfModel(
        algorithm=selected,
        params=effective,
        task=task,
        random_state=seed,
        config=root,
        name=str(node.get("name") or spec.display_name),
    )
    logger.debug("Model built | {} | lexical={}", model.summary(), lexical)
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
            f"scikit-learn (TF-IDF / BM25) stack "
            f"(expected one of {sorted(SUPPORTED_SUFFIXES)})"
        )
        raise ValueError(msg)
    return TfidfModel.load(source, config=config)


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
    """Return the documentation of one algorithm (used by the reports and notebooks).

    Args:
        name: Algorithm identifier.

    Returns:
        A mapping with ``name``, ``display_name``, ``scoring``, ``rationale`` and ``defaults``.

    Raises:
        ValueError: When the algorithm is unknown.
    """
    spec = _algorithm_spec(str(name))
    return {
        "name": spec.name,
        "display_name": spec.display_name,
        "scoring": spec.scoring,
        "rationale": spec.rationale,
        "defaults": dict(spec.default_params),
    }


def _algorithm_spec(name: str) -> AlgorithmSpec:
    """Return the specification of an algorithm, or raise a helpful error."""
    if name not in ALGORITHMS:
        msg = f"Unknown algorithm '{name}'. Available: {sorted(ALGORITHMS)}"
        raise ValueError(msg)
    return ALGORITHMS[name]


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


def supported_tasks() -> Sequence[str]:
    """Return the learning tasks this stack serves (documented in the README)."""
    return ("retrieval", "generation")


__all__ = [
    "ALGORITHMS",
    "AlgorithmSpec",
    "CONFIGURED_ALGORITHM",
    "MODEL_CLASS",
    "MODEL_FILE",
    "PROJECT_TASK",
    "SUPPORTED_SUFFIXES",
    "FitResult",
    "ModelCard",
    "available_algorithms",
    "build_model",
    "describe_algorithm",
    "load_model",
    "supported_tasks",
]
