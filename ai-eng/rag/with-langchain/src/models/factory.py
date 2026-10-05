"""Model factory of the **LangChain (LCEL)** stack.

Le retrieval reste BM25 — la référence de la famille, et le concurrent à battre — mais la
réponse n'est plus produite par le seul retriever : un graphe LCEL assemble le contexte, le
prompt et l'appel au générateur, puis rattache la réponse aux passages qui la fondent. L'intérêt
de cette variante n'est pas le score de recherche (identique à celui de `with-tfidf`) mais ce
qu'elle rend explicite et testable : le prompt est un objet de configuration, l'adaptateur LLM
est une couture (extractive hors ligne par défaut, client distant optionnel), les citations
traversent toute la chaîne et la précision des citations devient une métrique qu'on peut suivre.
Les deux autres stratégies de la stack — embeddings hachés et fusion RRF des classements —
existent pour mesurer, sur le split de validation, ce que le sémantique et l'hybride apportent
réellement avant de payer leur coût.

The factory is the single place where the configuration becomes an object:

* :func:`build_model` reads the ``model`` node (algorithm, hyper-parameters, seed) and returns a
  :class:`~src.models.model.LangChainModel` respecting the :class:`~src.models.base.BaseModel`
  contract,
* :func:`load_model` reloads a persisted index whatever the caller passes (file or directory) and
  rebuilds the LCEL chain around it,
* :func:`available_algorithms` lists the retrieval strategies this stack can serve — that list is
  what notebook 04 iterates over instead of hard-coding a name.

Algorithms are declared in :data:`ALGORITHMS` and selected here: adding a strategy is a registry
entry, not a code change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.base import BaseModel, FitResult, ModelCard
from src.models.model import LangChainModel
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Task learned by this project (injected from the manifest).
PROJECT_TASK = "retrieval"

#: Algorithm selected in `conf/model/default.yaml`.
CONFIGURED_ALGORITHM = "langchain_lexical"

#: Model class of this stack (documented in the README and the configuration).
MODEL_CLASS = "LangChainModel"

#: Artefact written by this stack (injected from `registry/stacks.yaml`).
MODEL_FILE = "rag_chain.joblib"

#: Extensions understood by :func:`load_model`.
SUPPORTED_SUFFIXES: tuple[str, ...] = (".joblib", ".pkl", ".pickle")

#: Lexical scorings understood by the lexical arm of the index.
LEXICAL_MODES: tuple[str, ...] = ("bm25", "tfidf")


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """Declaration of one retrieval strategy available in this stack.

    Attributes:
        name: Stable identifier used in ``conf/model/default.yaml`` and the notebooks.
        display_name: Human readable name (reports, notebooks).
        scoring: Ranking strategy (``bm25``, ``tfidf``, ``dense`` or ``hybrid``).
        default_params: Index parameters applied when the configuration is silent.
        rationale: Why (and when) to pick this strategy.
    """

    name: str
    display_name: str
    scoring: str
    default_params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


#: Registry of the algorithms served by this stack.
ALGORITHMS: dict[str, AlgorithmSpec] = {
    "langchain_lexical": AlgorithmSpec(
        name="langchain_lexical",
        display_name="Chaîne LCEL + retriever lexical BM25",
        scoring="bm25",
        default_params={
            "lexical": {"k1": 1.5, "b": 0.75, "ngram_range": (1, 1), "min_df": 1},
            "rrf_constant": 60,
        },
        rationale=(
            "Le même index BM25 que la baseline de la famille, mais servi par le graphe LCEL : "
            "prompt de synthèse, adaptateur LLM et réponse fondée sur les passages retrouvés. "
            "C'est la variante qui isole ce qu'apporte l'orchestration, à retrieval constant."
        ),
    ),
    "langchain_dense": AlgorithmSpec(
        name="langchain_dense",
        display_name="Chaîne LCEL + embeddings hachés (SVD)",
        scoring="dense",
        default_params={
            "lexical": {"mode": "bm25", "k1": 1.5, "b": 0.75, "ngram_range": (1, 1)},
            "embedding": {"n_features": 4096, "n_components": 192},
            "rrf_constant": 60,
        },
        rationale=(
            "Un espace dense appris sur le train (n-grammes hachés compressés par une SVD "
            "tronquée) : il rapproche des formulations qui ne partagent pas de mot, sans "
            "télécharger de modèle. C'est là que se joue le segment « paraphrase »."
        ),
    ),
    "langchain_hybrid": AlgorithmSpec(
        name="langchain_hybrid",
        display_name="Chaîne LCEL + recherche hybride (RRF)",
        scoring="hybrid",
        default_params={
            "lexical": {"mode": "bm25", "k1": 1.5, "b": 0.75, "ngram_range": (1, 1)},
            "embedding": {"n_features": 4096, "n_components": 192},
            "rrf_constant": 60,
        },
        rationale=(
            "Fusion par rangs réciproques des deux classements : le lexical garde les "
            "correspondances exactes de termes, le dense rattrape les paraphrases. La fusion se "
            "fait sur les rangs, jamais sur les scores (BM25 n'est pas borné, un cosinus si)."
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
    seed = int(node.get("random_state") or root.get("seed", 42))

    configured = dict(node.get("params") or {})
    effective = dict(configured) if params is None else dict(params)
    effective["lexical"] = _lexical_params(spec, configured, effective)
    effective["embedding"] = _embedding_params(spec, configured, effective, seed=seed)
    effective.setdefault("rrf_constant", int(spec.default_params.get("rrf_constant", 60)))
    effective.setdefault("model_file", MODEL_FILE)

    model = LangChainModel(
        algorithm=selected,
        params=effective,
        task=task,
        random_state=seed,
        config=root,
        name=str(node.get("name") or spec.display_name),
    )
    logger.debug("Model built | {} | index={}", model.summary(), spec.scoring)
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
            f"LangChain (LCEL) stack "
            f"(expected one of {sorted(SUPPORTED_SUFFIXES)})"
        )
        raise ValueError(msg)
    return LangChainModel.load(source, config=config)


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


def _lexical_params(
    spec: AlgorithmSpec,
    configured: Mapping[str, Any],
    effective: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the parameters of the lexical arm (mode included)."""
    lexical: dict[str, Any] = dict(spec.default_params.get("lexical") or {})
    lexical.update(dict(configured.get("lexical") or {}))
    lexical.update(dict(effective.get("lexical") or {}))
    lexical["mode"] = spec.scoring if spec.scoring in LEXICAL_MODES else lexical.get("mode", "bm25")
    if lexical["mode"] not in LEXICAL_MODES:
        msg = f"lexical.mode must be one of {list(LEXICAL_MODES)}, got '{lexical['mode']}'"
        raise ValueError(msg)
    return lexical


def _embedding_params(
    spec: AlgorithmSpec,
    configured: Mapping[str, Any],
    effective: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Resolve the parameters of the dense arm (hashing space, SVD width, seed)."""
    embedding: dict[str, Any] = dict(spec.default_params.get("embedding") or {})
    embedding.update(dict(configured.get("embedding") or {}))
    embedding.update(dict(effective.get("embedding") or {}))
    embedding.setdefault("n_features", 4096)
    embedding.setdefault("n_components", 192)
    embedding["random_state"] = seed
    return embedding


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
    "LEXICAL_MODES",
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
