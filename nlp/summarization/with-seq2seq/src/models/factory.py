"""Fabrique des stratégies de résumé : l'encodeur-décodeur appris et les baselines extractives.

``build_model`` est le seul point d'entrée du projet pour construire une stratégie : les pipelines,
le trainer et les notebooks ne connaissent que :class:`~src.models.contract.BaseTextGenerator`.
Ajouter une architecture, c'est ajouter une entrée dans :data:`ALGORITHMS` et la classe
correspondante — jamais toucher un pipeline.

Les baselines ``lead`` et ``textrank`` vivent dans la couche de tâche du dépôt de templates,
pas dans la stack : elles ne dépendent d'aucun framework et servent de **références publiées**.
La fabrique les construit donc aussi, ce qui permet à la configuration de comparer trois
stratégies sur les mêmes lignes sans qu'aucun pipeline n'ait à savoir laquelle est apprise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from src.models.contract import BaseTextGenerator
from src.models.lead import LeadSummarizer
from src.models.summarizer import EncoderDecoderSummarizer
from src.models.textrank import TextRankSummarizer
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Algorithm served by the project when the configuration does not say (published in the manifest).
CONFIGURED_ALGORITHM = "transformer_tiny"

#: Task declared by the stack (checked against the configuration).
PROJECT_TASK = "generation"

#: Blocks of ``model.params`` that belong to the stack; foreign blocks are ignored.
PARAM_BLOCKS: tuple[str, ...] = ("training", "pretraining", "budget")

#: Budget keys forwarded to :class:`~src.models.contract.BaseTextGenerator` as keyword arguments.
BUDGET_KEYS: tuple[str, ...] = ("min_output_tokens", "max_output_tokens", "compression")

#: Training keys the model reads for itself, taken from the ``train`` node when absent elsewhere.
TRAIN_KEYS: tuple[str, ...] = ("epochs", "batch_size", "learning_rate", "logging_every")


@dataclass(frozen=True)
class AlgorithmSpec:
    """Description of one summarization strategy.

    Attributes:
        key: Name used in the configuration.
        display_name: Human name published in the model card and the reports.
        family: ``learned`` or ``extractive`` — the two families the project compares.
        requires_fit: Whether the strategy learns parameters (the extractive ones do not).
        notes: One sentence describing what the strategy does.
        default_params: Default parameters, by block (``training``, ``pretraining``, ``budget``).
    """

    key: str
    display_name: str
    family: str
    requires_fit: bool
    notes: str
    default_params: dict[str, dict[str, Any]] = field(default_factory=dict)


#: Strategies served by the stack, in reading order.
ALGORITHMS: dict[str, AlgorithmSpec] = {
    "transformer_tiny": AlgorithmSpec(
        key="transformer_tiny",
        display_name="Encodeur-décodeur transformer (2 couches, 128 unités)",
        family="learned",
        requires_fit=True,
        notes=(
            "Vocabulaire WordPiece partagé et poids appris sur le split d'entraînement, "
            "pré-entraînement dénoising puis affinage supervisé, décodage glouton."
        ),
        default_params={
            "training": {
                "layers": 2,
                "units": 128,
                "heads": 4,
                "dropout": 0.1,
                "max_input_tokens": 160,
                "vocab_size": 1200,
                "min_frequency": 2,
            },
            "pretraining": {"epochs": 2, "mask_rate": 0.25},
            "budget": {"compression": 0.45, "max_output_tokens": 90, "min_output_tokens": 12},
        },
    ),
    "textrank": AlgorithmSpec(
        key="textrank",
        display_name="Baseline extractive TextRank (graphe de phrases + MMR)",
        family="extractive",
        requires_fit=False,
        notes=(
            "Sélectionne des phrases du document, sans les réécrire : son texte est toujours "
            "fidèle, et son poids MMR est calibré sur le split de validation."
        ),
        default_params={"budget": {"compression": 0.45, "max_output_tokens": 90}},
    ),
    "lead": AlgorithmSpec(
        key="lead",
        display_name="Baseline extractive « premières phrases du document »",
        family="extractive",
        requires_fit=False,
        notes=(
            "Prend les phrases du début du document jusqu'au budget de longueur : le plancher que "
            "tout lecteur a en tête, publié pour être mesuré."
        ),
        default_params={"budget": {"compression": 0.45, "max_output_tokens": 90}},
    ),
}


def build_model(
    config: Mapping[str, Any] | Any,
    *,
    algorithm: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> BaseTextGenerator:
    """Build the summarization strategy declared by the configuration.

    ``params`` a deux sens :

    * ``None`` ou ``{}`` — les hyper-paramètres de ``conf/model/default.yaml``, complétés par les
      défauts du registre (chemin de production ; c'est ce que fait un notebook qui suit la
      configuration du projet) ;
    * un mapping — une surcharge **explicite**, appliquée après la configuration (grille
      d'hyper-paramètres : comparaison loyale entre deux valeurs d'un même réglage).

    Args:
        config: Application configuration (mapping or :class:`~src.schemas.config.AppConfig`).
        algorithm: Algorithm override (defaults to ``model.algorithm``).
        params: Hyper-parameter override.

    Returns:
        An unfitted :class:`~src.models.contract.BaseTextGenerator`.

    Raises:
        ValueError: When the configuration has no ``model`` node, or the algorithm is unknown.
    """
    node = _model_node(config)
    root = _as_mapping(config)
    selected = str(algorithm or node.get("algorithm") or CONFIGURED_ALGORITHM)
    spec = _algorithm_spec(selected)
    effective = _effective_params(spec, node, root, params)
    budget = dict(effective.get("budget") or {})
    base_budget = {key: budget[key] for key in BUDGET_KEYS if key in budget}
    seed = int(node.get("random_state") or root.get("seed", 42))
    if spec.key == "transformer_tiny":
        model: BaseTextGenerator = EncoderDecoderSummarizer(params=effective, seed=seed)
    elif spec.key == "textrank":
        model = TextRankSummarizer(
            mmr_lambda=float(budget.get("mmr_lambda", 0.75)),
            seed=seed,
            **base_budget,
        )
    else:
        model = LeadSummarizer(seed=seed, **base_budget)
    logger.debug("Stratégie construite | {} | famille={}", model.summary(), spec.family)
    return model


def supported_tasks() -> Sequence[str]:
    """Tasks this stack serves.

    Returns:
        ``("generation",)``.
    """
    return (PROJECT_TASK,)


def load_model(
    path: str | Path,
    config: Mapping[str, Any] | Any | None = None,
    **overrides: Any,
) -> BaseTextGenerator:
    """Reload a persisted strategy.

    Args:
        path: Artefact written by :meth:`~src.models.contract.BaseTextGenerator.save`.
        config: Optional configuration (its hyper-parameters take precedence over the artefact's, so
            that an artefact moved to another project obeys the new configuration).
        overrides: Values that take precedence over both.

    Returns:
        The restored strategy.

    Raises:
        FileNotFoundError: When the artefact does not exist.
        ValueError: When the artefact declares an unknown strategy.
    """
    target = Path(path)
    if not target.is_file():
        msg = (
            f"Model artefact not found: {target}. Run `make train` (or "
            "`python -m src.main mode=train`) to produce it."
        )
        raise FileNotFoundError(msg)
    strategy = str(_peek_payload(target).get("strategy") or "")
    mapping: dict[str, type[BaseTextGenerator]] = {
        "transformer_tiny": EncoderDecoderSummarizer,
        "textrank": TextRankSummarizer,
        "lead": LeadSummarizer,
    }
    model_class = mapping.get(strategy)
    if model_class is None:
        msg = f"Unknown strategy '{strategy}' in {target}. Known: {sorted(mapping)}"
        raise ValueError(msg)
    if config is not None:
        node = _model_node(config)
        root = _as_mapping(config)
        effective = _effective_params(_algorithm_spec(strategy), node, root, None)
        budget = dict(effective.get("budget") or {})
        overrides = {
            "params": effective,
            "seed": int(node.get("random_state") or root.get("seed", 42)),
            **{key: budget[key] for key in BUDGET_KEYS if key in budget},
            **overrides,
        }
    return model_class.load(target, **overrides)


def available_algorithms(task: str | None = None) -> list[str]:
    """List the strategies the stack serves.

    Args:
        task: Optional task filter (the stack serves a single task).

    Returns:
        The algorithm names, sorted.
    """
    if task is not None and str(task) not in supported_tasks():
        return []
    return sorted(ALGORITHMS)


def describe_algorithm(name: str) -> dict[str, Any]:
    """Describe one strategy (used by the notebooks and the model card).

    Args:
        name: Algorithm name.

    Returns:
        The description of that algorithm.

    Raises:
        ValueError: When the name is unknown.
    """
    spec = _algorithm_spec(name)
    return {
        "key": spec.key,
        "display_name": spec.display_name,
        "family": spec.family,
        "requires_fit": spec.requires_fit,
        "notes": spec.notes,
        "default_params": spec.default_params,
    }


def _algorithm_spec(name: str) -> AlgorithmSpec:
    """Return the spec of an algorithm, with an explicit error when it is unknown.

    Args:
        name: Algorithm name.

    Returns:
        The algorithm specification.

    Raises:
        ValueError: When the name is unknown.
    """
    spec = ALGORITHMS.get(str(name))
    if spec is None:
        msg = f"Unknown algorithm '{name}'. Available: {sorted(ALGORITHMS)}"
        raise ValueError(msg)
    return spec


def _effective_params(
    spec: AlgorithmSpec,
    node: Mapping[str, Any],
    root: Mapping[str, Any],
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge the three sources of hyper-parameters, in the order the project documents.

    La précédence est toujours la même — défauts du registre, puis configuration du projet, puis
    surcharge explicite —, et elle s'applique **bloc par bloc** : un réglage écrit sous
    ``model.params.pretraining`` ne remplace que le pré-entraînement, jamais l'architecture.

    Les projets texte écrivent leurs réglages **à plat** dans ``conf/model/default.yaml``
    (``model.params.units``, ``model.params.compression``) : les clés hors blocs sont donc
    conservées telles quelles, et les réglages que la configuration ne donne pas sont lus dans le
    nœud ``train`` — le modèle doit lire la même boucle d'entraînement que le trainer.

    Args:
        spec: Specification of the selected algorithm (its registry defaults).
        node: The ``model`` node of the configuration.
        root: The whole configuration (for the ``train`` node and the seed).
        override: Explicit hyper-parameters, or ``None`` for the configured ones.

    Returns:
        The effective parameters: one block per entry of :data:`PARAM_BLOCKS`, plus the flat keys.
    """
    configured = dict(node.get("params") or {})
    explicit = None if override is None else dict(override)
    # Un réglage écrit **à plat** (``model.params.vocab_size``) appartient au bloc qui déclare ce nom
    # dans le registre : c'est ce qui permet à un notebook de réduire l'architecture sans connaître
    # la disposition interne des blocs. Sans cette table, le défaut du registre l'emporterait
    # silencieusement sur la configuration du projet.
    owner = {key: block for block in PARAM_BLOCKS for key in (spec.default_params.get(block) or {})}
    flat_configured = {key: value for key, value in configured.items() if key not in PARAM_BLOCKS}
    flat_explicit = (
        {key: value for key, value in explicit.items() if key not in PARAM_BLOCKS}
        if explicit is not None
        else {}
    )
    effective: dict[str, Any] = {}
    for block in PARAM_BLOCKS:
        merged = dict(spec.default_params.get(block) or {})
        # Défauts du registre, puis configuration à plat, puis bloc explicite, puis surcharge : la
        # dernière source qui parle d'un réglage gagne.
        merged.update(
            {key: value for key, value in flat_configured.items() if owner.get(key) == block}
        )
        merged.update(dict(configured.get(block) or {}))
        merged.update(
            {key: value for key, value in flat_explicit.items() if owner.get(key) == block}
        )
        if explicit is not None:
            merged.update(dict(explicit.get(block) or {}))
        effective[block] = merged
    for source in (configured, explicit):
        for key, value in (source or {}).items():
            if key not in PARAM_BLOCKS:
                effective[key] = value
    train = dict(root.get("train") or {})
    callbacks = dict(train.get("callbacks") or {})
    from_train: dict[str, Any] = {
        "epochs": int(train.get("epochs", 12)),
        "batch_size": int(train.get("batch_size", 16)),
        "learning_rate": float(train.get("learning_rate", 2e-3)),
        "logging_every": int(callbacks.get("logging_every", 2)),
    }
    for key in TRAIN_KEYS:
        effective.setdefault(key, from_train[key])
    return effective


def _model_node(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Return the ``model`` node of a configuration, whatever its concrete type.

    Args:
        config: Mapping or ``AppConfig``-like object.

    Returns:
        The ``model`` node as a plain mapping.

    Raises:
        ValueError: When the configuration has no ``model`` node.
    """
    mapping = _as_mapping(config)
    node = mapping.get("model")
    if node is None:
        msg = (
            "No 'model' node in the configuration: build the strategy with "
            "`build_model({'model': {'algorithm': ...}})`."
        )
        raise ValueError(msg)
    return dict(node) if isinstance(node, Mapping) else dict(_as_mapping(node))


def _as_mapping(config: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    """Convert a configuration object into a plain mapping.

    Args:
        config: Mapping, ``AppConfig``-like object or ``None``.

    Returns:
        A plain mapping (empty when ``config`` is ``None``).
    """
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    for attribute in ("model_dump", "dict"):
        method = getattr(config, attribute, None)
        if callable(method):
            try:
                return dict(method())
            except TypeError:  # pragma: no cover - defensive, depends on the pydantic version
                continue
    return {}


def _peek_payload(path: Path) -> dict[str, Any]:
    """Read the metadata block of an artefact without rebuilding the model.

    Args:
        path: Artefact written by ``save``.

    Returns:
        The artefact payload (an empty mapping when it cannot be read).
    """
    try:
        payload = joblib.load(path)
    except Exception as exc:  # pragma: no cover - defensive, depends on the artefact
        logger.warning("Artefact illisible ({}) : {}", path, exc)
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


__all__ = [
    "ALGORITHMS",
    "BUDGET_KEYS",
    "CONFIGURED_ALGORITHM",
    "PARAM_BLOCKS",
    "PROJECT_TASK",
    "TRAIN_KEYS",
    "AlgorithmSpec",
    "available_algorithms",
    "build_model",
    "describe_algorithm",
    "load_model",
    "supported_tasks",
]
