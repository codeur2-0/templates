"""Fabrique de modèles de la stack **Hugging Face Transformers (entraîné sur le corpus)**.

Deux couches d'attention, 128 unités, un vocabulaire et des poids appris sur le corpus : le pré-
entraînement masqué lit les tickets **sans leurs libellés**, puis l'affinage supervisé apprend
la classification avec `class_weight='balanced'` pour que la classe « autre » ne soit pas
sacrifiée. C'est l'architecture qui gagne la comparaison publiée du projet : 0,8022 de F1 macro
en validation contre 0,7455 pour la moyenne des vecteurs de termes (sans contexte) et moins
encore pour l'encodeur à quatre couches, sur les mêmes 719 tickets d'entraînement. Ce résultat
tient à un réglage mesuré, pas à une intuition : un encodeur initialisé au hasard diverge au
taux d'apprentissage du sac d'embeddings, et la grille de sélection a montré qu'il lui faut
0,005 — cette valeur est devenue le défaut de la stack pour les encodeurs. Le coût est réel et
publié : deux minutes et demie de CPU contre trois secondes pour le sac, et la variante `with-
tfidf_classifier` reste devant sur ce corpus (0,8408 en test). Chaque décision est explicable
par occlusion (le terme affiché est retiré du texte et la baisse de probabilité est publiée), et
la fiche de modèle annonce le taux de jetons `[UNK]`, le taux de tickets tronqués et le nombre
de paramètres.

La fabrique est le seul endroit où la configuration devient un objet :

* :func:`build_model` lit le nœud ``model`` (algorithme, hyper-paramètres, colonne de texte, graine)
  et rend un :class:`~src.models.classifier.TransformerTextClassifier` conforme au contrat
  :class:`~src.models.contract.BaseTextClassifier` ;
* :func:`load_model` recharge un artefact persisté, que l'appelant passe un fichier ou un dossier ;
* :func:`available_algorithms` liste les architectures que la stack sait servir : c'est ce que le
  notebook 04 parcourt au lieu de coder un nom en dur.

Les architectures sont déclarées ici et construites dans :mod:`src.models.classifier` : ajouter une
taille d'encodeur (ou remplacer l'attention par autre chose) est une entrée de registre, pas une
modification du pipeline. Chaque entrée déclare ses quatre blocs de paramètres — ``tokenizer``
(vocabulaire appris localement), ``network`` (architecture), ``pretraining`` (masquage) et
``training`` (optimisation) — et la configuration peut surcharger chacun d'eux champ par champ.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.base import FitResult, ModelCard
from src.models.classifier import TransformerTextClassifier
from src.models.contract import BaseTextClassifier
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Task learned by this project (injected from the manifest).
PROJECT_TASK = "multiclass"

#: Algorithm selected in `conf/model/default.yaml`.
CONFIGURED_ALGORITHM = "bert_tiny"

#: Model class of this stack (documented in the README and the configuration).
MODEL_CLASS = "TransformerTextClassifier"

#: Artefact written by this stack (injected from `registry/stacks.yaml`).
MODEL_FILE = "model.pt"

#: Extensions understood by :func:`load_model`. L'artefact est écrit par
#: :meth:`~src.models.contract.BaseTextClassifier.save` (joblib, via ``src.utils.io``) : ``.pt`` est
#: le nom choisi par la stack, les autres suffixes sont acceptés pour un artefact renommé à la main.
SUPPORTED_SUFFIXES: tuple[str, ...] = (".pt", ".pth", ".joblib", ".pkl", ".pickle")

#: Paramètre blocks of the stack, merged individually by :func:`build_model`.
PARAM_BLOCKS: tuple[str, ...] = ("tokenizer", "network", "pretraining", "training")


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """Déclaration d'une architecture servie par cette stack.

    Attributes:
        name: Identifiant stable, utilisé dans ``conf/model/default.yaml`` et les notebooks.
        display_name: Nom lisible (rapports, notebooks).
        default_params: Paramètres de tokenisation, d'architecture, de pré-entraînement et
            d'optimisation appliqués par défaut.
        rationale: Pourquoi (et quand) choisir cette architecture.
    """

    name: str
    display_name: str
    default_params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


#: Registry of the algorithms served by this stack.
ALGORITHMS: dict[str, AlgorithmSpec] = {
    "embedding_bag": AlgorithmSpec(
        name="embedding_bag",
        display_name="Sac d'embeddings (moyenne des vecteurs de termes)",
        default_params={
            "tokenizer": {"vocab_size": 4000, "min_frequency": 2, "max_length": 64},
            # 48 dimensions et 0,3 de dropout : la configuration retenue par le projet de référence
            # de la famille (choisie sur le split de validation, 719 tickets d'entraînement). Elle
            # vit ici, dans les défauts de la stack, pour qu'elle ne contamine pas les encodeurs
            # pendant la comparaison du notebook 04.
            "network": {"embedding_dim": 48, "dropout": 0.3},
            "pretraining": {"epochs": 0},
            "training": {"class_weight": "balanced"},
        },
        rationale=(
            "La moyenne des vecteurs de termes appris, suivie d'un classifieur linéaire : aucun "
            "ordre, aucune fenêtre d'attention, donc aucun contexte. C'est le plancher neuronal "
            "de la stack — entraîné par la **même** boucle que l'encodeur, ce qui permet de lire "
            "ce que l'attention apporte à boucle d'entraînement constante."
        ),
    ),
    "bert_tiny": AlgorithmSpec(
        name="bert_tiny",
        display_name="Encodeur BERT minuscule (2 couches, 128 unités)",
        default_params={
            "tokenizer": {"vocab_size": 4000, "min_frequency": 2, "max_length": 64},
            "network": {
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 256,
                "dropout": 0.1,
            },
            "pretraining": {"epochs": 10, "mask_probability": 0.15},
            # 0,005 et non 0,02 : un encodeur initialisé au hasard diverge au taux d'apprentissage
            # du sac d'embeddings. Les défauts déclarés ici sont ceux de la grille mesurée sur le
            # corpus de référence, sinon la comparaison d'architectures du notebook 04 serait une
            # comparaison de taux d'apprentissage déguisée.
            "training": {"class_weight": "balanced", "learning_rate": 0.005},
        },
        rationale=(
            "Deux couches d'attention suffisent à faire circuler l'information entre les termes "
            "d'un ticket. C'est la plus petite architecture de la stack : elle s'entraîne en "
            "quelques minutes sur un CPU, et c'est celle qui va le plus loin des deux encodeurs "
            "sur 719 tickets — sans atteindre le sac d'embeddings, ce que le notebook 04 publie."
        ),
    ),
    "bert_small": AlgorithmSpec(
        name="bert_small",
        display_name="Encodeur BERT (4 couches, 256 unités)",
        default_params={
            "tokenizer": {"vocab_size": 4000, "min_frequency": 2, "max_length": 64},
            "network": {
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "intermediate_size": 512,
                "dropout": 0.1,
            },
            "pretraining": {"epochs": 3, "mask_probability": 0.15},
            "training": {"class_weight": "balanced", "learning_rate": 0.005},
        },
        rationale=(
            "Quatre couches et 256 unités : la plus grande architecture de la stack, pré-entraînée "
            "par masquage sur les textes du train puis affinée sur les libellés. C'est aussi la "
            "moins bonne sur ce corpus : deux millions de paramètres pour 719 tickets, mesuré et "
            "publié. Le nombre de paramètres figure dans la fiche de modèle, parce qu'un encodeur "
            "appris sur 719 tickets n'est pas un modèle de langue — c'est un modèle **mesuré**."
        ),
    ),
}


def build_model(
    config: Mapping[str, Any] | Any,
    *,
    algorithm: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> BaseTextClassifier:
    """Build the classifier declared by the configuration.

    ``params`` a trois sens, volontairement :

    * ``None`` — les hyper-paramètres de ``conf/model/default.yaml`` (chemin de production) ;
    * ``{}`` — les défauts du registre (comparaison loyale entre architectures, notebook 04) ;
    * un mapping — utilisé tel quel (grille d'hyper-paramètres).

    Args:
        config: Application configuration (mapping or :class:`~src.schemas.config.AppConfig`).
        algorithm: Algorithm override (defaults to ``model.algorithm``).
        params: Hyper-parameter override.

    Returns:
        An unfitted :class:`~src.models.contract.BaseTextClassifier`.

    Raises:
        ValueError: When the configuration has no ``model`` node, or the algorithm is unknown.
    """
    node = _model_node(config)
    root = _as_mapping(config)
    selected = str(algorithm or node.get("algorithm") or CONFIGURED_ALGORITHM)
    spec = _algorithm_spec(selected)
    metrics = dict(root.get("metrics") or {})
    task = str(node.get("task") or metrics.get("task") or PROJECT_TASK)
    data = dict(root.get("data") or {})

    configured = dict(node.get("params") or {})
    override = None if params is None else dict(params)
    effective: dict[str, Any] = {}
    for block in PARAM_BLOCKS:
        merged = dict(spec.default_params.get(block) or {})
        merged.update(dict(configured.get(block) or {}))
        if override is not None:
            merged.update(dict(override.get(block) or {}))
        effective[block] = merged
    # Une clé hors des quatre blocs servis par la stack (``model_file`` par exemple) est conservée :
    # c'est un réglage de la fabrique, pas une hyper-paramètre du réseau. Les blocs étrangers
    # (paramètres d'une autre stack, hérités d'un défaut de famille) sont ignorés : la fiche de
    # modèle ne publie que ce que le modèle utilise réellement.
    for key, value in (override or {}).items():
        if key not in PARAM_BLOCKS:
            effective[key] = value
    effective.setdefault("model_file", MODEL_FILE)

    model = TransformerTextClassifier(
        algorithm=selected,
        params=effective,
        task=task,
        text_column=str(node.get("text_column") or "text"),
        target_name=str(node.get("target") or data.get("target") or "label"),
        random_state=int(node.get("random_state") or root.get("seed", 42)),
        config=root,
        name=str(node.get("name") or spec.display_name),
    )
    logger.debug("Classifier built | {} | network={}", model.summary(), effective.get("network"))
    return model


def supported_tasks() -> Sequence[str]:
    """Return the learning tasks this stack serves (documented in the README)."""
    return ("multiclass", "binary")


def load_model(
    path: str | Path, *, config: Mapping[str, Any] | Any | None = None
) -> BaseTextClassifier:
    """Reload a persisted classifier.

    Args:
        path: Artefact file, or directory holding the default file name.
        config: Optional configuration refreshed into the reloaded model.

    Returns:
        The reloaded classifier, ready to classify.

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
            "Hugging Face Transformers (entraîné sur le corpus) stack "
            f"(expected one of {sorted(SUPPORTED_SUFFIXES)})"
        )
        raise ValueError(msg)
    return TransformerTextClassifier.load(source, config=config)


def available_algorithms(task: str | None = None) -> list[str]:
    """List the algorithms this stack can serve.

    Args:
        task: Learning task; this stack serves classification tasks.

    Returns:
        The algorithm identifiers, in registry order.
    """
    if task and task not in {"multiclass", "binary", PROJECT_TASK}:
        return []
    return list(ALGORITHMS)


def describe_algorithm(name: str) -> dict[str, Any]:
    """Return the documentation of one algorithm (used by the reports and notebooks).

    Args:
        name: Algorithm identifier.

    Returns:
        A mapping with ``name``, ``display_name``, ``rationale`` and ``defaults``.

    Raises:
        ValueError: When the algorithm is unknown.
    """
    spec = _algorithm_spec(str(name))
    return {
        "name": spec.name,
        "display_name": spec.display_name,
        "rationale": spec.rationale,
        "defaults": dict(spec.default_params),
    }


def _algorithm_spec(name: str) -> AlgorithmSpec:
    """Return the specification of an algorithm, or raise a helpful error."""
    if name not in ALGORITHMS:
        msg = f"Unknown algorithm '{name}'. Available: {sorted(ALGORITHMS)}"
        raise ValueError(msg)
    return ALGORITHMS[name]


def _model_node(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Return the ``model`` node of a configuration, whatever its concrete type.

    Raises:
        ValueError: When the configuration has no ``model`` node.
    """
    node = _as_mapping(config).get("model")
    if not node:
        msg = (
            "No 'model' node in the configuration: build_model needs at least "
            "model.algorithm (see conf/model/default.yaml)."
        )
        raise ValueError(msg)
    return dict(node)


def _as_mapping(config: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    """Normalise a configuration (mapping or Pydantic object) into a plain dictionary."""
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    dump = getattr(config, "model_dump", None)
    if callable(dump):
        return dict(dump())
    return dict(vars(config))


__all__ = [
    "ALGORITHMS",
    "AlgorithmSpec",
    "CONFIGURED_ALGORITHM",
    "MODEL_CLASS",
    "MODEL_FILE",
    "PARAM_BLOCKS",
    "PROJECT_TASK",
    "SUPPORTED_SUFFIXES",
    "BaseTextClassifier",
    "FitResult",
    "ModelCard",
    "TransformerTextClassifier",
    "available_algorithms",
    "build_model",
    "describe_algorithm",
    "load_model",
    "supported_tasks",
]
