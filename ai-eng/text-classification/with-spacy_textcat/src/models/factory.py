"""Fabrique de modèles de la stack **spaCy (classification de texte)**.

Le sac de n-grammes de spaCy hache les unigrammes et les bigrammes du ticket dans une table fixe
et apprend une couche linéaire par classe : pas de vocabulaire à construire, pas de mot hors
vocabulaire, et un ajustement d'une dizaine de secondes sur 719 tickets. Il a été choisi au
terme d'une étude rejouable (`scripts/architecture_study.py`) faite au protocole du projet —
lots de 64, défauts du registre pour chaque architecture, split de validation, cinq graines : il
plafonne à 0,841 de F1 macro dès la septième époque, avec la même valeur sur les cinq graines,
quand l'encodeur de jetons réduit atteint au mieux 0,8177 (écart-type 0,011). L'avance (0,023)
dépasse deux fois la dispersion de l'encodeur : c'est un écart réel, pas un tirage. Le budget de
huit époques se place sur ce plateau. La limite est mesurée : le modèle est **sous-confiant** —
confiance moyenne 0,561 pour une exactitude de 0,819 en test, d'où un écart de calibration de
0,258 — et il ne faut donc pas router automatiquement sur sa confiance brute sans la recalibrer
sur le split `calibration` que la famille réserve à cet usage. Chaque décision s'explique par
occlusion : le jeton est retiré du ticket, et la baisse de probabilité est publiée.

La fabrique est le seul endroit où la configuration devient un objet :

* :func:`build_model` lit le nœud ``model`` (algorithme, hyper-paramètres, colonne de texte, graine)
  et rend un :class:`~src.models.classifier.SpacyTextClassifier` conforme au contrat
  :class:`~src.models.contract.BaseTextClassifier` ;
* :func:`load_model` recharge un artefact persisté, que l'appelant passe un fichier ou un dossier ;
* :func:`available_algorithms` liste les classifieurs que la stack sait servir : c'est ce que le
  notebook 04 parcourt au lieu de coder un nom en dur.

Les algorithmes sont déclarés ici ; chacun choisit une architecture spaCy de
:mod:`src.models.classifier` et ses réglages par défaut. Ajouter une architecture est une entrée de
registre, pas une modification du pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.base import FitResult, ModelCard
from src.models.classifier import SpacyTextClassifier
from src.models.contract import BaseTextClassifier
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Task learned by this project (injected from the manifest).
PROJECT_TASK = "multiclass"

#: Algorithm selected in `conf/model/default.yaml`.
CONFIGURED_ALGORITHM = "textcat_bow"

#: Model class of this stack (documented in the README and the configuration).
MODEL_CLASS = "SpacyTextClassifier"

#: Artefact written by this stack (injected from `registry/stacks.yaml`).
MODEL_FILE = "classifier.joblib"

#: Extensions understood by :func:`load_model`.
SUPPORTED_SUFFIXES: tuple[str, ...] = (".joblib", ".pkl", ".pickle")


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """Déclaration d'un classifieur spaCy disponible dans cette stack.

    Attributes:
        name: Identifiant stable, utilisé dans ``conf/model/default.yaml`` et les notebooks.
        display_name: Nom lisible (rapports, notebooks).
        default_params: Architecture, réseau et entraînement appliqués par défaut.
        rationale: Pourquoi (et quand) choisir ce classifieur.
    """

    name: str
    display_name: str
    default_params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


#: Registry of the algorithms served by this stack.
ALGORITHMS: dict[str, AlgorithmSpec] = {
    "textcat_bow": AlgorithmSpec(
        name="textcat_bow",
        display_name="spaCy TextCatBOW (sac de n-grammes haché, couche linéaire)",
        default_params={
            "architecture": "bow",
            "network": {"ngram_size": 2, "length": 262144},
            # Le sac de n-grammes n'a aucune couche de dropout : le régler ne changerait rien.
            "training": {"dropout": 0.0, "learning_rate": 0.001},
        },
        rationale=(
            "Le sac de n-grammes de spaCy : unigrammes et bigrammes hachés, une couche linéaire, "
            "aucun contexte. C'est l'analogue spaCy de la variante TF-IDF, et le plus rapide à "
            "entraîner : il dit ce que la stack obtient sans encodeur."
        ),
    ),
    "textcat_reduce": AlgorithmSpec(
        name="textcat_reduce",
        display_name="spaCy TextCatReduce (encodeur de jetons, moyenne et maximum)",
        default_params={
            "architecture": "reduce",
            "network": {"width": 64, "depth": 2, "embed_rows": 2000, "window_size": 1},
            "training": {"dropout": 0.2, "learning_rate": 0.001},
        },
        rationale=(
            "Un encodeur de jetons appris sur le corpus : plongements hachés sur la forme "
            "normalisée, la forme en minuscules, le préfixe et le suffixe, puis une "
            "convolution. Les sous-mots le rendent tolérant aux fautes et aux flexions "
            "qu'un vocabulaire exact ignore."
        ),
    ),
    "textcat_ensemble": AlgorithmSpec(
        name="textcat_ensemble",
        display_name="spaCy TextCatEnsemble (sac de n-grammes + encodeur)",
        default_params={
            "architecture": "ensemble",
            "network": {
                "ngram_size": 2,
                "length": 262144,
                "width": 64,
                "depth": 2,
                "embed_rows": 2000,
                "window_size": 1,
            },
            "training": {"dropout": 0.2, "learning_rate": 0.003},
        },
        rationale=(
            "L'architecture par défaut de spaCy : le sac de n-grammes et l'encodeur de jetons, "
            "combinés. Elle coûte l'entraînement des deux et ne se justifie que si elle bat chacun "
            "des deux sur la validation."
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
    * ``{}`` — les défauts du registre (comparaison loyale entre algorithmes, notebook 04) ;
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
    # `params=None` : chemin de production, la configuration complète les défauts du registre.
    # `params={}` : défauts du registre **seuls** (comparaison loyale du notebook 04) — la
    # configuration du modèle servi ne doit pas fuir vers les autres architectures.
    # Un mapping : surcharge appliquée par-dessus les défauts (grille d'hyper-paramètres).
    overrides = configured if params is None else dict(params)
    effective: dict[str, Any] = {
        key: value for key, value in overrides.items() if key not in {"network", "training"}
    }
    for block in ("network", "training"):
        merged = dict(spec.default_params.get(block) or {})
        merged.update(dict(overrides.get(block) or {}))
        effective[block] = merged
    # L'architecture appartient à l'algorithme : `textcat_reduce` construit toujours un encodeur,
    # quoi que dise un `params.architecture` laissé dans la configuration.
    effective["architecture"] = str(spec.default_params.get("architecture", "bow"))
    effective.setdefault("language", configured.get("language", "fr"))
    effective.setdefault("model_file", MODEL_FILE)

    model = SpacyTextClassifier(
        algorithm=selected,
        params=effective,
        task=task,
        text_column=str(node.get("text_column") or "text"),
        target_name=str(node.get("target") or data.get("target") or "label"),
        random_state=int(node.get("random_state") or root.get("seed", 42)),
        config=root,
        name=str(node.get("name") or spec.display_name),
    )
    logger.debug("Classifier built | {} | params={}", model.summary(), effective)
    return model


def load_model(path: str | Path, *, config: Mapping[str, Any] | None = None) -> BaseTextClassifier:
    """Reload a classifier artefact written by :meth:`BaseTextClassifier.save`.

    Args:
        path: Artefact path (a directory receives the default file name).
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
            "spaCy (classification de texte) stack "
            f"(expected one of {sorted(SUPPORTED_SUFFIXES)})"
        )
        raise ValueError(msg)
    return SpacyTextClassifier.load(source, config=config)


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
    return ("multiclass", "binary")


__all__ = [
    "ALGORITHMS",
    "AlgorithmSpec",
    "CONFIGURED_ALGORITHM",
    "MODEL_CLASS",
    "MODEL_FILE",
    "PROJECT_TASK",
    "SUPPORTED_SUFFIXES",
    "BaseTextClassifier",
    "FitResult",
    "ModelCard",
    "SpacyTextClassifier",
    "available_algorithms",
    "build_model",
    "describe_algorithm",
    "load_model",
    "supported_tasks",
]
