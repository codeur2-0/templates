"""Fabrique de modèles de la stack **spaCy**.

Le tagger est un pipeline `spacy.blank("fr")` entraîné **sur le corpus** : le tok2vec et le
composant `ner` (un classifieur à transitions) apprennent sur les 720 messages du split
d'entraînement, sans aucun poids pré-entraîné, donc sans dépendance réseau. C'est le seul modèle
du projet capable de généraliser au-delà du vocabulaire vu : les surfaces réservées des splits
d'évaluation ne peuvent pas être retrouvées par une liste, mais une partie est retrouvée par la
*forme* des noms — et l'écart est mesuré. Le modèle servi est hybride (`algorithm: hybrid`) : le
tagger décide, la couche de règles corrobore et comble les trous. Les règles n'apportent rien en
F1 sur ce corpus (le tagger les couvre), et c'est publié ; en revanche elles donnent à chaque
mention une **provenance** et une **confiance** — le taux de recouvrement entre le span prédit
et la couche de règles, dont la précision est mesurée par niveau dans le rapport d'évaluation.
Un tagger à transitions n'expose pas de probabilité par mention : publier une fausse probabilité
serait pire que publier une mesure de corroboration documentée.

La fabrique est le seul endroit où la configuration devient un objet :

* :func:`build_model` lit le nœud ``model`` (algorithme, hyper-paramètres, colonne de texte,
graine)
  et rend un :class:`~src.models.tagger.SpacyEntityTagger` conforme au contrat
  :class:`~src.models.contract.BaseEntityTagger` ;
* :func:`load_model` recharge un artefact persisté ; un artefact spaCy est un **répertoire** (le
  pipeline écrit par ``nlp.to_disk``, l'index de règles appris sur le train et un manifeste) ;
* :func:`available_algorithms` liste les algorithmes que la stack sait servir : c'est ce que le
  notebook 04 parcourt au lieu de coder un nom en dur.

Les trois algorithmes sont déclarés ici et implémentés dans :mod:`src.models.tagger` et
:mod:`src.models.rules` : ajouter un algorithme est une entrée de registre, pas une modification
du
pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.base import FitResult, ModelCard
from src.models.contract import BaseEntityTagger, EntityMention
from src.models.tagger import GAZETTEER, HYBRID, TAGGER, SpacyEntityTagger
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Task learned by this project (injected from the manifest).
PROJECT_TASK = "multiclass"

#: Algorithm selected in `conf/model/default.yaml`.
CONFIGURED_ALGORITHM = "hybrid"

#: Model class of this stack (documented in the README and the configuration).
MODEL_CLASS = "SpacyEntityTagger"

#: Artefact written by this stack (injected from `registry/stacks.yaml`).
MODEL_FILE = "tagger"


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """Déclaration d'un algorithme d'extraction d'entités disponible dans cette stack.

    Attributes:
        name: Identifiant stable, utilisé dans ``conf/model/default.yaml`` et les notebooks.
        display_name: Nom lisible (rapports, notebooks).
        learns_weights: ``True`` quand l'algorithme entraîne un réseau (sinon il applique des
            règles) : c'est ce que les rapports publient pour distinguer un plancher d'un modèle.
        default_params: Paramètres appliqués par défaut.
        rationale: Pourquoi (et quand) choisir cet algorithme.
    """

    name: str
    display_name: str
    learns_weights: bool
    default_params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


#: Registry of the algorithms served by this stack.
ALGORITHMS: dict[str, AlgorithmSpec] = {
    GAZETTEER: AlgorithmSpec(
        name=GAZETTEER,
        display_name="Couche de règles déclarées (motifs + surfaces vues au train)",
        learns_weights=False,
        default_params={},
        rationale=(
            "Aucun poids n'est appris : les références de commande, les montants et les dates sont "
            "reconnus par des motifs déclarés, et les produits et transporteurs par les surfaces "
            "annotées dans le split d'entraînement. C'est le plancher **explicable** du projet — "
            "chaque mention peut dire quelle règle l'a produite — et il ne peut pas connaître une "
            "surface réservée aux splits d'évaluation, ce qui rend la généralisation du tagger "
            "mesurable au lieu d'être supposée."
        ),
    ),
    TAGGER: AlgorithmSpec(
        name=TAGGER,
        display_name="Tagger spaCy appris sur le corpus (tok2vec + transitions)",
        learns_weights=True,
        default_params={},
        rationale=(
            "Un pipeline `spacy.blank(...)` dont le tok2vec et le classifieur à transitions "
            "apprennent sur les messages annotés du split d'entraînement, sans aucun poids "
            "pré-entraîné : c'est le seul algorithme du projet qui généralise hors du "
            "vocabulaire vu à l'entraînement, donc le seul qui puisse retrouver une partie "
            "d'un nom réservé à l'évaluation."
        ),
    ),
    HYBRID: AlgorithmSpec(
        name=HYBRID,
        display_name="Hybride : tagger appris, règles déclarées en corroboration",
        learns_weights=True,
        default_params={},
        rationale=(
            "Le tagger décide, la couche de règles corrobore et comble les trous. Sur le corpus de "
            "référence, les règles n'apportent pas de F1 que le tagger n'ait déjà (l'écart est "
            "publié) ; en revanche elles donnent à chaque mention sa **provenance** et une "
            "**confiance** dont la précision est mesurée par niveau. C'est l'algorithme servi."
        ),
    ),
}


def build_model(
    config: Mapping[str, Any] | Any,
    *,
    algorithm: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> BaseEntityTagger:
    """Build the entity extractor declared by the configuration.

    ``params`` a trois sens, volontairement :

    * ``None`` — les hyper-paramètres de ``conf/model/default.yaml`` (chemin de production) ;
    * ``{}`` — les défauts du registre (comparaison loyale entre algorithmes, notebook 04) ;
    * un mapping — utilisé tel quel (grille de réglages).

    Args:
        config: Application configuration (mapping or :class:`~src.schemas.config.AppConfig`).
        algorithm: Algorithm override (defaults to ``model.algorithm``).
        params: Hyper-parameter override.

    Returns:
        An unfitted :class:`~src.models.contract.BaseEntityTagger`.

    Raises:
        ValueError: When the configuration has no ``model`` node, or the algorithm is unknown.
    """
    root = _as_mapping(config)
    node = _model_node(config)
    selected = str(algorithm or node.get("algorithm") or CONFIGURED_ALGORITHM)
    spec = _algorithm_spec(selected)
    metrics = dict(root.get("metrics") or {})
    task = str(node.get("task") or metrics.get("task") or PROJECT_TASK)
    data = dict(root.get("data") or {})

    effective = dict(node.get("params") or {}) if params is None else dict(params)
    training = dict(spec.default_params.get("training") or {})
    training.update(dict(effective.get("training") or {}))
    if training:
        effective["training"] = training

    model = SpacyEntityTagger(
        algorithm=selected,
        params=effective,
        task=task,
        text_column=str(node.get("text_column") or "text"),
        target_name=str(node.get("target") or "label"),
        id_column=str(node.get("id_column") or data.get("id_column") or "msg_id"),
        random_state=int(node.get("random_state") or root.get("seed", 42)),
        config=root,
        name=str(node.get("name") or spec.display_name),
    )
    logger.debug(
        "Entity extractor built | {} | learns_weights={}", model.summary(), spec.learns_weights
    )
    return model


def load_model(path: str | Path, *, config: Mapping[str, Any] | None = None) -> BaseEntityTagger:
    """Reload an extractor artefact written by :meth:`BaseEntityTagger.save`.

    Args:
        path: Artefact directory (a file name is turned into its stem, since an artefact is a
            directory: the pipeline, the rule index and a manifest).
        config: Optional configuration refreshed into the reloaded model.

    Returns:
        The reloaded extractor, ready to predict.

    Raises:
        FileNotFoundError: When the artefact directory or its manifest is missing.
    """
    source = Path(path)
    if not source.is_dir():
        source = source.with_suffix("") if source.suffix else source
    if not source.exists():
        msg = f"Model artefact not found: {source}"
        raise FileNotFoundError(msg)
    return SpacyEntityTagger.load(source, config=config)


def available_algorithms(task: str | None = None) -> list[str]:
    """List the algorithms this stack can serve.

    Args:
        task: Learning task; this stack serves entity typing (a per-token classification).

    Returns:
        The algorithm identifiers, in registry order.
    """
    if task and task not in {"multiclass", PROJECT_TASK}:
        return []
    return list(ALGORITHMS)


def describe_algorithm(name: str) -> dict[str, Any]:
    """Return the documentation of one algorithm (used by the reports and notebooks).

    Args:
        name: Algorithm identifier.

    Returns:
        A mapping with ``name``, ``display_name``, ``learns_weights``, ``rationale`` and
        ``defaults``.

    Raises:
        ValueError: When the algorithm is unknown.
    """
    spec = _algorithm_spec(str(name))
    return {
        "name": spec.name,
        "display_name": spec.display_name,
        "learns_weights": spec.learns_weights,
        "rationale": spec.rationale,
        "defaults": dict(spec.default_params),
    }


def _algorithm_spec(name: str) -> AlgorithmSpec:
    """Return the specification of an algorithm, or raise a helpful error."""
    normalised = str(name).strip().lower().removeprefix("spacy_")
    aliases = {"rules": GAZETTEER, "regles": GAZETTEER, "ner": TAGGER, "hybride": HYBRID}
    key = aliases.get(normalised, normalised)
    if key not in ALGORITHMS:
        msg = f"Unknown algorithm '{name}'. Available: {sorted(ALGORITHMS)}"
        raise ValueError(msg)
    return ALGORITHMS[key]


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
    return ("multiclass",)


__all__ = [
    "ALGORITHMS",
    "AlgorithmSpec",
    "CONFIGURED_ALGORITHM",
    "GAZETTEER",
    "HYBRID",
    "MODEL_CLASS",
    "MODEL_FILE",
    "PROJECT_TASK",
    "TAGGER",
    "BaseEntityTagger",
    "EntityMention",
    "FitResult",
    "ModelCard",
    "SpacyEntityTagger",
    "available_algorithms",
    "build_model",
    "describe_algorithm",
    "load_model",
    "supported_tasks",
]
