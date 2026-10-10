"""Fabrique de modèles de la stack **scikit-learn (TF-IDF, classification de texte)**.

Les tickets d'une même catégorie partagent un vocabulaire (« facture », « prélèvement » pour la
facturation ; « colis », « livraison » pour la livraison) et unigrammes comme bigrammes
suffisent à les séparer. La régression logistique apprend un coefficient par terme et par classe
(`class_weight='balanced'`, sinon la classe « autre » — 12 % du corpus — serait sacrifiée au
profit des cinq autres), et le modèle expose ses poids : chaque décision peut être expliquée par
les trois termes qui l'ont emporté. Sa limite est mesurée, pas supposée : un ticket qui parle de
« colis jamais reçu » et un qui parle de « livraison en retard » n'ont aucun terme en commun, et
le segment paraphrase existe pour chiffrer ce que cela coûte. La variante `with-transformers` a
été construite pour répondre à cette limite, et la réponse est publiée : sur ce corpus, un
encodeur appris sur le train réduit l'écart de 0,09 sur les paraphrases (0,8285 contre 0,7352 de
F1 macro) mais perd 0,14 sur les tickets canoniques (0,7892 contre 0,9262) — au total, le
lexical reste devant (0,8408 contre 0,7942 en test), et c'est l'arbitrage que les deux variantes
documentent.

La fabrique est le seul endroit où la configuration devient un objet :

* :func:`build_model` lit le nœud ``model`` (algorithme, hyper-paramètres, colonne de texte, graine)
  et rend un :class:`~src.models.classifier.TfidfClassifier` conforme au contrat
  :class:`~src.models.contract.BaseTextClassifier` ;
* :func:`load_model` recharge un artefact persisté, que l'appelant passe un fichier ou un dossier ;
* :func:`available_algorithms` liste les classifieurs que la stack sait servir : c'est ce que le
  notebook 04 parcourt au lieu de coder un nom en dur.

Les algorithmes sont déclarés ici et implémentés dans :mod:`src.models.classifier` : ajouter un
classifieur linéaire est une entrée de registre, pas une modification du pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.models.base import FitResult, ModelCard
from src.models.classifier import TfidfClassifier
from src.models.contract import BaseTextClassifier
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Task learned by this project (injected from the manifest).
PROJECT_TASK = "multiclass"

#: Algorithm selected in `conf/model/default.yaml`.
CONFIGURED_ALGORITHM = "tfidf_logreg"

#: Model class of this stack (documented in the README and the configuration).
MODEL_CLASS = "TfidfClassifier"

#: Artefact written by this stack (injected from `registry/stacks.yaml`).
MODEL_FILE = "classifier.joblib"

#: Extensions understood by :func:`load_model`.
SUPPORTED_SUFFIXES: tuple[str, ...] = (".joblib", ".pkl", ".pickle")


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """Déclaration d'un classifieur linéaire disponible dans cette stack.

    Attributes:
        name: Identifiant stable, utilisé dans ``conf/model/default.yaml`` et les notebooks.
        display_name: Nom lisible (rapports, notebooks).
        default_params: Paramètres de vectorisation et d'estimateur appliqués par défaut.
        rationale: Pourquoi (et quand) choisir ce classifieur.
    """

    name: str
    display_name: str
    default_params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


#: Registry of the algorithms served by this stack.
ALGORITHMS: dict[str, AlgorithmSpec] = {
    "tfidf_logreg": AlgorithmSpec(
        name="tfidf_logreg",
        display_name="Régression logistique sur TF-IDF",
        default_params={
            "lexical": {"mode": "tfidf", "ngram_range": (1, 2), "min_df": 2, "sublinear_tf": True},
            "estimator": {"C": 4.0, "max_iter": 400, "class_weight": "balanced"},
        },
        rationale=(
            "Un poids par terme et par classe : le classifieur s'explique terme à terme, gère "
            "nativement six classes et reste instantané à l'entraînement comme à l'inférence. "
            "`class_weight='balanced'` est le réglage par défaut, parce que la classe `autre` est "
            "minoritaire et qu'une F1 macro récompense le fait de ne pas l'ignorer."
        ),
    ),
    "tfidf_nb": AlgorithmSpec(
        name="tfidf_nb",
        display_name="Bayésien naïf complémentaire sur TF-IDF",
        default_params={
            "lexical": {"mode": "tfidf", "ngram_range": (1, 1), "min_df": 2, "sublinear_tf": True},
            "estimator": {"alpha": 0.3},
        },
        rationale=(
            "Très rapide et robuste au déséquilibre des classes, au prix d'une hypothèse fausse "
            "(l'indépendance des termes). Il sert de témoin : ce qu'il perd face à la régression "
            "logistique mesure ce que la pondération apprise apporte réellement."
        ),
    ),
    "tfidf_centroid": AlgorithmSpec(
        name="tfidf_centroid",
        display_name="Centroïde de classe sur TF-IDF",
        default_params={
            "lexical": {"mode": "tfidf", "ngram_range": (1, 1), "min_df": 1, "sublinear_tf": True},
            "estimator": {"temperature": 8.0},
        },
        rationale=(
            "Aucun poids n'est appris : chaque classe est la moyenne de ses vecteurs, et le "
            "document va au centroïde le plus proche. C'est le plancher de la famille — il montre "
            "ce qu'une simple moyenne obtient déjà, et il est déterministe par construction."
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
    effective = dict(configured) if params is None else dict(params)
    lexical = dict(spec.default_params.get("lexical") or {})
    lexical.update(dict(configured.get("lexical") or {}))
    lexical.update(dict(effective.get("lexical") or {}))
    estimator = dict(spec.default_params.get("estimator") or {})
    estimator.update(dict(configured.get("estimator") or {}))
    estimator.update(dict(effective.get("estimator") or {}))
    effective["lexical"] = lexical
    effective["estimator"] = estimator
    effective.setdefault("model_file", MODEL_FILE)

    model = TfidfClassifier(
        algorithm=selected,
        params=effective,
        task=task,
        text_column=str(node.get("text_column") or "text"),
        target_name=str(node.get("target") or data.get("target") or "label"),
        random_state=int(node.get("random_state") or root.get("seed", 42)),
        config=root,
        name=str(node.get("name") or spec.display_name),
    )
    logger.debug("Classifier built | {} | lexical={}", model.summary(), lexical)
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
            "scikit-learn (TF-IDF, classification de texte) stack "
            f"(expected one of {sorted(SUPPORTED_SUFFIXES)})"
        )
        raise ValueError(msg)
    return TfidfClassifier.load(source, config=config)


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
    "TfidfClassifier",
    "available_algorithms",
    "build_model",
    "describe_algorithm",
    "load_model",
    "supported_tasks",
]
