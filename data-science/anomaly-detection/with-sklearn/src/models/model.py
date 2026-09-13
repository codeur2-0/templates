"""scikit-learn implementation of the model contract.

The class is a thin, honest wrapper: it owns **one** estimator, built from an explicit registry
(:data:`ESTIMATORS`) so that the algorithm names used in the configuration, the notebooks and the
reports are stable strings rather than import paths.

Three details matter for the rest of the project:

* ``estimator_`` exposes the native object after ``fit`` — the evaluator reads
  ``feature_importances_`` / ``coef_`` / ``cluster_centers_`` from it, and the predictor reads the
  centroids of a clustering;
* hyper-parameters are filtered against the estimator signature, so a grid may pass
  ``{"n_clusters": 6, "n_components": 6}`` and each estimator keeps only what it understands;
* ``class_weight="auto"`` / ``scale_pos_weight="auto"`` are resolved from the training labels: the
  imbalance handling is data-driven, never hard-coded.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModel, FitResult, ModelCard, library_versions
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Tasks handled by the classification estimators of this registry.
_CLASSIFICATION = frozenset({"binary", "multiclass"})
#: Tasks handled by the regression estimators of this registry.
_REGRESSION = frozenset({"regression", "forecasting"})
#: Tasks handled by the unsupervised estimators of this registry.
_CLUSTERING = frozenset({"clustering"})
#: Tasks handled by the anomaly estimators of this registry.
_ANOMALY = frozenset({"anomaly"})


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """Declaration of one algorithm available in this stack.

    Attributes:
        name: Stable identifier used in ``conf/model/default.yaml`` and the notebooks.
        display_name: Human readable name (reports, notebooks).
        tasks: Learning tasks the algorithm can serve.
        builder: Factory receiving the resolved parameters and returning an estimator.
        rationale: Why (and when) to pick this algorithm — shown by ``describe_algorithm``.
        iterative: Whether the estimator trains by rounds / epochs (used for the history).
        variant_by_task: Whether the builder selects a classifier / regressor variant through the
            ``__classifier__`` flag (tree ensembles, SVM).
    """

    name: str
    display_name: str
    tasks: frozenset[str]
    builder: Callable[..., Any]
    rationale: str = ""
    iterative: bool = False
    variant_by_task: bool = False
    defaults: dict[str, Any] = field(default_factory=dict)


def _logistic_regression(**params: Any) -> Any:
    """Build a regularised logistic regression (``saga`` when the data is large)."""
    from sklearn.linear_model import LogisticRegression

    params.setdefault("max_iter", 2000)
    return LogisticRegression(**params)


def _ridge(**params: Any) -> Any:
    """Build a ridge regression (linear baseline for a regression task)."""
    from sklearn.linear_model import Ridge

    return Ridge(**params)


def _svm(**params: Any) -> Any:
    """Build a support vector machine (SVC for a classification task, SVR otherwise).

    Args:
        **params: Estimator parameters; ``__classifier__`` selects the variant.

    Returns:
        The estimator.
    """
    from sklearn.svm import SVC, SVR

    if params.pop("__classifier__", False):
        return SVC(**params)
    return SVR(**params)


def _kmeans(**params: Any) -> Any:
    """Build a KMeans clustering."""
    from sklearn.cluster import KMeans

    return KMeans(**params)


def _minibatch_kmeans(**params: Any) -> Any:
    """Build a MiniBatchKMeans clustering (constant memory footprint)."""
    from sklearn.cluster import MiniBatchKMeans

    return MiniBatchKMeans(**params)


def _gaussian_mixture(**params: Any) -> Any:
    """Build a Gaussian mixture model (soft assignments, elliptical clusters)."""
    from sklearn.mixture import GaussianMixture

    return GaussianMixture(**params)


def _isolation_forest(**params: Any) -> Any:
    """Build an isolation forest (anomaly detection)."""
    from sklearn.ensemble import IsolationForest

    return IsolationForest(**params)


def _random_forest(**params: Any) -> Any:
    """Build a random forest (classifier or regressor, chosen by task)."""
    return _ensemble("random_forest", **params)


def _extra_trees(**params: Any) -> Any:
    """Build an extremely randomized trees ensemble (classifier or regressor)."""
    return _ensemble("extra_trees", **params)


def _gradient_boosting(**params: Any) -> Any:
    """Build a gradient boosting ensemble (classifier or regressor)."""
    return _ensemble("gradient_boosting", **params)


def _hist_gradient_boosting(**params: Any) -> Any:
    """Build a histogram-based gradient boosting ensemble (classifier or regressor)."""
    return _ensemble("hist_gradient_boosting", **params)


# Les constructeurs du registre sont des fonctions de module (jamais des lambdas) : le modèle
# archivé par joblib emporte `self.spec`, donc `builder`, qui doit rester picklable.
#: Registry of every algorithm this stack can build, keyed by stable identifier.
ESTIMATORS: dict[str, AlgorithmSpec] = {
    spec.name: spec
    for spec in (
        AlgorithmSpec(
            name="logistic_regression",
            display_name="Régression logistique régularisée",
            tasks=_CLASSIFICATION,
            builder=_logistic_regression,
            rationale=(
                "Baseline interprétable (coefficients signés), très rapide ; suppose des features "
                "standardisées et un signal plutôt linéaire/additif."
            ),
        ),
        AlgorithmSpec(
            name="ridge",
            display_name="Ridge (régression linéaire L2)",
            tasks=_REGRESSION,
            builder=_ridge,
            rationale=(
                "Plancher interprétable d'une régression : effets linéaires, coefficients "
                "régularisés. À battre par tout modèle non linéaire."
            ),
        ),
        AlgorithmSpec(
            name="random_forest",
            display_name="Forêt aléatoire",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_random_forest,
            variant_by_task=True,
            rationale=(
                "Ensemble d'arbres baggés : robuste au bruit et aux échelles, importance native "
                "des features, peu de réglages. Coûteux en mémoire au-delà de quelques millions "
                "de lignes."
            ),
        ),
        AlgorithmSpec(
            name="extra_trees",
            display_name="Extremely randomized trees",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_extra_trees,
            variant_by_task=True,
            rationale=(
                "Variante de la forêt aléatoire avec des seuils de split tirés aléatoirement : "
                "plus de variance injectée, souvent plus robuste au bruit, un peu moins précise "
                "sur un signal net."
            ),
        ),
        AlgorithmSpec(
            name="gradient_boosting",
            display_name="Gradient boosting (arbres)",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_gradient_boosting,
            variant_by_task=True,
            rationale=(
                "Variante historique du boosting, séquentielle et plus lente que "
                "HistGradientBoosting ; utile pour comparer deux stratégies d'agrégation."
            ),
            iterative=True,
        ),
        AlgorithmSpec(
            name="hist_gradient_boosting",
            display_name="HistGradientBoosting (histogrammes)",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_hist_gradient_boosting,
            variant_by_task=True,
            rationale=(
                "Boosting par histogrammes, natif valeurs manquantes, très rapide sur gros "
                "volumes : le meilleur rapport précision/coût de scikit-learn en tabulaire."
            ),
            iterative=True,
        ),
        AlgorithmSpec(
            name="svm",
            display_name="SVM (noyau RBF / SVR)",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_svm,
            variant_by_task=True,
            rationale=(
                "Marge maximale avec noyau RBF : excellent sur petits volumes, coût quadratique "
                "au-delà de ~100k lignes. Pas de probabilités sans `probability=True` (coûteux)."
            ),
        ),
        AlgorithmSpec(
            name="kmeans",
            display_name="KMeans",
            tasks=_CLUSTERING,
            builder=_kmeans,
            rationale=(
                "Référence du clustering partitionnel : rapide, centroïdes interprétables, "
                "suppose des groupes convexes et isotropes. Sensible à l'initialisation "
                "(`n_init`) et aux outliers (d'où le winsorising du pré-traitement)."
            ),
            defaults={"n_init": 10},
        ),
        AlgorithmSpec(
            name="minibatch_kmeans",
            display_name="MiniBatchKMeans",
            tasks=_CLUSTERING,
            builder=_minibatch_kmeans,
            rationale=(
                "Même géométrie que KMeans, mais par lots : coût mémoire constant et entraînement "
                "quasi linéaire. L'option des bases de plusieurs millions de clients, au prix "
                "d'une inertie légèrement supérieure."
            ),
            defaults={"batch_size": 1024, "n_init": 10},
        ),
        AlgorithmSpec(
            name="gaussian_mixture",
            display_name="Mélange gaussien (GMM)",
            tasks=_CLUSTERING,
            builder=_gaussian_mixture,
            rationale=(
                "Groupes ellipsoïdaux et **probabilités d'appartenance** : la confiance d'une "
                "affectation devient mesurable. Plus expressif que KMeans, plus sensible aux "
                "outliers et à l'initialisation."
            ),
            defaults={"n_init": 3, "reg_covar": 1e-4},
        ),
        AlgorithmSpec(
            name="isolation_forest",
            display_name="Isolation Forest",
            tasks=_ANOMALY,
            builder=_isolation_forest,
            rationale=(
                "Détection d'anomalies par isolation aléatoire : score continu, aucune hypothèse "
                "de distribution. Réservé aux tâches `anomaly`, pas à une segmentation."
            ),
        ),
    )
}


def _ensemble(kind: str, **params: Any) -> Any:
    """Build a tree ensemble, choosing the classifier or regressor variant.

    Args:
        kind: Ensemble identifier (``random_forest``, ``extra_trees``, ``gradient_boosting``,
            ``hist_gradient_boosting``).
        **params: Estimator parameters (``__classifier__`` selects the variant).

    Returns:
        The fitted-able estimator.
    """
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )

    variants: dict[str, tuple[type, type]] = {
        "random_forest": (RandomForestClassifier, RandomForestRegressor),
        "extra_trees": (ExtraTreesClassifier, ExtraTreesRegressor),
        "gradient_boosting": (GradientBoostingClassifier, GradientBoostingRegressor),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        ),
    }
    classifier_class, regressor_class = variants[kind]
    is_classifier = bool(params.pop("__classifier__", False))
    return (classifier_class if is_classifier else regressor_class)(**params)


def filter_params(target: Any, params: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the parameters the estimator understands.

    A pedagogical grid may explore ``n_clusters`` (KMeans) and ``n_components`` (GMM) at once;
    silently dropping the inapplicable key is what makes such a comparison possible — and logging
    it keeps the behaviour auditable.

    Args:
        target: Estimator class or instance.
        params: Candidate parameters.

    Returns:
        The applicable subset.
    """
    signature = inspect.signature(target if isinstance(target, type) else type(target).__init__)
    accepted = set(signature.parameters)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(params)
    kept = {str(key): value for key, value in params.items() if key in accepted}
    dropped = sorted(set(params) - set(kept))
    if dropped:
        logger.debug(
            "{}: paramètre(s) ignoré(s) {} (absents de la signature de {})",
            type(target).__name__ if not isinstance(target, type) else target.__name__,
            dropped,
            getattr(target, "__name__", type(target).__name__),
        )
    return kept


class SklearnModel(BaseModel):
    """A scikit-learn estimator behind the project's model contract."""

    framework = "sklearn"
    default_model_file = "model.joblib"
    epochs_based = False

    def __init__(
        self,
        *,
        algorithm: str = "random_forest",
        params: Mapping[str, Any] | None = None,
        task: str = "binary",
        feature_names: Sequence[str] | None = None,
        target_name: str | None = None,
        random_state: int = 42,
        supports_proba: bool | None = None,
        config: Mapping[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        """Store the contract and keep the estimator lazy.

        Args:
            algorithm: Identifier present in :data:`ESTIMATORS`.
            params: Hyper-parameters (filtered against the estimator signature).
            task: Learning task.
            feature_names: Features expected at inference time.
            target_name: Business target column (``None`` for unsupervised tasks).
            random_state: Reproducibility seed.
            supports_proba: Explicit probability support (``None`` = detected after fit).
            config: Full application configuration.
            name: Human readable name.

        Raises:
            ValueError: When the algorithm is unknown or does not serve the task.
        """
        super().__init__(
            algorithm=algorithm,
            params=params,
            task=task,
            feature_names=feature_names,
            target_name=target_name,
            random_state=random_state,
            supports_proba=supports_proba,
            config=config,
            name=name,
        )
        self.spec = resolve_algorithm(algorithm, task)
        self.estimator_: Any = None

    # ------------------------------------------------------------------ construction ------
    def build_estimator(self) -> Any:
        """Instantiate the estimator declared by ``algorithm`` with the resolved parameters.

        The estimator is built in two steps: a **prototype** with default values (which fixes the
        classifier / regressor variant), then ``set_params`` with only the parameters its real
        signature accepts. This is what lets one grid explore ``n_clusters`` (KMeans) and
        ``n_components`` (GaussianMixture) without a TypeError.

        Returns:
            A fresh (unfitted) estimator.
        """
        return self._new_estimator(self._resolved_params())

    def _new_estimator(self, params: Mapping[str, Any]) -> Any:
        """Build an unfitted estimator from explicit parameters.

        Args:
            params: Candidate hyper-parameters (``random_state`` is added when absent).

        Returns:
            The estimator, with only the applicable parameters set.
        """
        resolved = dict(params)
        resolved.setdefault("random_state", self.random_state)
        variant: dict[str, Any] = (
            {"__classifier__": self.task in _CLASSIFICATION} if self.spec.variant_by_task else {}
        )
        prototype = self.spec.builder(**variant)
        kept = filter_params(prototype, resolved)
        return prototype.set_params(**kept)

    def _resolved_params(self) -> dict[str, Any]:
        """Merge the registry defaults with the configured parameters."""
        return {**self.spec.defaults, **self._effective_params()}

    # ------------------------------------------------------------------ entraînement ------
    # Cette méthode est longue par choix : la résolution des valeurs « auto » (classes pondérées,
    # seuils, paramètres déduits des données) est une séquence linéaire, commentée étape par étape.
    # La factoriser en sous-méthodes disperserait la logique de réglage sans la simplifier.
    def _fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None,
        *,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        context: Any = None,
        callbacks: Sequence[Any] | None = None,
    ) -> None:
        """Fit the estimator, resolve the imbalance parameters and fire one epoch event.

        Args:
            X: Aligned training features.
            y: Training target (``None`` for clustering / anomaly).
            X_val: Validation features (used by the cross-validation diagnostic).
            y_val: Validation target.
            context: Callback context.
            callbacks: Callbacks to notify.
        """
        self._seed_everything()
        matrix = self._matrix(X)
        labels = None if y is None else np.asarray(y).ravel()
        params = self._resolve_auto_params(self._resolved_params(), labels)
        estimator = self._new_estimator(params)

        if labels is None:
            estimator.fit(matrix)
        else:
            estimator.fit(matrix, labels)

        self.estimator_ = estimator
        self.params = dict(params)
        self.classes_ = getattr(estimator, "classes_", None)
        if self._supports_proba is None:
            self._supports_proba = hasattr(estimator, "predict_proba")

        logs = self._epoch_logs(matrix, labels, X_val, y_val)
        if context is not None:
            context.extra["estimator"] = type(estimator).__name__
            context.extra["n_params"] = len(self.params)
            self.emit_epoch(context, list(callbacks or []), logs, epoch=0)
        self._cross_validate(matrix, labels, context)

    def _resolve_auto_params(
        self, params: Mapping[str, Any], labels: np.ndarray | None
    ) -> dict[str, Any]:
        """Replace the ``auto`` sentinels by values computed from the training labels.

        Args:
            params: Configured parameters.
            labels: Training target (``None`` for unsupervised tasks).

        Returns:
            The resolved parameters.
        """
        resolved = dict(params)
        if labels is None:
            return {key: value for key, value in resolved.items() if value != "auto"}
        counts = pd.Series(labels).value_counts().to_numpy(dtype="float64")
        majority = float(counts[0]) if counts.size else 1.0
        others = float(counts[1:].sum()) if counts.size > 1 else 0.0
        ratio = max(others / max(majority, 1.0), 1.0)
        for key, value in list(resolved.items()):
            if value != "auto":
                continue
            if key == "class_weight":
                resolved[key] = "balanced"
            elif key == "scale_pos_weight":
                resolved[key] = round(ratio, 4)
            else:
                logger.debug("Paramètre '{}' à 'auto' non pris en charge : valeur ignorée", key)
                resolved.pop(key)
        return resolved

    def _epoch_logs(
        self,
        matrix: np.ndarray,
        labels: np.ndarray | None,
        X_val: pd.DataFrame | np.ndarray | None,
        y_val: pd.Series | np.ndarray | None,
    ) -> dict[str, float]:
        """Compute the ``train_*`` / ``val_*`` metrics emitted to the callbacks."""
        calculator = self._metric_calculator()
        if calculator is None:
            return {}
        logs: dict[str, float] = {}
        train_values = calculator.evaluate(
            self._metric_inputs(
                y_true=labels,
                y_pred=self.estimator_.predict(matrix),
                y_proba=self._safe_proba(matrix),
                X=matrix,
            )
        )
        logs.update({f"train_{name}": float(value) for name, value in train_values.items()})
        if X_val is not None:
            validation = self._matrix(X_val)
            validation_labels = None if y_val is None else np.asarray(y_val).ravel()
            val_values = calculator.evaluate(
                self._metric_inputs(
                    y_true=validation_labels,
                    y_pred=self.estimator_.predict(validation),
                    y_proba=self._safe_proba(validation),
                    X=validation,
                )
            )
            logs.update({f"val_{name}": float(value) for name, value in val_values.items()})
        primary = str(dict(self.config.get("metrics") or {}).get("primary", ""))
        if primary:
            # `loss` et `val_loss` sont les clés surveillées par les callbacks (early stopping,
            # seuil de qualité). Sans métrique primaire calculable — détecteur non supervisé
            # entraîné sans cible — on ne publie rien : un NaN ferait mentir les courbes et
            # déclencherait des alertes de seuil dénuées de sens.
            training_loss = logs.get(f"train_{primary}", logs.get("train_loss"))
            if training_loss is not None and np.isfinite(training_loss):
                logs["loss"] = float(training_loss)
            validation_loss = logs.get(f"val_{primary}")
            if validation_loss is not None and np.isfinite(validation_loss):
                logs["val_loss"] = float(validation_loss)
        return logs

    def _safe_proba(self, matrix: np.ndarray) -> np.ndarray | None:
        """Return probabilities when the estimator exposes them (``None`` otherwise)."""
        if not hasattr(self.estimator_, "predict_proba"):
            return None
        try:
            return np.asarray(self.estimator_.predict_proba(matrix), dtype="float64")
        except (ValueError, AttributeError) as error:  # pragma: no cover - défensif
            logger.debug("predict_proba indisponible : {}", error)
            return None

    def _cross_validate(self, matrix: np.ndarray, labels: np.ndarray | None, context: Any) -> None:
        """Run the cross-validation declared in ``model.cross_validation`` (diagnostic).

        Args:
            matrix: Training features.
            labels: Training target (``None`` for unsupervised tasks).
            context: Callback context receiving the result in ``extra``.
        """
        node = dict(dict(self.config.get("model") or {}).get("cross_validation") or {})
        if not bool(node.get("enabled", False)):
            return
        folds = max(int(node.get("folds", 5)), 2)
        strategy = str(node.get("strategy", "kfold"))
        calculator = self._metric_calculator()
        if calculator is None or len(matrix) < folds * 5:
            logger.debug("Validation croisée ignorée (jeu trop petit ou métriques indisponibles)")
            return
        try:
            from sklearn.model_selection import KFold, StratifiedKFold

            splitter: Any
            if labels is not None and strategy == "stratified" and len(np.unique(labels)) > 1:
                splitter = StratifiedKFold(
                    n_splits=folds, shuffle=True, random_state=self.random_state
                )
                splits = splitter.split(matrix, labels)
            else:
                splitter = KFold(n_splits=folds, shuffle=True, random_state=self.random_state)
                splits = splitter.split(matrix)
            scores: list[float] = []
            primary = str(dict(self.config.get("metrics") or {}).get("primary", ""))
            for train_index, hold_index in splits:
                candidate = self._new_estimator(self.params)
                hold_labels = None if labels is None else labels[hold_index]
                if labels is None or hold_labels is None:
                    candidate.fit(matrix[train_index])
                else:
                    candidate.fit(matrix[train_index], labels[train_index])
                values = calculator.evaluate(
                    self._metric_inputs(
                        y_true=hold_labels,
                        y_pred=candidate.predict(matrix[hold_index]),
                        y_proba=(
                            np.asarray(candidate.predict_proba(matrix[hold_index]), dtype="float64")
                            if hasattr(candidate, "predict_proba")
                            else None
                        ),
                        X=matrix[hold_index],
                    )
                )
                scores.append(float(values.get(primary, float("nan"))))
        except (ImportError, ValueError) as error:
            logger.warning("Validation croisée interrompue : {}", error)
            return
        finite = [score for score in scores if np.isfinite(score)]
        summary = {
            "folds": folds,
            "strategy": strategy,
            "metric": primary,
            "mean": float(np.mean(finite)) if finite else float("nan"),
            "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "scores": [float(score) for score in scores],
        }
        if context is not None:
            context.extra["cross_validation"] = summary
        logger.info(
            "Cross-validation {} plis ({}) | {} = {:.5f} ± {:.5f}",
            folds,
            strategy,
            primary,
            summary["mean"],
            summary["std"],
        )

    # ------------------------------------------------------------------ prédiction --------
    def _predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict with the fitted estimator.

        Args:
            X: Aligned features.

        Returns:
            The predictions.

        Raises:
            RuntimeError: When no estimator has been fitted or loaded.
        """
        if self.estimator_ is None:
            msg = "No estimator is attached to this model (fit or load it first)"
            raise RuntimeError(msg)
        matrix = self._matrix(X)
        if self.task in _ANOMALY:
            # Un détecteur d'anomalies doit rendre un **score ordonnable** (plus élevé = plus
            # anormal), pas l'étiquette binaire +/-1 de `predict` : c'est ce score qui alimente
            # le recall@budget, la PR AUC et le choix du seuil d'alerte côté métier.
            return self._anomaly_score(matrix)
        return np.asarray(self.estimator_.predict(matrix)).ravel()

    def _anomaly_score(self, matrix: np.ndarray) -> np.ndarray:
        """Return a continuous anomaly score (higher = more anomalous).

        scikit-learn exposes the ranking quantity under ``score_samples`` (isolation forest,
        one-class SVM) or ``decision_function`` (covariance-based detectors); both are oriented
        towards *normality*, hence the sign flip.

        Args:
            matrix: Aligned float matrix.

        Returns:
            A 1-D array of anomaly scores.
        """
        estimator = self.estimator_
        for attribute in ("score_samples", "decision_function"):
            scorer = getattr(estimator, attribute, None)
            if callable(scorer):
                return -np.asarray(scorer(matrix), dtype="float64").ravel()
        # Repli défensif : un estimateur sans score continu ne peut pas servir de détecteur.
        return np.asarray(estimator.predict(matrix), dtype="float64").ravel()

    def _predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict probabilities with the fitted estimator.

        Args:
            X: Aligned features.

        Returns:
            A ``(n_samples, n_classes)`` array.

        Raises:
            NotImplementedError: When the estimator exposes no ``predict_proba``.
        """
        if self.estimator_ is None:
            msg = "No estimator is attached to this model (fit or load it first)"
            raise RuntimeError(msg)
        if not hasattr(self.estimator_, "predict_proba"):
            msg = f"{type(self.estimator_).__name__} exposes no predict_proba"
            raise NotImplementedError(msg)
        return np.asarray(self.estimator_.predict_proba(self._matrix(X)), dtype="float64")

    # ------------------------------------------------------------------ persistance -------
    def save(self, path: str | Path) -> Path:
        """Persist the whole wrapper (contract + estimator) with joblib.

        Args:
            path: Destination file or directory.

        Returns:
            The written path.
        """
        import joblib

        self.check_is_fitted()
        destination = self._resolve_path(path)
        joblib.dump(self, destination, compress=3)
        logger.info("Model saved: {} ({} algorithm)", destination, self.algorithm)
        return destination

    @classmethod
    def load(cls, path: str | Path, *, config: Mapping[str, Any] | None = None) -> BaseModel:
        """Reload a model persisted by :meth:`save`.

        Args:
            path: Artefact written by :meth:`save`.
            config: Optional configuration refreshed into the reloaded model.

        Returns:
            The reloaded :class:`SklearnModel`.

        Raises:
            FileNotFoundError: When the artefact does not exist.
            TypeError: When the artefact is not a model of this contract.
        """
        import joblib

        source = Path(path)
        if not source.exists():
            msg = f"Model artefact not found: {source}"
            raise FileNotFoundError(msg)
        restored = joblib.load(source)
        if not isinstance(restored, BaseModel):
            msg = (
                f"{source} does not contain a BaseModel (got {type(restored).__name__}). "
                "Point model_file at the artefact written by Trainer.save_artifacts."
            )
            raise TypeError(msg)
        if config:
            restored.config = dict(config)
        restored.fit_result_ = restored.fit_result_ or FitResult(
            model_name=type(restored).__name__, algorithm=restored.algorithm
        )
        return restored

    # ------------------------------------------------------------------ identité ----------
    def _repr_fields(self) -> list[str]:
        """Add the native estimator class to the representation."""
        fields = [f"algorithm={self.algorithm or '-'}"]
        if self.estimator_ is not None:
            fields.append(f"estimator={type(self.estimator_).__name__}")
        fields += [
            f"task={self.task}",
            f"features={len(self.feature_names)}",
            f"state={self.state}",
        ]
        return fields

    def model_card(
        self,
        *,
        metrics: Mapping[str, Any] | None = None,
        artifact: str | None = None,
        notes: Iterable[str] | None = None,
    ) -> ModelCard:
        """Build the model card, enriched with the native estimator identity.

        Args:
            metrics: Metrics of the run.
            artifact: Serialised model file name.
            notes: Extra remarks.

        Returns:
            The :class:`ModelCard`.
        """
        card = super().model_card(metrics=metrics, artifact=artifact, notes=notes)
        card.library_versions = library_versions()
        if self.estimator_ is not None:
            card.notes = [
                *card.notes,
                f"estimateur natif : {type(self.estimator_).__name__}",
            ]
        return card


def resolve_algorithm(algorithm: str, task: str) -> AlgorithmSpec:
    """Return the registry entry of an algorithm, validating the task.

    Args:
        algorithm: Algorithm identifier.
        task: Learning task of the project.

    Returns:
        The :class:`AlgorithmSpec`.

    Raises:
        ValueError: When the algorithm is unknown or does not serve ``task``.
    """
    spec = ESTIMATORS.get(str(algorithm))
    if spec is None:
        msg = (
            f"Unknown sklearn algorithm '{algorithm}'. Available: {sorted(ESTIMATORS)} "
            f"(for task '{task}': {available_for_task(task)})"
        )
        raise ValueError(msg)
    if task not in spec.tasks:
        msg = (
            f"Algorithm '{algorithm}' does not serve task '{task}' "
            f"(it serves {sorted(spec.tasks)}). Pick one of {available_for_task(task)}."
        )
        raise ValueError(msg)
    return spec


def available_for_task(task: str) -> list[str]:
    """Return the algorithm names serving a task, in registry order.

    Args:
        task: Learning task (``None``-like values return every algorithm).

    Returns:
        The algorithm identifiers.
    """
    if not task:
        return sorted(ESTIMATORS)
    return [name for name, spec in ESTIMATORS.items() if task in spec.tasks]


__all__ = [
    "ESTIMATORS",
    "AlgorithmSpec",
    "FitResult",
    "ModelCard",
    "SklearnModel",
    "available_for_task",
    "filter_params",
    "resolve_algorithm",
]
