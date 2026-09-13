"""LightGBM implementation of the model contract.

LightGBM grows trees **leaf-wise** with histogram binning: for a fixed budget it is usually faster
than depth-wise boosting, at the price of a stronger exposure to overfitting on small datasets —
which is exactly why ``num_leaves``, ``min_child_samples`` and the L1/L2 penalties are declared in
the configuration rather than hard-coded.

The wrapper keeps the project's contract intact:

* **native early stopping** through ``lightgbm.early_stopping`` on the validation split (absent for
  the DART booster, which does not support it),
* **one project callback event per boosting round** (throttled), so history, logging and quality
  threshold callbacks behave exactly like they do with the other stacks,
* **explicit parameter allow-list**, because the sklearn wrapper accepts ``**kwargs`` and would
  otherwise fail deep inside the C++ engine,
* ``importance_type`` (``gain`` by default) drives ``feature_importances_``, which the evaluator
  turns into the report's importance ranking.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import (
    EarlyStopException,
    LGBMClassifier,
    LGBMRegressor,
    early_stopping,
    log_evaluation,
)

from src.models.base import BaseModel, FitResult, ModelCard, library_versions
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Tasks served by the classification variants.
_CLASSIFICATION = frozenset({"binary", "multiclass"})
#: Tasks served by the regression variants.
_REGRESSION = frozenset({"regression", "forecasting"})

#: Hyper-parameters accepted by every booster of this stack (aliases sklearn + natifs).
_COMMON_PARAMS: frozenset[str] = frozenset(
    {
        "boosting_type",
        "n_estimators",
        "num_leaves",
        "max_depth",
        "learning_rate",
        "min_child_samples",
        "min_child_weight",
        "min_data_in_leaf",
        "subsample",
        "subsample_freq",
        "bagging_fraction",
        "bagging_freq",
        "colsample_bytree",
        "feature_fraction",
        "reg_lambda",
        "reg_alpha",
        "lambda_l1",
        "lambda_l2",
        "min_split_gain",
        "max_bin",
        "objective",
        "random_state",
        "n_jobs",
        "verbose",
        "importance_type",
        "class_weight",
    }
)

#: Paramètres réservés aux tâches de classification.
_CLASSIFICATION_PARAMS: frozenset[str] = frozenset({"scale_pos_weight", "num_class"})

#: Paramètres propres au booster DART (dropout sur les arbres).
_DART_PARAMS: frozenset[str] = frozenset(
    {"drop_rate", "skip_drop", "max_drop", "uniform_drop", "xgboost_dart_mode"}
)

#: Traduction de la métrique native LightGBM vers le nom du registre du projet.
_METRIC_ALIASES: dict[str, str] = {
    "auc": "roc_auc",
    "average_precision": "pr_auc",
    "binary_logloss": "log_loss",
    "multi_logloss": "log_loss",
    "logloss": "log_loss",
    "rmse": "rmse",
    "l2": "rmse",
    "l1": "mae",
    "mae": "mae",
    "map": "map_at_k",
}

#: Nombre de boosting rounds entre deux notifications des callbacks du projet.
EMIT_EVERY_ROUNDS = 10


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """Declaration of one algorithm available in this stack.

    Attributes:
        name: Stable identifier used in the configuration and the notebooks.
        display_name: Human readable name.
        tasks: Learning tasks the algorithm can serve.
        builder: Factory receiving the resolved parameters and returning an estimator.
        rationale: Why (and when) to pick this algorithm.
        iterative: Whether the estimator trains round by round (boosting: yes).
        defaults: Parameters added unless the configuration overrides them.
        accepted: Hyper-parameter allow-list for this algorithm.
        supports_early_stopping: Whether the booster honours a validation-based stop.
    """

    name: str
    display_name: str
    tasks: frozenset[str]
    builder: Callable[..., Any]
    rationale: str = ""
    iterative: bool = True
    defaults: dict[str, Any] = field(default_factory=dict)
    accepted: frozenset[str] = _COMMON_PARAMS | _CLASSIFICATION_PARAMS
    supports_early_stopping: bool = True


def _booster(**params: Any) -> Any:
    """Build an LGBMClassifier or LGBMRegressor according to ``__classifier__``.

    Args:
        **params: Booster parameters.

    Returns:
        The unfitted estimator.
    """
    if params.pop("__classifier__", False):
        return LGBMClassifier(**params)
    return LGBMRegressor(**params)


#: Registry of every algorithm this stack can build.
ESTIMATORS: dict[str, AlgorithmSpec] = {
    spec.name: spec
    for spec in (
        AlgorithmSpec(
            name="lightgbm",
            display_name="LightGBM (gbdt, leaf-wise)",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_booster,
            rationale=(
                "Boosting par histogrammes avec croissance leaf-wise : le plus rapide à "
                "entraîner sur gros volumes. `num_leaves` et `min_child_samples` sont les deux "
                "leviers qui évitent le sur-apprentissage sur les petits jeux."
            ),
            defaults={"boosting_type": "gbdt", "n_jobs": -1, "verbose": -1},
        ),
        AlgorithmSpec(
            name="lightgbm_dart",
            display_name="LightGBM DART (dropout sur les arbres)",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_booster,
            rationale=(
                "Applique du dropout aux arbres déjà construits : régularise les gros ensembles "
                "et limite la spécialisation des derniers arbres. Early stopping **indisponible** "
                "— le nombre de rounds doit être fixé à l'avance, d'où un coût plus élevé."
            ),
            defaults={
                "boosting_type": "dart",
                "drop_rate": 0.1,
                "skip_drop": 0.5,
                "n_jobs": -1,
                "verbose": -1,
            },
            accepted=_COMMON_PARAMS | _CLASSIFICATION_PARAMS | _DART_PARAMS,
            supports_early_stopping=False,
        ),
        AlgorithmSpec(
            name="lightgbm_goss",
            display_name="LightGBM GOSS (échantillonnage par gradient)",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_booster,
            rationale=(
                "Conserve les exemples à fort gradient et sous-échantillonne les autres : encore "
                "plus rapide sur les très gros volumes, sans bagging (`subsample` désactivé). "
                "Utile quand le coût d'entraînement domine la recherche de précision."
            ),
            defaults={"boosting_type": "goss", "n_jobs": -1, "verbose": -1},
        ),
    )
}


def filter_params(spec: AlgorithmSpec, params: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the hyper-parameters this algorithm accepts.

    GOSS does not bag rows: ``subsample`` is forced to ``1.0`` to avoid a LightGBM warning that
    would otherwise pollute every run.

    Args:
        spec: Registry entry of the algorithm.
        params: Candidate parameters.

    Returns:
        The applicable subset (rejected keys are logged, never silently dropped).
    """
    kept = {str(key): value for key, value in params.items() if key in spec.accepted}
    dropped = sorted(set(params) - set(kept))
    if dropped:
        logger.debug("{} : paramètre(s) ignoré(s) {}", spec.name, dropped)
    if spec.name == "lightgbm_goss":
        kept["subsample"] = 1.0
        kept.pop("subsample_freq", None)
    return kept


def resolve_algorithm(algorithm: str, task: str) -> AlgorithmSpec:
    """Return the registry entry of an algorithm, validating the task.

    Args:
        algorithm: Algorithm identifier.
        task: Learning task.

    Returns:
        The :class:`AlgorithmSpec`.

    Raises:
        ValueError: When the algorithm is unknown or does not serve ``task``.
    """
    spec = ESTIMATORS.get(str(algorithm))
    if spec is None:
        msg = (
            f"Unknown LightGBM algorithm '{algorithm}'. Available: {sorted(ESTIMATORS)} "
            f"(for task '{task}': {available_for_task(task)})"
        )
        raise ValueError(msg)
    if task not in spec.tasks:
        msg = (
            f"Algorithm '{algorithm}' does not serve task '{task}' "
            f"(it serves {sorted(spec.tasks)})."
            f" Pick one of {available_for_task(task)}."
        )
        raise ValueError(msg)
    return spec


def available_for_task(task: str) -> list[str]:
    """Return the algorithm names serving a task, in registry order.

    Args:
        task: Learning task (an empty value returns every algorithm).

    Returns:
        The algorithm identifiers.
    """
    if not task:
        return sorted(ESTIMATORS)
    return [name for name, spec in ESTIMATORS.items() if task in spec.tasks]


class RoundBridge:
    """Bridge between LightGBM's boosting rounds and the project's callbacks.

    LightGBM invokes any callable with its callback environment (``env.iteration``,
    ``env.evaluation_result_list``); we translate it into the ``loss`` / ``val_loss`` /
    ``val_<metric>`` keys the project expects, throttled to keep the history readable, and raise
    :class:`EarlyStopException` when a project callback asks to stop.
    """

    def __init__(
        self,
        model: LightGBMModel,
        context: Any,
        callbacks: Sequence[Any],
        *,
        primary: str,
        every: int = EMIT_EVERY_ROUNDS,
    ) -> None:
        """Store the emission target.

        Args:
            model: Model being trained (fires the project callbacks through it).
            context: Mutable project callback context.
            callbacks: Project callbacks to notify.
            primary: Primary metric name of the project.
            every: Emit one event every ``every`` rounds.
        """
        self.model = model
        self.context = context
        self.callbacks = list(callbacks)
        self.primary = primary
        self.every = max(int(every), 1)
        self.rounds = 0

    def __call__(self, env: Any) -> None:
        """Emit the project events of one boosting round.

        Args:
            env: LightGBM callback environment.

        Raises:
            EarlyStopException: When a project callback requested an early stop.
        """
        self.rounds = int(getattr(env, "iteration", self.rounds)) + 1
        logs = _read_evaluation_results(getattr(env, "evaluation_result_list", []), self.primary)
        logs["round"] = float(self.rounds)
        stop = False
        if self.rounds % self.every == 0 or self.context.stopped_early:
            stop = self.model.emit_epoch(self.context, self.callbacks, logs, epoch=self.rounds - 1)
        if stop:
            best = self.context.best_epoch if self.context.best_epoch >= 0 else self.rounds - 1
            raise EarlyStopException(int(best), [])


def _read_evaluation_results(results: Any, primary: str) -> dict[str, float]:
    """Translate LightGBM's ``evaluation_result_list`` into project log keys.

    Args:
        results: Sequence of ``(data_name, metric_name, value, is_higher_better)`` tuples.
        primary: Primary metric name of the project (used for the ``val_<primary>`` alias).

    Returns:
        The logs of the round.
    """
    logs: dict[str, float] = {}
    for entry in results or ():
        try:
            data_name, metric_name, value, _ = entry[:4]
            number = float(value)
        except (TypeError, ValueError, IndexError):  # pragma: no cover - défensif
            continue
        prefix = "val" if str(data_name).startswith("valid") else "train"
        logs[f"{prefix}_{metric_name}"] = number
        if prefix == "train":
            logs["loss"] = number
        else:
            logs["val_loss"] = number
            alias = _METRIC_ALIASES.get(str(metric_name))
            if alias and alias == primary:
                logs[f"val_{primary}"] = number
    return logs


class LightGBMModel(BaseModel):
    """A LightGBM booster behind the project's model contract."""

    framework = "lightgbm"
    default_model_file = "model.joblib"
    epochs_based = True

    def __init__(
        self,
        *,
        algorithm: str = "lightgbm",
        params: Mapping[str, Any] | None = None,
        task: str = "binary",
        feature_names: Sequence[str] | None = None,
        target_name: str | None = None,
        random_state: int = 42,
        supports_proba: bool | None = None,
        config: Mapping[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        """Store the contract and resolve the algorithm.

        Args:
            algorithm: Identifier present in :data:`ESTIMATORS`.
            params: Hyper-parameters (filtered against the algorithm's allow-list).
            task: Learning task.
            feature_names: Features expected at inference time.
            target_name: Business target column.
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
        self.best_iteration_: int | None = None

    # ------------------------------------------------------------------ construction ------
    def _resolved_params(self, labels: np.ndarray | None) -> dict[str, Any]:
        """Merge the registry defaults, the configuration and the data-driven ``auto`` values.

        Args:
            labels: Training target (used to resolve ``scale_pos_weight`` / ``class_weight``).

        Returns:
            The parameters handed to the booster.
        """
        resolved: dict[str, Any] = {**self.spec.defaults, **self._effective_params()}
        resolved.setdefault("random_state", self.random_state)
        if labels is not None:
            counts = pd.Series(labels).value_counts().to_numpy(dtype="float64")
            majority = float(counts[0]) if counts.size else 1.0
            others = float(counts[1:].sum()) if counts.size > 1 else 0.0
            ratio = round(max(others / max(majority, 1.0), 1.0), 4)
            if str(resolved.get("scale_pos_weight")) == "auto":
                resolved["scale_pos_weight"] = ratio
            if str(resolved.get("class_weight")) == "auto":
                resolved["class_weight"] = "balanced"
        return filter_params(self.spec, {k: v for k, v in resolved.items() if v != "auto"})

    def build_estimator(self, labels: np.ndarray | None = None) -> Any:
        """Instantiate the booster declared by ``algorithm``.

        Args:
            labels: Training target, used to resolve the imbalance parameters.

        Returns:
            A fresh (unfitted) estimator.
        """
        return _booster(
            __classifier__=self.task in _CLASSIFICATION,
            **self._resolved_params(labels),
        )

    def _planned_epochs(self) -> int:
        """Announce the number of boosting rounds to the callbacks."""
        try:
            return max(int(self.params.get("n_estimators", 100)), 1)
        except (TypeError, ValueError):  # pragma: no cover - défensif
            return 100

    def _early_stopping_rounds(self) -> int | None:
        """Number of rounds without improvement before stopping (``None`` when disabled)."""
        if not self.spec.supports_early_stopping:
            return None
        rounds = dict(self.config.get("model") or {}).get("early_stopping_rounds")
        try:
            return max(int(rounds), 1) if rounds else None
        except (TypeError, ValueError):  # pragma: no cover - défensif
            return None

    def _eval_metric(self) -> Any:
        """Evaluation metric(s) declared in the configuration (name or list of names)."""
        metric = dict(self.config.get("model") or {}).get("eval_metric")
        if not metric:
            return None
        if isinstance(metric, (list, tuple)):
            return [str(name) for name in metric]
        return str(metric)

    # ------------------------------------------------------------------ entraînement ------
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
        """Train the booster with native early stopping and per-round events.

        Args:
            X: Aligned training features.
            y: Training target.
            X_val: Validation features (enables ``eval_set`` and early stopping).
            y_val: Validation target.
            context: Project callback context.
            callbacks: Project callbacks.

        Raises:
            ValueError: When the task is supervised but no target is provided.
        """
        self._seed_everything()
        matrix = self._matrix(X)
        labels = None if y is None else np.asarray(y).ravel()
        if labels is None and self.is_supervised:
            msg = f"Task '{self.task}' requires a target; got y=None"
            raise ValueError(msg)

        params = self._resolved_params(labels)
        estimator = _booster(__classifier__=self.task in _CLASSIFICATION, **params)

        fit_kwargs: dict[str, Any] = {}
        lightgbm_callbacks: list[Any] = [log_evaluation(period=0)]
        validation = None
        if X_val is not None and y_val is not None:
            validation = (self._matrix(X_val), np.asarray(y_val).ravel())
            fit_kwargs["eval_set"] = [(matrix, labels), validation]
            metric = self._eval_metric()
            if metric:
                fit_kwargs["eval_metric"] = metric
            rounds = self._early_stopping_rounds()
            if rounds:
                lightgbm_callbacks.append(early_stopping(rounds, verbose=False))
        bridge: RoundBridge | None = None
        if context is not None:
            primary = str(dict(self.config.get("metrics") or {}).get("primary", ""))
            bridge = RoundBridge(self, context, list(callbacks or []), primary=primary)
            lightgbm_callbacks.append(bridge)
        fit_kwargs["callbacks"] = lightgbm_callbacks

        try:
            if labels is None:  # pragma: no cover - garde-fou (tâche supervisée vérifiée plus haut)
                estimator.fit(matrix, **fit_kwargs)
            else:
                estimator.fit(matrix, labels, **fit_kwargs)
        except EarlyStopException as stopped:
            # Arrêt demandé par un callback du projet : le booster conserve son meilleur état.
            logger.info(
                "Entraînement interrompu au round {}", getattr(stopped, "best_iteration", -1)
            )

        self.estimator_ = estimator
        self.params = dict(params)
        self.classes_ = getattr(estimator, "classes_", None)
        if self._supports_proba is None:
            self._supports_proba = bool(hasattr(estimator, "predict_proba"))
        self.best_iteration_ = _best_iteration(estimator)
        if context is not None:
            context.extra["best_iteration"] = self.best_iteration_
            context.extra["n_rounds"] = bridge.rounds if bridge else None
            context.extra["boosting_type"] = params.get(
                "boosting_type", self.spec.defaults.get("boosting_type")
            )
        logger.info(
            "LightGBM fitted | {} rounds | best_iteration={} | val={}",
            bridge.rounds if bridge else params.get("n_estimators"),
            self.best_iteration_,
            validation is not None,
        )

    # ------------------------------------------------------------------ prédiction --------
    def _predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict with the fitted booster (best iteration when early stopping fired).

        Args:
            X: Aligned features.

        Returns:
            The predictions.

        Raises:
            RuntimeError: When no booster is attached.
        """
        estimator = self._require_estimator()
        return np.asarray(estimator.predict(self._matrix(X), **self._predict_kwargs())).ravel()

    def _predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities with the fitted booster.

        Args:
            X: Aligned features.

        Returns:
            A ``(n_samples, n_classes)`` array.

        Raises:
            NotImplementedError: When the booster is a regressor.
        """
        estimator = self._require_estimator()
        if not hasattr(estimator, "predict_proba"):
            msg = f"{type(estimator).__name__} exposes no predict_proba"
            raise NotImplementedError(msg)
        return np.asarray(
            estimator.predict_proba(self._matrix(X), **self._predict_kwargs()), dtype="float64"
        )

    def _predict_kwargs(self) -> dict[str, Any]:
        """Restrict prediction to the rounds kept by early stopping, when applicable."""
        if self.best_iteration_:
            return {"num_iteration": self.best_iteration_}
        return {}

    def _require_estimator(self) -> Any:
        """Return the fitted booster or raise a diagnostic error."""
        if self.estimator_ is None:
            msg = "No booster is attached to this model (fit or load it first)"
            raise RuntimeError(msg)
        return self.estimator_

    # ------------------------------------------------------------------ persistance -------
    def save(self, path: str | Path) -> Path:
        """Persist the wrapper (contract + booster) with joblib.

        Args:
            path: Destination file or directory.

        Returns:
            The written path.
        """
        import joblib

        self.check_is_fitted()
        destination = self._resolve_path(path)
        joblib.dump(self, destination, compress=3)
        logger.info("Model saved: {} ({} booster)", destination, self.algorithm)
        return destination

    @classmethod
    def load(cls, path: str | Path, *, config: Mapping[str, Any] | None = None) -> BaseModel:
        """Reload a model persisted by :meth:`save`.

        Args:
            path: Artefact path.
            config: Optional configuration refreshed into the reloaded model.

        Returns:
            The reloaded :class:`LightGBMModel`.

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
        """Add the booster identity and the best iteration to the representation."""
        fields = [f"algorithm={self.algorithm or '-'}"]
        if self.estimator_ is not None:
            fields.append(f"booster={type(self.estimator_).__name__}")
        fields += [f"task={self.task}", f"features={len(self.feature_names)}"]
        if self.best_iteration_ is not None:
            fields.append(f"best_iteration={self.best_iteration_}")
        fields.append(f"state={self.state}")
        return fields

    def model_card(
        self,
        *,
        metrics: Mapping[str, Any] | None = None,
        artifact: str | None = None,
        notes: Iterable[str] | None = None,
    ) -> ModelCard:
        """Build the model card, enriched with the boosting diagnostics.

        Args:
            metrics: Metrics to archive.
            artifact: Serialised model file name.
            notes: Extra remarks.

        Returns:
            The :class:`ModelCard`.
        """
        card = super().model_card(metrics=metrics, artifact=artifact, notes=notes)
        card.library_versions = library_versions()
        extra: list[str] = []
        if self.best_iteration_ is not None:
            extra.append(f"meilleur round (early stopping) : {self.best_iteration_}")
        if self.estimator_ is not None:
            extra.append(f"booster natif : {type(self.estimator_).__name__}")
        card.notes = [*card.notes, *extra]
        return card


def _best_iteration(estimator: Any) -> int | None:
    """Read the boosting round selected by early stopping, when available."""
    booster = getattr(estimator, "booster_", None)
    for holder in (booster, estimator):
        value = getattr(holder, "best_iteration", None) if holder is not None else None
        if value:
            try:
                return int(value)
            except (TypeError, ValueError):  # pragma: no cover - défensif
                continue
    return None


__all__ = [
    "EMIT_EVERY_ROUNDS",
    "ESTIMATORS",
    "AlgorithmSpec",
    "FitResult",
    "LightGBMModel",
    "ModelCard",
    "available_for_task",
    "filter_params",
    "resolve_algorithm",
]
