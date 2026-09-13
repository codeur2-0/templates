"""XGBoost implementation of the model contract.

XGBoost is a **regularised gradient boosting** library: second-order gradients, shrinkage, L1/L2
penalties and column/row subsampling. On structured tabular data it is the reference to beat, which
is why this project keeps it behind exactly the same contract as scikit-learn
(:class:`~src.models.base.BaseModel`): the trainer, the evaluator and the predictor do not know
which framework they are talking to.

Three implementation choices matter here:

* **native early stopping** — ``early_stopping_rounds`` + ``eval_set`` let XGBoost stop as soon as
  the validation metric plateaus, and ``best_iteration`` is archived in the fit result;
* **one callback per boosting round** — the project's callbacks (history, logging, early stopping,
  quality threshold) receive the same events as they would from an epoch-based framework;
* **explicit parameter allow-list** — the sklearn wrappers of XGBoost accept ``**kwargs``, so an
  inapplicable hyper-parameter would only fail deep inside the booster: filtering up front gives an
  immediate, actionable diagnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from xgboost.callback import TrainingCallback

from src.models.base import BaseModel, FitResult, ModelCard, library_versions
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Tasks served by the classification variants.
_CLASSIFICATION = frozenset({"binary", "multiclass"})
#: Tasks served by the regression variants.
_REGRESSION = frozenset({"regression", "forecasting"})

#: Hyper-parameters accepted by every booster of this stack.
_COMMON_PARAMS: frozenset[str] = frozenset(
    {
        "n_estimators",
        "max_depth",
        "max_leaves",
        "max_bin",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "colsample_bylevel",
        "min_child_weight",
        "reg_lambda",
        "reg_alpha",
        "gamma",
        "objective",
        "tree_method",
        "grow_policy",
        "random_state",
        "n_jobs",
        "verbosity",
        "importance_type",
        "early_stopping_rounds",
        "eval_metric",
        "booster",
    }
)

#: Paramètres réservés aux tâches de classification.
_CLASSIFICATION_PARAMS: frozenset[str] = frozenset({"scale_pos_weight", "num_class"})

#: Paramètres propres au booster DART (dropout sur les arbres).
_DART_PARAMS: frozenset[str] = frozenset(
    {"rate_drop", "skip_drop", "sample_type", "normalize_type", "one_drop"}
)

#: Traduction de la métrique native XGBoost vers le nom du registre du projet.
_METRIC_ALIASES: dict[str, str] = {
    "auc": "roc_auc",
    "aucpr": "pr_auc",
    "logloss": "log_loss",
    "mlogloss": "log_loss",
    "error": "accuracy",
    "rmse": "rmse",
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
    """

    name: str
    display_name: str
    tasks: frozenset[str]
    builder: Callable[..., Any]
    rationale: str = ""
    iterative: bool = True
    defaults: dict[str, Any] = field(default_factory=dict)
    accepted: frozenset[str] = _COMMON_PARAMS | _CLASSIFICATION_PARAMS


def _booster(**params: Any) -> Any:
    """Build an XGBClassifier or XGBRegressor according to ``__classifier__``.

    Args:
        **params: Booster parameters.

    Returns:
        The unfitted estimator.
    """
    from xgboost import XGBClassifier, XGBRegressor

    if params.pop("__classifier__", False):
        return XGBClassifier(**params)
    return XGBRegressor(**params)


#: Registry of every algorithm this stack can build.
ESTIMATORS: dict[str, AlgorithmSpec] = {
    spec.name: spec
    for spec in (
        AlgorithmSpec(
            name="xgboost",
            display_name="XGBoost (arbres, gbtree)",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_booster,
            rationale=(
                "Boosting régularisé par gradients d'ordre 2 : la référence sur données "
                "tabulaires. Early stopping natif sur `eval_set`, subsampling des lignes et des "
                "colonnes, pénalités L1/L2."
            ),
            defaults={"tree_method": "hist", "n_jobs": -1, "verbosity": 0},
        ),
        AlgorithmSpec(
            name="xgboost_dart",
            display_name="XGBoost DART (dropout sur les arbres)",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_booster,
            rationale=(
                "Applique du dropout aux arbres déjà construits : régularise les gros ensembles et "
                "réduit la spécialisation des derniers arbres. Early stopping non supporté par le "
                "booster DART — le nombre de rounds doit être fixé à l'avance."
            ),
            defaults={
                "tree_method": "hist",
                "booster": "dart",
                "rate_drop": 0.1,
                "skip_drop": 0.5,
                "n_jobs": -1,
                "verbosity": 0,
            },
            accepted=_COMMON_PARAMS | _CLASSIFICATION_PARAMS | _DART_PARAMS,
        ),
        AlgorithmSpec(
            name="xgboost_linear",
            display_name="XGBoost linéaire (gblinear)",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_booster,
            rationale=(
                "Booster linéaire : vérifie qu'une part du signal est bien additive avant d'empiler "
                "des arbres. Sert de plancher interprétable, au même titre qu'une régression "
                "logistique ou un Ridge."
            ),
            defaults={"booster": "gblinear", "reg_lambda": 1.0, "n_jobs": -1, "verbosity": 0},
        ),
    )
}


def filter_params(spec: AlgorithmSpec, params: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the hyper-parameters this algorithm accepts.

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
            f"Unknown XGBoost algorithm '{algorithm}'. Available: {sorted(ESTIMATORS)} "
            f"(for task '{task}': {available_for_task(task)})"
        )
        raise ValueError(msg)
    if task not in spec.tasks:
        msg = (
            f"Algorithm '{algorithm}' does not serve task '{task}' (it serves {sorted(spec.tasks)})."
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


class RoundBridge(TrainingCallback):
    """Bridge between XGBoost's boosting rounds and the project's callbacks.

    XGBoost calls :meth:`after_iteration` once per round; the bridge translates the evaluation
    history into the ``loss`` / ``val_loss`` / ``val_<metric>`` keys the project callbacks expect,
    throttled to keep the history readable, and returns ``True`` to stop training when a project
    callback (early stopping) asks for it.

    The class is defined at module level on purpose: the estimator keeps a reference to its
    callbacks, and a locally defined class cannot be pickled when the model is archived.
    """

    def __init__(
        self,
        model: XGBoostModel,
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
        super().__init__()
        self.model = model
        self.context = context
        self.callbacks = list(callbacks)
        self.primary = primary
        self.every = max(int(every), 1)
        self.rounds = 0

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
        """Emit the project events of one boosting round.

        Args:
            model: Booster being trained.
            epoch: 0-based boosting round.
            evals_log: Evaluation history ``{data_name: {metric_name: [values]}}``.

        Returns:
            ``True`` when training must stop (a project callback requested it).
        """
        self.rounds = int(epoch) + 1
        stop_requested = bool(self.context.stopped_early)
        if not stop_requested and self.rounds % self.every:
            return False
        logs = _read_evals_log(evals_log, self.primary)
        logs["round"] = float(self.rounds)
        return self.model.emit_epoch(self.context, self.callbacks, logs, epoch=epoch)


def _read_evals_log(evals_log: Any, primary: str) -> dict[str, float]:
    """Extract the last train / validation metric of an XGBoost evaluation history.

    Args:
        evals_log: Mapping ``{data_name: {metric_name: [values]}}``.
        primary: Primary metric name of the project (used for the ``val_<primary>`` alias).

    Returns:
        The logs of the round (``loss``, ``val_loss``, optionally ``val_<primary>``).
    """
    logs: dict[str, float] = {}
    try:
        per_set = dict(evals_log or {})
    except (TypeError, ValueError):  # pragma: no cover - défensif
        return logs
    for index, (_, metrics) in enumerate(sorted(per_set.items())):
        prefix = "train" if index == 0 else "val"
        for metric_name, values in dict(metrics or {}).items():
            if not values:
                continue
            try:
                value = float(np.asarray(values).ravel()[-1])
            except (TypeError, ValueError):  # pragma: no cover - défensif
                continue
            logs[f"{prefix}_{metric_name}"] = value
            if prefix == "train":
                logs["loss"] = value
            else:
                logs["val_loss"] = value
                alias = _METRIC_ALIASES.get(str(metric_name))
                if alias and alias == primary:
                    logs[f"val_{primary}"] = value
    return logs


class XGBoostModel(BaseModel):
    """An XGBoost booster behind the project's model contract."""

    framework = "xgboost"
    default_model_file = "model.joblib"
    epochs_based = True

    def __init__(
        self,
        *,
        algorithm: str = "xgboost",
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
            labels: Training target (used to resolve ``scale_pos_weight="auto"``).

        Returns:
            The parameters handed to the booster.
        """
        model_node = dict(self.config.get("model") or {})
        resolved: dict[str, Any] = {**self.spec.defaults, **self._effective_params()}
        resolved.setdefault("random_state", self.random_state)
        if labels is not None and str(resolved.get("scale_pos_weight")) == "auto":
            counts = pd.Series(labels).value_counts().to_numpy(dtype="float64")
            majority = float(counts[0]) if counts.size else 1.0
            others = float(counts[1:].sum()) if counts.size > 1 else 0.0
            resolved["scale_pos_weight"] = round(max(others / max(majority, 1.0), 1.0), 4)
        resolved = {key: value for key, value in resolved.items() if value != "auto"}
        eval_metric = model_node.get("eval_metric")
        if eval_metric:
            # XGBoost accepte un nom unique ou une liste (l'early stopping surveille le dernier).
            resolved.setdefault(
                "eval_metric",
                [str(name) for name in eval_metric]
                if isinstance(eval_metric, (list, tuple))
                else str(eval_metric),
            )
        return filter_params(self.spec, resolved)

    def _early_stopping_rounds(self) -> int | None:
        """Number of rounds without improvement before stopping (``None`` when disabled).

        XGBoost refuses ``early_stopping_rounds`` without an ``eval_set``: the value is therefore
        only injected when a validation split is actually provided (see :meth:`_fit`).
        """
        if self.spec.name == "xgboost_dart":
            return None  # le booster DART ne supporte pas l'early stopping
        rounds = dict(self.config.get("model") or {}).get("early_stopping_rounds")
        try:
            return max(int(rounds), 1) if rounds else None
        except (TypeError, ValueError):  # pragma: no cover - défensif
            return None

    def build_estimator(self, labels: np.ndarray | None = None) -> Any:
        """Instantiate the booster declared by ``algorithm``.

        Args:
            labels: Training target, used to resolve the imbalance parameters.

        Returns:
            A fresh (unfitted) estimator.
        """
        return _build(
            self.spec,
            self._resolved_params(labels),
            is_classifier=self.task in _CLASSIFICATION,
        )

    def _planned_epochs(self) -> int:
        """Announce the number of boosting rounds to the callbacks."""
        try:
            return max(int(self.params.get("n_estimators", self.spec.defaults.get("n_estimators", 100))), 1)
        except (TypeError, ValueError):  # pragma: no cover - défensif
            return 100

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
        is_classifier = self.task in _CLASSIFICATION

        # L'API sklearn d'XGBoost prend les callbacks au **constructeur** : un `set_params`
        # postérieur reconfigure le booster C++ et sérialise mal `eval_metric` (bug observé en
        # 3.2.0). Le pont est donc créé avant l'estimateur, puis détaché après l'entraînement.
        bridge: RoundBridge | None = None
        if context is not None:
            primary = str(dict(self.config.get("metrics") or {}).get("primary", ""))
            bridge = RoundBridge(self, context, list(callbacks or []), primary=primary)

        fit_kwargs: dict[str, Any] = {"verbose": False}
        validation = None
        if X_val is not None and y_val is not None:
            validation = (self._matrix(X_val), np.asarray(y_val).ravel())
            fit_kwargs["eval_set"] = [(matrix, labels), validation]
            rounds = self._early_stopping_rounds()
            if rounds:
                params["early_stopping_rounds"] = rounds

        estimator = _build(
            self.spec,
            params,
            is_classifier=is_classifier,
            callbacks=None if bridge is None else [bridge],
        )

        if labels is None:  # pragma: no cover - garde-fou (tâche non supervisée refusée plus haut)
            estimator.fit(matrix, **fit_kwargs)
        else:
            estimator.fit(matrix, labels, **fit_kwargs)

        # Le booster conserve une référence à ses callbacks : on la retire (par affectation
        # directe, jamais via `set_params`) pour que l'artefact joblib reste léger et indépendant
        # du contexte d'entraînement.
        estimator.callbacks = None
        self.estimator_ = estimator
        self.params = dict(params)
        self.classes_ = getattr(estimator, "classes_", None)
        if self._supports_proba is None:
            self._supports_proba = bool(hasattr(estimator, "predict_proba"))
        self.best_iteration_ = _best_iteration(estimator)
        if context is not None:
            context.extra["best_iteration"] = self.best_iteration_
            context.extra["n_rounds"] = bridge.rounds if bridge else None
            context.extra["early_stopping_rounds"] = params.get("early_stopping_rounds")
        logger.info(
            "XGBoost fitted | {} rounds | best_iteration={} | val={}",
            bridge.rounds if bridge else params.get("n_estimators"),
            self.best_iteration_,
            validation is not None,
        )

    # ------------------------------------------------------------------ prédiction --------
    def _predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict with the fitted booster.

        Args:
            X: Aligned features.

        Returns:
            The predictions.

        Raises:
            RuntimeError: When no booster is attached.
        """
        estimator = self._require_estimator()
        return np.asarray(estimator.predict(self._matrix(X))).ravel()

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
        return np.asarray(estimator.predict_proba(self._matrix(X)), dtype="float64")

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
            The reloaded :class:`XGBoostModel`.

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


def _build(
    spec: AlgorithmSpec,
    params: Mapping[str, Any],
    *,
    is_classifier: bool,
    callbacks: Sequence[Any] | None = None,
) -> Any:
    """Instantiate the estimator of a registry entry.

    Args:
        spec: Algorithm entry.
        params: Filtered parameters (``__classifier__`` is ignored here).
        is_classifier: Whether to build the classification variant.
        callbacks: XGBoost training callbacks (passed to the constructor, never to ``fit``).

    Returns:
        The unfitted estimator.
    """
    cleaned = {key: value for key, value in params.items() if key != "__classifier__"}
    if callbacks:
        cleaned["callbacks"] = list(callbacks)
    return spec.builder(__classifier__=is_classifier, **cleaned)


def _best_iteration(estimator: Any) -> int | None:
    """Read the boosting round selected by early stopping, when available."""
    for attribute in ("best_iteration", "best_ntree_limit"):
        value = getattr(estimator, attribute, None)
        if value is None:
            continue
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
    "ModelCard",
    "XGBoostModel",
    "available_for_task",
    "filter_params",
    "resolve_algorithm",
]
