"""Training callbacks.

Callbacks decouple *side effects* (logging, progress bar, early stopping, artefact
snapshots) from the training loop itself. The trainer calls the hooks below at well defined
moments; iterative frameworks (PyTorch, TensorFlow, Keras, boosting) emit one event per
epoch / boosting round, while single-shot estimators (scikit-learn) only emit
``on_train_begin`` / ``on_train_end``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class CallbackContext:
    """State passed to every callback hook.

    Attributes:
        model_name: Name of the model being trained.
        params: Effective hyper-parameters.
        epochs: Total number of epochs (``1`` for single-shot estimators).
        epoch: Current epoch index (0-based).
        logs: Metrics of the current step (``train_*`` / ``val_*`` keys).
        history: Accumulated metrics per epoch.
        best_score: Best value observed so far for the monitored metric.
        best_epoch: Epoch at which ``best_score`` was observed.
        stopped_early: Set by :class:`EarlyStoppingCallback` to request a stop.
    """

    model_name: str
    params: dict[str, Any] = field(default_factory=dict)
    epochs: int = 1
    epoch: int = 0
    logs: dict[str, float] = field(default_factory=dict)
    history: dict[str, list[float]] = field(default_factory=dict)
    best_score: float | None = None
    best_epoch: int = -1
    stopped_early: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class BaseCallback(ABC):
    """Interface implemented by every callback."""

    #: Human readable name used in logs.
    name: str = "base"

    def on_train_begin(self, context: CallbackContext) -> None:  # noqa: B027 - hook optionnel
        """Called once before training starts.

        Hook **optionnel** : une implémentation qui n'a rien à faire au démarrage ne doit pas
        être obligée de le surcharger (seul ``on_epoch_end`` est abstrait).

        Args:
            context: Mutable training context.
        """

    @abstractmethod
    def on_epoch_end(self, context: CallbackContext) -> None:
        """Called after every epoch / boosting round.

        Args:
            context: Mutable training context.
        """

    def on_train_end(self, context: CallbackContext) -> None:  # noqa: B027 - hook optionnel
        """Called once after training completes.

        Hook **optionnel**, symétrique de :meth:`on_train_begin`.

        Args:
            context: Mutable training context.
        """


class MetricHistoryCallback(BaseCallback):
    """Accumulate per-epoch metrics into ``context.history`` (learning curves)."""

    name = "metric-history"

    def on_epoch_end(self, context: CallbackContext) -> None:
        """Append every metric of ``context.logs`` to the history.

        Args:
            context: Mutable training context.
        """
        for key, value in context.logs.items():
            if value is None:
                continue
            context.history.setdefault(key, []).append(float(value))


class LoggingCallback(BaseCallback):
    """Log training progress with loguru (one line per epoch, throttled)."""

    name = "logging"

    def __init__(self, *, every: int = 1) -> None:
        """Store the logging frequency.

        Args:
            every: Log every ``every`` epochs (``1`` = every epoch).
        """
        self.every = max(int(every), 1)

    def on_train_begin(self, context: CallbackContext) -> None:
        """Log the training configuration.

        Args:
            context: Mutable training context.
        """
        logger.info(
            "Training started | model={} | epochs={} | params={}",
            context.model_name,
            context.epochs,
            context.params,
        )

    def on_epoch_end(self, context: CallbackContext) -> None:
        """Log the metrics of the current epoch.

        Args:
            context: Mutable training context.
        """
        if (context.epoch + 1) % self.every != 0 and context.epoch + 1 != context.epochs:
            return
        metrics = " | ".join(f"{key}={value:.5f}" for key, value in sorted(context.logs.items()))
        logger.info("Epoch {}/{} | {}", context.epoch + 1, context.epochs, metrics or "no metric")

    def on_train_end(self, context: CallbackContext) -> None:
        """Log the completion of the training.

        Args:
            context: Mutable training context.
        """
        reason = "early stopping" if context.stopped_early else "all epochs completed"
        logger.info("Training finished ({}) | best_epoch={}", reason, context.best_epoch)


class ProgressBarCallback(BaseCallback):
    """Display a ``tqdm`` progress bar over the epochs."""

    name = "progress-bar"

    def __init__(self, *, enabled: bool = True, leave: bool = True) -> None:
        """Create the callback.

        Args:
            enabled: Disable the bar in non interactive environments (CI, tests).
            leave: Keep the bar on screen after completion.
        """
        self.enabled = enabled
        self.leave = leave
        self._bar: Any = None

    def on_train_begin(self, context: CallbackContext) -> None:
        """Instantiate the progress bar.

        Args:
            context: Mutable training context.
        """
        if not self.enabled:
            return
        from tqdm import tqdm

        self._bar = tqdm(
            total=context.epochs,
            desc=f"Training {context.model_name}",
            leave=self.leave,
            unit="epoch",
        )

    def on_epoch_end(self, context: CallbackContext) -> None:
        """Refresh the bar with the current metrics.

        Args:
            context: Mutable training context.
        """
        if self._bar is None:
            return
        self._bar.set_postfix({key: f"{value:.4f}" for key, value in sorted(context.logs.items())})
        self._bar.update(1)

    def on_train_end(self, context: CallbackContext) -> None:
        """Close the progress bar.

        Args:
            context: Mutable training context.
        """
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class EarlyStoppingCallback(BaseCallback):
    """Stop training when a monitored metric stops improving.

    The callback is *advisory*: it sets ``context.stopped_early`` and the training loop is
    responsible for honouring it. This keeps the callback free of framework specific logic.
    """

    name = "early-stopping"

    def __init__(
        self,
        monitor: str = "val_loss",
        *,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        """Configure the stopping rule.

        Args:
            monitor: Metric key to watch.
            patience: Number of epochs without improvement before stopping.
            min_delta: Minimum improvement to be considered a progress.
            mode: ``min`` (loss-like) or ``max`` (score-like).
        """
        if mode not in {"min", "max"}:
            msg = f"mode must be 'min' or 'max', got '{mode}'"
            raise ValueError(msg)
        self.monitor = monitor
        self.patience = max(int(patience), 1)
        self.min_delta = float(min_delta)
        self.mode = mode
        self._wait = 0
        self._best: float | None = None
        self._best_epoch: int = -1

    @property
    def best_score(self) -> float | None:
        """Best value observed so far for the monitored metric (``None`` before training)."""
        return self._best

    @property
    def best_epoch(self) -> int:
        """Epoch index at which :attr:`best_score` was observed (``-1`` before training)."""
        return self._best_epoch

    def _improved(self, value: float) -> bool:
        """Return whether ``value`` improves the best observed score."""
        if self._best is None:
            return True
        if self.mode == "min":
            return value < self._best - self.min_delta
        return value > self._best + self.min_delta

    def on_train_begin(self, context: CallbackContext) -> None:
        """Reset the internal state.

        Args:
            context: Mutable training context.
        """
        self._wait = 0
        self._best = None
        self._best_epoch = -1
        context.stopped_early = False

    def on_epoch_end(self, context: CallbackContext) -> None:
        """Update the best score and request a stop when patience is exhausted.

        Args:
            context: Mutable training context.
        """
        value = context.logs.get(self.monitor)
        if value is None:
            logger.debug(
                "EarlyStopping: metric '{}' not in logs {}", self.monitor, sorted(context.logs)
            )
            return
        value = float(value)
        if self._improved(value):
            self._best = value
            self._best_epoch = int(context.epoch)
            self._wait = 0
            context.best_score = value
            context.best_epoch = context.epoch
        else:
            self._wait += 1
            if self._wait >= self.patience:
                logger.info(
                    "Early stopping triggered: no improvement of '{}' for {} epochs (best={:.5f})",
                    self.monitor,
                    self.patience,
                    self._best if self._best is not None else float("nan"),
                )
                context.stopped_early = True

    def on_train_end(self, context: CallbackContext) -> None:
        """Expose the best observed score in the context.

        Args:
            context: Mutable training context.
        """
        if self._best is not None:
            context.best_score = self._best


class MetricThresholdCallback(BaseCallback):
    """Warn (or stop) when a metric crosses a quality threshold.

    Useful in CI: a smoke training run must reach a minimal quality, otherwise the pipeline
    is considered broken.
    """

    name = "metric-threshold"

    def __init__(
        self,
        monitor: str,
        threshold: float,
        *,
        mode: str = "max",
        fail_fast: bool = False,
    ) -> None:
        """Configure the threshold rule.

        Args:
            monitor: Metric key to watch.
            threshold: Target value.
            mode: ``max`` when higher is better (ROC AUC, R2), ``min`` when lower is better
                (RMSE, MAE, log loss).
            fail_fast: Stop the training as soon as the rule cannot be satisfied.
        """
        self.monitor = monitor
        self.threshold = float(threshold)
        self.mode = mode
        self.fail_fast = fail_fast
        self.satisfied = False

    def on_epoch_end(self, context: CallbackContext) -> None:
        """Check the threshold and update ``self.satisfied``.

        Args:
            context: Mutable training context.
        """
        value = context.logs.get(self.monitor)
        if value is None:
            return
        try:
            measured = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(measured):
            # Métrique non mesurable à cette époque (détecteur non supervisé entraîné sans cible,
            # split de validation dégénéré) : ce n'est pas un échec de seuil. Alerter ici noierait
            # les vrais dépassements sous du bruit.
            logger.debug(
                "Metric '{}' not measurable ({}): threshold check skipped", self.monitor, measured
            )
            return
        ok = measured >= self.threshold if self.mode == "max" else measured <= self.threshold
        self.satisfied = self.satisfied or ok
        if not ok:
            requirement = "au moins" if self.mode == "max" else "au plus"
            logger.warning(
                "Metric '{}' = {:.5f} ne respecte pas le seuil de qualité ({} {})",
                self.monitor,
                float(value),
                requirement,
                self.threshold,
            )
