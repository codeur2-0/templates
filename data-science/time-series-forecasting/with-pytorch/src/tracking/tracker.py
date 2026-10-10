"""No-op experiment tracker (the default of every stack without a tracking backend).

Keeping a real object — rather than ``if tracker is not None`` checks in every pipeline — makes the
tracking seam explicit and testable: a stack that tracks experiments replaces this file, the
pipelines stay identical.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.utils.paths import ProjectPaths


class Tracker(Protocol):
    """What the pipelines expect from an experiment tracker."""

    enabled: bool

    def log_training(
        self,
        *,
        config: Mapping[str, Any],
        metrics: Mapping[str, float],
        model: Any,
        artifacts: Iterable[str | Path],
        sample: pd.DataFrame | None = None,
    ) -> None:
        """Record one training run (parameters, metrics, artefacts, model)."""

    def log_evaluation(
        self, *, metrics: Mapping[str, float], artifacts: Iterable[str | Path]
    ) -> None:
        """Attach the held-out evaluation to the training run."""


class NullTracker:
    """Tracker that records nothing (artefacts on disk remain the source of truth)."""

    enabled = False

    def log_training(
        self,
        *,
        config: Mapping[str, Any],
        metrics: Mapping[str, float],
        model: Any,
        artifacts: Iterable[str | Path],
        sample: pd.DataFrame | None = None,
    ) -> None:
        """Ignore the training run.

        Args:
            config: Resolved configuration.
            metrics: Training / validation metrics.
            model: Fitted model.
            artifacts: Files produced by the pipeline.
            sample: A few rows of the model input (signature example).
        """
        del config, metrics, model, artifacts, sample

    def log_evaluation(
        self, *, metrics: Mapping[str, float], artifacts: Iterable[str | Path]
    ) -> None:
        """Ignore the evaluation.

        Args:
            metrics: Test metrics.
            artifacts: Files produced by the evaluation.
        """
        del metrics, artifacts


def build_tracker(config: Mapping[str, Any], paths: ProjectPaths) -> Tracker:
    """Return the tracker of this stack.

    Args:
        config: Resolved configuration (``model_dump()`` of the application config).
        paths: Project layout.

    Returns:
        A :class:`NullTracker`: this stack does not track experiments.
    """
    del config, paths
    return NullTracker()
