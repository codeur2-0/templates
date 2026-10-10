"""Experiment tracking behind a single seam.

The pipelines call :func:`build_tracker` once and hand it what they produced (configuration,
training outcome, fitted model, metrics, artefacts). Most stacks ship the no-op
:class:`~src.tracking.tracker.NullTracker`; a tracking stack (MLflow) overrides
``src/tracking/tracker.py`` without touching a single pipeline.
"""

from src.tracking.tracker import Tracker, build_tracker

__all__ = ["Tracker", "build_tracker"]
