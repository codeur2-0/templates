"""Common contract for the end-to-end pipelines of the project.

Four pipelines cover the whole lifecycle and are the only entry points used by
``src/main.py`` and the ``scripts/`` helpers:

``DataGenerationPipeline``  create the synthetic example dataset
``TrainPipeline``           load -> validate -> split -> preprocess -> fit -> evaluate -> save
``EvaluationPipeline``      reload a trained artefact and produce reports / figures
``InferencePipeline``       score new records (file or sample) with a trained artefact

Every pipeline returns a :class:`PipelineResult`, which makes the outcome machine readable
(CI gates, notebooks, dashboards) instead of relying on side effects and print statements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths
from src.utils.utils import timer

logger = get_logger(__name__)


@dataclass(slots=True)
class PipelineResult:
    """Structured outcome of a pipeline execution.

    Attributes:
        name: Pipeline name.
        status: ``success`` or ``failure``.
        metrics: Metrics produced (may be empty).
        artifacts: Paths of the files written.
        duration_seconds: Wall-clock duration.
        payload: Arbitrary result object (predictions, report, fitted model, ...).
        messages: Human readable summary lines.
    """

    name: str
    status: str = "success"
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    payload: Any = None
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result, dropping the non-serialisable payload reference."""
        data = asdict(self)
        data["payload"] = type(self.payload).__name__ if self.payload is not None else None
        return data

    def add_artifact(self, path: str | Path) -> None:
        """Register a produced artefact.

        Args:
            path: Artefact path.
        """
        self.artifacts.append(str(path))

    @property
    def succeeded(self) -> bool:
        """Whether the pipeline completed successfully."""
        return self.status == "success"


class BasePipeline(ABC):
    """Base class of every pipeline: configuration in, structured result out."""

    #: Name used in logs and results.
    name: str = "pipeline"

    def __init__(self, config: Any, paths: ProjectPaths | None = None) -> None:
        """Store the typed configuration and resolve the filesystem layout.

        Args:
            config: Validated :class:`src.schemas.config.AppConfig` instance.
            paths: Optional explicit layout (defaults to the config-derived one).
        """
        self.config = config
        self.paths = paths or ProjectPaths.from_config(config.model_dump())

    @abstractmethod
    def _execute(self) -> PipelineResult:
        """Pipeline specific logic, implemented by subclasses.

        Returns:
            The pipeline result.
        """

    def run(self) -> PipelineResult:
        """Execute the pipeline with timing, logging and error encapsulation.

        Returns:
            The :class:`PipelineResult` of the run.
        """
        logger.info("=== Pipeline '{}' started ===", self.name)
        with timer(self.name) as elapsed:
            try:
                result = self._execute()
            except Exception as exc:  # noqa: BLE001 - the pipeline is the error boundary
                logger.exception("Pipeline '{}' failed", self.name)
                return PipelineResult(
                    name=self.name,
                    status="failure",
                    duration_seconds=elapsed["seconds"],
                    messages=[f"{type(exc).__name__}: {exc}"],
                )
        result.duration_seconds = elapsed["seconds"]
        logger.info(
            "=== Pipeline '{}' finished in {:.2f}s (status={}, artifacts={}) ===",
            self.name,
            result.duration_seconds,
            result.status,
            len(result.artifacts),
        )
        return result
