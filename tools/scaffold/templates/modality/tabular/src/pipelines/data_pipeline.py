"""Data generation pipeline: produce the synthetic example dataset.

This pipeline is the first step of ``mode=all`` and the target of ``make data``. It
instantiates the family-specific :class:`src.data.generators.SyntheticDataGenerator` with the
values of ``conf/data/default.yaml``, writes the files, validates them against the raw
contract and persists a generation metadata file (fingerprint, distributions, missing rate).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.generators import SyntheticDataGenerator
from src.data.schemas import RawDataSchema
from src.pipelines.base import BasePipeline, PipelineResult
from src.utils.io import read_table, write_json
from src.utils.logging import get_logger
from src.utils.utils import frame_fingerprint

logger = get_logger(__name__)


class DataGenerationPipeline(BasePipeline):
    """Generate ``data/raw/*`` and its metadata."""

    name = "generate-data"

    def _execute(self) -> PipelineResult:
        """Generate, export, validate and document the dataset.

        Returns:
            The pipeline result (files written + profile metrics).
        """
        data_config = self.config.data
        self.paths.ensure()

        options = _generator_options(data_config)
        generator = SyntheticDataGenerator(
            n_samples=int(data_config.n_samples),
            seed=int(self.config.seed),
            dataset_name=str(data_config.dataset_name),
            output_dir=self.paths.raw_dir,
            formats=tuple(data_config.formats),
            **options,
        )

        logger.info(
            "Generating synthetic dataset '{}' | rows={} seed={} options={}",
            data_config.dataset_name,
            data_config.n_samples,
            self.config.seed,
            options or "{}",
        )
        frame = generator.generate()
        written = generator.export(frame)

        validated = RawDataSchema.validate(frame, lazy=bool(data_config.validation.lazy))
        metadata = generator.metadata(validated)
        metadata["fingerprint"] = frame_fingerprint(validated)
        metadata["files"] = {fmt: str(path) for fmt, path in written.items()}
        metadata_path = write_json(self.paths.raw_dir / "generation_metadata.json", metadata)

        result = PipelineResult(name=self.name)
        for path in [*written.values(), metadata_path]:
            result.add_artifact(path)
        result.payload = {"frame": validated, "metadata": metadata}
        result.metrics = {
            "n_rows": float(len(validated)),
            "n_columns": float(validated.shape[1]),
            "missing_cells": float(int(validated.isna().to_numpy().sum())),
            "missing_rate": float(validated.isna().to_numpy().mean()),
        }
        result.messages.append(
            f"{len(validated)} rows x {validated.shape[1]} columns written to {self.paths.raw_dir}"
        )
        logger.info("Dataset generated | {}", result.messages[-1])
        return result


def _generator_options(data_config: Any) -> dict[str, Any]:
    """Extract generator-specific options from the extra keys of the data config.

    ``DataConfig`` allows extra fields, so a manifest can expose any knob of the generator
    (positive rate, number of users, seasonality, ...) directly in ``conf/data/default.yaml``.

    Args:
        data_config: Validated data configuration.

    Returns:
        The options accepted by the generator.
    """
    extra = dict(getattr(data_config, "model_extra", {}) or {})
    supported = set(getattr(SyntheticDataGenerator, "SUPPORTED_OPTIONS", ()))
    if not supported:
        return {}
    ignored = sorted(set(extra) - supported)
    if ignored:
        logger.debug("Ignoring unknown generator options: {}", ignored)
    return {key: value for key, value in extra.items() if key in supported}


def load_generated_dataset(config: Any, paths: Any) -> pd.DataFrame:
    """Convenience helper used by notebooks: load (or generate) the raw dataset.

    Args:
        config: Validated application configuration.
        paths: Project paths.

    Returns:
        The raw dataset.
    """
    from src.data.loaders import RawDataLoader

    loader = RawDataLoader(
        paths,
        dataset_name=str(config.data.dataset_name),
        formats=tuple(config.data.formats),
        validate=bool(config.data.validation.raw),
        lazy_validation=bool(config.data.validation.lazy),
        strict=bool(config.data.validation.strict),
        limit=config.data.sampling.limit,
    )
    try:
        return loader.load()
    except FileNotFoundError:
        logger.warning("Dataset missing: generating it on the fly")
        DataGenerationPipeline(config, paths).run()
        return loader.load()
