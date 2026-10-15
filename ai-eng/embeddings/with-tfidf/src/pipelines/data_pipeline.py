"""Data generation pipeline: build the synthetic corpus and its annotated questions.

``mode=generate-data`` (and ``make data``) runs this pipeline. It is the only place where the
family generator is called, and the only place that writes into ``data/raw``:

1. instantiate the family generator from the ``data`` node of the configuration,
2. generate documents, questions and metadata,
3. validate both tables against their Pandera contracts **before** writing anything,
4. persist them (Parquet for machines, CSV for humans) plus the metadata JSON.

The metadata is not decoration: it records the generator's seed, the number of documents per
source and — most importantly — the *latent* information (which passages answer which question,
how many questions are unanswerable). The evaluation report uses it to state the ceiling a
system can reach, rather than pretending that a recall of 1.0 is attainable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.data.generators import SyntheticCorpusGenerator
from src.data.loaders import TextCorpusLoader
from src.pipelines.base import BasePipeline, PipelineResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataGenerationPipeline(BasePipeline):
    """Generate -> validate -> persist the synthetic corpus."""

    name = "generate-data"

    def _execute(self) -> PipelineResult:
        """Run the generation flow.

        Returns:
            The pipeline result, whose metrics summarise the generated dataset.
        """
        self.paths.ensure()
        data_config = self.config.data
        generator = SyntheticCorpusGenerator.from_config(data_config)
        bundle = generator.generate()

        loader = TextCorpusLoader(
            self.paths,
            formats=data_config.formats,
            validation_enabled=data_config.validation.raw,
            lazy_validation=data_config.validation.lazy,
        )
        documents_written = loader.save_documents(bundle.documents)
        queries_written = loader.save_queries(bundle.queries)
        metadata_path = loader.save_metadata(bundle.metadata)

        metrics = self._summarise(bundle.documents, bundle.queries)
        artifacts = [str(path) for path in [*documents_written.values(), *queries_written.values()]]
        artifacts.append(str(metadata_path))
        logger.info(
            "Corpus '{}' generated: {} documents, {} questions ({} unanswerable)",
            data_config.dataset_name,
            metrics["n_documents"],
            metrics["n_questions"],
            metrics["n_unanswerable"],
        )
        return PipelineResult(
            name=self.name,
            metrics=metrics,
            artifacts=artifacts,
            payload=bundle,
            messages=[
                f"{metrics['n_documents']:.0f} documents, "
                f"{metrics['n_chunks_expected']:.0f} passages attendus, "
                f"{metrics['n_questions']:.0f} questions annotées "
                f"dont {metrics['n_unanswerable']:.0f} sans réponse dans le corpus."
            ],
        )

    @staticmethod
    def _summarise(documents: Any, queries: Any) -> dict[str, float]:
        """Compute the summary metrics of a generated corpus.

        Args:
            documents: Generated corpus frame.
            queries: Generated question frame.

        Returns:
            Finite metrics describing the dataset.
        """
        tokens = documents["n_tokens"]
        return {
            "n_documents": float(len(documents)),
            "n_sources": float(documents["source"].nunique()),
            "mean_document_tokens": float(tokens.mean()),
            "n_chunks_expected": float(int(tokens.sum() / 120) + len(documents)),
            "n_questions": float(len(queries)),
            "n_unanswerable": float((queries["answer_type"] == "unanswerable").sum()),
            "n_multi_document": float((queries["difficulty"] == "multi_document").sum()),
            "mean_gold_documents": float(queries["n_gold_docs"].mean()),
        }


def summarise_metadata(payload: Mapping[str, Any]) -> dict[str, float]:
    """Flatten the numeric entries of the generation metadata (used in the report).

    Args:
        payload: Metadata mapping written by the generator.

    Returns:
        Mapping of ``key`` to finite float values.
    """
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            flattened[str(key)] = float(value)
    return flattened


__all__ = ["DataGenerationPipeline", "summarise_metadata"]
