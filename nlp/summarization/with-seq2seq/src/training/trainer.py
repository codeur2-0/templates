"""Entraînement : ajuster la stratégie déclarée, la surveiller, la comparer, la persister.

Le trainer ne réimplémente pas la boucle d'apprentissage — elle vit dans la stack ``seq2seq`` — il
orchestre ce qui doit être identique d'une stratégie à l'autre :

1. **l'ajustement** sur le split ``train`` uniquement, la validation n'étant lue que pour suivre ;
2. **la comparaison** des algorithmes déclarés par la configuration
   (``strategies_to_compare``) sur un échantillon **déterministe** du split de validation,
   ROUGE-1 et couverture des faits en main. La grille est publiée telle quelle : c'est ce
   qui permet de dire *pourquoi* l'encodeur-décodeur est servi plutôt qu'asserté ;
3. **la sélection** du meilleur selon la métrique contractuelle, puis la persistance de l'artefact,
   de ses métriques, de sa fiche de modèle et de la configuration résolue.

Deux garde-fous sont explicites : un modèle est refusé s'il ne bat pas la baseline triviale sur
l'échantillon de comparaison (le trainer le **signale**, il ne l'empêche pas — un résultat honnête
vaut mieux qu'un silence), et le split de test n'est jamais lu ici.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from src.data.schemas import FACT_TYPES, fact_columns
from src.evaluation.rouge import rouge_scores
from src.models.contract import BaseTextGenerator
from src.training.metrics import coverage_scores, latency_stats, verdict_from_metrics
from src.utils.config_access import node
from src.utils.io import write_json
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Number of validation documents used by the algorithm
#: comparison (deterministic head of the split).
COMPARISON_DOCUMENTS = 40


@dataclass(slots=True)
class SummaryTrainingOutcome:
    """Everything one training run produced.

    Attributes:
        model: The fitted strategy that will be served.
        algorithm: Name of that strategy.
        fit_metrics: Metrics returned by the fit (grid traces included).
        validation_metrics: Metrics measured on the validation split.
        comparison: One row per compared algorithm (ROUGE, coverage, latency, duration).
        model_path: Path of the persisted artefact.
        model_card_path: Path of the model card.
        metrics_path: Path of the training metrics.
        config_path: Path of the resolved configuration.
        warnings: Readings the trainer refuses to hide.
        duration_seconds: Wall-clock duration of the run.
    """

    model: BaseTextGenerator
    algorithm: str
    fit_metrics: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_path: Path | None = None
    model_card_path: Path | None = None
    metrics_path: Path | None = None
    config_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary of the run.

        Returns:
            The metrics, the comparison table (as records) and the artefact paths.
        """
        return {
            "algorithm": self.algorithm,
            "strategy": self.model.strategy,
            "fit_metrics": dict(self.fit_metrics),
            "validation_metrics": dict(self.validation_metrics),
            "comparison": self.comparison.to_dict(orient="records"),
            "artifacts": {
                "model": str(self.model_path) if self.model_path else None,
                "model_card": str(self.model_card_path) if self.model_card_path else None,
                "metrics": str(self.metrics_path) if self.metrics_path else None,
                "resolved_config": str(self.config_path) if self.config_path else None,
            },
            "warnings": list(self.warnings),
            "duration_seconds": round(self.duration_seconds, 3),
        }


class SummaryTrainer:
    """Fit a summary strategy, compare it to its alternatives and persist the result."""

    def __init__(
        self,
        model: BaseTextGenerator,
        *,
        config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        algorithm: str | None = None,
        text_column: str = "text",
        id_column: str = "doc_id",
        compare: Sequence[str] | None = None,
        comparison_documents: int = COMPARISON_DOCUMENTS,
        builders: Mapping[str, BaseTextGenerator] | None = None,
        baseline: BaseTextGenerator | None = None,
        callbacks: Sequence[Any] | None = None,
    ) -> None:
        """Configure the trainer.

        Args:
            model: Strategy to fit and serve.
            config: Full application configuration.
            paths: Project filesystem layout.
            algorithm: Name of the served algorithm (defaults to the configuration).
            text_column: Column holding the documents' text.
            id_column: Column holding the document identifiers.
            compare: Algorithms compared on the validation sample before serving.
            comparison_documents: Size of that deterministic sample.
            builders: Alternative strategies, provided by the pipeline (the trainer never
                imports the factory itself, so it stays testable with hand-built models).
            baseline: Published reference measured on the same lines (``lead`` or ``textrank``).
            callbacks: Callbacks fired after each validation measurement.
        """
        self.model = model
        self.config = dict(config or {})
        self.paths = paths or ProjectPaths.from_root()
        self.algorithm = str(algorithm or self.config.get("algorithm", model.strategy))
        self.text_column = str(text_column)
        self.id_column = str(id_column)
        self.compare = tuple(compare or ())
        self.comparison_documents = max(int(comparison_documents), 1)
        self.builders = dict(builders or {})
        self.baseline = baseline
        self.callbacks = list(callbacks or [])

    # ------------------------------------------------------------------ exécution -----------
    def run(
        self,
        documents: pd.DataFrame,
        references: pd.DataFrame,
        facts: pd.DataFrame,
    ) -> SummaryTrainingOutcome:
        """Fit, compare, measure and persist.

        Args:
            documents: Full document table (the ``split`` column selects the rows).
            references: Reference summaries of the same documents.
            facts: Fact table of the same documents.

        Returns:
            The :class:`SummaryTrainingOutcome` of the run.

        Raises:
            ValueError: When the training split is empty.
        """
        started = perf_counter()
        train_documents = self.split(documents, "train")
        val_documents = self.split(documents, "val")
        if train_documents.empty:
            msg = (
                "Training split is empty: the corpus must carry a 'train' split. "
                "Run `make data` first."
            )
            raise ValueError(msg)
        train_references = self.references_of(references, train_documents)
        val_references = self.references_of(references, val_documents)
        logger.info(
            "Entraînement '{}' sur {} document(s), validation sur {}",
            self.algorithm,
            len(train_documents),
            len(val_documents),
        )

        validation_pair = (
            (val_documents, val_references)
            if not val_documents.empty and not val_references.empty
            else None
        )
        fit_metrics = self.model.fit(
            train_documents,
            train_references,
            validation=validation_pair,
        )
        validation_metrics = (
            self.score(self.model, val_documents, val_references, facts) if validation_pair else {}
        )
        self._fire_callbacks(validation_metrics)

        comparison = self.compare_algorithms(
            train_documents,
            train_references,
            val_documents,
            val_references,
            facts,
        )
        warnings = self._warnings(validation_metrics, comparison)
        outcome = SummaryTrainingOutcome(
            model=self.model,
            algorithm=self.algorithm,
            fit_metrics=fit_metrics,
            validation_metrics=validation_metrics,
            comparison=comparison,
            warnings=warnings,
            duration_seconds=perf_counter() - started,
        )
        return outcome

    def compare_algorithms(
        self,
        train_documents: pd.DataFrame,
        train_references: pd.DataFrame,
        val_documents: pd.DataFrame,
        val_references: pd.DataFrame,
        facts: pd.DataFrame,
    ) -> pd.DataFrame:
        """Measure every declared algorithm on the same validation sample.

        Args:
            train_documents: Training documents (used to fit the alternatives).
            train_references: Training references.
            val_documents: Validation documents.
            val_references: Validation references.
            facts: Fact table of the validation documents.

        Returns:
            One row per algorithm: ROUGE-1/2/L, coverage, compression, latency and fitting time. The
            served model is measured too, so the table is complete even when nothing is compared.
        """
        if val_documents.empty or val_references.empty:
            return pd.DataFrame()
        sample = val_documents.head(self.comparison_documents).reset_index(drop=True)
        golden = dict(zip(val_references["doc_id"], val_references["summary"], strict=False))
        sample = sample.loc[sample[self.id_column].isin(golden)].reset_index(drop=True)
        if sample.empty:
            return pd.DataFrame()

        candidates: dict[str, BaseTextGenerator] = {self.algorithm: self.model}
        for name, builder in self.builders.items():
            if name in candidates:
                continue
            candidates[name] = builder
        rows: list[dict[str, Any]] = []
        for name, candidate in candidates.items():
            already_fitted = candidate.is_fitted
            started = perf_counter()
            if not already_fitted:
                candidate.fit(train_documents, train_references)
            fit_seconds = perf_counter() - started
            row: dict[str, Any] = {
                "algorithm": name,
                "strategy": candidate.strategy,
                "fitted_here": not already_fitted,
                "fit_seconds": round(fit_seconds, 3),
            }
            row.update(self.score(candidate, sample, val_references, facts))
            rows.append(row)
        frame = pd.DataFrame(rows).sort_values(["rouge1_f", "algorithm"], ascending=[False, True])
        return frame.reset_index(drop=True)

    def score(
        self,
        model: BaseTextGenerator,
        documents: pd.DataFrame,
        references: pd.DataFrame,
        facts: pd.DataFrame,
    ) -> dict[str, float]:
        """Measure one strategy on a set of documents.

        Args:
            model: Fitted strategy.
            documents: Documents to summarise.
            references: Reference summaries of the same documents.
            facts: Fact table of the same documents.

        Returns:
            ROUGE-1/2/L, coverage, compression, the share of truncated summaries and the latency
            percentiles of that measurement.
        """
        if documents.empty or references.empty:
            return {}
        golden = dict(zip(references["doc_id"], references["summary"], strict=False))
        summaries = model.summarize(
            [str(value) for value in documents[self.text_column]],
            doc_ids=[str(value) for value in documents[self.id_column]],
        )
        evaluated = [
            (summary, str(golden[summary.doc_id]))
            for summary in summaries
            if summary.doc_id in golden
        ]
        if not evaluated:
            return {}
        scores = [rouge_scores(reference, summary.summary) for summary, reference in evaluated]
        coverage = [
            coverage_scores(
                summary.summary,
                facts.loc[facts["doc_id"] == summary.doc_id],
                document=str(
                    documents.loc[
                        documents[self.id_column] == summary.doc_id, self.text_column
                    ].iloc[0]
                ),
            )
            for summary, _ in evaluated
        ]
        n = len(evaluated)
        metrics: dict[str, float] = {
            "rouge1_f": round(sum(item.rouge1.f1 for item in scores) / n, 4),
            "rouge2_f": round(sum(item.rouge2.f1 for item in scores) / n, 4),
            "rouge_l_f": round(sum(item.rouge_l.f1 for item in scores) / n, 4),
            "fact_coverage": round(sum(item["fact_coverage"] for item in coverage) / n, 4),
            "fact_precision": round(sum(item["fact_precision"] for item in coverage) / n, 4),
            "unsupported_facts_mean": round(sum(item["n_unsupported"] for item in coverage) / n, 3),
            "compression_mean": round(sum(summary.compression for summary, _ in evaluated) / n, 4),
            "hit_max_length_share": round(
                sum(1.0 for summary, _ in evaluated if summary.hit_max_length) / n, 4
            ),
            "n_documents": float(n),
        }
        metrics.update(latency_stats([summary.latency_ms for summary, _ in evaluated]))
        for name in FACT_TYPES:
            key = f"covered_{name}"
            metrics[key] = round(sum(item.get(key, 0.0) for item in coverage) / n, 4)
        return metrics

    # ------------------------------------------------------------------ helpers ------------
    def split(self, documents: pd.DataFrame, name: str) -> pd.DataFrame:
        """Select one split of the corpus.

        Args:
            documents: Full document table (``split`` column).
            name: Split name.

        Returns:
            The rows of that split, index reset.

        Raises:
            ValueError: When the corpus carries no ``split`` column.
        """
        if "split" not in documents.columns:
            msg = (
                "Document table is missing the 'split' column: the corpus must be generated by "
                "`make data` so that two runs measure the same rows."
            )
            raise ValueError(msg)
        return documents.loc[documents["split"] == name].reset_index(drop=True)

    def references_of(self, references: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
        """Restrict the reference summaries to a set of documents.

        Args:
            references: Reference summary table.
            documents: Documents to keep.

        Returns:
            The matching reference rows, index reset.
        """
        keys = set(documents[self.id_column])
        return references.loc[references["doc_id"].isin(keys)].reset_index(drop=True)

    def persist(
        self,
        outcome: SummaryTrainingOutcome,
        *,
        model_file: str,
        resolved_config: Mapping[str, Any] | None = None,
    ) -> SummaryTrainingOutcome:
        """Write the artefact, its card, its metrics and the resolved configuration.

        Args:
            outcome: Outcome to persist (its paths are filled in place).
            model_file: Name of the artefact inside ``artifacts/models``.
            resolved_config: Configuration actually used, archived next to the artefact.

        Returns:
            The same outcome, with its paths set.
        """
        self.paths.ensure()
        target = self.paths.models_dir / model_file
        outcome.model_path = outcome.model.save(target)
        card_path = target.with_name("model_card.json")
        outcome.model_card_path = card_path
        write_json(
            card_path,
            {
                **outcome.model.model_card(),
                "algorithm": outcome.algorithm,
                "validation_metrics": dict(outcome.validation_metrics),
                "comparison": outcome.comparison.to_dict(orient="records"),
                "warnings": list(outcome.warnings),
                "duration_seconds": round(outcome.duration_seconds, 3),
            },
        )
        outcome.metrics_path = write_json(
            self.paths.metrics_dir / "training_metrics.json",
            {
                "algorithm": outcome.algorithm,
                "strategy": outcome.model.strategy,
                "fit_metrics": dict(outcome.fit_metrics),
                "validation_metrics": dict(outcome.validation_metrics),
                "comparison": outcome.comparison.to_dict(orient="records"),
                "warnings": list(outcome.warnings),
                "duration_seconds": round(outcome.duration_seconds, 3),
            },
        )
        if resolved_config is not None:
            outcome.config_path = write_json(
                self.paths.models_dir / "resolved_config.json",
                dict(resolved_config),
            )
        return outcome

    def _warnings(
        self, validation_metrics: Mapping[str, float], comparison: pd.DataFrame
    ) -> list[str]:
        """Read the run and write down what a reader must not miss.

        Args:
            validation_metrics: Metrics of the validation split.
            comparison: Comparison table of the algorithms.

        Returns:
            The warnings, in publication order (empty when there is nothing to say).
        """
        warnings: list[str] = []
        if not validation_metrics:
            warnings.append(
                "Aucune mesure de validation : le split 'val' est vide, donc rien ne prouve que le "
                "modèle généralise — le test reste la seule lecture."
            )
        if not comparison.empty and "rouge1_f" in comparison.columns:
            trivial = comparison.loc[comparison["algorithm"] == self.algorithm, "rouge1_f"]
            if not trivial.empty and float(trivial.iloc[0]) <= 0.0:
                warnings.append(
                    "L'algorithme servi obtient un ROUGE-1 nul sur l'échantillon de comparaison : "
                    "la sortie du modèle est vide ou non alignée sur les références."
                )
            if self.baseline is not None and self.baseline.is_fitted:
                rows = comparison.loc[comparison["algorithm"] == self.baseline.strategy, "rouge1_f"]
                if (
                    not rows.empty
                    and not trivial.empty
                    and float(trivial.iloc[0]) < float(rows.iloc[0])
                ):
                    warnings.append(
                        f"La baseline '{self.baseline.strategy}' obtient un meilleur ROUGE-1 "
                        f"({float(rows.iloc[0]):.4f}) que l'algorithme servi "
                        f"({float(trivial.iloc[0]):.4f}) : c'est un résultat à publier, pas "
                        "à corriger "
                        "en silence."
                    )
        return warnings

    def _fire_callbacks(self, metrics: Mapping[str, float]) -> None:
        """Fire the validation measurement through the shared callbacks.

        Args:
            metrics: Metrics just measured on the validation split.
        """
        if not self.callbacks:
            return
        from src.training.callbacks import CallbackContext

        context = CallbackContext(
            model_name=type(self.model).__name__,
            params=dict(self.model.params),
            epochs=1,
            epoch=1,
            logs={f"val_{name}": float(value) for name, value in metrics.items()},
        )
        for callback in self.callbacks:
            callback.on_train_begin(context)
            callback.on_epoch_end(context)
            callback.on_train_end(context)

    @classmethod
    def from_config(cls, config: Any, **overrides: Any) -> SummaryTrainer:
        """Build a trainer from the configuration, without touching the model factory.

        Args:
            config: Full application configuration.
            overrides: Values that take precedence over the configuration.

        Returns:
            The configured trainer (its ``builders`` and ``baseline`` are set by the pipeline).
        """
        train_node = node(config, "train")
        model_node = node(config, "model")
        data_node = node(config, "data")
        compare = list(model_node.get("strategies_to_compare", []) or [])
        params = {
            "config": config.model_dump() if hasattr(config, "model_dump") else dict(config),
            "algorithm": str(
                model_node.get("algorithm", overrides.pop("algorithm", "transformer_tiny"))
            ),
            "text_column": str(model_node.get("text_column", "text")),
            "id_column": str(data_node.get("id_column", "doc_id")),
            "compare": compare,
            "comparison_documents": int(
                train_node.get("comparison_documents", COMPARISON_DOCUMENTS)
            ),
            **overrides,
        }
        return cls(**params)

    def verdict(self, metrics: Mapping[str, float]) -> tuple[str, dict[str, Any]]:
        """Read the contractual verdict of a metrics block.

        Args:
            metrics: Flat metrics.

        Returns:
            ``(verdict, detail)`` as produced by the family's metric helper.
        """
        metrics_config = dict(self.config.get("metrics", {}) or {})
        return verdict_from_metrics(
            metrics,
            primary=str(metrics_config.get("primary", "rouge1_f")),
            minimum=(
                None
                if metrics_config.get("min_primary") is None
                else float(metrics_config["min_primary"])
            ),
            direction=str(metrics_config.get("direction", "maximize")),
        )


def default_model_file(config: Mapping[str, Any] | None = None) -> str:
    """Return the artefact file name declared by the configuration.

    Args:
        config: Full application configuration.

    Returns:
        The file name inside ``artifacts/models`` (``summarizer.pt`` by default).
    """
    train = dict((config or {}).get("train", {}) or {})
    artifacts = dict(train.get("artifacts", {}) or {})
    return str(artifacts.get("model_file", "summarizer.pt"))


def comparison_columns() -> tuple[str, ...]:
    """Metric columns published in the algorithm comparison.

    Returns:
        The column names, in reading order.
    """
    return (
        "algorithm",
        "strategy",
        "rouge1_f",
        "rouge2_f",
        "rouge_l_f",
        "fact_coverage",
        "compression_mean",
        "hit_max_length_share",
        "latency_p50_ms",
        "fit_seconds",
    )


def coverage_columns() -> tuple[str, ...]:
    """Per-type coverage columns published by :meth:`SummaryTrainer.score`.

    Returns:
        The ``covered_<type>`` column names.
    """
    return fact_columns()


__all__ = [
    "COMPARISON_DOCUMENTS",
    "SummaryTrainer",
    "SummaryTrainingOutcome",
    "comparison_columns",
    "coverage_columns",
    "default_model_file",
]
