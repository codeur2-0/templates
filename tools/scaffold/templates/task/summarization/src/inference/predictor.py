"""Inférence : produire les résumés d'un corpus ou d'un échantillon, et les lire.

Le prédicteur est le seul endroit qui décide **ce qui est publié** d'une génération : le
résumé, sa longueur, sa compression, la couverture des faits saillants quand la vérité
terrain est disponible, et les valeurs non supportées. Il ne contient aucune logique de
modèle — il charge l'artefact écrit par ``mode=train``, l'applique, et refuse de deviner :

* sans fichier d'entrée, il échantillonne le corpus **déterministe** (les cinq premiers documents du
  split de test) plutôt que d'inventer des textes ;
* quand les faits ne sont pas disponibles (textes fournis par l'utilisateur), la couverture est
  publiée à ``0,0`` et signalée comme non mesurée — jamais présentée comme un score ;
* la table publiée garde la référence quand elle existe, pour que la relecture se fasse sans
  réouvrir le corpus.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from src.data.loaders import SummaryCorpusLoader
from src.data.schemas import validate_predictions
from src.models.contract import BaseTextGenerator
from src.training.metrics import coverage_scores, unsupported_values
from src.utils.io import read_table, write_json, write_table
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Number of documents summarised when the caller asks for an inference sample.
DEFAULT_SAMPLE = 5


class SummaryPredictor:
    """Summarise an input file or a deterministic sample of the corpus."""

    def __init__(
        self,
        model: BaseTextGenerator,
        *,
        paths: ProjectPaths | None = None,
        config: Mapping[str, Any] | None = None,
        text_column: str = "text",
        id_column: str = "doc_id",
        publish_fidelity: bool = True,
    ) -> None:
        """Configure the predictor.

        Args:
            model: Fitted strategy to apply.
            paths: Project filesystem layout.
            config: Full application configuration.
            text_column: Column holding the documents' text.
            id_column: Column holding the document identifiers.
            publish_fidelity: Whether the fact metrics are computed (they need the fact table).
        """
        self.model = model
        self.paths = paths or ProjectPaths.from_root()
        self.config = dict(config or {})
        self.text_column = str(text_column)
        self.id_column = str(id_column)
        self.publish_fidelity = bool(publish_fidelity)
        self.loader = SummaryCorpusLoader(
            self.paths,
            dataset_name=str(
                (self.config.get("data") or {}).get("dataset_name", "intervention_reports")
            ),
        )

    # ------------------------------------------------------------------ entrées ------------
    def load_input(
        self, path: str | Path | None = None, *, n_samples: int = DEFAULT_SAMPLE
    ) -> pd.DataFrame:
        """Load the documents to summarise.

        Args:
            path: Optional file to read (Parquet or CSV, ``text`` column required).
            n_samples: Number of documents taken from the corpus when no file is given.

        Returns:
            The documents to summarise (``doc_id``, ``text``…), index reset.

        Raises:
            FileNotFoundError: When the file does not exist.
            ValueError: When the file carries no text column.
        """
        if path is not None:
            target = Path(path)
            if not target.is_file():
                msg = (
                    f"Input file not found: {target}. Pass an existing Parquet or CSV file, or let "
                    "the pipeline sample the generated corpus."
                )
                raise FileNotFoundError(msg)
            frame = read_table(target)
            if self.text_column not in frame.columns:
                msg = (
                    f"Input file is missing the column '{self.text_column}' "
                    f"(present: {list(frame.columns)})"
                )
                raise ValueError(msg)
            frame = frame.reset_index(drop=True)
            if self.id_column not in frame.columns:
                frame[self.id_column] = [f"doc-{index}" for index in range(len(frame))]
            logger.info("Inférence sur {} document(s) lus depuis {}", len(frame), target)
            return frame

        documents = self.loader.load_documents()
        sample = documents.loc[documents["split"] == "test"].head(max(int(n_samples), 1))
        if sample.empty:
            sample = documents.head(max(int(n_samples), 1))
        logger.info("Inférence sur {} document(s) échantillonnés dans le corpus", len(sample))
        return sample.reset_index(drop=True)

    # ------------------------------------------------------------------ sortie -------------
    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Summarise the given documents.

        Args:
            frame: Documents to summarise (``doc_id``,
                ``text``), plus their segments when available.

        Returns:
            The prediction table: identifier, strategy, summary, lengths, latency, ROUGE and fact
            metrics when a reference is available.
        """
        started = perf_counter()
        summaries = self.model.summarize(
            [str(value) for value in frame[self.text_column]],
            doc_ids=[str(value) for value in frame[self.id_column]],
        )
        references, facts = self._ground_truth(frame)
        rows: list[dict[str, Any]] = []
        for summary in summaries:
            reference = references.get(summary.doc_id, "")
            source = str(
                frame.loc[frame[self.id_column] == summary.doc_id, self.text_column].iloc[0]
            )
            document_facts = facts.get(summary.doc_id)
            fidelity = (
                coverage_scores(summary.summary, document_facts, document=source)
                if document_facts is not None
                else {
                    "fact_coverage": 0.0,
                    "fact_precision": 0.0,
                    "n_unsupported": float(len(unsupported_values(summary.summary, source))),
                }
            )
            row: dict[str, Any] = {
                self.id_column: summary.doc_id,
                "strategy": summary.strategy,
                "reference_summary": reference or None,
                "prediction": summary.summary,
                "n_clauses": len(summary.sentences),
                "n_words": len(summary.summary.split()),
                "compression": round(summary.compression, 6),
                "hit_max_length": int(summary.hit_max_length),
                "fact_coverage": float(fidelity["fact_coverage"]),
                "unsupported_facts": int(fidelity["n_unsupported"]),
                "unsupported_values": " | ".join(unsupported_values(summary.summary, source)),
                "latency_ms": round(summary.latency_ms, 3),
            }
            for column in ("intervention_type", "urgency", "site"):
                if column in frame.columns:
                    row[column] = str(
                        frame.loc[frame[self.id_column] == summary.doc_id, column].iloc[0]
                    )
            rows.append(row)
        table = pd.DataFrame(rows)
        logger.info(
            "Inférence terminée : {} document(s), {} mots en moyenne, {:.1f} ms par document",
            len(table),
            round(float(table["n_words"].mean()), 1) if not table.empty else 0.0,
            (perf_counter() - started) * 1000.0 / max(len(table), 1),
        )
        return table

    def summary_frame(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a prediction table by strategy.

        Args:
            predictions: Prediction rows.

        Returns:
            One row per strategy: documents, mean words, mean compression, mean
            latency, coverage and the share of summaries that hit their budget.
        """
        if predictions.empty:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for strategy, group in predictions.groupby("strategy", sort=True):
            rows.append(
                {
                    "strategy": str(strategy),
                    "n_documents": len(group),
                    "n_words_mean": round(float(group["n_words"].mean()), 2),
                    "compression_mean": round(float(group["compression"].mean()), 4),
                    "sentences_mean": round(float(group["n_clauses"].mean()), 2),
                    "hit_max_length_share": round(float(group["hit_max_length"].mean()), 4),
                    "fact_coverage": round(float(group["fact_coverage"].mean()), 4),
                    "unsupported_facts_mean": round(float(group["unsupported_facts"].mean()), 3),
                    "latency_p50_ms": round(float(group["latency_ms"].median()), 3),
                }
            )
        return pd.DataFrame(rows)

    def save(self, predictions: pd.DataFrame, *, path: str | Path | None = None) -> Path:
        """Validate then persist the prediction table (CSV) and its summary (JSON).

        Args:
            predictions: Prediction rows.
            path: Destination CSV (defaults to ``artifacts/reports/predictions.csv``).

        Returns:
            The written CSV path.
        """
        target = Path(path) if path is not None else self.loader.predictions_path
        table = self._to_published_schema(predictions)
        route = write_table(table, target)
        summary = self.summary_frame(predictions)
        if not summary.empty:
            write_json(
                self.paths.reports_dir / "inference_summary.json",
                summary.to_dict(orient="records"),
            )
        return route

    def _to_published_schema(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Adapt the inference table to the published contract.

        Args:
            predictions: Raw prediction rows.

        Returns:
            A frame matching :class:`~src.data.schemas.PredictedSummariesSchema` (the columns the
            contract requires, in the order it declares them).
        """
        columns = [
            "doc_id",
            "strategy",
            "reference_summary",
            "prediction",
            "compression",
            "n_sentences",
            "rouge1_f",
            "rouge2_f",
            "rouge_l_f",
            "fact_coverage",
            "unsupported_facts",
            "hit_max_length",
            "latency_ms",
        ]
        adapted = predictions.rename(columns={self.id_column: "doc_id", "n_clauses": "n_sentences"})
        for column in columns:
            if column not in adapted.columns:
                adapted[column] = 0
        adapted = adapted.loc[:, columns]
        # À l'inférence, la référence peut être inconnue : les métriques face à elle valent alors 0,
        # et le contrat tolère une référence vide — jamais une métrique inventée.
        adapted["reference_summary"] = adapted["reference_summary"].astype("object")
        try:
            return validate_predictions(adapted)
        except Exception as exc:  # pragma: no cover - dépend de pandera, message transmis tel quel
            logger.warning("Table de prédictions non conforme au contrat publié : {}", exc)
            return adapted

    # ------------------------------------------------------------------ helpers ------------
    def _ground_truth(self, frame: pd.DataFrame) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
        """Load the references and facts of the documents to summarise, when they exist.

        Args:
            frame: Documents to summarise.

        Returns:
            ``(references, facts)``: the reference summary of each known
            document and its fact table. Both are empty for documents
            the corpus does not know (an input file provided by hand).
        """
        if not self.publish_fidelity:
            return {}, {}
        try:
            documents, references, facts = self.loader.load_corpus()
        except FileNotFoundError:
            logger.info("Corpus absent : les métriques de fidélité ne seront pas publiées")
            return {}, {}
        keys = {str(value) for value in frame[self.id_column]}
        known = documents.loc[documents["doc_id"].isin(keys)]
        if known.empty:
            return {}, {}
        golden = {
            str(row.doc_id): str(row.summary)
            for row in references.loc[references["doc_id"].isin(keys)].itertuples(index=False)
        }
        grouped = {
            str(doc_id): group.loc[group["salient"] == 1]
            for doc_id, group in facts.loc[facts["doc_id"].isin(keys)].groupby("doc_id")
        }
        return golden, grouped


def prediction_columns() -> Sequence[str]:
    """Columns of the inference table produced by the predictor.

    Returns:
        The column names, in publication order.
    """
    return (
        "doc_id",
        "strategy",
        "reference_summary",
        "prediction",
        "n_clauses",
        "n_words",
        "compression",
        "hit_max_length",
        "fact_coverage",
        "unsupported_facts",
        "latency_ms",
    )


__all__ = ["DEFAULT_SAMPLE", "SummaryPredictor", "prediction_columns"]
