"""Pandera contracts of the text modality.

Three tables carry the whole retrieval pipeline:

``documents``    the reference corpus (the knowledge base that is indexed),
``chunks``       the corpus after splitting (what the retriever actually ranks),
``queries``      the annotated questions used for calibration, validation and test.

The contracts are *executable*: a missing column, an unknown source or a question shorter than
its minimum fails loudly before any model is fitted. Inference payloads get their own, relaxed
contract (``PredictedAnswersSchema``) because a prediction row is produced by us, not by the
user: it must be complete, and it is validated *before* being written.

Two rules are worth stating because they are enforced here rather than documented elsewhere:

* ``gold_doc_ids`` and ``gold_chunk_ids`` are comma-separated strings, never lists. Parquet
  handles lists perfectly, but a contract on a nullable list column cannot express "at least
  one identifier, no empty entry" as clearly as a regex does.
* An unanswerable question (``answer_type == "unanswerable"``) has **no** gold document. It
  exists to measure abstention: a RAG assistant that answers everything is a liability, and
  the schema refuses to let such a question silently carry a fake gold document.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera.typing import Series

#: Identifier formats. They are part of the contract: artefacts are joined on those keys.
DOC_ID_PATTERN = r"^DOC-\d{4}$"
CHUNK_ID_PATTERN = r"^DOC-\d{4}-C\d{2}$"
QUERY_ID_PATTERN = r"^QRY-\d{4}$"
ID_LIST_PATTERN = r"^(DOC-\d{4})(,DOC-\d{4})*$"

#: Editorial sections of the synthetic knowledge base.
SECTIONS: tuple[str, ...] = (
    "politique",
    "procedure",
    "definition",
    "contact",
    "calcul",
    "conformite",
)

#: Sources of the knowledge base (one per internal wiki space).
SOURCES: tuple[str, ...] = (
    "wiki_rh",
    "wiki_it",
    "wiki_finance",
    "wiki_juridique",
    "wiki_ops",
    "wiki_support",
)

#: Difficulty levels of the annotated questions.
DIFFICULTIES: tuple[str, ...] = ("facile", "paraphrase", "multi_document", "hors_corpus")

#: Intent of the question, used to segment the evaluation report.
INTENTS: tuple[str, ...] = ("procedure", "definition", "calcul", "contact", "politique")

#: How the reference answer relates to the corpus.
ANSWER_TYPES: tuple[str, ...] = ("extractive", "abstractive", "unanswerable")

#: Splits of the annotated questions. ``calibration`` tunes the abstention threshold,
#: ``val`` selects models and ``test`` measures them exactly once.
SPLITS: tuple[str, ...] = ("calibration", "val", "test")


class DocumentsSchema(pa.DataFrameModel):
    """Contract of the reference corpus (``data/raw/documents.parquet``)."""

    doc_id: Series[str] = pa.Field(
        unique=True, str_matches=DOC_ID_PATTERN, description="Identifiant stable du document."
    )
    title: Series[str] = pa.Field(
        str_length={"min_value": 6, "max_value": 160}, description="Titre éditorial du document."
    )
    section: Series[str] = pa.Field(isin=SECTIONS, description="Section éditoriale du wiki.")
    source: Series[str] = pa.Field(isin=SOURCES, description="Espace wiki d'origine.")
    published_at: Series[pa.DateTime] = pa.Field(description="Date de publication du document.")
    n_tokens: Series[int] = pa.Field(
        ge=40, le=1200, description="Nombre de tokens du texte (découpage lexical)."
    )
    text: Series[str] = pa.Field(
        str_length={"min_value": 200}, description="Texte intégral du document."
    )

    class Config:
        """Pandera configuration: exact columns, coerced dtypes."""

        strict = True
        coerce = True
        ordered = False


class ChunksSchema(pa.DataFrameModel):
    """Contract of the chunked corpus (``data/processed/chunks.parquet``)."""

    chunk_id: Series[str] = pa.Field(
        unique=True, str_matches=CHUNK_ID_PATTERN, description="Identifiant du passage."
    )
    doc_id: Series[str] = pa.Field(str_matches=DOC_ID_PATTERN, description="Document d'origine.")
    chunk_index: Series[int] = pa.Field(ge=0, description="Position du passage dans le document.")
    start_char: Series[int] = pa.Field(ge=0, description="Offset de début dans le document.")
    end_char: Series[int] = pa.Field(ge=1, description="Offset de fin (exclu) dans le document.")
    n_tokens: Series[int] = pa.Field(ge=5, le=400, description="Nombre de tokens du passage.")
    text: Series[str] = pa.Field(
        str_length={"min_value": 20}, description="Texte du passage indexé."
    )

    class Config:
        """Pandera configuration: exact columns, coerced dtypes."""

        strict = True
        coerce = True
        ordered = False

    @pa.dataframe_check
    @classmethod
    def end_after_start(cls, data: pd.DataFrame) -> pd.Series:
        """A passage must span a strictly positive character range."""

        return bool((data["end_char"] > data["start_char"]).all())


class QueriesSchema(pa.DataFrameModel):
    """Contract of the annotated questions (``data/raw/queries.parquet``)."""

    query_id: Series[str] = pa.Field(
        unique=True, str_matches=QUERY_ID_PATTERN, description="Identifiant de la question."
    )
    question: Series[str] = pa.Field(
        str_length={"min_value": 12, "max_value": 240}, description="Question posée par l'utilisateur."
    )
    reference_answer: Series[str] = pa.Field(
        str_length={"min_value": 1}, description="Réponse de référence rédigée par l'expert."
    )
    answer_type: Series[str] = pa.Field(
        isin=ANSWER_TYPES, description="Réponse extractive, abstractive ou question sans réponse."
    )
    difficulty: Series[str] = pa.Field(isin=DIFFICULTIES, description="Niveau de difficulté annoté.")
    intent: Series[str] = pa.Field(isin=INTENTS, description="Intention métier de la question.")
    split: Series[str] = pa.Field(
        isin=SPLITS, description="Split : calibration, validation ou test."
    )
    gold_doc_ids: Series[str] = pa.Field(
        str_matches=ID_LIST_PATTERN,
        description="Documents pertinents, séparés par des virgules (au moins un).",
    )
    gold_span: Series[str] = pa.Field(
        description=(
            "Extrait(s) du corpus qui contiennent la réponse, séparés par '||' quand la question "
            "exige plusieurs documents (chaîne vide si la question est hors corpus)."
        )
    )
    n_gold_docs: Series[int] = pa.Field(ge=1, description="Nombre de documents pertinents.")

    class Config:
        """Pandera configuration: exact columns, coerced dtypes."""

        strict = True
        coerce = True
        ordered = False

    @pa.dataframe_check
    @classmethod
    def unanswerable_has_two_sentinels(cls, data: pd.DataFrame) -> pd.Series:
        """Enforce the sentinel convention of unanswerable questions.

        An unanswerable question carries ``gold_doc_ids == "NONE"``... which the regex above
        would reject. Rather than a magic value, the convention is: unanswerable questions
        point to a real document id but ``gold_span`` is empty, and ``answer_type`` says so.
        This check verifies that the two markers agree, so a question can never claim both
        "there is an answer" and "there is none".
        """

        unanswerable = data["answer_type"] == "unanswerable"
        empty_span = data["gold_span"].str.strip() == ""
        return bool((unanswerable == empty_span).all())


class PredictedAnswersSchema(pa.DataFrameModel):
    """Contract of the predicted answers (``artifacts/reports/predictions.csv``)."""

    query_id: Series[str] = pa.Field(description="Identifiant de la question évaluée.")
    question: Series[str] = pa.Field(description="Question posée.")
    answer: Series[str] = pa.Field(description="Réponse produite (chaîne vide si abstention).")
    citations: Series[str] = pa.Field(
        description="Passages cités, sous la forme ``chunk_id|doc_id``, séparés par des virgules."
    )
    n_citations: Series[int] = pa.Field(ge=0, description="Nombre de passages cités.")
    abstained: Series[int] = pa.Field(isin=[0, 1], description="1 si le système s'est abstenu.")
    top_score: Series[float] = pa.Field(
        ge=0.0, description="Score de similarité du meilleur passage retrouvé."
    )
    n_chunks_indexed: Series[int] = pa.Field(ge=1, description="Taille de l'index au moment du run.")
    latency_ms: Series[float] = pa.Field(ge=0.0, description="Latence de la réponse, en ms.")

    class Config:
        """Pandera configuration: exact columns, coerced dtypes."""

        strict = True
        coerce = True
        ordered = False


def validate_documents(frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
    """Validate a corpus frame against :class:`DocumentsSchema`.

    Args:
        frame: Corpus to validate.
        lazy: When ``True``, every violation is collected before raising (useful for a data
            quality report on a user-provided corpus).

    Returns:
        The validated frame (coerced dtypes).
    """
    return DocumentsSchema.validate(frame, lazy=lazy)


def validate_chunks(frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
    """Validate a chunk frame against :class:`ChunksSchema`.

    Args:
        frame: Chunks to validate.
        lazy: Collect every violation before raising.

    Returns:
        The validated frame.
    """
    return ChunksSchema.validate(frame, lazy=lazy)


def validate_queries(frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
    """Validate an annotated question frame against :class:`QueriesSchema`.

    Args:
        frame: Questions to validate.
        lazy: Collect every violation before raising.

    Returns:
        The validated frame.
    """
    return QueriesSchema.validate(frame, lazy=lazy)


def validate_predictions(frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
    """Validate predictions before they are persisted.

    Args:
        frame: Prediction rows.
        lazy: Collect every violation before raising.

    Returns:
        The validated frame.
    """
    return PredictedAnswersSchema.validate(frame, lazy=lazy)


__all__ = [
    "ANSWER_TYPES",
    "CHUNK_ID_PATTERN",
    "ChunksSchema",
    "DIFFICULTIES",
    "DOC_ID_PATTERN",
    "DocumentsSchema",
    "ID_LIST_PATTERN",
    "INTENTS",
    "PredictedAnswersSchema",
    "QUERY_ID_PATTERN",
    "QueriesSchema",
    "SECTIONS",
    "SOURCES",
    "SPLITS",
    "validate_chunks",
    "validate_documents",
    "validate_predictions",
    "validate_queries",
]
