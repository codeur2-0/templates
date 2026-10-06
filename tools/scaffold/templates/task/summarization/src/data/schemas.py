"""Contrats Pandera du corpus de résumé automatique.

Un projet de résumé tient dans **trois tables** qui se lisent ensemble : les documents sources, leurs
résumés de référence et les **faits** que ces résumés ont le droit de rapporter. Les contrats sont
exécutables et disent trois choses qu'un fichier de données ne dit jamais de lui-même :

* le **découpage est une donnée** (`split`) : il est écrit par le générateur, jamais tiré au moment de
  l'entraînement, ce qui rend deux exécutions comparables ;
* la **longueur est contrainte des deux côtés** : un document trop court n'apprend rien et un
  document de 5 000 caractères coûte plus cher à entraîner qu'il n'apporte, tandis qu'un résumé de
  référence hors bornes rendrait le ROUGE incomparable d'une ligne à l'autre ;
* les **faits portent leur propre contrat** (`fact_type`, `salient`) : c'est cette table qui permet
  de mesurer la *fidélité* d'un résumé — quels faits du document il rapporte, et lesquels il invente
  — au lieu de publier un ROUGE qui monte quand le modèle recopie des chiffres au hasard.

Les payloads d'inférence ont leur propre contrat, relaxé : une prédiction est produite par le projet,
elle doit être complète, et elle est validée **avant** d'être écrite.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera.typing import Series

#: Identifiers are part of the contract: artefacts are joined on those keys.
DOC_ID_PATTERN = r"^CR-\d{4}$"

#: Nature de l'intervention décrite par le compte-rendu. Sert de segment d'analyse : un modèle peut
#: résumer correctement un dépannage et échouer sur une expertise, et le rapport doit le montrer.
INTERVENTION_TYPES: tuple[str, ...] = ("depannage", "maintenance", "installation", "expertise")

#: Urgence déclarée du compte-rendu. Distracteur assumé côté modèle, segment côté rapport : elle
#: explique une partie de la longueur des documents, pas la qualité du résumé.
URGENCIES: tuple[str, ...] = ("basse", "normale", "haute")

#: Sites anonymisés. Le site est un distracteur déclaré : il change le vocabulaire, jamais les faits.
SITES: tuple[str, ...] = ("SITE-A", "SITE-B", "SITE-C", "SITE-D", "SITE-E", "SITE-F")

#: Type de fait rapportable. C'est le vocabulaire de l'évaluation de fidélité : une *cause* absente
#: du résumé est une omission, une *durée* absente du document est une hallucination.
FACT_TYPES: tuple[str, ...] = ("equipement", "symptome", "cause", "action", "piece", "duree", "statut")

#: Splits of the corpus. ``train`` fits, ``val`` monitors, ``calibration`` is used by the notebook to
#: settle the decoding budget, ``test`` measures once.
SPLITS: tuple[str, ...] = ("train", "val", "calibration", "test")


def fact_columns(types: tuple[str, ...] = FACT_TYPES) -> tuple[str, ...]:
    """Return the per-type coverage column names of an evaluation table.

    Args:
        types: Fact types in display order.

    Returns:
        The ``covered_<type>`` column names, in the same order.
    """
    return tuple(f"covered_{name}" for name in types)


class DocumentsSchema(pa.DataFrameModel):
    """Contract of the source corpus (``data/raw/intervention_reports.parquet``)."""

    doc_id: Series[str] = pa.Field(
        unique=True, str_matches=DOC_ID_PATTERN, description="Identifiant stable du compte-rendu."
    )
    text: Series[str] = pa.Field(
        str_length={"min_value": 300, "max_value": 2400},
        description="Compte-rendu complet, tel qu'il a été rédigé.",
    )
    intervention_type: Series[str] = pa.Field(
        isin=INTERVENTION_TYPES, description="Nature de l'intervention décrite."
    )
    urgency: Series[str] = pa.Field(
        isin=URGENCIES, description="Urgence déclarée : un distracteur, pas un signal du résumé."
    )
    site: Series[str] = pa.Field(isin=SITES, description="Site d'intervention (anonymisé).")
    n_sentences: Series[int] = pa.Field(ge=6, le=40, description="Nombre de phrases du document.")
    n_tokens: Series[int] = pa.Field(ge=60, le=600, description="Nombre de tokens du document.")
    published_at: Series[pa.DateTime] = pa.Field(description="Date de rédaction du compte-rendu.")
    split: Series[str] = pa.Field(
        isin=SPLITS, description="Découpage du corpus : une donnée, pas un tirage."
    )

    class Config:
        """Pandera configuration: exact columns, coerced dtypes."""

        strict = True
        coerce = True
        ordered = False


class ReferenceSummariesSchema(pa.DataFrameModel):
    """Contract of the reference summaries (``data/raw/reference_summaries.parquet``)."""

    doc_id: Series[str] = pa.Field(
        str_matches=DOC_ID_PATTERN, description="Document résumé (clé de jointure)."
    )
    summary: Series[str] = pa.Field(
        str_length={"min_value": 80, "max_value": 600},
        description="Résumé de référence rédigé à partir des faits saillants du document.",
    )
    n_sentences: Series[int] = pa.Field(ge=2, le=6, description="Nombre de phrases du résumé.")
    n_tokens: Series[int] = pa.Field(ge=15, le=140, description="Nombre de tokens du résumé.")
    n_salient_facts: Series[int] = pa.Field(
        ge=2, le=9, description="Nombre de faits saillants que le résumé doit rapporter."
    )
    compression: Series[float] = pa.Field(
        gt=0.0, le=0.6, description="Longueur du résumé rapportée à celle du document."
    )

    class Config:
        """Pandera configuration: exact columns, coerced dtypes."""

        strict = True
        coerce = True
        ordered = False


class FactsSchema(pa.DataFrameModel):
    """Contract of the fact table (``data/raw/salient_facts.parquet``).

    One row per fact extracted from a document by the generator. ``salient`` says whether the
    reference summary must report it; ``surface`` is the exact span of the document that carries it,
    which is what makes the coverage measurable by string containment instead of by a fuzzy score.
    """

    doc_id: Series[str] = pa.Field(str_matches=DOC_ID_PATTERN, description="Document porteur du fait.")
    fact_id: Series[str] = pa.Field(
        str_matches=r"^F-\d{5}$", description="Identifiant stable du fait."
    )
    fact_type: Series[str] = pa.Field(isin=FACT_TYPES, description="Type du fait rapportable.")
    value: Series[str] = pa.Field(
        str_length={"min_value": 2, "max_value": 80},
        description="Valeur normalisée du fait (durée en minutes, référence de pièce…).",
    )
    surface: Series[str] = pa.Field(
        str_length={"min_value": 2, "max_value": 120},
        description="Formulation exacte du fait dans le document.",
    )
    salient: Series[int] = pa.Field(
        isin=[0, 1], description="1 lorsque le résumé de référence doit rapporter le fait."
    )
    sentence_index: Series[int] = pa.Field(
        ge=0, le=39, description="Phrase du document qui porte le fait."
    )

    class Config:
        """Pandera configuration: exact columns, coerced dtypes."""

        strict = True
        coerce = True
        ordered = False


class PredictedSummariesSchema(pa.DataFrameModel):
    """Contract of the prediction table (``artifacts/reports/predictions.csv``)."""

    doc_id: Series[str] = pa.Field(str_matches=DOC_ID_PATTERN, description="Document résumé.")
    strategy: Series[str] = pa.Field(
        str_length={"min_value": 3, "max_value": 40},
        description="Stratégie de résumé : baseline extractive ou architecture apprise.",
    )
    reference_summary: Series[str] = pa.Field(
        nullable=True, description="Résumé de référence, vide à l'inférence."
    )
    prediction: Series[str] = pa.Field(
        str_length={"min_value": 20, "max_value": 800}, description="Résumé produit."
    )
    compression: Series[float] = pa.Field(
        gt=0.0, le=1.0, description="Longueur du résumé produit rapportée à celle du document."
    )
    n_sentences: Series[int] = pa.Field(ge=1, le=12, description="Nombre de phrases produites.")
    rouge1_f: Series[float] = pa.Field(ge=0.0, le=1.0, description="ROUGE-1 F1 face à la référence.")
    rouge2_f: Series[float] = pa.Field(ge=0.0, le=1.0, description="ROUGE-2 F1 face à la référence.")
    rouge_l_f: Series[float] = pa.Field(ge=0.0, le=1.0, description="ROUGE-L F1 face à la référence.")
    fact_coverage: Series[float] = pa.Field(
        ge=0.0, le=1.0, description="Part des faits saillants rapportés par le résumé."
    )
    unsupported_facts: Series[int] = pa.Field(
        ge=0, description="Faits présents dans le résumé et absents du document."
    )
    hit_max_length: Series[int] = pa.Field(
        isin=[0, 1], description="1 lorsque le décodeur a atteint sa longueur maximale."
    )
    latency_ms: Series[float] = pa.Field(ge=0.0, description="Latence de la génération, en ms.")

    class Config:
        """Pandera configuration: exact columns, coerced dtypes."""

        strict = True
        coerce = True
        ordered = False


def validate_documents(frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
    """Validate a source corpus against :class:`DocumentsSchema`.

    Args:
        frame: Corpus to validate.
        lazy: Collect every violation before raising (useful on a user-provided file).

    Returns:
        The validated frame (coerced dtypes).
    """
    return DocumentsSchema.validate(frame, lazy=lazy)


def validate_references(frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
    """Validate the reference summary table against :class:`ReferenceSummariesSchema`.

    Args:
        frame: Reference summaries.
        lazy: Collect every violation before raising.

    Returns:
        The validated frame.
    """
    return ReferenceSummariesSchema.validate(frame, lazy=lazy)


def validate_facts(frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
    """Validate the fact table against :class:`FactsSchema`.

    Args:
        frame: Facts to validate.
        lazy: Collect every violation before raising.

    Returns:
        The validated frame.
    """
    return FactsSchema.validate(frame, lazy=lazy)


def validate_predictions(frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
    """Validate the prediction table against :class:`PredictedSummariesSchema`.

    Args:
        frame: Prediction rows.
        lazy: Collect every violation before raising.

    Returns:
        The validated frame.
    """
    return PredictedSummariesSchema.validate(frame, lazy=lazy)


def validate_corpus(
    documents: pd.DataFrame,
    references: pd.DataFrame,
    facts: pd.DataFrame,
    *,
    lazy: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate the three tables **and** the links between them.

    A per-table contract cannot see that a summary references an unknown document, that a document
    has no reference summary, or that a fact belongs to a document it does not exist in. Those three
    failures are exactly the ones that silently bias a ROUGE computation, so they are checked here,
    once, for every caller.

    Args:
        documents: Source corpus.
        references: Reference summaries.
        facts: Fact table.
        lazy: Collect every per-table violation before raising.

    Returns:
        The three validated frames, in the same order.

    Raises:
        ValueError: When a table references a document that is not in the corpus, when a document has
            no reference summary, or when a fact lies outside the sentences of its document.
    """
    documents = validate_documents(documents, lazy=lazy)
    references = validate_references(references, lazy=lazy)
    facts = validate_facts(facts, lazy=lazy)

    known = set(documents["doc_id"])
    unknown_refs = sorted(set(references["doc_id"]) - known)
    if unknown_refs:
        msg = (
            f"Reference summary table mentions {len(unknown_refs)} unknown document(s): "
            f"{unknown_refs[:5]}"
        )
        raise ValueError(msg)
    missing_refs = sorted(known - set(references["doc_id"]))
    if missing_refs:
        msg = (
            f"Corpus has {len(missing_refs)} document(s) without a reference summary: "
            f"{missing_refs[:5]}"
        )
        raise ValueError(msg)
    unknown_facts = sorted(set(facts["doc_id"]) - known)
    if unknown_facts:
        msg = f"Fact table mentions {len(unknown_facts)} unknown document(s): {unknown_facts[:5]}"
        raise ValueError(msg)

    n_sentences = documents.set_index("doc_id")["n_sentences"]
    facts = facts.copy()
    facts["_n_sentences"] = facts["doc_id"].map(n_sentences).astype("int64")
    outside = facts["sentence_index"] >= facts["_n_sentences"]
    if bool(outside.any()):
        offenders = facts.loc[outside, ["fact_id", "doc_id", "sentence_index"]].head(5)
        msg = f"{int(outside.sum())} fact(s) point outside the sentences of their document: {offenders.to_dict('records')}"
        raise ValueError(msg)
    facts = facts.drop(columns="_n_sentences")
    return documents, references, facts


def split_sizes(documents: pd.DataFrame) -> dict[str, int]:
    """Count the documents of each split.

    Args:
        documents: Validated corpus.

    Returns:
        Mapping ``split -> number of documents``, in :data:`SPLITS` order.
    """
    counts = documents["split"].value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in SPLITS}


def salient_fact_counts(facts: pd.DataFrame) -> dict[str, int]:
    """Count the salient facts of each type.

    Args:
        facts: Validated fact table.

    Returns:
        Mapping ``fact_type -> number of salient facts``, in :data:`FACT_TYPES` order.
    """
    salient = facts.loc[facts["salient"] == 1]
    counts = salient["fact_type"].value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in FACT_TYPES}


__all__ = [
    "DOC_ID_PATTERN",
    "FACT_TYPES",
    "INTERVENTION_TYPES",
    "SPLITS",
    "SITES",
    "URGENCIES",
    "DocumentsSchema",
    "FactsSchema",
    "PredictedSummariesSchema",
    "ReferenceSummariesSchema",
    "fact_columns",
    "salient_fact_counts",
    "split_sizes",
    "validate_corpus",
    "validate_documents",
    "validate_facts",
    "validate_predictions",
    "validate_references",
]
