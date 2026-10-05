"""Synthetic catalogue of the embedding-pipeline family: every fact written three times.

The corpus of this family is deliberately **redundant**, because that is what an embedding index
has to cope with in a real company: the same information is written again — a constructor notice,
a commercial sheet, a repair-desk note — and nobody owns the duplicate. Two facts are planted per
reference (the battery autonomy and the legal warranty); the two sentences of a fact — the one
that carries the value and the one that contextualises it — are **copy-pasted** into the three
fiches, while the style sentence, the editorial section, the wiki space and the title differ. The
three fiches are therefore three genuinely equivalent answers to the same question, and an index
that returns any of them has done its job; they are also near-duplicates, which is what makes the
second use case of the family — ``nearest_neighbours``, run over the whole corpus — measurable.

That design has a measurable consequence the family states instead of hiding: every answerable
question has **three** relevant documents (six for the multi-document segment: two facts x three
fiches), so ``recall_at_1`` is capped at ``1/3`` and reads as a "precision-like" recall, while
``recall_at_5`` says whether the whole set was retrieved. Each fact yields a canonical question,
and half of them also yield a paraphrase that changes the vocabulary without removing the
reference name:

``facile``
    the question names the reference and the fact ("Quelle est la durée de garantie du casque
    Aria ?"), with the vocabulary of the planted sentence.
``paraphrase``
    the same question asked with other words — 34 % of the content words shared with the planted
    sentence against 71 % for the canonical form, measured on the generated corpus: it measures
    what a representation buys over exact word matching.
``multi_document``
    two facts of the same reference at once: the answer needs two sentences, from two different
    fiches, so the segment is capped by construction.
``hors_corpus``
    a neighbouring fact the catalogue never writes down (a price, a delivery delay, a repair
    cost): only abstention is correct there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.data.generator_base import BaseCorpusGenerator, GeneratedCorpus
from src.preprocessing.transformers import tokenize
from src.utils.config_access import as_mapping
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Reference:
    """One catalogue reference, with the two values its six fiches quote.

    Attributes:
        code: Four-letter reference code (identifies the product folder).
        name: Name used inside the sentences and the questions ("casque Aria").
        gender: Grammatical gender of the name (``m`` or ``f``), used to agree the articles.
        category: Product family (audio, mobilite, informatique, maison).
        autonomy_hours: Battery life promised by the constructor, in hours.
        warranty_months: Legal warranty, in months.
    """

    code: str
    name: str
    gender: str
    category: str
    autonomy_hours: int
    warranty_months: int

    @property
    def article(self) -> str:
        """Definite article that agrees with the name (``le``, ``la`` or ``l'``)."""
        if self.name[:1].lower() in "aeiouéèêàâîïôû":
            return "l'"
        return "le" if self.gender == "m" else "la"

    @property
    def of_the(self) -> str:
        """Contraction of ``de`` with the article (``du``, ``de la``, ``de l'``)."""
        return {"le": "du", "la": "de la", "l'": "de l'"}[self.article]

    @property
    def with_article(self) -> str:
        """Name preceded by its article (``la souris Sour``, ``l'enceinte Onda``)."""
        separator = "" if self.article.endswith("'") else " "
        return f"{self.article}{separator}{self.name}"

    @property
    def of_name(self) -> str:
        """Name preceded by the contraction of ``de`` (``du casque Aria``, ``de l'enceinte``)."""
        return f"{self.of_the}{self.with_article.removeprefix(self.article)}"

    @property
    def titled(self) -> str:
        """Name preceded by its article, sentence-initial (``La souris Sour``, ``L'enceinte``)."""
        return self.with_article[:1].upper() + self.with_article[1:]


@dataclass(frozen=True, slots=True)
class Style:
    """One of the three ways a fact is published, and by which team.

    Attributes:
        key: Stable identifier of the style.
        label: Lower-case label, used in prose.
        title_label: Title-case label, used in the title of the fiche.
        section: Editorial section declared in the contract.
        source: Wiki space the fiche belongs to.
        support: Sentence identifying the style of the fiche. It carries no fact word on purpose:
            the answer is always the planted sentence, never this one.
    """

    key: str
    label: str
    title_label: str
    section: str
    source: str
    support: str


@dataclass(frozen=True, slots=True)
class Fact:
    """One planted fact: the sentence that carries it, its consequence, and the questions.

    Attributes:
        key: Stable identifier of the fact.
        label: Lower-case label, used in prose.
        intent: Business intent of the questions written about this fact.
        clause: The sentence that carries the value; it is copy-pasted into the three fiches.
        conditions: Consequence sentence, reused by every reference quoting the same fact.
        question: Factual question whose answer is the planted sentence.
        paraphrasis: Alternative wording of that question, deliberately far from the fiches.
    """

    key: str
    label: str
    intent: str
    clause: str
    conditions: str
    question: str
    paraphrasis: str


#: The catalogue: four product families, twenty-eight references in total. Every name carries the
#: word a customer would use ("casque", "souris", "imprimante"): it is the term the questions repeat.
REFERENCES: tuple[Reference, ...] = (
    Reference("ARIA", "casque Aria", "m", "audio", 24, 24),
    Reference("ONDA", "enceinte Onda", "f", "audio", 14, 24),
    Reference("PIKO", "kit d'écouteurs Piko", "m", "audio", 8, 12),
    Reference("VELA", "barre de son Vela", "f", "audio", 10, 36),
    Reference("MIRA", "montre Mira", "f", "mobilite", 40, 24),
    Reference("TRAC", "traceur Trac", "m", "mobilite", 72, 12),
    Reference("ROAM", "trottinette Roam", "f", "mobilite", 35, 24),
    Reference("GALA", "vélo Gala", "m", "mobilite", 90, 36),
    Reference("GYRO", "gyropode Gyro", "m", "mobilite", 28, 24),
    Reference("DRON", "drone Dron", "m", "mobilite", 26, 12),
    Reference("NEXA", "clavier Nexa", "m", "informatique", 120, 24),
    Reference("SOUR", "souris Sour", "f", "informatique", 200, 24),
    Reference("VIA", "webcam Via", "f", "informatique", 6, 12),
    Reference("ROUT", "routeur Rout", "m", "informatique", 30, 36),
    Reference("IMPR", "imprimante Impr", "f", "informatique", 48, 24),
    Reference("TABL", "tablette Tabl", "f", "informatique", 12, 24),
    Reference("CAFE", "cafetière Cafe", "f", "maison", 4, 24),
    Reference("MIXE", "robot Mixe", "m", "maison", 6, 36),
    Reference("ASPI", "aspirateur Aspi", "m", "maison", 45, 24),
    Reference("PUR", "purificateur Pur", "m", "maison", 12, 24),
    Reference("LAMP", "lampe Lamp", "f", "maison", 20, 12),
    Reference("JARD", "robot d'arrosage Jard", "m", "maison", 60, 24),
    Reference("SPA", "spa gonflable Spa", "m", "maison", 24, 24),
    Reference("BOUI", "bouilloire Boui", "f", "maison", 3, 24),
    Reference("FRIT", "friteuse Frit", "f", "maison", 5, 24),
    Reference("VENT", "ventilateur Vent", "m", "maison", 18, 12),
    Reference("CHAU", "chauffage Chau", "m", "maison", 30, 24),
    Reference("ANTI", "anti-nuisibles Anti", "m", "maison", 90, 24),
)

#: The three styles. The sentence that carries the fact is identical in the three fiches; only the
#: framing sentence, the section and the title tell the reader which team published it.
STYLES: tuple[Style, ...] = (
    Style(
        "notice",
        "notice constructeur",
        "Notice",
        "procedure",
        "wiki_it",
        "Cette valeur figure dans la notice constructeur du produit.",
    ),
    Style(
        "commerciale",
        "fiche commerciale",
        "Fiche commerciale",
        "definition",
        "wiki_ops",
        "Cette valeur est reprise dans la fiche commerciale de l'enseigne.",
    ),
    Style(
        "sav",
        "note SAV",
        "Note SAV",
        "contact",
        "wiki_support",
        "Cette valeur est confirmée par les dossiers traités en atelier.",
    ),
)

#: The two facts planted for every reference. ``conditions`` never carries the quoted value: it
#: contextualises the fact without competing with the answer sentence.
FACTS: tuple[Fact, ...] = (
    Fact(
        key="autonomie",
        label="autonomie",
        intent="definition",
        clause="L'autonomie {of_name} atteint {value} en lecture continue.",
        conditions=(
            "La mesure de référence est faite écran éteint, à température ambiante, sur un "
            "appareil chargé à cent pour cent."
        ),
        question="Quelle est l'autonomie {of_name} en lecture continue ?",
        paraphrasis="Combien de temps fonctionne {with_article} sans recharge ?",
    ),
    Fact(
        key="garantie",
        label="garantie légale",
        intent="politique",
        clause="La garantie {of_name} couvre {value} pièces et main-d'œuvre à compter de l'achat.",
        conditions=(
            "La prise en charge se fait en atelier agréé, sur présentation de la facture d'achat."
        ),
        question="Quelle est la durée de garantie {of_name} ?",
        paraphrasis="Pendant combien de mois la couverture {of_name} court-elle après l'achat ?",
    ),
)

#: Questions the catalogue cannot answer: (question template, intent). They stay on the vocabulary
#: of the catalogue — the topic exists, the fact does not — which is the hard half of abstention.
HORS_CORPUS_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("Combien coûte la réparation {of_name} hors garantie ?", "calcul"),
    ("Livrez-vous {with_article} le samedi ?", "procedure"),
    ("Puis-je payer {with_article} en plusieurs fois ?", "calcul"),
    ("Reprenez-vous {with_article} après deux ans d'usage ?", "politique"),
    ("Le remplacement de la batterie {of_name} est-il gratuit à vie ?", "politique"),
    ("Vendez-vous les pièces détachées {of_name} au détail ?", "calcul"),
)

#: Questions asking the two planted facts at once (the answer needs one sentence per fact).
MULTI_DOCUMENT_QUESTIONS: tuple[str, ...] = (
    "Quelle est l'autonomie {of_name} en lecture continue et quelle est la durée de garantie ?",
    "Quelle garantie couvre {with_article} et combien de temps tient sa batterie en usage continu ?",
)

#: Neutral sentence opening every fiche: it must not steal the answer from the planted sentence, and
#: it is the only sentence shared by the whole catalogue.
NEUTRAL: str = "Cette fiche interne est révisée à chaque évolution de gamme."


def _render(template: str, reference: Reference, *, value: str | None = None) -> str:
    """Render a template of the catalogue for a given reference.

    Every sentence and every question of the family goes through this helper: the agreement and
    the elision of the articles are written once, and a template can never produce a sentence the
    reference does not agree with.

    Args:
        template: Template using the ``{name}``, ``{with_article}``, ``{of_name}`` and
            ``{titled}`` placeholders, plus ``{value}`` when the sentence quotes a value.
        reference: Reference the sentence is written for.
        value: Value quoted by a planted sentence, when the template quotes one.

    Returns:
        The rendered sentence.
    """
    placeholders: dict[str, str] = {
        "name": reference.name,
        "with_article": reference.with_article,
        "of_name": reference.of_name,
        "titled": reference.titled,
    }
    if value is not None:
        placeholders["value"] = value
    return template.format(**placeholders)


def _parse_date(value: Any) -> date | None:
    """Parse an ISO date from the configuration, returning ``None`` when it is absent."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


@dataclass
class SyntheticCorpusGenerator(BaseCorpusGenerator):
    """Generate the redundant catalogue, its questions and the metadata.

    The volume of questions is derived from the corpus rather than hard-coded: one factual question
    per (reference, fact), one paraphrase per reference for the first
    ``paraphrase_share * n_factual`` references, then double questions and unanswerable questions as
    a share of what has been written so far. Every share is clamped, so a small ``n_samples``
    produces a small but well-formed corpus instead of an empty table.

    Attributes:
        dataset_name: Name archived in the metadata.
        n_documents: Target number of fiches (truncated to whole references).
        seed: Seed of the random generator.
        reference_date: Date used to date the fiches.
        paraphrase_share: Share of the factual questions re-asked with another wording.
        multi_document_share: Share of the questions asking both planted facts at once.
        hors_corpus_share: Share of the questions the catalogue cannot answer.
        val_size: Share of questions in the validation split.
        test_size: Share of questions in the test split.
        references: Catalogue of the family.
        styles: Ways a fact is published.
        facts: Facts planted in every reference.
        hors_corpus_questions: Unanswerable question templates and their intent.
        random_state: Random generator, seeded in :meth:`__post_init__`.
    """

    dataset_name: str = "equipment_catalogue_editions"
    n_documents: int = 168
    seed: int = 42
    reference_date: date = date(2025, 1, 6)
    paraphrase_share: float = 0.5
    multi_document_share: float = 0.15
    hors_corpus_share: float = 0.15
    val_size: float = 0.20
    test_size: float = 0.35
    references: tuple[Reference, ...] = field(default=REFERENCES, repr=False)
    styles: tuple[Style, ...] = field(default=STYLES, repr=False)
    facts: tuple[Fact, ...] = field(default=FACTS, repr=False)
    hors_corpus_questions: tuple[tuple[str, str], ...] = field(
        default=HORS_CORPUS_QUESTIONS, repr=False
    )
    random_state: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Seed the random generator and validate the shares."""
        self.random_state = np.random.default_rng(self.seed)
        for name in ("paraphrase_share", "multi_document_share", "hors_corpus_share"):
            share = float(getattr(self, name))
            if not 0.0 <= share < 1.0:
                msg = f"{name} must be a share in [0, 1), got {share}"
                raise ValueError(msg)
        if not 0.0 < self.test_size < 1.0 or not 0.0 <= self.val_size < 1.0:
            msg = f"invalid split sizes: val_size={self.val_size}, test_size={self.test_size}"
            raise ValueError(msg)

    @classmethod
    def from_config(cls, config: Any, *, seed: int | None = None) -> SyntheticCorpusGenerator:
        """Build the generator from the ``data`` node of the configuration.

        Args:
            config: ``data`` configuration node (``seed``, ``n_samples`` and the ``corpus``
                sub-node are read).
            seed: Optional seed override.

        Returns:
            The configured generator.
        """
        settings = as_mapping(config)
        corpus = as_mapping(settings.get("corpus"))
        return cls(
            dataset_name=str(settings.get("dataset_name", "equipment_catalogue_editions")),
            n_documents=int(settings.get("n_samples", 168)),
            seed=int(seed if seed is not None else settings.get("seed", 42)),
            reference_date=_parse_date(corpus.get("reference_date")) or date(2025, 1, 6),
            paraphrase_share=float(corpus.get("paraphrase_share", 0.5)),
            multi_document_share=float(corpus.get("multi_document_share", 0.15)),
            hors_corpus_share=float(corpus.get("hors_corpus_share", 0.15)),
            val_size=float(corpus.get("val_size", 0.20)),
            test_size=float(corpus.get("test_size", 0.35)),
        )

    # ------------------------------------------------------------------ génération --------
    def generate(self) -> GeneratedCorpus:
        """Generate the corpus, the questions and the metadata.

        Returns:
            The :class:`GeneratedCorpus`.
        """
        documents = self._documents()
        questions = self._questions(documents)
        metadata = self._metadata(documents, questions)
        logger.info(
            "Corpus '{}' generated | {} documents | {} questions | {} unanswerable",
            self.dataset_name,
            len(documents),
            len(questions),
            int((questions["answer_type"] == "unanswerable").sum()),
        )
        return GeneratedCorpus(documents=documents, queries=questions, metadata=metadata)

    def _catalogue(self) -> tuple[Reference, ...]:
        """Return the references kept by the configuration (six fiches per reference)."""
        per_reference = len(self.styles) * len(self.facts)
        wanted = max(min(self.n_documents // per_reference, len(self.references)), 2)
        return self.references[:wanted]

    def _documents(self) -> pd.DataFrame:
        """Build the corpus: one fiche per (reference, fact, style).

        Returns:
            The corpus frame, validated by :class:`src.data.schemas.DocumentsSchema`.
        """
        rows: list[dict[str, Any]] = []
        number = 0
        for reference in self._catalogue():
            for fact in self.facts:
                for style in self.styles:
                    number += 1
                    rows.append(self._document_row(number, reference, fact, style))
        frame = pd.DataFrame(rows)
        return frame.sort_values("doc_id", kind="stable").reset_index(drop=True)

    @staticmethod
    def _title(reference: Reference, style: Style, fact: Fact) -> str:
        """Return the title of a fiche (the identity used to annotate the questions)."""
        return f"{style.title_label} — {fact.label} — {reference.name} ({reference.code})"

    def _document_row(
        self, number: int, reference: Reference, fact: Fact, style: Style
    ) -> dict[str, Any]:
        """Build one fiche: the planted sentence, its consequence, the style and the reference.

        The planted sentence comes first, then its consequence, then the sentence that identifies
        the style, and the fiche closes on the reference of its own folder. Order matters: the
        sentence a question has the most words in common with must stay the planted one, otherwise
        an extractive answer would quote another sentence and the measured exact match would say
        more about the generator than about the model. The consequence and the folder line name the
        reference on purpose — that is what keeps the three fiches of a reference close to each
        other in a vector space, and therefore what makes deduplication measurable.
        """
        value = self._value(reference, fact)
        text = " ".join(
            [
                _render(fact.clause, reference, value=value),
                _render(fact.conditions, reference),
                style.support,
                (
                    f"Le dossier de suivi {reference.of_name} porte la référence "
                    f"{reference.code}-{style.key.upper()}."
                ),
            ]
        )
        published = self.reference_date - timedelta(days=int(self.random_state.integers(15, 400)))
        return {
            "doc_id": f"DOC-{number:04d}",
            "title": self._title(reference, style, fact),
            "section": style.section,
            "source": style.source,
            "published_at": pd.Timestamp(published),
            "n_tokens": len(tokenize(text)),
            "text": text,
        }

    @staticmethod
    def _value(reference: Reference, fact: Fact) -> str:
        """Return the value quoted by the planted sentence of a fact.

        The value is a *phrase* — "24 heures", "24 mois" — so that the sentence reads like a real
        fiche and the extractive answer reproduces it verbatim.
        """
        values = {
            "autonomie": f"{reference.autonomy_hours} heures",
            "garantie": f"{reference.warranty_months} mois",
        }
        return values[fact.key]

    def _questions(self, documents: pd.DataFrame) -> pd.DataFrame:
        """Write the annotated questions from the planted sentences.

        Args:
            documents: Generated corpus.

        Returns:
            The questions, with their split and their reference answer.
        """
        sentences = self._sentence_index(documents)
        rows: list[dict[str, Any]] = []
        rows.extend(self._factual_questions(sentences))
        rows.extend(self._paraphrase_questions(sentences, len(rows)))
        frame = pd.DataFrame(rows)
        frame = self._append_multi_document(frame, sentences)
        frame = self._append_hors_corpus(frame, sentences)
        return self._assign_splits(frame)

    def _sentence_index(
        self, documents: pd.DataFrame
    ) -> dict[tuple[str, str, str], tuple[str, str]]:
        """Map every ``(reference, fact, style)`` triple to its document id and planted sentence.

        The sentence is read back **from the generated text** and not from the template: the
        reference answer is therefore, character for character, what the corpus contains, and a
        fiche whose sentence failed to render is left out of the annotations instead of being
        annotated with a sentence it does not contain.
        """
        rows = documents.to_dict(orient="records")
        by_title = {str(row["title"]): str(row["doc_id"]) for row in rows}
        texts = {str(row["doc_id"]): str(row["text"]) for row in rows}
        index: dict[tuple[str, str, str], tuple[str, str]] = {}
        for reference in self._catalogue():
            for fact in self.facts:
                statement = _render(fact.clause, reference, value=self._value(reference, fact))
                for style in self.styles:
                    doc_id = by_title.get(self._title(reference, style, fact))
                    if doc_id is None or statement not in texts.get(doc_id, ""):
                        continue
                    index[(reference.code, fact.key, style.key)] = (doc_id, statement)
        return index

    def _styles_of(
        self, sentences: dict[tuple[str, str, str], tuple[str, str]], code: str, fact_key: str
    ) -> list[tuple[str, str]]:
        """Return the ``(doc_id, sentence)`` pairs of every style holding a fact, in style order."""
        return [
            sentences[(code, fact_key, style.key)]
            for style in self.styles
            if (code, fact_key, style.key) in sentences
        ]

    def _annotate(
        self,
        question: str,
        entries: list[tuple[str, str]],
        *,
        difficulty: str,
        intent: str,
        answer_type: str = "extractive",
        answer: str | None = None,
    ) -> dict[str, Any]:
        """Build one annotated question from the fiches that answer it.

        Args:
            question: The question text.
            entries: ``(doc_id, sentence)`` pairs that answer it.
            difficulty: Annotated difficulty.
            intent: Business intent.
            answer_type: How the reference answer relates to the corpus.
            answer: Reference answer (the concatenation of the sentences when omitted).

        Returns:
            The question row.
        """
        doc_ids = [doc_id for doc_id, _ in entries]
        sentences = list(dict.fromkeys(sentence for _, sentence in entries))
        return {
            "query_id": "",
            "question": question,
            "reference_answer": answer or " ".join(sentences),
            "answer_type": answer_type,
            "difficulty": difficulty,
            "intent": intent,
            "split": "",
            "gold_doc_ids": ",".join(doc_ids),
            "gold_span": "||".join(sentences),
            "n_gold_docs": len(doc_ids),
        }

    def _factual_questions(
        self, sentences: dict[tuple[str, str, str], tuple[str, str]]
    ) -> list[dict[str, Any]]:
        """Write one factual question per reference and per fact (the ``facile`` difficulty)."""
        rows: list[dict[str, Any]] = []
        for reference in self._catalogue():
            for fact in self.facts:
                entries = self._styles_of(sentences, reference.code, fact.key)
                if not entries:
                    continue
                rows.append(
                    self._annotate(
                        _render(fact.question, reference),
                        entries,
                        difficulty="facile",
                        intent=fact.intent,
                        answer=entries[0][1],
                    )
                )
        return rows

    def _paraphrase_questions(
        self, sentences: dict[tuple[str, str, str], tuple[str, str]], factual_count: int
    ) -> list[dict[str, Any]]:
        """Write the paraphrase questions (same answer, other wording).

        One paraphrase per reference is enough to measure what a lexical index loses: the fact asked
        alternates between the two planted facts, so the segment is not a proxy for one of them.
        """
        wanted = min(int(round(factual_count * self.paraphrase_share)), len(self._catalogue()))
        rows: list[dict[str, Any]] = []
        for position, reference in enumerate(self._catalogue()[: max(wanted, 0)]):
            fact = self.facts[position % len(self.facts)]
            entries = self._styles_of(sentences, reference.code, fact.key)
            if not entries:
                continue
            rows.append(
                self._annotate(
                    _render(fact.paraphrasis, reference),
                    entries,
                    difficulty="paraphrase",
                    intent=fact.intent,
                )
            )
        return rows

    def _append_multi_document(
        self,
        questions: pd.DataFrame,
        sentences: dict[tuple[str, str, str], tuple[str, str]],
    ) -> pd.DataFrame:
        """Add the questions asking both planted facts at once.

        The reference answer concatenates one sentence per fact and every fiche that states them is
        annotated relevant: a system that returns a single passage is capped on this segment.
        """
        wanted = max(int(len(questions) * self.multi_document_share), 1)
        rows: list[dict[str, Any]] = []
        for reference in self._catalogue():
            if len(rows) >= wanted:
                break
            entries: list[tuple[str, str]] = []
            for fact in self.facts:
                entries.extend(self._styles_of(sentences, reference.code, fact.key))
            if len(entries) < len(self.facts) * len(self.styles):
                continue
            answer = " ".join(
                sentences[(reference.code, fact.key, "notice")][1] for fact in self.facts
            )
            rows.append(
                self._annotate(
                    _render(
                        MULTI_DOCUMENT_QUESTIONS[len(rows) % len(MULTI_DOCUMENT_QUESTIONS)],
                        reference,
                    ),
                    entries,
                    difficulty="multi_document",
                    intent="procedure",
                    answer_type="abstractive",
                    answer=answer,
                )
            )
        if not rows:
            return questions
        return pd.concat([questions, pd.DataFrame(rows)], ignore_index=True)

    def _append_hors_corpus(
        self,
        questions: pd.DataFrame,
        sentences: dict[tuple[str, str, str], tuple[str, str]],
    ) -> pd.DataFrame:
        """Add the questions the catalogue cannot answer.

        ``gold_doc_ids`` points at the fiche a retriever will return (the notice of the first fact
        of the reference), ``gold_span`` stays empty and ``answer_type`` says ``unanswerable``: the
        only correct behaviour is to abstain.
        """
        catalogue = self._catalogue()
        wanted = min(
            max(int(len(questions) * self.hors_corpus_share), 1),
            len(self.hors_corpus_questions) * len(catalogue),
        )
        rows: list[dict[str, Any]] = []
        for position in range(wanted):
            reference = catalogue[(position // len(self.hors_corpus_questions)) % len(catalogue)]
            template, intent = self.hors_corpus_questions[position % len(self.hors_corpus_questions)]
            entry = sentences.get((reference.code, self.facts[0].key, "notice"))
            if entry is None:
                continue
            rows.append(
                {
                    "query_id": "",
                    "question": _render(template, reference),
                    "reference_answer": "Aucune réponse dans le catalogue.",
                    "answer_type": "unanswerable",
                    "difficulty": "hors_corpus",
                    "intent": intent,
                    "split": "",
                    "gold_doc_ids": entry[0],
                    "gold_span": "",
                    "n_gold_docs": 1,
                }
            )
        return pd.concat([questions, pd.DataFrame(rows)], ignore_index=True)

    def _assign_splits(self, questions: pd.DataFrame) -> pd.DataFrame:
        """Assign an identifier and a split to every question.

        The split is assigned **within each difficulty**, not globally: every difficulty must be
        represented in the calibration split (the abstention threshold is learned there), in the
        validation split and in the test split, otherwise a segment would be measured on a handful
        of questions or not at all.
        """
        frame = questions.reset_index(drop=True).copy()
        frame["split"] = "calibration"
        for difficulty in sorted(frame["difficulty"].unique()):
            positions = frame.index[frame["difficulty"] == difficulty].to_numpy()
            shuffled = self.random_state.permutation(positions)
            count = len(shuffled)
            n_test = int(round(count * self.test_size))
            n_val = int(round(count * self.val_size))
            if count >= 3:
                n_test = min(max(n_test, 1), count - 2)
                n_val = min(max(n_val, 1), count - n_test - 1)
            else:
                n_test = min(n_test, count)
                n_val = min(n_val, count - n_test)
            frame.loc[shuffled[:n_test], "split"] = "test"
            frame.loc[shuffled[n_test : n_test + n_val], "split"] = "val"
        frame["query_id"] = [f"QRY-{index + 1:04d}" for index in range(len(frame))]
        return frame

    def _metadata(self, documents: pd.DataFrame, questions: pd.DataFrame) -> dict[str, Any]:
        """Build the generation metadata archived next to the corpus."""
        by_difficulty = questions["difficulty"].value_counts().to_dict()
        answerable = questions.loc[questions["answer_type"] != "unanswerable"]
        extractive = questions.loc[questions["answer_type"] == "extractive"]
        return {
            "dataset_name": self.dataset_name,
            "seed": self.seed,
            "reference_date": self.reference_date.isoformat(),
            "n_documents": int(len(documents)),
            "n_questions": int(len(questions)),
            "n_answerable": int(len(answerable)),
            "n_unanswerable": int(len(questions) - len(answerable)),
            "n_products": int(len(self._catalogue())),
            "n_facts": int(len(self.facts)),
            "n_styles": int(len(self.styles)),
            "n_sources": int(documents["source"].nunique()),
            "mean_document_tokens": float(documents["n_tokens"].mean()),
            "extractive_share": round(float(len(extractive) / max(len(questions), 1)), 4),
            "questions_by_difficulty": {
                str(key): int(value) for key, value in by_difficulty.items()
            },
            "questions_by_split": {
                str(key): int(value)
                for key, value in questions["split"].value_counts().to_dict().items()
            },
            "retrievable_ceiling": round(
                float(len(answerable) / max(len(questions), 1)) * 100.0, 2
            ),
            "generator": "SyntheticCorpusGenerator",
            "notes": [
                "Chaque fait est écrit trois fois (notice constructeur, fiche commerciale, note "
                "SAV) : les deux phrases du fait sont recopiées dans les trois fiches, la phrase "
                "de style, la section, l'espace d'origine et le titre diffèrent. Les trois fiches "
                "sont annotées pertinentes : recall@1 est plafonné au tiers, ce qui en fait une "
                "lecture de précision, et c'est recall@5 qui dit si le jeu d'exemplaires a été "
                "retrouvé.",
                "Les questions sont écrites à partir des phrases plantées : la réponse de "
                "référence est une phrase du corpus, mot pour mot, et elle est identique dans les "
                "trois fiches qui l'écrivent.",
                "Les paraphrases changent le vocabulaire de la question, jamais l'entité qui "
                "identifie la référence : sans elle, plusieurs fiches répondraient et la "
                "difficulté mesurerait l'ambiguïté du générateur.",
                "Les questions hors corpus portent sur un fait voisin que le catalogue n'écrit "
                "nulle part : seule l'abstention y est correcte.",
                "Le découpage (110 tokens, 30 de recouvrement) laisse une fiche — une "
                "cinquantaine de tokens — dans un seul passage : le dédoublonnage porte donc sur "
                "les fiches, pas sur des fragments.",
            ],
        }


__all__ = [
    "FACTS",
    "Fact",
    "HORS_CORPUS_QUESTIONS",
    "MULTI_DOCUMENT_QUESTIONS",
    "NEUTRAL",
    "REFERENCES",
    "STYLES",
    "Reference",
    "Style",
    "SyntheticCorpusGenerator",
]
