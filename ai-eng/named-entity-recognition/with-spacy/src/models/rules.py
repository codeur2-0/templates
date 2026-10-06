r"""Couche de règles déclarées : motifs explicites et index de surfaces appris sur le train.

Cette couche est la partie **explicable** de la stack, et elle est volontairement séparée du
tagger :

* les **motifs** décrivent la *forme* d'une entité — référence de commande (``CMD-1234`` ou
  ``cmd 1234``), montant (``89,90 €``, ``89.90 EUR``, ``45 euros``), date (``12 mars 2025``,
  ``12/03/2025``). Ils ne dépendent d'aucun apprentissage et couvrent les trois types dont
  l'écriture
  est régulière ;
* l'**index de surfaces** ne contient que les noms annotés dans le split d'**entraînement** —
jamais
  le vocabulaire de la famille. C'est ce qui rend l'effet des surfaces réservées *mesurable* : la
  couche de règles ne peut pas connaître un produit réservé, donc elle le manque, et l'écart entre
  le train et le test est un résultat publié plutôt qu'une supposition.

Les deux règles de construction qui ont coûté le plus cher, et qui sont testées :

* les surfaces de l'index sont cherchées avec des **frontières de mot** (``(?<![\\w-])`` / ``(?!``
  ``[\\w-])``) et triées de la plus longue à la plus courte : sans cela, « GLS » est reconnu à
  l'intérieur de « GLS Express » — une mention plus courte que celle du corpus, donc deux erreurs
  au
  lieu d'une ;
* les types sont résolus par ordre de priorité déclaré (référence de commande, date, montant,
  produit, transporteur) et les spans ne se recouvrent jamais : le métier lit une liste de
  mentions,
  pas un graphe d'ambiguïtés.

Le module ne connaît ni spaCy ni le projet : il prend un texte, une index, et rend des spans.
C'est
ce qui permet de tester la référence explicable sans entraîner quoi que ce soit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Final

import pandas as pd

from src.models.contract import EntityMention, sort_mentions

#: Types d'entités reconnus par les règles, dans l'ordre de résolution des chevauchements (du plus
#: spécifique au plus général) : les motifs d'abord, les noms ensuite.
LABEL_PRIORITY: Final[tuple[str, ...]] = (
    "commande",
    "date",
    "montant",
    "produit",
    "transporteur",
)

#: Types dont la reconnaissance repose sur un **nom** (donc sur un index appris) et non sur une
#: forme régulière : ce sont les types pour lesquels une surface réservée est un obstacle.
NAME_LABELS: Final[tuple[str, ...]] = ("produit", "transporteur")

#: Mois français utilisés par les dates rédigées.
MOIS: Final[tuple[str, ...]] = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)

#: Motifs **déclarés**. Chaque motif est une définition au sens du corpus : ce qui suit est
#: annoté, tout le reste ne l'est pas — variantes rédigée et abrégée comprises.
PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    "commande": (
        r"\bCMD[- ]\d{4}\b",
        r"\bcmd[ ]\d{4}\b",
    ),
    "montant": (
        r"\d{1,3}(?:[ \u00a0]\d{3})*,\d{2}\s?(?:€|EUR)",
        r"\d{1,3}(?:[ \u00a0]\d{3})*\.\d{2}\s?(?:EUR|€)",
        r"\d{1,3}(?:[ \u00a0]\d{3})*\s?(?:euros?|€)",
    ),
    "date": (
        r"\d{1,2}\s+(?:" + "|".join(MOIS) + r")\s+\d{4}",
        r"\d{1,2}/\d{1,2}/\d{4}",
    ),
}


def _compile_patterns() -> dict[str, tuple[re.Pattern[str], ...]]:
    """Compile the declared patterns, case-insensitively.

    Returns:
        Mapping of entity type to its compiled patterns.
    """
    return {
        label: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
        for label, patterns in PATTERNS.items()
    }


#: Compiled patterns, used by :func:`pattern_spans`.
COMPILED_PATTERNS: Final[dict[str, tuple[re.Pattern[str], ...]]] = _compile_patterns()


@dataclass(frozen=True, slots=True)
class RuleSpan:
    """A span proposed by the rule layer, with the rule that produced it.

    Attributes:
        start: Index of the first character of the span.
        end: Index of the first character after the span.
        label: Entity type proposed by the rule.
        rule: Identifier of the rule (``motif:<type>`` or ``index:<type>``), published with the
            mention so a reader can see *why* a span exists.
    """

    start: int
    end: int
    label: str
    rule: str

    def as_tuple(self) -> tuple[int, int, str]:
        """Return the ``(start, end, label)`` tuple used by the metrics.

        Returns:
            The span coordinates and its type.
        """
        return (int(self.start), int(self.end), str(self.label))


@dataclass(slots=True)
class GazetteerIndex:
    """Surfaces annotated in the **training** split, indexed by entity type.

    L'index est appris sur le train, et rien d'autre : c'est ce qui distingue une référence
    explicable d'une liste codée en dur. Il est sérialisé avec le modèle (il fait partie de
    l'artefact), ce qui garantit qu'un rechargement prédit exactement comme le modèle ajusté.

    Attributes:
        surfaces: ``label -> tuple of surfaces``, sorted by decreasing length so that the longest
            match wins.
        lookup: Normalised surface to label mapping (built in :meth:`from_spans`).
    """

    surfaces: dict[str, tuple[str, ...]] = field(default_factory=dict)
    lookup: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_spans(
        cls, spans: pd.DataFrame, *, labels: tuple[str, ...] = NAME_LABELS
    ) -> GazetteerIndex:
        """Build the index from the annotated spans of the training split.

        Args:
            spans: Annotation table of the training split (``label``, ``surface``).
            labels: Entity types indexed by name (the regular ones are handled by patterns).

        Returns:
            The index, ready to match. An empty annotation table gives an empty index, which is a
            legitimate state: the rule layer then only relies on its patterns.
        """
        surfaces: dict[str, tuple[str, ...]] = {}
        lookup: dict[str, str] = {}
        if not spans.empty and {"label", "surface"}.issubset(spans.columns):
            for label in labels:
                found = sorted(
                    {
                        str(surface)
                        for surface in spans.loc[spans["label"] == label, "surface"]
                        if str(surface).strip()
                    },
                    key=lambda surface: (-len(surface), surface),
                )
                surfaces[label] = tuple(found)
                for surface in found:
                    lookup.setdefault(surface.casefold(), label)
        return cls(surfaces=surfaces, lookup=lookup)

    @property
    def size(self) -> int:
        """Number of indexed surfaces."""
        return int(sum(len(values) for values in self.surfaces.values()))

    def match(self, text: str) -> list[RuleSpan]:
        """Find every indexed surface of a text, longest surface first.

        Args:
            text: Text to scan.

        Returns:
            The spans proposed by the index, sorted by start offset, without overlap.
        """
        found: list[RuleSpan] = []
        folded = text.casefold()
        entries = sorted(self.lookup.items(), key=lambda item: (-len(item[0]), item[0]))
        for surface, label in entries:
            # Frontières de mot obligatoires : « GLS » ne doit pas être reconnu dans
            # « GLX Express », sans quoi les règles inventeraient un transporteur plus court
            # que celui du corpus.
            # celle du corpus — et la borne est justement ce qu'un modèle doit apprendre.
            pattern = re.compile(rf"(?<![\w-]){re.escape(surface)}(?![\w-])")
            for match in pattern.finditer(folded):
                span = RuleSpan(match.start(), match.end(), label, f"index:{label}")
                if not _overlaps_any(span, found):
                    found.append(span)
        return sorted(found, key=lambda span: (span.start, span.end))

    def describe(self) -> dict[str, Any]:
        """Return a documentation payload for the reports and notebooks.

        Returns:
            The number of surfaces per type, with a sample of the longest ones.
        """
        return {
            "size": self.size,
            "per_label": {label: len(values) for label, values in self.surfaces.items()},
            "longest": {label: list(values[:5]) for label, values in self.surfaces.items()},
            "definition": (
                "Surfaces annotées dans le split d'entraînement, jamais le vocabulaire de la "
                "famille : une surface réservée aux splits d'évaluation est donc invisible ici."
            ),
        }


def _overlaps_any(candidate: RuleSpan, others: list[RuleSpan]) -> bool:
    """Return whether a candidate span overlaps any span of a list.

    Args:
        candidate: Span to test.
        others: Spans already accepted.

    Returns:
        ``True`` when at least one pair shares a character.
    """
    return any(candidate.start < other.end and other.start < candidate.end for other in others)


def pattern_spans(text: str) -> list[RuleSpan]:
    """Run the declared patterns on a text.

    Args:
        text: Text to scan.

    Returns:
        The spans proposed by the patterns, in priority order and without overlap.
    """
    found: list[RuleSpan] = []
    for label in LABEL_PRIORITY:
        for pattern in COMPILED_PATTERNS.get(label, ()):
            for match in pattern.finditer(text):
                span = RuleSpan(match.start(), match.end(), label, f"motif:{label}")
                if not _overlaps_any(span, found):
                    found.append(span)
    return sorted(found, key=lambda span: (span.start, span.end))


def rule_spans(text: str, index: GazetteerIndex | None = None) -> list[RuleSpan]:
    """Apply the declared rule layer: patterns first, then the training index.

    Args:
        text: Text to scan.
        index: Surfaces observed in the training split (``None`` disables the name lookup).

    Returns:
        The spans of the rule layer, without overlap, sorted by start offset.
    """
    found = pattern_spans(text)
    if index is not None and index.lookup:
        for span in index.match(text):
            if not _overlaps_any(span, found):
                found.append(span)
    return sorted(found, key=lambda span: (span.start, span.end))


def rule_mentions(
    text: str,
    msg_id: str,
    index: GazetteerIndex | None = None,
) -> list[EntityMention]:
    """Extract the mentions proposed by the rule layer.

    Args:
        text: Text to scan.
        msg_id: Identifier of the document (carried by every mention).
        index: Surfaces observed in the training split.

    Returns:
        The mentions of the rule layer, with source ``regle`` and a confidence of 1.0: une règle
        déclarée est exacte par construction, et sa précision est mesurée sur le split de
        validation
        — elle n'est donc pas supposée.
    """
    return sort_mentions(
        [
            EntityMention(
                msg_id=msg_id,
                start=span.start,
                end=span.end,
                label=span.label,
                surface=text[span.start : span.end],
                source="regle",
                confidence=1.0,
            )
            for span in rule_spans(text, index)
        ]
    )


def describe_rules() -> dict[str, Any]:
    """Document the rule layer (used by the reports, the notebooks and the model card).

    Returns:
        The declared patterns, the indexed types and the overlap policy.
    """
    return {
        "patterns": {label: list(patterns) for label, patterns in PATTERNS.items()},
        "name_labels": list(NAME_LABELS),
        "priority": list(LABEL_PRIORITY),
        "definition": (
            "Motifs déclarés pour les types à écriture régulière (référence, montant, date) "
            "et index des noms annotés dans le train (produits, transporteurs). Les spans ne "
            "se recouvrent jamais : l'ordre de priorité est publié."
            "Les spans ne se recouvrent jamais : l'ordre de priorité est publié."
        ),
    }


__all__ = [
    "COMPILED_PATTERNS",
    "LABEL_PRIORITY",
    "MOIS",
    "NAME_LABELS",
    "PATTERNS",
    "GazetteerIndex",
    "RuleSpan",
    "describe_rules",
    "pattern_spans",
    "rule_mentions",
    "rule_spans",
]
