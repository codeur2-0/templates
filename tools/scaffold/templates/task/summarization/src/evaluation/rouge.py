"""ROUGE-1, ROUGE-2 et ROUGE-L, implémentés ici plutôt qu'importés.

Trois raisons de ne pas dépendre d'un paquet d'évaluation :

* **la définition doit être publiée**. Un rapport qui annonce « ROUGE-2 » sans dire quel tokeniseur,
  quel lissage et quelle agrégation a été utilisé annonce un chiffre qu'on ne peut pas comparer ;
* **les données sont synthétiques et françaises** : la tokenisation conserve les accents (``durée``
  et ``duree`` sont deux mots) et les chiffres (``45 minutes`` porte un fait), ce que les
  implémentations par défaut ne font pas toujours ;
* **le déterminisme** : deux exécutions doivent rendre le même score au bit près, y compris quand
  plusieurs n-grams ont la même fréquence.

Les trois variantes suivent l'article d'origine : ROUGE-N compte les n-grams communs (recouvrement de
multi-ensembles), ROUGE-L la plus longue sous-séquence commune, et les trois publient précision,
rappel et F1. Le corpus publie **deux références par document** (un résumé de référence et sa
variante), et l'évaluateur calcule le score face à la meilleure des deux — c'est la définition de
ROUGE sur plusieurs références, et c'est aussi ce qui empêche un modèle de mémoriser une seule
formulation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections import Counter

#: Token pattern of the evaluation: letters (accents included) and digits, nothing else. A decimal
#: number is split in two tokens (``4``, ``5``), which is the convention of the original package and
#: makes ``4,5`` and ``45`` different strings.
TOKEN_PATTERN = re.compile(r"[0-9a-zà-öø-ÿœæ]+")

#: N-gram orders computed by default: unigrams and bigrams, as in the ROUGE paper.
DEFAULT_ORDERS: tuple[int, ...] = (1, 2)


@dataclass(frozen=True, slots=True)
class RougeScore:
    """Precision, recall and F1 of one ROUGE variant.

    Attributes:
        precision: Matched n-grams divided by the n-grams of the generated summary.
        recall: Matched n-grams divided by the n-grams of the reference.
        f1: Harmonic mean of precision and recall (zero when both are zero).
    """

    precision: float
    recall: float
    f1: float

    def to_dict(self, prefix: str) -> dict[str, float]:
        """Flatten the score with a metric prefix.

        Args:
            prefix: Metric name (``rouge1``, ``rouge2``, ``rougeL``).

        Returns:
            ``{prefix_p, prefix_r, prefix_f}``, rounded to four decimals.
        """
        return {
            f"{prefix}_p": round(self.precision, 4),
            f"{prefix}_r": round(self.recall, 4),
            f"{prefix}_f": round(self.f1, 4),
        }


@dataclass(frozen=True, slots=True)
class RougeScores:
    """The three variants, as one row of the evaluation table.

    Attributes:
        rouge1: Unigram score.
        rouge2: Bigram score.
        rouge_l: Longest common subsequence score.
        n_reference_tokens: Length of the reference used (the best one when several are given).
    """

    rouge1: RougeScore
    rouge2: RougeScore
    rouge_l: RougeScore
    n_reference_tokens: int = 0

    def to_metrics(self) -> dict[str, float]:
        """Flatten the three variants into the columns of the prediction table.

        Returns:
            ``rouge1_f``, ``rouge2_f``, ``rouge_l_f`` and the full precision/recall triplets.
        """
        metrics: dict[str, float] = {}
        metrics.update(self.rouge1.to_dict("rouge1"))
        metrics.update(self.rouge2.to_dict("rouge2"))
        metrics.update(self.rouge_l.to_dict("rouge_l"))
        return metrics


def tokenize_for_rouge(text: str) -> list[str]:
    """Tokenise a text the way ROUGE compares it.

    Args:
        text: Text to tokenise.

    Returns:
        The lower-cased tokens, accents and digits kept, punctuation dropped.
    """
    return TOKEN_PATTERN.findall(str(text).lower())


def _count_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    """Count the n-grams of a token list.

    Args:
        tokens: Token list.
        n: Order of the n-grams.

    Returns:
        A counter of n-gram tuples (empty when the text is shorter than ``n``).
    """
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def _score(matches: int, n_predicted: int, n_reference: int) -> RougeScore:
    """Turn a match count into precision, recall and F1.

    Args:
        matches: Number of matched n-grams.
        n_predicted: Number of n-grams of the prediction.
        n_reference: Number of n-grams of the reference.

    Returns:
        The score; a division by zero yields ``0.0`` rather than ``nan`` so the metric stays finite.
    """
    precision = matches / n_predicted if n_predicted else 0.0
    recall = matches / n_reference if n_reference else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return RougeScore(precision=precision, recall=recall, f1=f1)


def rouge_n(reference: str, prediction: str, n: int = 1) -> RougeScore:
    """ROUGE-N of one prediction against one reference.

    Args:
        reference: Reference summary.
        prediction: Generated summary.
        n: Order of the n-grams (1 for ROUGE-1, 2 for ROUGE-2).

    Returns:
        The score of that order.
    """
    reference_tokens = tokenize_for_rouge(reference)
    prediction_tokens = tokenize_for_rouge(prediction)
    reference_ngrams = _count_ngrams(reference_tokens, n)
    prediction_ngrams = _count_ngrams(prediction_tokens, n)
    matches = sum((reference_ngrams & prediction_ngrams).values())
    return _score(matches, sum(prediction_ngrams.values()), sum(reference_ngrams.values()))


def longest_common_subsequence(reference: list[str], prediction: list[str]) -> int:
    """Length of the longest common subsequence of two token lists.

    Args:
        reference: Reference tokens.
        prediction: Prediction tokens.

    Returns:
        The length of the longest common subsequence (dynamic programming, one row of memory).
    """
    if not reference or not prediction:
        return 0
    previous = [0] * (len(prediction) + 1)
    for token in reference:
        current = [0]
        for column, other in enumerate(prediction, start=1):
            if token == other:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[column - 1]))
        previous = current
    return previous[-1]


def rouge_l(reference: str, prediction: str) -> RougeScore:
    """ROUGE-L of one prediction against one reference.

    Args:
        reference: Reference summary.
        prediction: Generated summary.

    Returns:
        The score of the longest common subsequence (F-beta with beta = 1).
    """
    reference_tokens = tokenize_for_rouge(reference)
    prediction_tokens = tokenize_for_rouge(prediction)
    matches = longest_common_subsequence(reference_tokens, prediction_tokens)
    return _score(matches, len(prediction_tokens), len(reference_tokens))


def rouge_scores(
    reference: str | list[str],
    prediction: str,
    *,
    orders: tuple[int, ...] = DEFAULT_ORDERS,
) -> RougeScores:
    """Score one prediction against one or several references.

    Args:
        reference: Reference summary, or list of references. With several references, the best
            ROUGE-1 F1 wins and its reference is used for ROUGE-2 and ROUGE-L, which is the
            multi-reference convention of the original package.
        prediction: Generated summary.
        orders: N-gram orders to compute (the first one is used to pick the best reference).

    Returns:
        The three scores of the winning reference.
    """
    references = [reference] if isinstance(reference, str) else list(reference)
    if not references:
        msg = "rouge_scores needs at least one reference summary"
        raise ValueError(msg)
    unigram_order = orders[0]
    best: tuple[float, str] | None = None
    for candidate in references:
        score = rouge_n(candidate, prediction, unigram_order)
        if best is None or score.f1 > best[0]:
            best = (score.f1, candidate)
    assert best is not None  # références non vides : la boucle a posé ``best``
    chosen = best[1]
    return RougeScores(
        rouge1=rouge_n(chosen, prediction, unigram_order),
        rouge2=rouge_n(chosen, prediction, 2),
        rouge_l=rouge_l(chosen, prediction),
        n_reference_tokens=len(tokenize_for_rouge(chosen)),
    )


def corpus_rouge(
    references: list[str],
    predictions: list[str],
    *,
    orders: tuple[int, ...] = DEFAULT_ORDERS,
) -> dict[str, float]:
    """Average ROUGE over a corpus (micro-aggregation is not used: documents are the unit).

    Args:
        references: Reference summaries, one per document.
        predictions: Generated summaries, in the same order.
        orders: N-gram orders to compute.

    Returns:
        ``rouge1_f``, ``rouge2_f``, ``rouge_l_f`` and the mean reference length, rounded to four
        decimals.

    Raises:
        ValueError: When the two lists do not have the same length.
    """
    if len(references) != len(predictions):
        msg = f"{len(references)} references for {len(predictions)} predictions"
        raise ValueError(msg)
    if not references:
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rouge_l_f": 0.0, "reference_tokens_mean": 0.0}
    scores = [
        rouge_scores(reference, prediction, orders=orders)
        for reference, prediction in zip(references, predictions, strict=True)
    ]
    n = len(scores)
    return {
        "rouge1_f": round(sum(item.rouge1.f1 for item in scores) / n, 4),
        "rouge2_f": round(sum(item.rouge2.f1 for item in scores) / n, 4),
        "rouge_l_f": round(sum(item.rouge_l.f1 for item in scores) / n, 4),
        "reference_tokens_mean": round(sum(item.n_reference_tokens for item in scores) / n, 2),
    }


__all__ = [
    "DEFAULT_ORDERS",
    "TOKEN_PATTERN",
    "RougeScore",
    "RougeScores",
    "corpus_rouge",
    "longest_common_subsequence",
    "rouge_l",
    "rouge_n",
    "rouge_scores",
    "tokenize_for_rouge",
]
