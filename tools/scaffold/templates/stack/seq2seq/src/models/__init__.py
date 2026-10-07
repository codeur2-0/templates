"""Couche modèles de la stack **seq2seq** : la fabrique et l'encodeur-décodeur appris.

Surface publique, importée partout ailleurs dans le projet::

    from src.models import build_model, load_model        # fabrique + rechargement
    from src.models.contract import BaseTextGenerator     # contrat des stratégies
    from src.models.factory import describe_algorithm     # algorithmes servis par la stack

Rien en dehors de ce paquet n'importe PyTorch ni la bibliothèque ``tokenizers`` : le framework reste
un détail d'implémentation derrière :class:`~src.models.contract.BaseTextGenerator` (inversion de
dépendance). C'est ce qui permet de comparer un encodeur-décodeur appris et deux baselines
extractives sur le même corpus, sans qu'aucun pipeline n'ait à savoir laquelle apprend.

Deux familles cohabitent dans ce paquet, et c'est volontaire :

* les **stratégies apprises** (:class:`~src.models.summarizer.EncoderDecoderSummarizer` et le
  vocabulaire partagé de :mod:`src.models.tokenizer`), qui viennent de cette stack ;
* les **baselines extractives** (:class:`~src.models.lead.LeadSummarizer`,
  :class:`~src.models.textrank.TextRankSummarizer`) et le contrat, qui viennent de la couche de
  tâche et ne dépendent d'aucun framework.
"""

from src.models.contract import BaseTextGenerator, TextSummary
from src.models.factory import (
    ALGORITHMS,
    AlgorithmSpec,
    available_algorithms,
    build_model,
    describe_algorithm,
    load_model,
    supported_tasks,
)
from src.models.lead import LeadSummarizer
from src.models.summarizer import EncoderDecoderSummarizer
from src.models.textrank import MMR_GRID, TextRankSummarizer
from src.models.tokenizer import SPECIAL_TOKENS, TOKENIZER_VERSION, SharedWordPieceTokenizer

__all__ = [
    "ALGORITHMS",
    "MMR_GRID",
    "SPECIAL_TOKENS",
    "TOKENIZER_VERSION",
    "AlgorithmSpec",
    "BaseTextGenerator",
    "EncoderDecoderSummarizer",
    "LeadSummarizer",
    "SharedWordPieceTokenizer",
    "TextRankSummarizer",
    "TextSummary",
    "available_algorithms",
    "build_model",
    "describe_algorithm",
    "load_model",
    "supported_tasks",
]
