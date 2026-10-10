"""Vocabulaire WordPiece partagé par l'encodeur et le décodeur.

Un modèle de résumé a besoin d'un vocabulaire **commun** aux deux côtés : le document et son résumé
sont écrits dans la même langue, avec les mêmes unités (``45``, ``minutes``, ``JNT-204``). Un
vocabulaire appris sur le train, partagé, fait deux choses qu'un vocabulaire importé ne ferait pas :

* il rend le projet **hors ligne** — aucun fichier de vocabulaire n'est téléchargé, et la fiche de
  modèle publie sa taille et le taux de jetons inconnus réellement observé ;
* il rend les mots du corpus visibles : ``P-12``, ``JNT-204`` et ``45 minutes`` sont des unités
  fréquentes ici, ils ont donc leurs propres pièces, au lieu d'être découpés en caractères.

Deux points d'attention, tous les deux testés : la tokenisation est **déterministe** — le
vocabulaire est appris par le projet, pas par l'apprenti de la bibliothèque ``tokenizers`` (voir
:func:`learn_vocabulary`) — et les identifiants des jetons spéciaux sont fixés par le projet : un
identifiant qui se décale d'un caractère produit un modèle qui apprend un bruit parfaitement
reproductible, ce qui est la pire des situations.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from src.utils.logging import get_logger
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from tokenizers.processors import TemplateProcessing

logger = get_logger(__name__)

#: Spécial tokens of the shared vocabulary, in a fixed order (their ids must never move).
SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[UNK]", "[BOS]", "[EOS]")

#: Prefix marking a piece that continues the previous one inside a word (WordPiece convention).
CONTINUATION_PREFIX = "##"

#: Ponctuation que le corpus colle au mot qui la **précède** (``CBL-045.``, ``45 minutes,``).
PUNCTUATION_BEFORE = ".,;:!?%\"')]}-"

#: Ponctuation que le corpus colle au mot qui la **suit** (``l'automate``, ``(AU-7)``).
PUNCTUATION_AFTER = "\"'([{-"

#: Espaces à retirer avant la ponctuation qui appartient au mot précédent.
_SPACE_BEFORE = re.compile(rf"\s+([{re.escape(PUNCTUATION_BEFORE)}])")

#: Espaces à retirer après la ponctuation qui appartient au mot suivant.
_SPACE_AFTER = re.compile(rf"([{re.escape(PUNCTUATION_AFTER)}])\s+")

#: Découpe d'un texte en segments : un mot (lettres, chiffres, tirets internes) ou un signe.
_SEGMENT = re.compile(r"\w+|[^\w\s]")

#: Version of the local tokenizer implementation, published in the model card.
TOKENIZER_VERSION = "shared-wordpiece-v3"


def rejoin_punctuation(text: str) -> str:
    """Recoller la ponctuation que la segmentation en pièces a isolée.

    ``BertPreTokenizer`` sépare les espaces **et** la ponctuation : ``CBL-045.`` arrive au modèle en
    pièces ``CBL``, ``-``, ``045``, ``.``. Au décodage, la bibliothèque recolle les pièces de
    continuation mais laisse un espace autour de la ponctuation — ``CBL - 045 .`` —, ce qui casse la
    promesse du projet : un résumé doit pouvoir recopier l'identifiant d'un équipement **tel quel**,
    sinon la couverture des faits compte un oubli. La règle est écrite ici, minuscule et testée,
    plutôt que laissée au hasard d'un décodeur.

    Args:
        text: Decoded text.

    Returns:
        The text with punctuation glued back to its word.
    """
    return _SPACE_AFTER.sub(r"\1", _SPACE_BEFORE.sub(r"\1", text))


def _normalise(text: str, *, lowercase: bool) -> str:
    """Apply the normalisation the tokenizer applies at encoding time.

    Args:
        text: Raw text.
        lowercase: Whether the text is lower-cased.

    Returns:
        The normalised text (NFKC, optionally lower-cased).
    """
    normalised = unicodedata.normalize("NFKC", str(text))
    return normalised.lower() if lowercase else normalised


def segments_of(text: str, *, lowercase: bool) -> list[str]:
    """Split a text the way the tokenizer's encoder splits it.

    Args:
        text: Raw text.
        lowercase: Whether the text is lower-cased.

    Returns:
        The segments, in order: one per word, one per non-space punctuation mark.
    """
    return [match.group(0) for match in _SEGMENT.finditer(_normalise(text, lowercase=lowercase))]


def _characters(word: str) -> list[str]:
    """Split a word into single-character WordPiece pieces.

    Args:
        word: Normalised, whitespace-free word.

    Returns:
        Its characters, every one but the first prefixed with :data:`CONTINUATION_PREFIX`.
    """
    if not word:
        return []
    return [word[0], *(f"{CONTINUATION_PREFIX}{character}" for character in word[1:])]


def learn_vocabulary(
    segments: Sequence[str], *, vocab_size: int, min_frequency: int
) -> tuple[dict[str, int], dict[str, Any]]:
    """Learn a WordPiece vocabulary inside the process, with a documented tie-break.

    Le vocabulaire est appris **par le projet** et non par ``WordPieceTrainer`` de ``tokenizers`` :
    l'apprenti de la bibliothèque départage les fréquences égales par l'ordre d'une table de
    hachage interne (côté Rust), donc deux exécutions sur le même corpus produisent deux
    vocabulaires différents — même taille, mais une composition de pièces qui diverge. Un modèle
    entraîné là-dessus n'est pas reproductible, et aucun test de reproductibilité ne peut passer.

    La règle appliquée ici est celle de WordPiece, écrite noir sur blanc : la paire de pièces la
    plus fréquente fusionne, les ex æquo sont départagés par ordre lexicographique, et la fusion
    s'arrête quand la meilleure paire passe sous ``min_frequency`` ou quand le vocabulaire atteint
    sa taille cible. Les identifiants sont ensuite attribués de façon stable : les jetons spéciaux
    d'abord (leur identifiant ne doit jamais bouger), puis les pièces par ordre alphabétique — un
    mot garde le même identifiant d'un corpus à l'autre, ce que le hachage ne garantissait
    pas.

    Args:
        segments: Pre-tokenised segments of the training corpus (:func:`segments_of`), c'est-à-dire
            **exactement** la découpe que l'encodeur appliquera : un segment par mot, un par signe.
        vocab_size: Target size of the vocabulary, special tokens included.
        min_frequency: Minimum number of occurrences for a pair to be merged.

    Returns:
        ``(vocabulary, stats)`` : the ``piece -> id`` mapping and the sizes published in the model
        card (segments, unigrams, merges, learned pieces).
    """
    word_counts: Counter[str] = Counter(segments)
    pieces: dict[str, list[str]] = {word: _characters(word) for word in word_counts}
    counts: Counter[tuple[str, str]] = Counter()
    # Index inverse : pour chaque paire, les mots qui la contiennent. Un dictionnaire plutôt qu'un
    # `set` : l'ordre d'itération doit être celui, déterministe, de l'insertion.
    where: dict[tuple[str, str], dict[str, None]] = {}
    for word, weight in word_counts.items():
        row = pieces[word]
        for pair in pairwise(row):
            counts[pair] += weight
            where.setdefault(pair, {})[word] = None

    learned: set[str] = {piece for row in pieces.values() for piece in row}
    merged: list[str] = []
    budget = max(int(vocab_size) - len(SPECIAL_TOKENS), 0)
    while len(learned) < budget:
        candidates = [pair for pair, count in counts.items() if count >= min_frequency]
        if not candidates:
            break
        best = min(candidates, key=lambda pair: (-counts[pair], pair))
        merged.append(best[0] + best[1].removeprefix(CONTINUATION_PREFIX))
        learned.add(merged[-1])
        for word in list(where.get(best, {})):
            weight = word_counts[word]
            row = pieces[word]
            updated: list[str] = []
            index = 0
            while index < len(row):
                if index + 1 < len(row) and (row[index], row[index + 1]) == best:
                    updated.append(merged[-1])
                    index += 2
                else:
                    updated.append(row[index])
                    index += 1
            # Seuls les compteurs de **ce** mot changent : on retire ses anciennes paires et on
            # ajoute les nouvelles, au lieu de recompter tout le corpus à chaque fusion.
            for pair, count in Counter(pairwise(row)).items():
                counts[pair] -= count * weight
                members = where.get(pair)
                if members is not None:
                    members.pop(word, None)
                    if not members:
                        del where[pair]
                if counts[pair] <= 0:
                    del counts[pair]
            for pair, count in Counter(pairwise(updated)).items():
                counts[pair] += count * weight
                where.setdefault(pair, {})[word] = None
            pieces[word] = updated

    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for piece in sorted(learned):
        vocabulary.setdefault(piece, len(vocabulary))
    stats: dict[str, Any] = {
        "version": TOKENIZER_VERSION,
        "segments": len(word_counts),
        "unigrams": sum(
            1 for piece in learned if len(piece.removeprefix(CONTINUATION_PREFIX)) == 1
        ),
        "merges": len(merged),
        "pieces": len(vocabulary) - len(SPECIAL_TOKENS),
        "target_size": int(vocab_size),
        "min_frequency": int(min_frequency),
    }
    return vocabulary, stats


@dataclass(slots=True)
class SharedWordPieceTokenizer:
    """WordPiece tokenizer trained on the corpus, shared by the encoder and the decoder.

    Attributes:
        vocab_size: Target size of the vocabulary (special tokens included).
        min_frequency: Minimum number of occurrences for a **pair** of pieces to be merged. Les
            caractères, eux, entrent toujours au vocabulaire : c'est ce qui permet à un mot rare
            d'être lu en pièces plutôt que remplacé par ``[UNK]``.
        lowercase: Whether the corpus is lower-cased before tokenisation.
        max_input_tokens: Hard bound on the encoder sequence length.
        tokenizer: The underlying ``tokenizers`` object, ``None`` until :meth:`fit` runs.
        meta: Metadata published in the model card (trained sizes, unknown-token rates).
    """

    vocab_size: int = 1200
    min_frequency: int = 2
    lowercase: bool = True
    max_input_tokens: int = 160
    tokenizer: Tokenizer | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ état --------------
    @property
    def is_fitted(self) -> bool:
        """Whether the vocabulary has been learned."""
        return self.tokenizer is not None

    @property
    def vocabulary_size(self) -> int:
        """Number of entries in the learned vocabulary."""
        return int(self.tokenizer.get_vocab_size()) if self.tokenizer is not None else 0

    def special_id(self, token: str) -> int:
        """Return the id of a special token.

        Args:
            token: One of :data:`SPECIAL_TOKENS`.

        Returns:
            The identifier of that token.

        Raises:
            RuntimeError: When the vocabulary has not been learned.
            KeyError: When the token is not a special token of the vocabulary.
        """
        if self.tokenizer is None:
            msg = "The tokenizer is not fitted: call fit(texts) before encoding anything."
            raise RuntimeError(msg)
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            msg = (
                f"'{token}' is not part of the vocabulary (expected one of {list(SPECIAL_TOKENS)})"
            )
            raise KeyError(msg)
        return int(token_id)

    @property
    def pad_id(self) -> int:
        """Identifier of the padding token."""
        return self.special_id("[PAD]")

    @property
    def bos_id(self) -> int:
        """Identifier of the beginning-of-summary token."""
        return self.special_id("[BOS]")

    @property
    def eos_id(self) -> int:
        """Identifier of the end-of-summary token."""
        return self.special_id("[EOS]")

    @property
    def unk_id(self) -> int:
        """Identifier of the unknown token."""
        return self.special_id("[UNK]")

    # ------------------------------------------------------------------ apprentissage -----
    def fit(self, texts: Sequence[str]) -> SharedWordPieceTokenizer:
        """Learn the vocabulary on the training corpus.

        Args:
            texts: Training texts (documents and reference summaries together: the vocabulary is
                shared, so learning it on both is what makes the values of a summary expressible).

        Returns:
            The fitted tokenizer (fluent interface).

        Raises:
            ValueError: When the corpus is empty.
        """
        corpus = [str(text) for text in texts if str(text).strip()]
        if not corpus:
            msg = "Cannot learn a vocabulary from an empty corpus"
            raise ValueError(msg)
        # L'apprentissage se fait sur la **même** découpe que l'encodage : un mot, un signe de
        # ponctuation. Apprendre sur des espaces et encoder avec une découpe plus fine produit des
        # pièces que le modèle ne verra jamais à l'entraînement.
        segments = [
            segment
            for text in corpus
            for segment in segments_of(text, lowercase=bool(self.lowercase))
        ]
        vocabulary, stats = learn_vocabulary(
            segments,
            vocab_size=int(self.vocab_size),
            min_frequency=int(self.min_frequency),
        )
        backend = Tokenizer(
            models.WordPiece(
                vocab=vocabulary,
                unk_token="[UNK]",
                max_input_chars_per_word=40,
            )
        )
        backend.normalizer = normalizers.Sequence(
            [normalizers.NFKC(), *([normalizers.Lowercase()] if self.lowercase else [])]
        )
        # La ponctuation est un segment à part entière — c'est ce qui permet à un identifiant comme
        # ``CBL-045`` de rester composé de pièces connues, et donc d'être recopié par le modèle.
        backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        backend.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[("[BOS]", 2), ("[EOS]", 3)],
        )
        # Le décodeur recolle les pièces de continuation et nettoie les espaces ; la ponctuation
        # isolée est recollée par :func:`rejoin_punctuation`, côté projet.
        backend.decoder = decoders.WordPiece(prefix=CONTINUATION_PREFIX, cleanup=True)
        self.tokenizer = backend
        self.meta = {
            **stats,
            "corpus_texts": len(corpus),
            "unknown_rate": self.unknown_rate(corpus),
        }
        logger.info(
            "Vocabulaire appris : {} pièces sur {} segments ({} fusions, fréquence minimale {})",
            self.vocabulary_size,
            stats["segments"],
            stats["merges"],
            self.min_frequency,
        )
        return self

    # ------------------------------------------------------------------ encodage ----------
    def encode(
        self, text: str, *, max_length: int | None = None, add_special: bool = True
    ) -> list[int]:
        """Encode one text as identifiers.

        Args:
            text: Text to encode.
            max_length: Maximum number of **content** tokens (defaults to :attr:`max_input_tokens`).
            add_special: Whether the ``[BOS]``/``[EOS]`` markers are added.

        Returns:
            The identifiers, truncated to the bound.

        Raises:
            RuntimeError: When the vocabulary has not been learned.
        """
        if self.tokenizer is None:
            msg = "The tokenizer is not fitted: call fit(texts) before encoding anything."
            raise RuntimeError(msg)
        limit = int(max_length or self.max_input_tokens)
        if add_special:
            encoded = self.tokenizer.encode(str(text)).ids
            return encoded[: limit + 2]
        pieces = self.tokenizer.encode(str(text), add_special_tokens=False).ids
        return pieces[:limit]

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        max_length: int | None = None,
        add_special: bool = True,
    ) -> list[list[int]]:
        """Encode several texts.

        Args:
            texts: Texts to encode.
            max_length: Maximum number of content tokens per text.
            add_special: Whether the markers are added.

        Returns:
            One identifier list per text.
        """
        return [self.encode(text, max_length=max_length, add_special=add_special) for text in texts]

    def special_ids(self) -> tuple[int, ...]:
        """Identifiers of the special tokens, in :data:`SPECIAL_TOKENS` order.

        Returns:
            ``(pad, unk, bos, eos)`` as identifiers.

        Raises:
            RuntimeError: When the vocabulary has not been learned.
        """
        return tuple(self.special_id(token) for token in SPECIAL_TOKENS)

    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str:
        """Decode identifiers back into text.

        Les jetons spéciaux sont retirés **par le projet**, à partir de sa propre table
        d'identifiants : la bibliothèque ne connaît comme spéciaux que les jetons qu'on lui a
        déclarés comme tels, et un ``[BOS]`` recraché en boucle par un petit modèle finirait publié
        au milieu d'un résumé. Un jeton spécial n'est pas du texte.

        Args:
            ids: Identifiers to decode.
            skip_special: Whether the special tokens are dropped (they are dropped by default: the
                decoded summary is shown to a human).

        Returns:
            The decoded text.
        """
        if self.tokenizer is None:
            msg = "The tokenizer is not fitted: call fit(texts) before decoding anything."
            raise RuntimeError(msg)
        values = [int(value) for value in ids]
        if skip_special:
            specials = set(self.special_ids())
            values = [value for value in values if value not in specials]
        # Les jetons inconnus sont retirés du texte publié : ``[UNK]`` n'est pas un mot, et un
        # résumé qui affiche ``[UNK]`` au milieu d'une phrase n'est pas lisible par un humain.
        unknown = self.unk_id
        text = self.tokenizer.decode(
            [value for value in values if value != unknown], skip_special_tokens=False
        )
        return rejoin_punctuation(text)

    def pieces(self, text: str) -> list[str]:
        """Return the WordPiece pieces of a text (used by the notebooks).

        Args:
            text: Text to split.

        Returns:
            The pieces, in order.
        """
        if self.tokenizer is None:
            msg = "The tokenizer is not fitted: call fit(texts) before asking for pieces."
            raise RuntimeError(msg)
        return list(self.tokenizer.encode(str(text), add_special_tokens=False).tokens)

    def unknown_rate(self, texts: Sequence[str]) -> float:
        """Share of content tokens that the vocabulary does not know.

        Args:
            texts: Texts to measure.

        Returns:
            The unknown-token rate in ``[0, 1]`` (``0.0`` on an empty corpus).
        """
        if self.tokenizer is None or not texts:
            return 0.0
        unknown = self.unk_id
        total = 0
        missed = 0
        for text in texts:
            ids = self.encode(text, add_special=False)
            total += len(ids)
            missed += sum(1 for value in ids if value == unknown)
        return round(missed / total, 4) if total else 0.0

    def truncation_rate(self, texts: Sequence[str], *, max_length: int | None = None) -> float:
        """Share of texts cut by the encoder bound.

        Args:
            texts: Texts to measure.
            max_length: Bound to test (defaults to :attr:`max_input_tokens`).

        Returns:
            The share of texts whose encoding was truncated.
        """
        if self.tokenizer is None or not texts:
            return 0.0
        limit = int(max_length or self.max_input_tokens)
        truncated = 0
        for text in texts:
            pieces = self.encode(text, add_special=False)
            if len(pieces) >= limit:
                truncated += 1
        return round(truncated / len(texts), 4)

    # ------------------------------------------------------------------ persistance -------
    def to_payload(self) -> dict[str, Any]:
        """Serialise the vocabulary (JSON-friendly, no pickle of a third-party object).

        Returns:
            The vocabulary mapping plus the settings that produced it.

        Raises:
            RuntimeError: When the vocabulary has not been learned.
        """
        if self.tokenizer is None:
            msg = "Cannot serialise an unfitted tokenizer"
            raise RuntimeError(msg)
        return {
            "version": TOKENIZER_VERSION,
            "vocab": dict(self.tokenizer.get_vocab()),
            "settings": {
                "vocab_size": int(self.vocab_size),
                "min_frequency": int(self.min_frequency),
                "lowercase": bool(self.lowercase),
                "max_input_tokens": int(self.max_input_tokens),
            },
            "meta": dict(self.meta),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SharedWordPieceTokenizer:
        """Rebuild a tokenizer from a serialised payload.

        Args:
            payload: Mapping written by :meth:`to_payload`.

        Returns:
            The restored tokenizer, ready to encode.
        """
        settings = dict(payload.get("settings") or {})
        instance = cls(
            vocab_size=int(settings.get("vocab_size", 1200)),
            min_frequency=int(settings.get("min_frequency", 2)),
            lowercase=bool(settings.get("lowercase", True)),
            max_input_tokens=int(settings.get("max_input_tokens", 160)),
        )
        backend = Tokenizer(
            models.WordPiece(
                vocab={
                    str(key): int(value) for key, value in dict(payload.get("vocab") or {}).items()
                },
                unk_token="[UNK]",
                max_input_chars_per_word=40,
            )
        )
        backend.normalizer = normalizers.Sequence(
            [normalizers.NFKC(), *([normalizers.Lowercase()] if instance.lowercase else [])]
        )
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            pair="[BOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[("[BOS]", 2), ("[EOS]", 3)],
        )
        instance.tokenizer = backend
        instance.meta = dict(payload.get("meta") or {})
        return instance


__all__ = ["SPECIAL_TOKENS", "TOKENIZER_VERSION", "SharedWordPieceTokenizer"]
