"""Encodeur de type BERT, appris **sur le corpus du projet** : ni téléchargement, ni clé d'API.

La stack porte le chaînon qui manque à la famille : ce qu'un modèle **contextuel** apporte là où le
TF-IDF s'arrête. Trois algorithmes, un seul contrat, deux architectures :

* ``embedding_bag`` — un sac d'embeddings : la moyenne des vecteurs de termes appris, suivie d'un
  classifieur linéaire. Ni ordre, ni contexte, ni fenêtre d'attention. C'est le **plancher
  neuronal** : tout ce qu'un modèle neuronal obtient sans lire la phrase ;
* ``bert_tiny`` / ``bert_small`` — un encodeur de type BERT (attention multi-têtes, connexions
  résiduelles, normalisation, masquage de jetons), de deux puis quatre couches. Le vocabulaire,
  comme les poids, sont appris sur le **train** du corpus : l'encodeur démarre à froid, ce qui est
  le seul réglage possible hors ligne, et c'est une information publiée, pas un détail caché.

Le **vocabulaire** est construit par le projet, pas par ``WordPieceTrainer`` : cet entraîneur
casse les égalités de fréquence dans l'ordre d'une table de hachage, donc deux entraînements sur le
même corpus produisent deux vocabulaires — et deux jeux d'identifiants — différents. La règle
retenue ici tient en trois lignes et se relit : un mot vu au moins ``min_frequency`` fois devient
une pièce (ordre ``(-fréquence, mot)``), l'alphabet du corpus est ajouté sous ses deux formes (``x``
et ``##x``), ``vocab_size`` borne le tout. Un mot rare se découpe donc en lettres connues, et
``[UNK]`` n'apparaît que pour un caractère que le train n'a jamais vu — un fait publié, pas une
surprise.

Le pipeline d'entraînement est celui de la littérature, en deux temps :

1. **pré-entraînement masqué** (*masked language modelling*) sur les textes du train, sans utiliser
   aucun libellé : 15 % des jetons sont masqués (80 % par ``[MASK]``, 10 % par un jeton tiré au
   hasard, 10 % laissés en place — remplacer par ``[MASK]`` à chaque fois créerait un écart entre
   l'entraînement et l'inférence que le modèle ne peut pas corriger) ;
2. **affinage supervisé** sur les libellés, avec ``AdamW``, warmup linéaire, décroissance linéaire,
   ``class_weight='balanced'`` (la classe ``autre`` est minoritaire) et troncature à ``max_length``.

Ce que la stack **ne** fait pas sous-entendre, et qu'elle publie donc dans la fiche de modèle : le
taux de jetons inconnus du vocabulaire, le taux de tickets tronqués, le nombre de paramètres et le
fait que l'encodeur n'a jamais vu d'autre français que celui du corpus. Un ``bert_small`` appris sur
719 tickets n'est pas un modèle de langue : c'est un modèle contextuel **mesuré**, et l'écart avec
le TF-IDF est exactement ce que le projet compare.

Côté contrat, deux points méritent d'être lus avant le code :

* :meth:`TransformerTextClassifier.explain` — un transformeur n'a pas de « poids par terme ». Pour
  garder la même API que la stack lexicale, l'explication est une **occlusion** : on retire un terme
  du texte et on mesure la baisse, sur la classe prédite, de sa probabilité. C'est plus lent qu'une
  lecture de coefficients et c'est assumé : le terme affiché est toujours dans le texte, et
  l'influence affichée est celle qui a réellement décidé cette prédiction ;
* l'artefact persiste le **vocabulaire** (JSON du tokenizer) et les **poids** (tableaux NumPy), avec
  les hyper-paramètres de l'architecture : un artefact rechargé prédit et explique exactement comme
  le modèle ajusté, y compris si la configuration a changé entre-temps.
"""

from __future__ import annotations

import random
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from torch import nn
from transformers import BertConfig, BertForMaskedLM, BertForSequenceClassification
from transformers import PreTrainedTokenizerFast

from src.models.contract import BaseTextClassifier
from src.training.metrics import classification_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Suffix appended to the algorithm identifier in the model card.
FRAMEWORK = "transformers"

#: Artefact written by :meth:`~src.models.contract.BaseTextClassifier.save` when it receives a
#: directory. Le fichier est un artefact du projet (joblib), pas un checkpoint Hugging Face : il ne
#: contient que des tableaux NumPy et le JSON du tokenizer, donc aucun code n'est exécuté au
#: chargement en dehors du `load_pickle` du projet.
DEFAULT_MODEL_FILE = "model.pt"

#: Special tokens of the local WordPiece vocabulary. ``[MASK]`` n'est utilisé que par le
#: pré-entraînement masqué ; il reste dans le vocabulaire pour que l'affinage et le rechargement
#: partagent un unique artefact.
SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

#: Valeur d'étiquette ignorée par la perte de pré-entraînement masqué.
IGNORE_INDEX = -100

#: Nombre maximal de termes testés par occlusion dans :meth:`TransformerTextClassifier.explain`.
MAX_OCCLUSION_TERMS = 24

#: Clés de métriques renvoyées par :meth:`TransformerTextClassifier._fit`.
_TRAIN_METRIC_KEYS = frozenset({"accuracy", "macro_f1", "balanced_accuracy"})


def _normalise(text: str, *, lowercase: bool, strip_accents: bool) -> str:
    """Normalise a text exactly like the tokenizer's normalizer chain.

    La chaîne Rust fait ``Lowercase`` puis ``NFD`` + ``StripAccents`` : cette fonction applique les
    mêmes règles en Python, pour que le vocabulaire soit compté sur les mots que le tokenizer verra
    réellement à l'encodage.

    Args:
        text: Raw text.
        lowercase: Lowercase the text.
        strip_accents: Remove diacritics.

    Returns:
        The normalised text.
    """
    normalised = str(text).lower() if lowercase else str(text)
    if strip_accents:
        normalised = "".join(
            character
            for character in unicodedata.normalize("NFD", normalised)
            if not unicodedata.combining(character)
        )
    return normalised


def _normalizer(*, lowercase: bool, strip_accents: bool) -> normalizers.Normalizer:
    """Build the tokenizer normaliser (case and accents, in this order).

    Args:
        lowercase: Lowercase the text before tokenisation.
        strip_accents: Remove diacritics (``é`` → ``e``).

    Returns:
        The normaliser chain applied to every text.
    """
    steps: list[normalizers.Normalizer] = []
    if lowercase:
        steps.append(normalizers.Lowercase())
    if strip_accents:
        steps.extend([normalizers.NFD(), normalizers.StripAccents()])
    return normalizers.Sequence(steps) if steps else normalizers.Sequence([])


@dataclass(slots=True)
class LocalWordPieceTokenizer:
    """WordPiece tokenizer built on the project corpus, persisted with the artefact.

    Le vocabulaire est appris sur le **train** uniquement, avec une règle déterministe (voir
    :meth:`fit`) : les mots fréquents sont des pièces entières, un mot rare se découpe en lettres,
    et seul un caractère absent du train devient ``[UNK]`` — taux publié dans la fiche de modèle.

    Attributes:
        vocab_size: Upper bound of the learned vocabulary.
        min_frequency: Minimum corpus frequency for a piece to enter the vocabulary.
        max_length: Number of tokens kept per document (longer documents are truncated).
        lowercase: Lowercase the text before tokenisation.
        strip_accents: Remove diacritics before tokenisation.
    """

    vocab_size: int = 4000
    min_frequency: int = 2
    max_length: int = 64
    lowercase: bool = True
    strip_accents: bool = True
    _fast: PreTrainedTokenizerFast | None = field(default=None, repr=False)
    _backend_json: str = field(default="", repr=False)

    @property
    def is_fitted(self) -> bool:
        """Whether the vocabulary has been learned."""
        return self._fast is not None

    @property
    def vocabulary_size(self) -> int:
        """Size of the learned vocabulary (``0`` before the fit)."""
        return int(self._fast.vocab_size) if self._fast is not None else 0

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        """The tokenizer itself (used by the notebooks to display tokens).

        Returns:
            The fast tokenizer backed by the local vocabulary.

        Raises:
            RuntimeError: When the vocabulary has not been learned yet.
        """
        if self._fast is None:
            msg = "The tokenizer has no vocabulary yet: call fit(documents) or load an artefact"
            raise RuntimeError(msg)
        return self._fast

    def fit(self, texts: Sequence[str]) -> LocalWordPieceTokenizer:
        """Learn the vocabulary on the training texts.

        Le vocabulaire est construit **par le projet**, pas par ``WordPieceTrainer`` : l'entraîneur
        de la bibliothèque casse les égalités de fréquence dans l'ordre d'une table de hachage Rust,
        donc deux entraînements sur le même corpus produisent deux vocabulaires (donc deux jeux
        d'identifiants) différents. Ici, la règle est explicite et rejouable terme pour terme :

        1. un **mot** vu au moins ``min_frequency`` fois devient une pièce, dans l'ordre
           ``(-fréquence, mot)`` — l'ordre des identifiants dépend du corpus, pas d'un tirage ;
        2. l'**alphabet** du corpus est ajouté (chaque caractère comme pièce ``x`` et comme
           continuation ``##x``) : un mot rare se découpe alors en lettres, et ``[UNK]`` n'apparaît
           que pour un caractère que le train n'a jamais vu ;
        3. ``vocab_size`` borne le tout ; l'alphabet, minuscule, passe toujours.

        Args:
            texts: Training texts (the vocabulary never sees the evaluation splits).

        Returns:
            ``self``.
        """
        vocabulary = self._build_vocabulary(texts)
        backend = Tokenizer(models.WordPiece(vocabulary, unk_token=UNK_TOKEN))
        backend.normalizer = _normalizer(lowercase=self.lowercase, strip_accents=self.strip_accents)
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        self._backend_json = backend.to_str()
        self._fast = self._wrap(Tokenizer.from_str(self._backend_json))
        logger.debug(
            "WordPiece vocabulary built | pieces={} | min_frequency={} | max_length={}",
            self._fast.vocab_size,
            self.min_frequency,
            self.max_length,
        )
        return self

    def _build_vocabulary(self, texts: Sequence[str]) -> dict[str, int]:
        """Build the deterministic WordPiece vocabulary (see :meth:`fit`).

        Args:
            texts: Training texts.

        Returns:
            The ``piece -> identifier`` mapping, special tokens first.
        """
        counts: Counter[str] = Counter()
        alphabet: set[str] = set()
        for text in texts:
            for word in _normalise(
                str(text), lowercase=self.lowercase, strip_accents=self.strip_accents
            ).split():
                counts[word] += 1
                alphabet.update(word)
        vocabulary: dict[str, int] = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        for character in sorted(alphabet):
            for piece in (character, f"##{character}"):
                if piece not in vocabulary:
                    vocabulary[piece] = len(vocabulary)
        capacity = max(int(self.vocab_size) - len(vocabulary), 0)
        frequent = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        for word, count in frequent[:capacity]:
            if count >= int(self.min_frequency) and word not in vocabulary:
                vocabulary[word] = len(vocabulary)
        return vocabulary

    def encode(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        """Encode texts into padded ``input_ids`` and ``attention_mask`` arrays.

        Args:
            texts: Texts to encode.

        Returns:
            Two ``(n_texts, max_length)`` integer arrays.

        Raises:
            RuntimeError: When the vocabulary has not been learned yet.
        """
        tokenizer = self.tokenizer
        encoded = tokenizer(
            [str(text) for text in texts],
            padding="max_length",
            truncation=True,
            max_length=int(self.max_length),
            return_tensors="np",
        )
        return {
            "input_ids": np.asarray(encoded["input_ids"], dtype="int64"),
            "attention_mask": np.asarray(encoded["attention_mask"], dtype="int64"),
        }

    def pieces(self, text: str) -> list[str]:
        """Return the WordPiece pieces of one text, special tokens excluded.

        Args:
            text: Text to tokenize.

        Returns:
            The pieces, truncated at ``max_length``.
        """
        return list(self.tokenizer.tokenize(str(text))[: int(self.max_length)])

    def unknown_token_rate(self, texts: Sequence[str]) -> float:
        """Share of pieces the vocabulary does not know (``[UNK]``).

        Args:
            texts: Texts to measure.

        Returns:
            The rate, between ``0`` and ``1`` (``0`` when no piece was produced).
        """
        unknown = 0
        total = 0
        for text in texts:
            pieces = self.pieces(text)
            unknown += sum(1 for piece in pieces if piece == UNK_TOKEN)
            total += len(pieces)
        return float(unknown) / float(total) if total else 0.0

    def truncated_rate(self, texts: Sequence[str]) -> float:
        """Share of documents longer than the model reads.

        Args:
            texts: Texts to measure.

        Returns:
            The rate, between ``0`` and ``1``.
        """
        if not texts:
            return 0.0
        truncated = sum(1 for text in texts if len(self.pieces(text)) >= int(self.max_length))
        return float(truncated) / float(len(texts))

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON vocabulary plus the tokenisation settings.

        Returns:
            A mapping restored by :meth:`from_payload`.
        """
        return {
            "backend_json": self._backend_json,
            "vocab_size": int(self.vocab_size),
            "min_frequency": int(self.min_frequency),
            "max_length": int(self.max_length),
            "lowercase": bool(self.lowercase),
            "strip_accents": bool(self.strip_accents),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LocalWordPieceTokenizer:
        """Rebuild a tokenizer from an artefact payload.

        Args:
            payload: Mapping produced by :meth:`to_payload`.

        Returns:
            The restored tokenizer, ready to encode.
        """
        tokenizer = cls(
            vocab_size=int(payload.get("vocab_size", 4000)),
            min_frequency=int(payload.get("min_frequency", 2)),
            max_length=int(payload.get("max_length", 64)),
            lowercase=bool(payload.get("lowercase", True)),
            strip_accents=bool(payload.get("strip_accents", True)),
        )
        backend_json = str(payload.get("backend_json") or "")
        if backend_json:
            tokenizer._backend_json = backend_json
            tokenizer._fast = tokenizer._wrap(Tokenizer.from_str(backend_json))
        return tokenizer

    def _wrap(self, backend: Tokenizer) -> PreTrainedTokenizerFast:
        """Wrap a low level tokenizer into the transformers interface."""
        return PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token=UNK_TOKEN,
            pad_token=PAD_TOKEN,
            cls_token=CLS_TOKEN,
            sep_token=SEP_TOKEN,
            mask_token=MASK_TOKEN,
        )


class _BagOfEmbeddings(nn.Module):
    """Mean of the token embeddings, followed by a linear head.

    Ni ordre ni contexte : chaque terme contribue indépendamment. C'est le plancher neuronal de la
    stack, entraîné par la même boucle que l'encodeur — ce qui permet de lire ce que l'attention
    apporte *à boucle d'entraînement constante*.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        n_labels: int,
        embedding_dim: int,
        dropout: float,
        padding_index: int,
        unknown_index: int,
    ) -> None:
        """Build the bag of embeddings.

        Args:
            vocab_size: Size of the vocabulary (rows of the embedding table).
            n_labels: Number of classes.
            embedding_dim: Dimension of the term embeddings.
            dropout: Dropout rate applied before the classifier.
            padding_index: Identifier of the padding token (excluded from the mean).
            unknown_index: Identifier of ``[UNK]``, substituted on an all-padding row.
        """
        super().__init__()
        self.embedding = nn.EmbeddingBag(
            vocab_size, embedding_dim, mode="mean", padding_idx=padding_index
        )
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(embedding_dim, n_labels)
        self.unknown_index = int(unknown_index)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Score a batch of encoded documents.

        Args:
            input_ids: ``(batch, max_length)`` token identifiers.
            attention_mask: Unused (the padding index already excludes the padding).

        Returns:
            A ``(batch, n_labels)`` matrix of logits.
        """
        del attention_mask
        # Une ligne entièrement padding (texte vide) n'a aucun jeton à moyenner : on force le jeton
        # inconnu, sinon la moyenne diviserait par zéro et produirait un NaN.
        padding_index = int(self.embedding.padding_idx or 0)
        empty = input_ids.eq(padding_index).all(dim=1)
        if bool(empty.any()):
            input_ids = input_ids.clone()
            input_ids[empty, 0] = self.unknown_index
        return self.head(self.dropout(self.embedding(input_ids)))


def _logits(output: Any) -> torch.Tensor:
    """Return the logits of a network output (models return an object or a tensor)."""
    logits = getattr(output, "logits", output)
    if not isinstance(logits, torch.Tensor):  # pragma: no cover - defensive guard
        msg = f"Network produced {type(logits)!r} instead of a tensor of logits"
        raise TypeError(msg)
    return logits


def _masked_language_model_inputs(
    input_ids: torch.Tensor,
    *,
    mask_index: int,
    special_ids: set[int],
    vocab_size: int,
    probability: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask a random share of the tokens for the masked language modelling objective.

    Args:
        input_ids: ``(batch, max_length)`` token identifiers.
        mask_index: Identifier of ``[MASK]``.
        special_ids: Identifiers never masked (padding, ``[CLS]``, ``[SEP]``, ``[UNK]``).
        vocab_size: Number of rows of the embedding table (random replacements stay inside it).
        probability: Share of the maskable tokens to hide.
        rng: Seeded generator, so that two runs mask exactly the same tokens.

    Returns:
        The corrupted identifiers and the labels (``IGNORE_INDEX`` outside the masked positions).
    """
    maskable = torch.ones_like(input_ids, dtype=torch.bool)
    for identifier in sorted(special_ids):
        maskable &= input_ids.ne(int(identifier))
    drawn = torch.tensor(rng.random(size=tuple(input_ids.shape)) < float(probability))
    selected = maskable & drawn
    if not bool(selected.any()) and bool(maskable.any()):
        # Toujours masquer au moins un jeton : sur un texte très court, un échantillon sans jeton
        # masqué produirait une perte nulle et un pas d'optimisation sans information.
        first = torch.nonzero(maskable, as_tuple=False)[0]
        selected[first[0], first[1]] = True
    labels = torch.where(selected, input_ids, torch.full_like(input_ids, IGNORE_INDEX))
    # 80 % [MASK], 10 % jeton aléatoire, 10 % inchangé : c'est la recette de BERT, et elle évite
    # que le modèle n'apprenne à ne prédire que sur le jeton [MASK].
    choice = torch.tensor(rng.random(size=tuple(input_ids.shape)))
    corrupted = input_ids.clone()
    corrupted[selected & (choice < 0.8)] = int(mask_index)
    random_positions = selected & (choice >= 0.8) & (choice < 0.9)
    if bool(random_positions.any()):
        drawn_ids = torch.tensor(
            rng.integers(0, max(int(vocab_size), 1), size=tuple(input_ids.shape)), dtype=torch.long
        )
        corrupted[random_positions] = drawn_ids[random_positions]
    return corrupted, labels


class TransformerTextClassifier(BaseTextClassifier):
    """Encodeur contextuel entraîné sur le corpus, derrière le contrat texte de la famille.

    Attributes:
        framework: Stack identifier archived in the model card.
    """

    framework = FRAMEWORK
    default_model_file = DEFAULT_MODEL_FILE

    def __init__(self, **kwargs: Any) -> None:
        """Build the classifier (see :class:`~src.models.contract.BaseTextClassifier`)."""
        super().__init__(**kwargs)
        tokenizer_config = dict(self.params.get("tokenizer") or {})
        network_config = dict(self.params.get("network") or {})
        pretraining_config = dict(self.params.get("pretraining") or {})
        training_config = dict(self.params.get("training") or {})

        self._tokenizer = LocalWordPieceTokenizer(
            vocab_size=int(tokenizer_config.get("vocab_size", 4000)),
            min_frequency=int(tokenizer_config.get("min_frequency", 2)),
            max_length=int(tokenizer_config.get("max_length", 64)),
            lowercase=bool(tokenizer_config.get("lowercase", True)),
            strip_accents=bool(tokenizer_config.get("strip_accents", True)),
        )
        self._network_config = network_config
        self._pretraining_config = {
            "epochs": int(pretraining_config.get("epochs", 3)),
            "mask_probability": float(pretraining_config.get("mask_probability", 0.15)),
        }
        self._training_config = training_config
        self._network: nn.Module | None = None
        self._classes: list[str] = []
        self._device = self._resolve_device(str(training_config.get("device", "auto")))
        self._facts: dict[str, float] = {}

    # ------------------------------------------------------------------ contrat ----------
    @property
    def n_features(self) -> int:
        """Size of the learned vocabulary (the representation the model reads)."""
        return int(self._tokenizer.vocabulary_size)

    @property
    def labels(self) -> list[str]:
        """Known labels, in the order of the probability columns."""
        return list(self._classes or self._labels)

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        """The local tokenizer, learned on the training split.

        Returns:
            The fast tokenizer (notebooks use it to display pieces).

        Raises:
            RuntimeError: When the model has not been fitted yet.
        """
        return self._tokenizer.tokenizer

    @property
    def network(self) -> nn.Module:
        """The trained network.

        Returns:
            The PyTorch module.

        Raises:
            RuntimeError: When the model has not been fitted yet.
        """
        if self._network is None:
            msg = (
                "TransformerTextClassifier has no network: call fit(documents) or load an artefact"
            )
            raise RuntimeError(msg)
        return self._network

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters of the network (``0`` before the fit)."""
        if self._network is None:
            return 0
        return int(sum(parameter.numel() for parameter in self._network.parameters()))

    @property
    def max_length(self) -> int:
        """Number of tokens the model reads per document."""
        return int(self._tokenizer.max_length)

    def _fit(
        self,
        documents: pd.DataFrame,
        *,
        callbacks: Sequence[Any] | None = None,
        context: Any = None,
    ) -> dict[str, float]:
        """Learn the vocabulary, pre-train the encoder, then fine-tune it on the labels.

        Args:
            documents: Training documents (labelled frame).
            callbacks: Project callbacks, fired once per fine-tuning epoch.
            context: Mutable callback context (an early stop request is honoured).

        Returns:
            Training metrics (accuracy, macro F1 and balanced accuracy on the train split).

        Raises:
            ValueError: When the label column is missing from the training frame.
        """
        target = str(self.target_name or "label")
        if target not in documents.columns:
            msg = f"Column '{target}' missing from the training frame: {sorted(documents.columns)}"
            raise ValueError(msg)
        texts = [str(value) for value in documents[self.text_column]]
        labels = [str(value) for value in documents[target]]

        self._seed_everything()
        self._tokenizer.fit(texts)
        self._classes = sorted(set(labels))
        self._labels = list(self._classes)
        index_of = {label: position for position, label in enumerate(self._classes)}
        targets = np.asarray([index_of[label] for label in labels], dtype="int64")
        encoded = self._tokenizer.encode(texts)

        self._network = self._build_network()
        backbone = self._pretrain(texts, encoded)
        if backbone is not None:
            self._load_backbone(backbone)

        self._network.to(self._device)
        self._network.train()
        batch_size = self._batch_size()
        optimizer = self._optimizer()
        steps_per_epoch = int(np.ceil(len(texts) / float(batch_size)))
        schedule = self._schedule(optimizer, steps_per_epoch=steps_per_epoch)
        loss_function = nn.CrossEntropyLoss(weight=self._class_weights(targets))
        rng = np.random.default_rng(int(self.random_state))
        metrics: dict[str, float] = {}

        for epoch in range(self._epochs()):
            learning_rate = float(optimizer.param_groups[0]["lr"])
            # ``_train_metrics`` passe le réseau en mode évaluation pour scorer le train : le mode
            # entraînement est donc rétabli à chaque époque, sans quoi le dropout s'arrêterait après
            # la première (le modèle apprendrait, mais plus dans les conditions qu'il publie).
            self.network.train()
            order = rng.permutation(len(texts))
            total = 0.0
            seen = 0
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                batch = self._batch(encoded, indices)
                optimizer.zero_grad(set_to_none=True)
                logits = _logits(self.network(**batch))
                loss = loss_function(logits, torch.as_tensor(targets[indices], dtype=torch.long))
                loss.backward()
                clip = float(self._training_config.get("clip_norm", 1.0))
                if clip > 0.0:
                    nn.utils.clip_grad_norm_(self.network.parameters(), clip)
                optimizer.step()
                schedule.step()
                total += float(loss.detach()) * len(indices)
                seen += len(indices)

            metrics = self._train_metrics(texts, labels)
            logs: dict[str, float] = {
                "loss": float(total) / float(seen or 1),
                "lr": learning_rate,
                "epoch": float(epoch),
                **{f"train_{key}": value for key, value in metrics.items()},
            }
            logger.info(
                "Époque {}/{} | loss={:.4f} | train_macro_f1={:.4f} | lr={:.5f}",
                epoch + 1,
                self._epochs(),
                logs["loss"],
                logs["train_macro_f1"],
                logs["lr"],
            )
            if context is not None and self._emit(context, callbacks, logs, epoch=epoch):
                logger.info("Arrêt anticipé demandé à l'époque {}", epoch)
                break

        self.network.eval()
        self._facts = {
            "unk_token_rate": float(self._tokenizer.unknown_token_rate(texts)),
            "truncated_rate": float(self._tokenizer.truncated_rate(texts)),
        }
        return {
            f"train_{key}": float(value)
            for key, value in metrics.items()
            if key in _TRAIN_METRIC_KEYS
        }

    def _predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Score raw texts.

        Args:
            texts: Texts to classify.

        Returns:
            A ``(n_texts, n_labels)`` probability matrix, in the order of :attr:`labels`.
        """
        if self._network is None:
            msg = (
                "TransformerTextClassifier has no network: call fit(documents) or load an artefact"
            )
            raise RuntimeError(msg)
        self._network.eval()
        batch_size = self._batch_size()
        encoded = self._tokenizer.encode(list(texts))
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                indices = np.arange(start, min(start + batch_size, len(texts)))
                batch = self._batch(encoded, indices)
                logits = _logits(self._network(**batch)).float()
                chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        if not chunks:
            return np.zeros((0, len(self.labels)), dtype="float64")
        return np.vstack(chunks).astype("float64")

    def _explain(self, texts: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        """Explain each prediction by occlusion, most influential term first.

        Le terme est retiré du texte, le document amputé est rescanné, et l'influence est la baisse
        de probabilité de la classe prédite. Un terme absent du texte n'est jamais cité, et deux
        textes identiques produisent deux explications identiques.

        Args:
            texts: Texts to explain.
            k: Number of terms per text.

        Returns:
            One list of ``(term, influence)`` pairs per text, sorted by decreasing influence.
        """
        probabilities = self._predict_proba(texts)
        candidates = [self._candidate_terms(text) for text in texts]
        variants: list[str] = []
        spans: list[tuple[int, int]] = []
        for text, terms in zip(texts, candidates, strict=True):
            start = len(variants)
            variants.extend(self._remove_term(str(text), term) for term in terms)
            spans.append((start, len(variants)))
        variant_probabilities = self._predict_proba(variants) if variants else None

        explanations: list[list[tuple[str, float]]] = []
        for position, terms in enumerate(candidates):
            predicted = int(np.argmax(probabilities[position])) if probabilities.size else 0
            baseline = float(probabilities[position, predicted]) if probabilities.size else 0.0
            start, stop = spans[position]
            scored = [
                (
                    term,
                    baseline
                    - (
                        float(variant_probabilities[index, predicted])
                        if variant_probabilities is not None
                        else 0.0
                    ),
                )
                for index, term in zip(range(start, stop), terms, strict=True)
            ]
            # Tri par influence absolue décroissante (le signe dit si le terme pousse *vers* ou
            # *contre* la classe prédite) ; ``sorted`` est stable, donc les ex æquo gardent l'ordre
            # d'apparition dans le texte.
            scored = sorted(scored, key=lambda item: -abs(item[1]))[: max(int(k), 0)]
            explanations.append([(term, weight) for term, weight in scored if weight != 0.0])
        return explanations

    def _payload(self) -> dict[str, Any]:
        """Return the state persisted with the artefact (vocabulary, weights, architecture)."""
        return {
            "tokenizer": self._tokenizer.to_payload(),
            "network": dict(self._network_config),
            "algorithm_family": self._algorithm_family(),
            "classes": list(self._classes),
            "weights": {
                name: value.detach().cpu().numpy()
                for name, value in self.network.state_dict().items()
            },
            "facts": dict(self._facts),
        }

    def _restore(self, payload: Mapping[str, Any]) -> None:
        """Restore the state archived in an artefact."""
        self._tokenizer = LocalWordPieceTokenizer.from_payload(dict(payload.get("tokenizer") or {}))
        self._network_config = dict(payload.get("network") or self._network_config)
        self._classes = [str(label) for label in payload.get("classes", [])]
        self._labels = list(self._classes)
        self._facts = {
            str(key): float(value) for key, value in dict(payload.get("facts") or {}).items()
        }
        weights = dict(payload.get("weights") or {})
        if not weights:
            msg = "Artefact without weights: the file was not written by this stack"
            raise ValueError(msg)
        network = self._build_network()
        network.load_state_dict(
            {name: torch.as_tensor(value) for name, value in weights.items()}, strict=True
        )
        network.to(self._device)
        network.eval()
        self._network = network

    def _effective_params(self) -> dict[str, Any]:
        """Return the hyper-parameters actually applied."""
        return {
            **dict(self.params),
            "vocabulary_size": self.n_features,
            "max_length": self.max_length,
            "n_parameters": self.n_parameters,
            "classes": list(self._classes),
        }

    def _extra_metadata(self) -> dict[str, Any]:
        """Return the facts archived next to the fit result."""
        return {
            "n_classes": float(len(self._classes)),
            "representation_size": float(self.n_features),
            "n_parameters": float(self.n_parameters),
            # Le contrat est partagé avec les estimateurs à passage unique : ``FitResult.epochs``
            # vaut donc 1 par défaut. Le budget réellement appliqué est publié ici, sinon la fiche
            # de modèle annoncerait une époque à un modèle qui en a vu quarante.
            "epochs_trained": float(self._epochs()),
            "pretraining_epochs": float(self._pretraining_config["epochs"]),
            # Première explication d'un rappel faible sur une tournure inédite : le vocabulaire
            # appris sur le train ne connaît pas le terme (il devient [UNK]).
            "unk_token_rate": float(self._facts.get("unk_token_rate", 0.0)),
            "truncated_rate": float(self._facts.get("truncated_rate", 0.0)),
            # Un encodeur contextuel n'a pas de poids par terme : l'explication est une occlusion.
            "is_transparent": 0.0,
        }

    # ------------------------------------------------------------------ réseau ------------
    def _algorithm_family(self) -> str:
        """Return the architecture family of the configured algorithm."""
        return (
            "bag"
            if str(self.algorithm).lower() in {"embedding_bag", "bag_of_embeddings"}
            else "bert"
        )

    def _build_network(
        self,
        *,
        network_config: Mapping[str, Any] | None = None,
        vocab_size: int | None = None,
        n_labels: int | None = None,
    ) -> nn.Module:
        """Instantiate the network declared by the algorithm.

        Args:
            network_config: Architecture override (defaults to the configured one).
            vocab_size: Vocabulary size (defaults to the fitted one).
            n_labels: Number of classes (defaults to the fitted ones).

        Returns:
            An unfitted PyTorch module.
        """
        config = dict(network_config if network_config is not None else self._network_config)
        vocabulary = int(vocab_size if vocab_size is not None else self.n_features)
        labels = int(n_labels if n_labels is not None else len(self._classes))
        seed = int(self.random_state)
        dropout = float(config.get("dropout", 0.1))
        torch.manual_seed(seed)

        if self._algorithm_family() == "bag":
            return _BagOfEmbeddings(
                vocab_size=vocabulary,
                n_labels=labels,
                embedding_dim=int(config.get("embedding_dim", 64)),
                dropout=dropout,
                padding_index=int(self._padding_index()),
                unknown_index=int(self._unknown_index()),
            )
        return BertForSequenceClassification(
            self._bert_config(vocabulary=vocabulary, n_labels=labels, dropout=dropout)
        )

    def _bert_config(self, *, vocabulary: int, n_labels: int, dropout: float) -> BertConfig:
        """Build the BERT configuration of the encoder (small, offline, from scratch)."""
        config = self._network_config
        bert = BertConfig(
            vocab_size=int(vocabulary),
            hidden_size=int(config.get("hidden_size", 128)),
            num_hidden_layers=int(config.get("num_hidden_layers", 2)),
            num_attention_heads=int(config.get("num_attention_heads", 4)),
            intermediate_size=int(config.get("intermediate_size", 256)),
            max_position_embeddings=int(self.max_length) + 2,
            hidden_dropout_prob=float(dropout),
            attention_probs_dropout_prob=float(dropout),
            pad_token_id=int(self._padding_index()),
        )
        # ``num_labels`` est un champ de la configuration, pas un argument du constructeur typé.
        bert.num_labels = int(n_labels)
        return bert

    def _unknown_index(self) -> int:
        """Return the identifier of the unknown token (``1`` before the vocabulary is learned)."""
        tokenizer = self._tokenizer.tokenizer if self._tokenizer.is_fitted else None
        return int(tokenizer.unk_token_id or 1) if tokenizer is not None else 1

    def _padding_index(self) -> int:
        """Return the identifier of the padding token (``0`` before the vocabulary is learned)."""
        tokenizer = self._tokenizer.tokenizer if self._tokenizer.is_fitted else None
        return int(tokenizer.pad_token_id or 0) if tokenizer is not None else 0

    def _pretrain(
        self, texts: Sequence[str], encoded: Mapping[str, np.ndarray]
    ) -> dict[str, Any] | None:
        """Pre-train the encoder with the masked language modelling objective.

        Le pré-entraînement n'utilise **aucun libellé** : il apprend la cooccurrence des termes du
        corpus d'entraînement, ce qui évite de partir de poids purement aléatoires. Les époques de
        pré-entraînement sont celles déclarées dans les paramètres du modèle ; elles ne déclenchent
        pas les callbacks, qui décrivent l'affine supervisé.

        Args:
            texts: Training texts (used for logging only).
            encoded: Their encoded form.

        Returns:
            The encoder weights, or ``None`` when the architecture has no pre-training step or the
            parameter disables it.
        """
        epochs = int(self._pretraining_config["epochs"])
        if self._algorithm_family() == "bag" or epochs <= 0:
            return None
        torch.manual_seed(int(self.random_state))
        model = BertForMaskedLM(
            self._bert_config(
                vocabulary=self.n_features,
                n_labels=2,
                dropout=float(self._network_config.get("dropout", 0.1)),
            )
        )
        module: nn.Module = model
        module.to(self._device)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self._training_config.get("pretrain_learning_rate", 0.005)),
            weight_decay=float(self._training_config.get("pretrain_weight_decay", 0.01)),
        )
        rng = np.random.default_rng(int(self.random_state))
        special_ids = self._special_ids()
        batch_size = self._batch_size()
        for epoch in range(epochs):
            order = rng.permutation(len(texts))
            total = 0.0
            seen = 0
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                batch = self._batch(encoded, indices)
                corrupted, labels = _masked_language_model_inputs(
                    batch["input_ids"],
                    mask_index=int(self.tokenizer.mask_token_id or 0),
                    special_ids=special_ids,
                    vocab_size=self.n_features,
                    probability=float(self._pretraining_config["mask_probability"]),
                    rng=rng,
                )
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    input_ids=corrupted, attention_mask=batch["attention_mask"], labels=labels
                )
                output.loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(output.loss.detach()) * len(indices)
                seen += len(indices)
            logger.info(
                "Pré-entraînement masqué {}/{} | loss={:.4f} | masque={:.0%}",
                epoch + 1,
                epochs,
                float(total) / float(seen or 1),
                float(self._pretraining_config["mask_probability"]),
            )
        return {
            name: value.detach().cpu().clone() for name, value in model.bert.state_dict().items()
        }

    def _load_backbone(self, backbone: Mapping[str, Any]) -> None:
        """Copy the pre-trained encoder weights into the classification network."""
        if not isinstance(self._network, BertForSequenceClassification):
            return
        missing, unexpected = self._network.bert.load_state_dict(
            {name: torch.as_tensor(value) for name, value in backbone.items()}, strict=False
        )
        # Le pooler n'existe pas dans le modèle de pré-entraînement masqué : sa tête est
        # réinitialisée puis affinée avec le classifieur. C'est le comportement attendu, et il est
        # tracé — toute autre clé manquante est un vrai avertissement.
        reinitialised = sorted(name for name in missing if name.startswith("pooler."))
        remaining = sorted(name for name in missing if not name.startswith("pooler."))
        if reinitialised:
            logger.debug("Pooler réinitialisé au transfert de l'encodeur : {}", reinitialised)
        if remaining or unexpected:
            logger.warning(
                "Poids d'encodeur partiellement transférés | missing={} | unexpected={}",
                remaining,
                sorted(unexpected),
            )

    def _special_ids(self) -> set[int]:
        """Return the identifiers that the masked pre-training never hides."""
        tokenizer = self.tokenizer
        ids = {
            tokenizer.pad_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.unk_token_id,
        }
        return {int(identifier) for identifier in ids if identifier is not None}

    def _optimizer(self) -> torch.optim.AdamW:
        """Build the optimiser of the fine-tuning loop."""
        return torch.optim.AdamW(
            self.network.parameters(),
            lr=self._learning_rate(),
            weight_decay=float(self._training_config.get("weight_decay", 0.01)),
        )

    def _schedule(
        self, optimizer: torch.optim.AdamW, *, steps_per_epoch: int
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Build the linear warmup + linear decay schedule (the recipe of the transformers).

        Args:
            optimizer: Optimiser to schedule.
            steps_per_epoch: Number of optimisation steps of one epoch.

        Returns:
            A per-step learning rate schedule.
        """
        steps = max(int(steps_per_epoch) * self._epochs(), 1)
        warmup = max(round(steps * float(self._training_config.get("warmup_ratio", 0.1))), 1)

        def factor(step: int) -> float:
            """Return the learning rate factor of one optimisation step."""
            if step < warmup:
                return float(step + 1) / float(warmup)
            remaining = max(steps - step, 0)
            return float(remaining) / float(max(steps - warmup, 1))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)

    def _class_weights(self, targets: np.ndarray) -> torch.Tensor | None:
        """Return the balanced class weights, or ``None`` when the parameter disables them."""
        if str(self._training_config.get("class_weight", "balanced")) != "balanced":
            return None
        counts = np.bincount(np.asarray(targets, dtype="int64"), minlength=len(self._classes))
        counts = np.where(counts == 0, 1, counts)
        weights = float(len(targets)) / (float(len(self._classes)) * counts)
        return torch.tensor(weights, dtype=torch.float32)

    def _batch(self, encoded: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, Any]:
        """Build a device-resident batch from the encoded corpus."""
        return {
            "input_ids": torch.as_tensor(
                encoded["input_ids"][indices], dtype=torch.long, device=self._device
            ),
            "attention_mask": torch.as_tensor(
                encoded["attention_mask"], dtype=torch.long, device=self._device
            )[indices],
        }

    def _train_metrics(self, texts: Sequence[str], labels: Sequence[str]) -> dict[str, float]:
        """Score the training texts and return accuracy, macro F1 and balanced accuracy."""
        probabilities = self._predict_proba(texts)
        predictions = [self._classes[int(index)] for index in np.argmax(probabilities, axis=1)]
        metrics = classification_metrics(labels, predictions, labels=list(self._classes))
        return {key: float(value) for key, value in metrics.items() if key in _TRAIN_METRIC_KEYS}

    def _emit(
        self,
        context: Any,
        callbacks: Sequence[Any] | None,
        logs: Mapping[str, float],
        *,
        epoch: int,
    ) -> bool:
        """Fire ``on_epoch_end`` for one epoch and report whether a callback asked to stop."""
        context.epoch = int(epoch)
        context.logs = {str(key): float(value) for key, value in logs.items()}
        for callback in callbacks or ():
            callback.on_epoch_end(context)
        return bool(context.stopped_early)

    # ------------------------------------------------------------------ réglages ----------
    def _epochs(self) -> int:
        """Number of fine-tuning epochs.

        Trois niveaux, du plus spécifique au plus général : ``params.epochs`` — le budget du modèle
        lui-même, qui permet à un notebook de comparer trois architectures à budget réduit, ou à un
        encodeur de demander plus d'époques qu'un sac d'embeddings — puis ``train.epochs``
        (l'orchestration), puis le défaut de la stack.
        """
        if self.params.get("epochs") is not None:
            return max(int(self.params["epochs"]), 1)
        train = dict(self.config.get("train") or {})
        if train.get("epochs") is not None:
            return max(int(train["epochs"]), 1)
        return max(int(self._training_config.get("epochs", 6)), 1)

    def _batch_size(self) -> int:
        """Fine-tuning batch size (`train.batch_size` wins, model parameters come second)."""
        train = dict(self.config.get("train") or {})
        if train.get("batch_size") is not None:
            return max(int(train["batch_size"]), 1)
        return max(int(self._training_config.get("batch_size", 16)), 1)

    def _learning_rate(self) -> float:
        """Fine-tuning learning rate.

        Le taux d'apprentissage appartient au **modèle** : il vit dans ``params.training`` et prend
        le pas sur ``train.learning_rate``, parce qu'une grille qui compare plusieurs architectures
        doit pouvoir donner à chacune le taux qui la fait converger (un encodeur entraîné de zéro
        diverge à 0,02 là où un sac d'embeddings converge). Le budget d'époques, lui, reste un
        réglage d'orchestration (voir :meth:`_epochs`).
        """
        if self._training_config.get("learning_rate") is not None:
            return float(self._training_config["learning_rate"])
        return float(dict(self.config.get("train") or {}).get("learning_rate", 0.004))

    def _resolve_device(self, requested: str) -> torch.device:
        """Resolve the device (``auto`` picks the GPU when one is available)."""
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested)

    def _seed_everything(self) -> None:
        """Fix every source of randomness, including the number of threads.

        Le déterminisme d'un entraînement PyTorch dépend aussi du **nombre de threads** : les
        réductions en virgule flottante ne sont pas associatives. Le modèle est minuscule, un seul
        thread suffit, et deux exécutions à graine fixée donnent alors les mêmes probabilités — ce
        que la suite de tests vérifie.
        """
        seed = int(self.random_state)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(int(self._training_config.get("threads", 1)))

    # ------------------------------------------------------------------ occlusion ---------
    def _candidate_terms(self, text: str) -> list[str]:
        """Return the distinct terms of a text, in order of appearance, bounded in number."""
        seen: list[str] = []
        for raw in str(text).split():
            term = raw.strip(".,;:!?()[]{}«»\"'")
            if len(term) < 2 or term in seen:
                continue
            seen.append(term)
            if len(seen) >= MAX_OCCLUSION_TERMS:
                break
        return seen

    def _remove_term(self, text: str, term: str) -> str:
        """Return the text with the first occurrence of ``term`` removed."""
        return text.replace(term, " ", 1)


__all__ = [
    "DEFAULT_MODEL_FILE",
    "FRAMEWORK",
    "SPECIAL_TOKENS",
    "LocalWordPieceTokenizer",
    "TransformerTextClassifier",
]
