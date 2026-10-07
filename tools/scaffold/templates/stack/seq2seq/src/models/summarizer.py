"""Encodeur-décodeur transformer entraîné **sur le corpus**, sans poids pré-entraîné.

Le modèle est volontairement minuscule (deux couches, 128 unités, quatre têtes) : le projet
entier — données, entraînement, six notebooks et pipeline complet — doit tourner sur un
ordinateur portable sans accélération matérielle. Ce qui rend la tâche possible à cette
taille n'est pas la puissance du modèle, c'est la **structure du corpus** : les résumés de
référence sont construits sur des gabarits de faits, et les valeurs à recopier (une durée,
une référence de pièce) sont présentes dans le document. Un encodeur-décodeur de deux
couches peut apprendre à les sélectionner ; c'est exactement ce que le projet mesure.

Trois étapes, toutes dans :meth:`EncoderDecoderSummarizer._fit` :

1. **vocabulaire partagé** — WordPiece appris sur le train (documents *et* résumés), donc les unités
   du domaine (``45``, ``minutes``, ``P-12``) ne sont pas découpées en caractères ;
2. **pré-entraînement dénoising** — le document est bruité (jetons remplacés, ordre local
   perturbé) et le modèle apprend à reconstruire le résumé de référence : c'est la tâche finale,
   avec du bruit en plus, ce qui stabilise les premiers pas d'un modèle initialisé au hasard ;
3. **affinage supervisé** — enseignant-forcé, avec décroissance linéaire du taux d'apprentissage.

Le **décodage** est glouton et déterministe : un résumé doit être reproductible ligne à ligne, donc
aucun échantillonnage, aucune température. Le décodeur s'arrête sur le jeton ``[EOS]`` ; le rapport
publie la part des résumés qui atteignent leur budget de longueur à la place — un décodeur qui bute
sur sa borne ne conclut pas, et c'est une information, pas un détail.

Rien n'est téléchargé : ni vocabulaire, ni poids, ni configuration. La fiche de modèle publie le
nombre de paramètres, la taille du vocabulaire, le taux de jetons inconnus et le taux de documents
tronqués par la borne de l'encodeur — les trois raisons pour lesquelles un ROUGE ne se lit pas seul.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from src.evaluation.rouge import rouge_scores
from src.models.contract import BaseTextGenerator, TextSummary
from src.models.tokenizer import SharedWordPieceTokenizer
from src.preprocessing.transformers import split_sentences
from src.utils.logging import get_logger
from torch import nn

logger = get_logger(__name__)

#: Value used by the tokenizer for an unknown piece (kept here for the corruption step).
UNKNOWN_TOKEN = "[UNK]"

#: Label smoothing of the cross-entropy: a summary may legitimately use a synonym, and a model
#: trained to be 100 % sure of one phrasing reproduces it word for word.
LABEL_SMOOTHING = 0.05

#: Number of validation documents scored during the fit (a full pass would multiply the fitting time
#: by the decoding cost; the same number is used for every run, so it is comparable).
FIT_VALIDATION_DOCUMENTS = 24


class _SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positions: no parameter, therefore no drift from one run to the next.

    Attributes:
        table: The pre-computed ``(1, max_positions, units)`` position table.
    """

    # Déclaration de type du tampon : `register_buffer` l'affecte dynamiquement, donc le typage du
    # module ne peut pas le déduire. Aucun attribut de classe n'est créé (annotations différées).
    table: torch.Tensor

    def __init__(self, units: int, max_positions: int, dropout: float) -> None:
        """Pre-compute the position table.

        Args:
            units: Model width (must be even).
            max_positions: Longest sequence the model accepts.
            dropout: Dropout applied after the addition.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        positions = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, units, 2, dtype=torch.float32) * (-math.log(10000.0) / units)
        )
        table = torch.zeros(max_positions, units)
        table[:, 0::2] = torch.sin(positions * frequencies)
        table[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("table", table.unsqueeze(0), persistent=False)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Add the position signal to a batch of embeddings.

        Args:
            embeddings: ``(batch, sequence, units)`` tensor.

        Returns:
            The shifted embeddings.
        """
        length = embeddings.size(1)
        return self.dropout(embeddings + self.table[:, :length, :])


class _SmallTransformer(nn.Module):
    """Encoder-decoder of two layers, trained from scratch on the corpus."""

    def __init__(
        self,
        *,
        vocab_size: int,
        units: int,
        heads: int,
        layers: int,
        dropout: float,
        max_positions: int,
        pad_id: int,
    ) -> None:
        """Build the network.

        Args:
            vocab_size: Size of the shared vocabulary.
            units: Model width.
            heads: Number of attention heads.
            layers: Number of encoder and decoder layers.
            dropout: Dropout rate.
            max_positions: Longest sequence accepted.
            pad_id: Identifier of the padding token.
        """
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.units = int(units)
        self.pad_id = int(pad_id)
        self.embedding = nn.Embedding(self.vocab_size, self.units, padding_idx=self.pad_id)
        self.encoding = _SinusoidalPositionalEncoding(
            self.units, int(max_positions), float(dropout)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.units,
            nhead=int(heads),
            dim_feedforward=self.units * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.units,
            nhead=int(heads),
            dim_feedforward=self.units * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(layers))
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(layers))
        self.projection = nn.Linear(self.units, self.vocab_size)

    @staticmethod
    def _key_padding_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        """Mask the padding positions of a batch.

        Args:
            ids: ``(batch, sequence)`` identifier tensor.
            pad_id: Identifier of the padding token.

        Returns:
            A boolean mask, ``True`` on the padding (the convention of ``nn.Transformer``).
        """
        return ids == pad_id

    @staticmethod
    def _causal_mask(length: int) -> torch.Tensor:
        """Upper-triangular mask that prevents the decoder from reading the future.

        Args:
            length: Length of the decoded sequence.

        Returns:
            A ``(length, length)`` boolean mask.
        """
        return torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)

    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Run one teacher-forced pass.

        Args:
            source_ids: ``(batch, source_sequence)`` document identifiers.
            target_ids: ``(batch, target_sequence)`` summary identifiers (teacher forcing).

        Returns:
            Logits of shape ``(batch, target_sequence, vocab_size)``.
        """
        memory = self.encode(source_ids)
        return self.decode(target_ids, memory)

    def encode(self, source_ids: torch.Tensor) -> torch.Tensor:
        """Encode a batch of documents.

        Args:
            source_ids: ``(batch, source_sequence)`` identifiers.

        Returns:
            The encoder memory, ``(batch, source_sequence, units)``.
        """
        mask = self._key_padding_mask(source_ids, self.pad_id)
        embedded = self.encoding(self.embedding(source_ids))
        return self.encoder(embedded, src_key_padding_mask=mask)

    def decode(self, target_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Decode a batch of summary prefixes against an encoder memory.

        Args:
            target_ids: ``(batch, target_sequence)`` identifiers.
            memory: Encoder output.

        Returns:
            Logits of shape ``(batch, target_sequence, vocab_size)``.
        """
        padding = self._key_padding_mask(target_ids, self.pad_id)
        embedded = self.encoding(self.embedding(target_ids))
        decoded = self.decoder(
            embedded,
            memory,
            tgt_mask=self._causal_mask(target_ids.size(1)),
            tgt_key_padding_mask=padding,
        )
        return self.projection(decoded)

    def n_parameters(self) -> int:
        """Number of trainable parameters (published in the model card).

        Returns:
            The parameter count.
        """
        return int(
            sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        )


class EncoderDecoderSummarizer(BaseTextGenerator):
    """Sequence-to-sequence summarizer trained on the corpus, decoded greedily.

    Attributes:
        layers: Number of encoder and decoder layers.
        units: Model width.
        heads: Number of attention heads.
        dropout: Dropout rate.
        max_input_tokens: Bound on the document length given to the encoder.
        vocab_size: Target size of the shared vocabulary.
        pretraining_epochs: Number of denoising epochs before the supervised fit.
        mask_rate: Share of document tokens corrupted during the denoising stage.
    """

    strategy = "transformer_tiny"

    def __init__(
        self,
        *,
        layers: int = 2,
        units: int = 128,
        heads: int = 4,
        dropout: float = 0.1,
        max_input_tokens: int = 160,
        vocab_size: int = 1200,
        min_frequency: int = 2,
        pretraining_epochs: int = 2,
        mask_rate: float = 0.25,
        params: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        """Configure the architecture and the training budget.

        Args:
            layers: Number of encoder and decoder layers.
            units: Model width.
            heads: Number of attention heads.
            dropout: Dropout rate.
            max_input_tokens: Bound on the document length.
            vocab_size: Target size of the shared vocabulary.
            min_frequency: Minimum frequency of a WordPiece pair.
            pretraining_epochs: Number of denoising epochs.
            mask_rate: Share of corrupted document tokens.
            params: Nested parameters of the stack (``training``, ``pretraining``…).
            extra: Decoding settings forwarded to :class:`BaseTextGenerator` (``compression``,
                ``max_output_tokens``…).
        """
        nested = dict(params or {})
        training = dict(nested.get("training") or {})
        pretraining = dict(nested.get("pretraining") or {})
        budget = {**nested.get("budget", {}), **training, **_strip_known(nested), **extra}
        super().__init__(
            max_output_tokens=int(budget.pop("max_output_tokens", 90)),
            min_output_tokens=int(budget.pop("min_output_tokens", 12)),
            compression=float(budget.pop("compression", 0.45)),
            seed=int(budget.pop("seed", 42)),
            **budget,
        )
        # Trois sources, dans cet ordre : le bloc explicite (``params.training``), la configuration
        # plate du projet (``model.params.units``, telle qu'elle est écrite dans
        # ``conf/model/default.yaml``) puis les défauts du constructeur. C'est la configuration du
        # projet qui gagne : une architecture se règle dans un fichier Hydra, pas dans du code.
        self.layers = int(training.get("layers", nested.get("layers", layers)))
        self.units = int(training.get("units", nested.get("units", units)))
        self.heads = int(training.get("heads", nested.get("heads", heads)))
        self.dropout = float(training.get("dropout", nested.get("dropout", dropout)))
        self.max_input_tokens = int(
            training.get("max_input_tokens", nested.get("max_input_tokens", max_input_tokens))
        )
        self.vocab_size = int(training.get("vocab_size", nested.get("vocab_size", vocab_size)))
        self.min_frequency = int(
            training.get("min_frequency", nested.get("min_frequency", min_frequency))
        )
        self.pretraining_epochs = int(
            pretraining.get(
                "epochs", nested.get("pretraining", {}).get("epochs", pretraining_epochs)
            )
        )
        self.mask_rate = float(
            pretraining.get("mask_rate", nested.get("pretraining", {}).get("mask_rate", mask_rate))
        )
        self.tokenizer = SharedWordPieceTokenizer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            max_input_tokens=self.max_input_tokens,
        )
        self.network: _SmallTransformer | None = None
        self._epochs_done = 0
        self._history: list[dict[str, float]] = []

    # ------------------------------------------------------------------ entraînement -------
    def _fit(
        self,
        documents: pd.DataFrame,
        references: pd.DataFrame,
        *,
        validation: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    ) -> dict[str, float]:
        """Learn the vocabulary, pre-train by denoising, then fine-tune supervised.

        Args:
            documents: Training documents (``doc_id``, ``text``, ``n_tokens``).
            references: Reference summaries of the same documents.
            validation: Optional ``(documents, references)`` pair used to measure a validation ROUGE
                on a fixed sample at the end of the fit.

        Returns:
            Pretraining loss, final training loss, epochs, vocabulary size, parameter count
            and, when a validation pair is given, the ROUGE-1 measured on its first documents.

        Raises:
            ValueError: When the training pair is empty or the two tables do not describe the same
                documents.
        """
        pairs = self._pairs(documents, references)
        if not pairs:
            msg = (
                "No (document, reference summary) pair to fit on: the two tables must share their "
                "doc_id column and the split must not be empty."
            )
            raise ValueError(msg)
        self._seed_everything()
        sources = [text for text, _ in pairs]
        targets = [summary for _, summary in pairs]
        self.tokenizer.fit([*sources, *targets])
        self.network = self._build_network()
        logger.info(
            "Encodeur-décodeur initialisé : vocabulaire {}, {} paramètres, {} paires",
            self.tokenizer.vocabulary_size,
            self.network.n_parameters(),
            len(pairs),
        )

        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self._learning_rate(), weight_decay=0.01
        )
        metrics: dict[str, float] = {}
        if self.pretraining_epochs > 0:
            metrics["pretrain_loss"] = self._denoise(pairs, optimizer)
        total_epochs = max(self._epochs(), 1)
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: max(0.05, 1.0 - epoch / total_epochs)
        )
        self.network.train()
        history: list[dict[str, float]] = []
        for epoch in range(total_epochs):
            loss = self._supervised_epoch(pairs, optimizer)
            schedule.step()
            history.append({"epoch": float(epoch + 1), "loss": round(loss, 6)})
            self._emit(epoch + 1, total_epochs, {"loss": loss})
        self._history = history
        self._epochs_done = total_epochs
        metrics.update(
            {
                "train_loss": round(float(history[-1]["loss"]), 6),
                "epochs": float(total_epochs),
                "vocab_size": float(self.tokenizer.vocabulary_size),
                "n_parameters": float(self.network.n_parameters()),
                "n_pairs": float(len(pairs)),
            }
        )
        if validation is not None and not validation[0].empty:
            metrics.update(self._validation_metrics(*validation))
        return metrics

    def _denoise(self, pairs: Sequence[tuple[str, str]], optimizer: torch.optim.Optimizer) -> float:
        """Pre-train by reconstructing the reference summary from a corrupted document.

        Args:
            pairs: ``(document, reference summary)`` pairs.
            optimizer: Optimiser shared with the supervised stage.

        Returns:
            The mean loss of the last denoising epoch.
        """
        assert self.network is not None  # posé par `_fit`
        generator = torch.Generator().manual_seed(self.seed)
        losses: list[float] = []
        for epoch in range(max(self.pretraining_epochs, 1)):
            self.network.train()
            epoch_losses: list[float] = []
            for batch in self._batches(pairs, corrupt=True):
                optimizer.zero_grad()
                logits = self.network(batch["source_ids"], batch["decoder_ids"])
                loss = self._loss(logits, batch["targets"])
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(float(loss.item()))
            mean = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            losses.append(mean)
            self._emit(
                epoch + 1,
                max(self.pretraining_epochs, 1),
                {"pretrain_loss": mean},
                stage="pretrain",
            )
        del generator
        return round(losses[-1], 6) if losses else 0.0

    def _supervised_epoch(
        self, pairs: Sequence[tuple[str, str]], optimizer: torch.optim.Optimizer
    ) -> float:
        """Run one supervised epoch with teacher forcing.

        Args:
            pairs: ``(document, reference summary)`` pairs.
            optimizer: Optimiser to update.

        Returns:
            The mean loss of the epoch.
        """
        assert self.network is not None  # posé par `_fit`
        self.network.train()
        losses: list[float] = []
        for batch in self._batches(pairs, corrupt=False):
            optimizer.zero_grad()
            logits = self.network(batch["source_ids"], batch["decoder_ids"])
            loss = self._loss(logits, batch["targets"])
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy of a teacher-forced pass, padding ignored.

        Args:
            logits: ``(batch, target_sequence, vocab_size)`` model output.
            targets: ``(batch, target_sequence)`` expected identifiers.

        Returns:
            The scalar loss.
        """
        assert self.network is not None  # posé par `_fit`
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.network.pad_id,
            label_smoothing=LABEL_SMOOTHING,
        )

    def _batches(
        self, pairs: Sequence[tuple[str, str]], *, corrupt: bool
    ) -> list[dict[str, torch.Tensor]]:
        """Turn the training pairs into padded batches.

        Args:
            pairs: ``(document, reference summary)`` pairs.
            corrupt: Whether the document tokens are corrupted (denoising stage) or given as-is.

        Returns:
            One mapping per batch: ``source_ids``, ``decoder_ids`` and ``targets``.
        """
        assert self.network is not None  # posé par `_fit`
        batch_size = self._batch_size()
        rng = random.Random(self.seed + (1 if corrupt else 0))
        order = list(range(len(pairs)))
        rng.shuffle(order)
        pad_id = self.network.pad_id
        batches: list[dict[str, torch.Tensor]] = []
        for start in range(0, len(order), batch_size):
            chunk = [pairs[index] for index in order[start : start + batch_size]]
            source_rows: list[list[int]] = []
            decoder_rows: list[list[int]] = []
            target_rows: list[list[int]] = []
            for document, summary in chunk:
                source = self.tokenizer.encode(
                    document, max_length=self.max_input_tokens, add_special=False
                )
                if corrupt:
                    source = self._corrupt(source, rng)
                target = self.tokenizer.encode(summary, add_special=True)
                decoder = [self.tokenizer.bos_id, *target[:-1]]
                source_rows.append(source)
                decoder_rows.append(decoder)
                target_rows.append(target)
            source_width = max(len(row) for row in source_rows)
            target_width = max(len(row) for row in target_rows)
            batches.append(
                {
                    "source_ids": self._pad(source_rows, source_width, pad_id),
                    "decoder_ids": self._pad(decoder_rows, target_width, pad_id),
                    "targets": self._pad(target_rows, target_width, pad_id),
                }
            )
        return batches

    def _corrupt(self, ids: list[int], rng: random.Random) -> list[int]:
        """Corrupt a document: replace tokens, then swap two neighbours locally.

        Args:
            ids: Identifier list of the document.
            rng: Random generator of the stage (seeded, so the corruption is reproducible).

        Returns:
            The corrupted identifiers; an empty document is left untouched.
        """
        if not ids:
            return ids
        unknown = self.tokenizer.unk_id
        corrupted = [unknown if rng.random() < self.mask_rate else value for value in ids]
        for index in range(len(corrupted) - 1):
            if rng.random() < self.mask_rate / 4:
                corrupted[index], corrupted[index + 1] = corrupted[index + 1], corrupted[index]
        return corrupted

    @staticmethod
    def _pad(rows: Sequence[Sequence[int]], width: int, pad_id: int) -> torch.Tensor:
        """Right-pad a batch of identifier lists.

        Args:
            rows: Identifier lists of unequal length.
            width: Target width.
            pad_id: Identifier of the padding token.

        Returns:
            The padded ``(batch, width)`` tensor.
        """
        return torch.tensor(
            [list(row) + [pad_id] * (width - len(row)) for row in rows], dtype=torch.long
        )

    def _pairs(self, documents: pd.DataFrame, references: pd.DataFrame) -> list[tuple[str, str]]:
        """Align documents and reference summaries on their identifier.

        Args:
            documents: Document table.
            references: Reference summary table.

        Returns:
            The ``(document text, reference summary)`` pairs, in document order.
        """
        golden = dict(zip(references["doc_id"], references["summary"], strict=False))
        pairs: list[tuple[str, str]] = []
        for row in documents.itertuples(index=False):
            summary = golden.get(str(row.doc_id))
            if summary is None:
                continue
            pairs.append((str(row.text), str(summary)))
        return pairs

    def _validation_metrics(
        self, documents: pd.DataFrame, references: pd.DataFrame
    ) -> dict[str, float]:
        """Measure a validation ROUGE on a fixed sample (same size for every run).

        Args:
            documents: Validation documents.
            references: Validation reference summaries.

        Returns:
            The mean ROUGE-1 F1 and the number of documents scored.
        """
        sample = documents.head(FIT_VALIDATION_DOCUMENTS)
        golden = dict(zip(references["doc_id"], references["summary"], strict=False))
        rows = [row for row in sample.itertuples(index=False) if str(row.doc_id) in golden]
        if not rows:
            return {}
        scores: list[float] = []
        for row in rows:
            summary = self._summarize(str(row.text), doc_id=str(row.doc_id))
            scores.append(rouge_scores(str(golden[str(row.doc_id)]), summary.summary).rouge1.f1)
        return {
            "val_rouge1_f": round(float(np.mean(scores)), 4),
            "val_documents": float(len(rows)),
        }

    # ------------------------------------------------------------------ décodage -----------
    def _summarize(self, text: str, *, doc_id: str) -> TextSummary:
        """Summarise one document by greedy decoding.

        Args:
            text: Document text.
            doc_id: Document identifier.

        Returns:
            The generated summary; ``hit_max_length`` is true when the decoder stopped on its budget
            instead of emitting the end-of-summary marker.
        """
        assert self.network is not None  # garanti par `check_is_fitted`
        started = perf_counter()
        document_tokens = len(self.tokenizer.pieces(text))
        budget = self._budget_tokens(document_tokens)
        source = self.tokenizer.encode(text, max_length=self.max_input_tokens, add_special=False)
        self.network.eval()
        # Un jeton spécial n'est pas une réponse : le décodeur peut choisir ``[EOS]`` (donc
        # s'arrêter), jamais ``[BOS]``, ``[PAD]`` ou ``[UNK]``. Sans ce masque, un modèle minuscule
        # encore peu entraîné met ``[BOS]`` en tête à chaque pas et publie une ligne de ``[BOS]``
        # comme résumé — mesuré, puis corrigé ici plutôt que filtré à l'affichage.
        forbidden = [
            identifier
            for identifier in self.tokenizer.special_ids()
            if identifier != self.tokenizer.eos_id
        ]
        ids = [self.tokenizer.bos_id]
        stopped_on_marker = False
        with torch.no_grad():
            memory = self.network.encode(torch.tensor([source], dtype=torch.long))
            for _ in range(max(int(budget), 1)):
                logits = self.network.decode(torch.tensor([ids], dtype=torch.long), memory)
                scores = logits[0, -1, :].clone()
                if forbidden:
                    scores[forbidden] = float("-inf")
                nxt = int(torch.argmax(scores).item())
                if nxt == self.tokenizer.eos_id:
                    stopped_on_marker = True
                    break
                ids.append(nxt)
        decoded = self.tokenizer.decode(ids[1:], skip_special=True)
        generated = len(self.tokenizer.pieces(decoded))
        sentences = split_sentences(decoded)
        return TextSummary(
            doc_id=str(doc_id),
            summary=decoded,
            sentences=sentences,
            strategy=self.strategy,
            budget_tokens=int(budget),
            n_tokens=generated,
            document_tokens=document_tokens,
            compression=generated / max(document_tokens, 1),
            # Un décodeur qui bute sur sa borne n'a pas conclu : c'est publié, jamais masqué.
            hit_max_length=not stopped_on_marker,
            latency_ms=(perf_counter() - started) * 1000.0,
            metadata={
                "decoded_tokens": float(generated),
                "source_tokens": float(len(source)),
                "truncated_source": float(len(source) >= self.max_input_tokens),
            },
        )

    # ------------------------------------------------------------------ construction -------
    def _build_network(self) -> _SmallTransformer:
        """Build the network from the learned vocabulary and the configured architecture.

        Returns:
            A fresh, randomly initialised network.
        """
        torch.manual_seed(self.seed)
        return _SmallTransformer(
            vocab_size=self.tokenizer.vocabulary_size,
            units=self.units,
            heads=self.heads,
            layers=self.layers,
            dropout=self.dropout,
            max_positions=max(self.max_input_tokens, self.max_output_tokens) + 4,
            pad_id=self.tokenizer.pad_id,
        )

    def _epochs(self) -> int:
        """Number of supervised epochs.

        Returns:
            ``train.epochs`` when the configuration provides it, ``12`` otherwise.
        """
        return int(self.params.get("epochs", self.params.get("train_epochs", 12)))

    def _batch_size(self) -> int:
        """Training batch size.

        Returns:
            ``train.batch_size`` when the configuration provides it, ``16`` otherwise.
        """
        return int(self.params.get("batch_size", 16))

    def _learning_rate(self) -> float:
        """Learning rate of the supervised fit.

        Returns:
            ``train.learning_rate`` when the configuration provides it, ``2e-3`` otherwise.
        """
        return float(self.params.get("learning_rate", 2e-3))

    def _seed_everything(self) -> None:
        """Seed every source of randomness, and pin the thread count.

        Le nombre de fils d'exécution change l'ordre des réductions flottantes, donc les derniers
        chiffres d'une perte — et, à la marge, la génération. Le fixer à un fil est ce qui rend deux
        exécutions identiques sur la même machine, ce que la suite de tests vérifie.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)

    # ------------------------------------------------------------------ persistance --------
    def _payload(self) -> dict[str, Any]:
        """Serialise the network, the vocabulary and the architecture.

        Returns:
            A mapping of NumPy arrays and plain values (no pickle of a third-party object).
        """
        if self.network is None:
            return {}
        return {
            "architecture": {
                "layers": self.layers,
                "units": self.units,
                "heads": self.heads,
                "dropout": self.dropout,
                "max_input_tokens": self.max_input_tokens,
            },
            "tokenizer": self.tokenizer.to_payload(),
            "state": {
                name: tensor.detach().numpy() for name, tensor in self.network.state_dict().items()
            },
            "history": list(self._history),
        }

    def _restore(self, payload: dict[str, Any]) -> None:
        """Restore the network and the vocabulary from an artefact.

        Args:
            payload: Mapping written by :meth:`_payload`.
        """
        architecture = dict(payload.get("architecture") or {})
        self.layers = int(architecture.get("layers", self.layers))
        self.units = int(architecture.get("units", self.units))
        self.heads = int(architecture.get("heads", self.heads))
        self.dropout = float(architecture.get("dropout", self.dropout))
        self.max_input_tokens = int(architecture.get("max_input_tokens", self.max_input_tokens))
        tokenizer_payload = payload.get("tokenizer")
        if tokenizer_payload:
            self.tokenizer = SharedWordPieceTokenizer.from_payload(tokenizer_payload)
        network = self._build_network()
        state = dict(payload.get("state") or {})
        if state:
            network.load_state_dict({name: torch.tensor(value) for name, value in state.items()})
        network.eval()
        self.network = network
        self._history = list(payload.get("history") or [])

    def _extra_metadata(self) -> dict[str, Any]:
        """Architecture, sizes and the three caveats a ROUGE should never be read without."""
        return {
            "architecture": {
                "type": "encoder-decoder-transformer",
                "layers": self.layers,
                "units": self.units,
                "heads": self.heads,
                "dropout": self.dropout,
                "max_input_tokens": self.max_input_tokens,
                "decoding": "greedy, deterministic",
            },
            "vocabulary": {
                "size": self.tokenizer.vocabulary_size,
                "min_frequency": self.min_frequency,
                "shared": True,
                "pretrained": False,
                "unknown_rate": self.tokenizer.meta.get("unknown_rate"),
                "version": self.tokenizer.meta.get("version"),
                "merges": self.tokenizer.meta.get("merges"),
                "learned_on_words": self.tokenizer.meta.get("words"),
            },
            "training": {
                "epochs": self._epochs_done,
                "pretraining_epochs": self.pretraining_epochs,
                "mask_rate": self.mask_rate,
                "label_smoothing": LABEL_SMOOTHING,
                "history": list(self._history),
            },
            "n_parameters": self.network.n_parameters() if self.network is not None else 0,
        }

    # ------------------------------------------------------------------ introspection -----
    def attention_shares(self, text: str) -> dict[str, float]:
        """Share of the encoder attention received by each sentence of a document.

        L'explication d'un résumé n'est pas une liste de mots pesés : c'est la **répartition de
        l'attention**. Le notebook 06 la publie pour montrer ce que le modèle regarde au premier pas
        de décodage — la phrase qui porte l'équipement, la durée, le statut.

        La sonde est volontairement simple et documentée : la similarité entre l'état encodé de
        chaque position et le vecteur du premier jeton choisi, normalisée en parts par un softmax.
        Les positions sont ensuite agrégées par phrase, en comptant les pièces de chaque phrase avec
        le même tokenizer que l'encodeur : la dernière phrase absorbe ce qui reste (les positions de
        bordure).

        Args:
            text: Document to analyse.

        Returns:
            ``{"0": part, "1": part, …}`` : une part par phrase, dans l'ordre de lecture. Le
            dictionnaire est vide quand le modèle n'est pas entraîné ou le document vide.
        """
        if self.network is None:
            msg = "The model is not fitted: call fit(documents, references) first."
            raise RuntimeError(msg)
        sentences = split_sentences(text)
        if not sentences:
            return {}
        source = self.tokenizer.encode(text, max_length=self.max_input_tokens, add_special=False)
        if not source:
            return {}
        # Bornes des phrases dans la séquence encodée : le nombre de pièces de chaque phrase découpe
        # la séquence dans le même ordre, la dernière phrase recevant ce qui reste.
        widths = [len(self.tokenizer.encode(sentence, add_special=False)) for sentence in sentences]
        bounds: list[tuple[int, int]] = []
        cursor = 0
        for index, width in enumerate(widths):
            end = cursor + width if index < len(widths) - 1 else len(source)
            bounds.append((cursor, max(end, cursor)))
            cursor = min(end, len(source))
        self.network.eval()
        with torch.no_grad():
            memory = self.network.encode(torch.tensor([source], dtype=torch.long))
            logits = self.network.decode(
                torch.tensor([[self.tokenizer.bos_id]], dtype=torch.long), memory
            )
            first_token = int(torch.argmax(logits[0, -1, :]).item())
            probe = self.network.embedding(torch.tensor(first_token))
            weights = torch.softmax(memory[0] @ probe / math.sqrt(self.units), dim=0)
        values = weights.tolist()
        shares = {
            str(index): round(float(sum(values[start:end])), 6)
            for index, (start, end) in enumerate(bounds)
        }
        return shares

    def _emit(
        self,
        epoch: int,
        total: int,
        metrics: Mapping[str, float],
        *,
        stage: str = "fit",
    ) -> None:
        """Log one epoch, with the frequency the configuration declares.

        Args:
            epoch: Current epoch (1-based).
            total: Total number of epochs.
            metrics: Loss of the epoch.
            stage: ``pretrain`` or ``fit`` (the label of the log line).
        """
        every = int(self.params.get("logging_every", 2) or 2)
        if epoch % max(every, 1) != 0 and epoch != total:
            return
        summary = ", ".join(f"{name}={value:.4f}" for name, value in sorted(metrics.items()))
        logger.info("[{}] époque {}/{} : {}", stage, epoch, total, summary)


def _strip_known(nested: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the parameters that are not training blocks of the stack.

    Args:
        nested: Nested parameters of the stack.

    Returns:
        The flat parameters (the ``training``, ``pretraining`` and ``budget`` blocks are removed).
    """
    return {
        key: value
        for key, value in nested.items()
        if key not in {"training", "pretraining", "budget"}
    }


__all__ = [
    "FIT_VALIDATION_DOCUMENTS",
    "LABEL_SMOOTHING",
    "EncoderDecoderSummarizer",
    "UNKNOWN_TOKEN",
]
