"""Tagger d'entités spaCy : trois algorithmes derrière le même contrat, dont l'artefact rechargé.

Le modèle porte la chaînon qui manque à la famille : ce qu'un **apprentissage** apporte là où une
liste de noms s'arrête. Trois algorithmes sont servis, et la comparaison entre eux est un résultat
du
projet plutôt qu'un réglage caché :

* ``gazetteer`` — la couche de règles seule (:mod:`src.models.rules`) : motifs déclarés plus
surfaces
  annotées dans le train. Aucun poids n'est appris, donc c'est le **plancher explicable** : il ne
  peut pas connaître une surface réservée aux splits d'évaluation, et c'est précisément ce que le
  projet mesure ;
* ``tagger`` — un pipeline ``spacy.blank(...)`` entraîné **sur le corpus** : le tok2vec et le
  classifieur à transitions du composant ``ner`` apprennent sur les messages annotés du split
  d'entraînement. Aucun poids pré-entraîné n'est téléchargé : le projet n'a aucune dépendance
  réseau ;
* ``hybrid`` — le tagger décide, la couche de règles corrobore et comble les trous. Servi par
défaut,
  parce qu'il publie la **provenance** de chaque mention et une **confiance** mesurable.

La confiance mérite une phrase, parce qu'elle est facile à falsifier : un tagger à transitions
n'expose aucune probabilité par mention. La valeur publiée est donc le **taux de recouvrement**
entre
le span prédit et la couche de règles (0,0 : aucune règle ne confirme le span ; 1,0 : le span est
entièrement confirmé), et sa précision est mesurée **par niveau** dans le rapport d'évaluation
(``confidence_gap``). Si les spans corroborés n'étaient pas plus précis que les autres, la table
le
montrerait — c'est un résultat publié, pas une promesse.

L'alignement des annotations est vérifié à l'entraînement : spaCy ignore silencieusement une
annotation qui ne tombe pas sur des frontières de tokens, et le projet publie le nombre
d'annotations ignorées plutôt que de laisser croire que tout a été appris.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import spacy
from spacy.training import Example, offsets_to_biluo_tags

from src.models.base import FitResult
from src.models.contract import BaseEntityTagger, EntityMention, sort_mentions
from src.models.rules import GazetteerIndex, RuleSpan, describe_rules, rule_mentions, rule_spans
from src.utils.io import load_pickle, save_pickle, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Algorithm identifiers served by this stack.
GAZETTEER = "gazetteer"
TAGGER = "tagger"
HYBRID = "hybrid"

#: Sub-directory of the artefact that holds the spaCy pipeline (``nlp.to_disk``).
PIPELINE_DIR = "pipeline"

#: File of the artefact that holds the rule index and the training history.
RULES_FILE = "rules.joblib"

#: File of the artefact that holds the published training history (human readable).
TRAINING_FILE = "training.json"

#: Language used to build the blank pipeline: it only selects the tokenizer rules.
DEFAULT_LANGUAGE = "fr"


class SpacyEntityTagger(BaseEntityTagger):
    """Extract typed mentions: a spaCy pipeline trained on the corpus, or declared rules."""

    framework = "spacy"
    default_model_file = "tagger"

    def __init__(self, **kwargs: Any) -> None:
        """Build an untrained extractor.

        Args:
            **kwargs: Keyword arguments of :class:`~src.models.contract.BaseEntityTagger`.
        """
        super().__init__(**kwargs)
        self._nlp: spacy.Language | None = None
        self._index = GazetteerIndex()
        self._losses: list[float] = []
        self._learning_rates: list[float] = []
        self._epochs_trained = 0

    # ------------------------------------------------------------------ configuration -----
    def _family(self) -> str:
        """Return the algorithm family (``gazetteer``, ``tagger`` or ``hybrid``).

        Returns:
            The normalised algorithm identifier.

        Raises:
            ValueError: When the configured algorithm is unknown.
        """
        name = str(self.algorithm or HYBRID).strip().lower().removeprefix("spacy_")
        if name in {GAZETTEER, "rules", "regles"}:
            return GAZETTEER
        if name in {TAGGER, "ner"}:
            return TAGGER
        if name in {HYBRID, "hybride"}:
            return HYBRID
        msg = f"Unknown algorithm '{self.algorithm}'. Available: {[GAZETTEER, TAGGER, HYBRID]}"
        raise ValueError(msg)

    def _training_config(self) -> dict[str, Any]:
        """Return the ``params.training`` node of the model."""
        node = self.params.get("training")
        return dict(node) if isinstance(node, Mapping) else {}

    def _train_node(self) -> dict[str, Any]:
        """Return the ``train`` node of the application configuration."""
        node = self.config.get("train")
        return dict(node) if isinstance(node, Mapping) else {}

    def _epochs(self) -> int:
        """Return the number of training epochs: ``params.epochs`` then ``train.epochs``."""
        if self.params.get("epochs") is not None:
            return max(int(self.params["epochs"]), 1)
        if self._train_node().get("epochs") is not None:
            return max(int(self._train_node()["epochs"]), 1)
        return 30

    def _batch_size(self) -> int:
        """Return the number of documents per optimisation step."""
        configured = self._training_config().get("batch_size")
        if configured is None:
            configured = self._train_node().get("batch_size")
        return max(int(configured or 8), 1)

    def _learning_rate(self) -> float:
        """Return the learning rate handed to the Thinc optimiser."""
        configured = self._training_config().get("learning_rate")
        if configured is None:
            configured = self._train_node().get("learning_rate")
        return float(configured if configured is not None else 0.001)

    def _dropout(self) -> float:
        """Return the dropout rate applied at each optimisation step."""
        return float(self._training_config().get("dropout", 0.2))

    def _shuffle(self) -> bool:
        """Return whether the training examples are shuffled at every epoch."""
        return bool(self._training_config().get("shuffle", True))

    def _language(self) -> str:
        """Return the language of the blank pipeline (tokenizer rules only)."""
        return str(self.params.get("language", DEFAULT_LANGUAGE))

    def _effective_params(self) -> dict[str, Any]:
        """Return the hyper-parameters actually served, epochs and learning rate included."""
        effective = dict(self.params)
        effective["algorithm_family"] = self._family()
        effective["language"] = self._language()
        effective["training"] = {
            **self._training_config(),
            "epochs": self._epochs(),
            "batch_size": self._batch_size(),
            "learning_rate": self._learning_rate(),
            "dropout": self._dropout(),
            "shuffle": self._shuffle(),
        }
        effective["resolved_architecture"] = self.architecture_report()
        effective["rules"] = self._index.describe()
        return effective

    def architecture_report(self) -> dict[str, Any]:
        """Describe the resolved spaCy architecture (published in the model card).

        L'architecture n'est pas redéclarée dans la configuration : une configuration partielle
        produirait un modèle différent de celui que la fiche décrit. Elle est donc **lue** sur le
        pipeline servi, et publiée telle quelle.

        Returns:
            The pipe names, the architecture of the entity recogniser and the embedding layer
            name.
        """
        if self._nlp is None:
            return {"pipe_names": [], "architecture": "rules-only"}
        report: dict[str, Any] = {"pipe_names": list(self._nlp.pipe_names)}
        if "ner" in self._nlp.pipe_names:
            component: Any = self._nlp.get_pipe("ner")
            model = component.model
            report["architecture"] = type(model).__name__
            tok2vec = model.get_ref("tok2vec")
            report["tok2vec"] = getattr(tok2vec, "name", type(tok2vec).__name__)
        return report

    # ------------------------------------------------------------------ entraînement ------
    def _fit(
        self,
        documents: pd.DataFrame,
        spans: pd.DataFrame,
        *,
        callbacks: Sequence[Any] | None = None,
        context: Any = None,
    ) -> FitResult:
        """Train the declared algorithm on the annotated training split.

        Args:
            documents: Training messages (text column and join key).
            spans: Training annotations (join key, offsets, entity type).
            callbacks: Project callbacks, fired once per epoch.
            context: Mutable callback context (an early stop request is honoured).

        Returns:
            The fit result: training metrics on the annotated split, the per-epoch loss history
            and
            the alignment diagnostics.

        Raises:
            ValueError: When the corpus cannot be turned into spaCy examples.
        """
        family = self._family()
        self._index = GazetteerIndex.from_spans(spans)
        texts, annotations, identifiers = self._training_rows(documents, spans)
        alignment = self._alignment_report(texts, annotations)
        if alignment["ignored"] > 0:
            logger.warning(
                "{} annotation(s) sur {} ne tombent pas sur des frontières de tokens et seront "
                "ignorées par spaCy : la fiche de modèle publie ce taux.",
                int(alignment["ignored"]),
                int(alignment["total"]),
            )

        metrics: dict[str, float] = {}
        history: dict[str, list[float]] = {}
        if family in {TAGGER, HYBRID}:
            self._nlp = self._train_pipeline(
                texts, annotations, callbacks=callbacks, context=context
            )
            metrics["loss"] = float(self._losses[-1]) if self._losses else 0.0
            history["loss"] = list(self._losses)
            history["learning_rate"] = list(self._learning_rates)
        else:
            self._nlp = None

        gold = [
            {(start, end, label) for start, end, label in annotation} for annotation in annotations
        ]
        predicted = [
            {mention.as_tuple() for mention in mentions}
            for mentions in self._predict(texts, identifiers)
        ]
        scored = _entity_scores(gold, predicted, labels=self.labels)
        metrics.update({key: float(value) for key, value in scored.items()})
        metrics["epochs_trained"] = float(self._epochs_trained if family != GAZETTEER else 0)
        metrics.update({f"alignment_{key}": float(value) for key, value in alignment.items()})
        metrics["gazetteer_size"] = float(self._index.size)
        logger.info(
            "{} fitted | {} messages | {} mentions | {} surfaces indexées | epochs={}",
            self.summary(),
            len(documents),
            len(spans),
            self._index.size,
            metrics["epochs_trained"],
        )
        return FitResult(
            model_name=type(self).__name__,
            algorithm=self.algorithm,
            metrics=metrics,
            history=history,
            epochs=max(self._epochs_trained, 1),
            extra={
                "algorithm_family": family,
                "architecture": self.architecture_report(),
                "rules": describe_rules(),
                "index": self._index.describe(),
                "alignment": alignment,
            },
        )

    def _training_rows(
        self, documents: pd.DataFrame, spans: pd.DataFrame
    ) -> tuple[list[str], list[list[tuple[int, int, str]]], list[str]]:
        """Turn the two corpus tables into the rows spaCy needs.

        Args:
            documents: Message table.
            spans: Annotation table.

        Returns:
            The texts, the annotations per text (``(start, end, label)``), and the document
            identifiers.
        """
        grouped: dict[str, list[tuple[int, int, str]]] = {}
        for row in spans.itertuples(index=False):
            grouped.setdefault(str(getattr(row, self.id_column)), []).append(
                (int(row.start), int(row.end), str(getattr(row, self.target_name)))
            )
        identifiers = [str(value) for value in documents[self.id_column]]
        texts = [str(value) for value in documents[self.text_column]]
        for identifier in identifiers:
            grouped.setdefault(identifier, [])
        annotations = [sorted(grouped[identifier]) for identifier in identifiers]
        return texts, annotations, identifiers

    def _train_pipeline(
        self,
        texts: Sequence[str],
        annotations: Sequence[Sequence[tuple[int, int, str]]],
        *,
        callbacks: Sequence[Any] | None,
        context: Any,
    ) -> spacy.Language:
        """Train the blank pipeline on the annotated messages.

        Args:
            texts: Training texts.
            annotations: Annotations of every text.
            callbacks: Project callbacks, fired at the end of each epoch.
            context: Mutable callback context.

        Returns:
            The trained pipeline.

        Raises:
            ValueError: When no example could be built (empty corpus).
        """
        spacy.util.fix_random_seed(self.random_state)
        nlp = spacy.blank(self._language())
        recogniser: Any = nlp.add_pipe("ner")
        for annotation in annotations:
            for _, _, label in annotation:
                recogniser.add_label(label)
        examples = [
            Example.from_dict(nlp.make_doc(text), {"entities": list(annotation)})
            for text, annotation in zip(texts, annotations, strict=True)
        ]
        if not examples:
            msg = "No training example could be built from the corpus"
            raise ValueError(msg)
        optimizer = nlp.initialize(lambda: examples)
        optimizer.learn_rate = self._learning_rate()
        epochs = self._epochs()
        batch_size = self._batch_size()
        dropout = self._dropout()
        shuffle = self._shuffle()
        generator = random.Random(self.random_state)
        self._losses = []
        self._learning_rates = []
        self._epochs_trained = 0
        for epoch in range(epochs):
            order = list(range(len(examples)))
            if shuffle:
                generator.shuffle(order)
            losses: dict[str, float] = {}
            for start in range(0, len(order), batch_size):
                chunk = [examples[index] for index in order[start : start + batch_size]]
                nlp.update(chunk, drop=dropout, sgd=optimizer, losses=losses)
            self._epochs_trained = epoch + 1
            self._losses.append(float(losses.get("ner", 0.0)))
            self._learning_rates.append(float(optimizer.learn_rate))
            logger.debug(
                "Époque {}/{} | loss={:.4f} | lr={:.6f}",
                epoch + 1,
                epochs,
                self._losses[-1],
                self._learning_rates[-1],
            )
            if not self._emit_epoch(context, callbacks, epoch=epoch, epochs=epochs):
                logger.info("Arrêt anticipé demandé à l'époque {}", epoch + 1)
                break
        return nlp

    def _emit_epoch(
        self,
        context: Any,
        callbacks: Sequence[Any] | None,
        *,
        epoch: int,
        epochs: int,
    ) -> bool:
        """Fire ``on_epoch_end`` for one epoch and report whether a callback asked to stop.

        Args:
            context: Mutable callback context (may be ``None`` in a direct unit test).
            callbacks: Project callbacks.
            epoch: 0-based epoch index.
            epochs: Planned number of epochs.

        Returns:
            ``False`` when a callback requested an early stop, ``True`` otherwise.
        """
        if context is None:
            return True
        logs = {
            "loss": float(self._losses[-1]) if self._losses else 0.0,
            "learning_rate": float(self._learning_rates[-1]) if self._learning_rates else 0.0,
            "epoch": float(epoch + 1),
            "epochs": float(epochs),
        }
        context.epoch = int(epoch)
        context.logs = logs
        context.history.setdefault("loss", []).append(logs["loss"])
        context.history.setdefault("learning_rate", []).append(logs["learning_rate"])
        for callback in callbacks or ():
            callback.on_epoch_end(context)
        return not bool(context.stopped_early)

    @staticmethod
    def _alignment_report(
        texts: Sequence[str], annotations: Sequence[Sequence[tuple[int, int, str]]]
    ) -> dict[str, float]:
        """Count the annotations spaCy can learn from.

        Une annotation qui ne tombe pas sur des frontières de tokens est **silencieusement
        ignorée**
        par spaCy : sans cette mesure, un corpus à moitié inapprenable produirait un score moyen
        sans
        que personne ne sache pourquoi.

        Args:
            texts: Texts of the corpus.
            annotations: Annotations of every text.

        Returns:
            The total number of annotations, the ignored ones and the resulting rate.
        """
        nlp = spacy.blank(DEFAULT_LANGUAGE)
        total = 0
        ignored = 0
        for text, annotation in zip(texts, annotations, strict=True):
            if not annotation:
                continue
            document = nlp.make_doc(text)
            tags = offsets_to_biluo_tags(document, list(annotation))
            outside = sum(1 for tag in tags if tag == "-")
            total += len(annotation)
            ignored += min(outside, len(annotation))
        rate = 1.0 - (ignored / total) if total else 1.0
        return {"total": float(total), "ignored": float(ignored), "aligned_rate": round(rate, 4)}

    # ------------------------------------------------------------------ prédiction --------
    def _predict(self, texts: list[str], ids: list[str]) -> list[list[EntityMention]]:
        """Extract the mentions of raw texts with the declared algorithm.

        Args:
            texts: Texts to annotate.
            ids: Identifiers of the documents.

        Returns:
            One list of mentions per text, in reading order and without overlap.
        """
        family = self._family()
        mentions: list[list[EntityMention]] = []
        for text, identifier in zip(texts, ids, strict=True):
            rules = rule_spans(text, self._index)
            if family == GAZETTEER:
                found = rule_mentions(text, identifier, self._index)
            else:
                found = self._model_mentions(text, identifier, rules)
                if family == HYBRID:
                    found.extend(self._rule_only_mentions(text, identifier, rules, found))
            mentions.append(sort_mentions(found))
        return mentions

    def _model_mentions(
        self, text: str, identifier: str, rules: Sequence[RuleSpan]
    ) -> list[EntityMention]:
        """Return the mentions proposed by the learned pipeline.

        Args:
            text: Text to annotate.
            identifier: Document identifier.
            rules: Spans of the rule layer (used for the corroboration level).

        Returns:
            The mentions of the tagger, with source ``modele`` and their corroboration level.
        """
        if self._nlp is None:
            return []
        document = self._nlp(text)
        known = set(self._labels)
        return [
            EntityMention(
                msg_id=identifier,
                start=int(entity.start_char),
                end=int(entity.end_char),
                label=str(entity.label_),
                surface=str(entity.text),
                source="modele",
                confidence=self._corroboration(int(entity.start_char), int(entity.end_char), rules),
            )
            for entity in document.ents
            if str(entity.label_) in known
        ]

    def _rule_only_mentions(
        self,
        text: str,
        identifier: str,
        rules: Sequence[RuleSpan],
        found: Sequence[EntityMention],
    ) -> list[EntityMention]:
        """Return the rule spans that the tagger did not propose (hybrid completion).

        Args:
            text: Text to annotate.
            identifier: Document identifier.
            rules: Spans of the rule layer.
            found: Mentions already proposed by the tagger.

        Returns:
            The additional mentions, with source ``regle`` and a confidence of 1.0.
        """
        additions: list[EntityMention] = []
        for span in rules:
            if any(span.start < mention.end and mention.start < span.end for mention in found):
                continue
            additions.append(
                EntityMention(
                    msg_id=identifier,
                    start=span.start,
                    end=span.end,
                    label=span.label,
                    surface=text[span.start : span.end],
                    source="regle",
                    confidence=1.0,
                )
            )
        return additions

    @staticmethod
    def _corroboration(start: int, end: int, rules: Sequence[RuleSpan]) -> float:
        """Return the share of a span that the declared rules also cover.

        Args:
            start: Start offset of the span.
            end: End offset of the span.
            rules: Spans of the rule layer.

        Returns:
            A level between 0.0 (no rule confirms the span) and 1.0 (the span is entirely
            confirmed). C'est une mesure de corroboration, **pas** une probabilité du modèle.
        """
        length = int(end - start)
        if length <= 0:
            return 0.0
        covered = 0
        for span in rules:
            overlap = min(end, span.end) - max(start, span.start)
            if overlap > 0:
                covered += int(overlap)
        return round(min(covered / length, 1.0), 4)

    # ------------------------------------------------------------------ persistance -------
    def _export(self, directory: Path) -> None:
        """Write the pipeline, the rule index and the training history.

        Args:
            directory: Artefact directory (already created).
        """
        if self._nlp is not None:
            self._nlp.to_disk(directory / PIPELINE_DIR)
        save_pickle(
            {
                "index": self._index,
                "losses": self._losses,
                "learning_rates": self._learning_rates,
                "epochs_trained": self._epochs_trained,
            },
            directory / RULES_FILE,
        )
        write_json(
            directory / TRAINING_FILE,
            {
                "epochs_trained": self._epochs_trained,
                "losses": list(self._losses),
                "learning_rates": list(self._learning_rates),
                "algorithm_family": self._family(),
                "index": self._index.describe(),
                "parameters": self._effective_params()["training"],
            },
        )

    def _restore(self, directory: Path) -> None:
        """Reload the pipeline, the rule index and the training history.

        Args:
            directory: Artefact directory written by :meth:`_export`.

        Raises:
            FileNotFoundError: When the rule payload is missing from the artefact.
        """
        pipeline = directory / PIPELINE_DIR
        self._nlp = spacy.load(pipeline) if pipeline.exists() else None
        payload_path = directory / RULES_FILE
        if not payload_path.exists():
            msg = f"Incomplete artefact: {RULES_FILE} missing from {directory}"
            raise FileNotFoundError(msg)
        payload = dict(load_pickle(payload_path))
        index = payload.get("index")
        self._index = index if isinstance(index, GazetteerIndex) else GazetteerIndex()
        self._losses = [float(value) for value in payload.get("losses", [])]
        self._learning_rates = [float(value) for value in payload.get("learning_rates", [])]
        self._epochs_trained = int(payload.get("epochs_trained", 0))
        logger.debug(
            "Artefact reloaded | {} | {} surfaces indexées | pipeline={}",
            self.summary(),
            self._index.size,
            self._nlp is not None,
        )

    def summary(self) -> str:
        """Return a one-line description of the model, algorithm and index included."""
        return (
            f"{type(self).__name__}(algorithm={self._family()}, labels={len(self._labels)}, "
            f"index={self._index.size}, pipeline={'oui' if self._nlp is not None else 'non'}, "
            f"state={self.state})"
        )


def _entity_scores(
    gold: Sequence[set[tuple[int, int, str]]],
    predicted: Sequence[set[tuple[int, int, str]]],
    *,
    labels: Sequence[str],
) -> dict[str, float]:
    """Score the training predictions with the entity metrics of the task layer.

    L'import est **local** : ``src.models`` est chargé par la fabrique avant ``src.training``, et un
    import au niveau du module fermerait le cycle
    ``models -> training.metrics -> training.__init__ -> trainer -> inference.predictor``. Le
    compteur d'entités reste le même que celui du rapport : une seule définition des scores.

    Args:
        gold: Reference spans, one set per message.
        predicted: Predicted spans, one set per message.
        labels: Declared entity types.

    Returns:
        The entity-level scores of the fitted tagger on its own training messages.
    """
    from src.training.metrics import entity_scores

    return entity_scores(gold, predicted, labels=labels)


__all__ = [
    "DEFAULT_LANGUAGE",
    "GAZETTEER",
    "HYBRID",
    "PIPELINE_DIR",
    "RULES_FILE",
    "TAGGER",
    "TRAINING_FILE",
    "SpacyEntityTagger",
]
