"""Classifieur de texte spaCy entraîné sur le corpus : le ``textcat`` d'un pipeline vierge.

Le pipeline part de ``spacy.blank(langue)`` — le tokeniseur à règles de la langue, sans aucun poids
téléchargé — et lui ajoute un composant ``textcat`` à classes exclusives (une seule catégorie par
ticket, sortie softmax). Trois architectures natives de spaCy, servies par la même boucle :

* ``textcat_bow`` — ``spacy.TextCatBOW`` : un sac de n-grammes **haché** et une couche linéaire.
  C'est l'analogue spaCy de la variante TF-IDF : pas de contexte, pas de vecteur de mot, et
  l'apprentissage le plus rapide de la stack ;
* ``textcat_reduce`` — ``spacy.TextCatReduce`` : un encodeur de jetons (plongements hachés sur la
  forme, le préfixe, le suffixe et la forme normalisée, puis une convolution) dont les vecteurs
  sont réduits par moyenne et maximum. Les sous-mots rendent le modèle tolérant aux fautes et aux
  flexions qu'un vocabulaire exact ne connaît pas ;
* ``textcat_ensemble`` — ``spacy.TextCatEnsemble`` : l'architecture par défaut de spaCy, qui
  combine les deux précédentes (le sac de n-grammes et l'encodeur avec attention).

Ce que la stack ne fait pas, et qu'elle publie : un pipeline vierge n'a aucun vecteur de mot
pré-entraîné, donc aucune connaissance des synonymes au-delà de ce que 719 tickets d'entraînement
lui apprennent. Le segment ``paraphrase`` du corpus mesure cette limite.

L'artefact persiste la configuration complète du pipeline (``nlp.config``) et ses poids
(``nlp.to_bytes()``) dans le même fichier joblib que les autres stacks : un artefact rechargé classe
exactement comme le modèle ajusté, et un test de round-trip le vérifie.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import spacy
from spacy.language import Language
from spacy.training import Example
from spacy.util import fix_random_seed, minibatch
from thinc.api import Config

from src.models.contract import BaseTextClassifier
from src.training.metrics import classification_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Identifiant de la stack, archivé dans la fiche de modèle.
FRAMEWORK = "spacy_textcat"

#: Nom de fichier par défaut quand l'artefact est un dossier.
DEFAULT_MODEL_FILE = "classifier.joblib"

#: Nombre maximal de termes testés par occlusion pour expliquer une décision.
MAX_OCCLUSION_TERMS = 24

#: Métriques d'entraînement publiées dans le résultat d'ajustement.
_TRAIN_METRIC_KEYS = {"accuracy", "macro_f1", "balanced_accuracy"}

#: Composant du pipeline qui porte la classification.
_COMPONENT = "textcat"


def _bow(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``TextCatBOW`` architecture block (hashed bag of n-grams, linear layer)."""
    return {
        "@architectures": "spacy.TextCatBOW.v3",
        "exclusive_classes": True,
        "ngram_size": int(params.get("ngram_size", 2)),
        "no_output_layer": False,
        "length": int(params.get("length", 262144)),
    }


def _tok2vec(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return a token encoder: hashed embeddings on four attributes, then a convolution."""
    width = int(params.get("width", 64))
    rows = int(params.get("embed_rows", 2000))
    return {
        "@architectures": "spacy.Tok2Vec.v2",
        "embed": {
            "@architectures": "spacy.MultiHashEmbed.v2",
            "width": width,
            "rows": [rows, rows, rows // 2, rows // 2],
            "attrs": ["NORM", "LOWER", "PREFIX", "SUFFIX"],
            "include_static_vectors": False,
        },
        "encode": {
            "@architectures": "spacy.MaxoutWindowEncoder.v2",
            "width": width,
            "window_size": int(params.get("window_size", 1)),
            "maxout_pieces": 3,
            "depth": int(params.get("depth", 2)),
        },
    }


def _reduce(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``TextCatReduce`` block (token encoder, mean and max pooling)."""
    return {
        "@architectures": "spacy.TextCatReduce.v1",
        "exclusive_classes": True,
        "use_reduce_first": False,
        "use_reduce_last": False,
        "use_reduce_max": True,
        "use_reduce_mean": True,
        "tok2vec": _tok2vec(params),
    }


def _ensemble(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``TextCatEnsemble`` block (bag of n-grams combined with the token encoder)."""
    return {
        "@architectures": "spacy.TextCatEnsemble.v2",
        "tok2vec": _tok2vec(params),
        "linear_model": _bow(params),
    }


#: Construction du bloc d'architecture, par famille d'algorithme.
ARCHITECTURES = {"bow": _bow, "reduce": _reduce, "ensemble": _ensemble}


def architecture_config(family: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Return the spaCy architecture block of an algorithm family.

    Args:
        family: ``bow``, ``reduce`` or ``ensemble``.
        params: Architecture parameters (``ngram_size``, ``width``, ``depth``...).

    Returns:
        The ``model`` block handed to ``nlp.add_pipe("textcat", config=...)``.

    Raises:
        ValueError: When the family is unknown.
    """
    if family not in ARCHITECTURES:
        msg = f"Unknown spaCy textcat architecture '{family}'. Expected {sorted(ARCHITECTURES)}"
        raise ValueError(msg)
    return ARCHITECTURES[family](params)


class SpacyTextClassifier(BaseTextClassifier):
    """A blank spaCy pipeline with an exclusive ``textcat`` component, trained on the corpus.

    Attributes:
        framework: Stack identifier archived in the model card.
        default_model_file: Artefact file name used when the destination is a directory.
    """

    framework = FRAMEWORK
    default_model_file = DEFAULT_MODEL_FILE

    def __init__(self, **kwargs: Any) -> None:
        """Build the classifier (see :class:`~src.models.contract.BaseTextClassifier`)."""
        super().__init__(**kwargs)
        self._nlp: Language | None = None
        self._classes: list[str] = []
        self._facts: dict[str, float] = {}

    # ------------------------------------------------------------------ contrat ----------
    @property
    def n_features(self) -> int:
        """Number of trainable parameters of the ``textcat`` model (0 before training)."""
        return int(self._facts.get("parameters", 0))

    @property
    def labels(self) -> list[str]:
        """Known labels, in the order of the probability columns."""
        return list(self._classes or self._labels)

    @property
    def nlp(self) -> Language:
        """The trained spaCy pipeline.

        Raises:
            RuntimeError: When the classifier has not been trained or reloaded.
        """
        if self._nlp is None:
            msg = "SpacyTextClassifier has no pipeline: call fit(documents) or load an artefact"
            raise RuntimeError(msg)
        return self._nlp

    def _fit(
        self,
        documents: pd.DataFrame,
        *,
        callbacks: Sequence[Any] | None = None,
        context: Any = None,
    ) -> dict[str, float]:
        """Train the ``textcat`` component on the training documents.

        Args:
            documents: Training documents (labelled frame).
            callbacks: Project callbacks, fired once per epoch.
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

        # Une seule graine pour spaCy, Thinc, NumPy et `random` : l'initialisation des poids, le
        # dropout et l'ordre des lots en dépendent tous les trois.
        fix_random_seed(int(self.random_state))
        self._classes = sorted(set(labels))
        self._labels = list(self._classes)
        nlp = self._blank_pipeline()
        examples = [
            Example.from_dict(nlp.make_doc(text), {"cats": self._one_hot(label)})
            for text, label in zip(texts, labels, strict=True)
        ]
        optimizer = nlp.initialize(lambda: examples)
        optimizer.learn_rate = self._learning_rate()
        self._nlp = nlp

        dropout = float(self._training_config().get("dropout", 0.2))
        batch_size = self._batch_size()
        rng = random.Random(int(self.random_state))
        metrics: dict[str, float] = {}
        epochs = self._epochs()
        for epoch in range(epochs):
            rng.shuffle(examples)
            losses: dict[str, float] = {}
            batches = 0
            for batch in minibatch(examples, size=batch_size):
                nlp.update(list(batch), sgd=optimizer, drop=dropout, losses=losses)
                batches += 1
            metrics = self._train_metrics(texts, labels)
            logs: dict[str, float] = {
                "loss": float(losses.get(_COMPONENT, 0.0)) / float(batches or 1),
                "lr": float(optimizer.learn_rate),
                "epoch": float(epoch),
                **{f"train_{key}": value for key, value in metrics.items()},
            }
            logger.info(
                "Époque {}/{} | loss={:.4f} | train_macro_f1={:.4f}",
                epoch + 1,
                epochs,
                logs["loss"],
                logs.get("train_macro_f1", float("nan")),
            )
            if context is not None and self._emit(context, callbacks, logs, epoch=epoch):
                logger.info("Arrêt anticipé demandé à l'époque {}", epoch)
                break

        self._facts = {
            "parameters": float(self._count_parameters()),
            "epochs_run": float(epoch + 1) if epochs else 0.0,
        }
        return {
            f"train_{key}": float(value) for key, value in metrics.items() if key in _TRAIN_METRIC_KEYS
        }

    def _predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Score raw texts.

        Args:
            texts: Texts to classify.

        Returns:
            A ``(n_texts, n_classes)`` probability matrix, columns in the order of :attr:`labels`.
        """
        if not texts:
            return np.zeros((0, len(self._classes)), dtype="float64")
        rows = [
            [float(doc.cats.get(label, 0.0)) for label in self._classes]
            for doc in self.nlp.pipe([str(text) for text in texts], batch_size=64)
        ]
        matrix = np.asarray(rows, dtype="float64")
        # Classes exclusives : la sortie est déjà un softmax. La renormalisation protège seulement
        # contre l'arrondi en simple précision de Thinc (les sommes doivent valoir 1 à 1e-9 près).
        totals = matrix.sum(axis=1, keepdims=True)
        return matrix / np.where(totals == 0.0, 1.0, totals)

    def _explain(self, texts: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        """Explain each prediction by occlusion, most influential term first.

        Le terme est retiré du texte, le document amputé est rescanné, et l'influence est la baisse
        de probabilité de la classe prédite : la même définition que la variante transformers, pour
        que les deux explications se comparent.

        Args:
            texts: Texts to explain.
            k: Number of terms per text.

        Returns:
            One list of ``(term, influence)`` pairs per text, sorted by decreasing influence.
        """
        probabilities = self._predict_proba(texts)
        candidates = [self._candidate_terms(str(text)) for text in texts]
        variants: list[str] = []
        spans: list[tuple[int, int]] = []
        for text, terms in zip(texts, candidates, strict=True):
            start = len(variants)
            variants.extend(str(text).replace(term, " ", 1) for term in terms)
            spans.append((start, len(variants)))
        scored_variants = self._predict_proba(variants) if variants else None

        explanations: list[list[tuple[str, float]]] = []
        for position, terms in enumerate(candidates):
            predicted = int(np.argmax(probabilities[position]))
            baseline = float(probabilities[position, predicted])
            start, stop = spans[position]
            scored = [
                (
                    term,
                    baseline
                    - (float(scored_variants[index, predicted]) if scored_variants is not None else 0.0),
                )
                for index, term in zip(range(start, stop), terms, strict=True)
            ]
            scored = sorted(scored, key=lambda item: -abs(item[1]))[: max(int(k), 0)]
            explanations.append([(term, weight) for term, weight in scored if weight != 0.0])
        return explanations

    def _payload(self) -> dict[str, Any]:
        """Return the state persisted with the artefact (pipeline configuration and weights)."""
        return {
            "config": self.nlp.config.to_str(),
            "weights": self.nlp.to_bytes(),
            "classes": list(self._classes),
            "facts": dict(self._facts),
        }

    def _restore(self, payload: Mapping[str, Any]) -> None:
        """Rebuild the pipeline from its configuration, then load its weights."""
        config = Config().from_str(str(payload["config"]))
        language = spacy.util.get_lang_class(str(config["nlp"]["lang"]))
        nlp = language.from_config(config)
        nlp.from_bytes(payload["weights"])
        self._nlp = nlp
        self._classes = [str(label) for label in payload.get("classes", [])]
        self._labels = list(self._classes)
        self._facts = {str(key): float(value) for key, value in dict(payload.get("facts") or {}).items()}

    def _effective_params(self) -> dict[str, Any]:
        """Return the hyper-parameters actually applied."""
        return {
            **dict(self.params),
            "epochs": self._epochs(),
            "batch_size": self._batch_size(),
            "learning_rate": self._learning_rate(),
            "classes": list(self._classes),
        }

    def _extra_metadata(self) -> dict[str, Any]:
        """Return the facts archived next to the fit result."""
        return {
            "n_classes": float(len(self._classes)),
            "representation_size": float(self.n_features),
            "parameters": float(self._facts.get("parameters", 0.0)),
            "epochs_run": float(self._facts.get("epochs_run", 0.0)),
            # Aucun poids par terme lisible : l'explication passe par l'occlusion.
            "is_transparent": 0.0,
        }

    # ------------------------------------------------------------------ interne ----------
    def _blank_pipeline(self) -> Language:
        """Return a blank pipeline of the configured language with an empty ``textcat``."""
        nlp = spacy.blank(str(self.params.get("language", "fr")))
        family = str(self.params.get("architecture", "bow"))
        nlp.add_pipe(
            _COMPONENT,
            config={"model": architecture_config(family, dict(self.params.get("network") or {}))},
        )
        textcat = nlp.get_pipe(_COMPONENT)
        for label in self._classes:
            textcat.add_label(label)  # type: ignore[attr-defined]
        return nlp

    def _one_hot(self, label: str) -> dict[str, float]:
        """Return the ``cats`` annotation of one label (exclusive classes)."""
        return {candidate: float(candidate == label) for candidate in self._classes}

    def _train_metrics(self, texts: Sequence[str], labels: Sequence[str]) -> dict[str, float]:
        """Score the training texts with the metrics of the project."""
        probabilities = self._predict_proba(texts)
        predictions = [self._classes[int(index)] for index in np.argmax(probabilities, axis=1)]
        metrics = classification_metrics(list(labels), predictions, labels=list(self._classes))
        return {key: float(value) for key, value in metrics.items() if key in _TRAIN_METRIC_KEYS}

    def _count_parameters(self) -> int:
        """Count the trainable parameters of the ``textcat`` model."""
        model = self.nlp.get_pipe(_COMPONENT).model  # type: ignore[attr-defined]
        total = 0
        for node in model.walk():
            for name in node.param_names:
                if node.has_param(name):
                    total += int(np.asarray(node.get_param(name)).size)
        return total

    def _training_config(self) -> dict[str, Any]:
        """Return the ``training`` block of the model parameters."""
        return dict(self.params.get("training") or {})

    def _epochs(self) -> int:
        """Number of training epochs (``params.epochs``, then ``train.epochs``, then 10)."""
        if self.params.get("epochs") is not None:
            return max(int(self.params["epochs"]), 1)
        train = dict(self.config.get("train") or {})
        if train.get("epochs") is not None:
            return max(int(train["epochs"]), 1)
        return 10

    def _batch_size(self) -> int:
        """Training batch size (``train.batch_size`` wins, model parameters come second)."""
        train = dict(self.config.get("train") or {})
        if train.get("batch_size") is not None:
            return max(int(train["batch_size"]), 1)
        return max(int(self._training_config().get("batch_size", 16)), 1)

    def _learning_rate(self) -> float:
        """Learning rate: it belongs to the model (``params.training``), then ``train``."""
        if self._training_config().get("learning_rate") is not None:
            return float(self._training_config()["learning_rate"])
        return float(dict(self.config.get("train") or {}).get("learning_rate", 0.001))

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
        return bool(getattr(context, "stopped_early", False))

    def _candidate_terms(self, text: str) -> list[str]:
        """Return the distinct alphabetic tokens of a text (spaCy tokenizer), bounded in number."""
        seen: list[str] = []
        for token in self.nlp.make_doc(text):
            term = token.text
            if len(term) < 2 or not token.is_alpha or term in seen:
                continue
            seen.append(term)
            if len(seen) >= MAX_OCCLUSION_TERMS:
                break
        return seen


__all__ = [
    "ARCHITECTURES",
    "DEFAULT_MODEL_FILE",
    "FRAMEWORK",
    "SpacyTextClassifier",
    "architecture_config",
]
