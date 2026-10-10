"""Classifieur linéaire sur représentation lexicale : la référence interprétable.

Trois algorithmes, une seule représentation (le vocabulaire du split d'entraînement) :

* ``tfidf_logreg`` — régression logistique multinomiale sur les poids TF-IDF. C'est le classifieur
  de référence : il apprend un **poids par terme et par classe**, donc il s'explique terme à terme,
  et il gère nativement plus de deux classes ;
* ``tfidf_nb`` — ``ComplementNB``, un bayésien naïf complémentaire : très rapide, robuste aux
  classes déséquilibrées, et volontairement naïf (il suppose les termes indépendants, ce qui est
  faux) — il sert de témoin « sans pondération apprise » ;
* ``tfidf_centroid`` — le centroïde de chaque classe dans l'espace lexical, sans apprentissage de
  poids : c'est le plancher de la famille, celui qui montre ce qu'une simple moyenne de vecteurs
  obtient déjà.

Ce que cette stack **ne** fait pas, et qu'elle publie : elle ne connaît aucun synonyme. Un ticket
qui parle de « colis jamais reçu » n'a aucun terme en commun avec « la livraison est en retard »,
et le segment ``paraphrase`` du corpus existe précisément pour chiffrer cette limite au lieu de la
supposer. Le vocabulaire est appris sur le train uniquement : un terme absent du train n'a pas de
poids à l'inférence, et le taux de termes hors vocabulaire est archivé dans la fiche de modèle.

Le modèle persiste la représentation **et** l'estimateur : un artefact rechargé classe exactement
comme le modèle ajusté (un test de round-trip le vérifie), y compris les probabilités par classe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB

from src.models.contract import BaseTextClassifier, softmax
from src.preprocessing.transformers import LexicalVectorizer, tokenize
from src.training.metrics import classification_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Suffix appended to the algorithm identifier in the model card.
FRAMEWORK = "tfidf_classifier"


@dataclass(slots=True)
class _CentroidModel:
    """Nearest-centroid classifier over a lexical space, persisted with the artefact.

    Attributes:
        classes: Labels, in the order of the centroid rows.
        centroids: ``(n_classes, n_features)`` matrix of mean term weights.
        temperatures: Softmax temperature applied to the cosine similarities.
    """

    classes: list[str] = field(default_factory=list)
    centroids: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype="float64"))
    temperature: float = 8.0

    def fit(self, matrix: np.ndarray, labels: Sequence[str]) -> _CentroidModel:
        """Compute the centroid of every class.

        Args:
            matrix: ``(n_documents, n_features)`` lexical weights.
            labels: One label per document.

        Returns:
            ``self``.
        """
        self.classes = sorted({str(label) for label in labels})
        centroids = np.zeros((len(self.classes), matrix.shape[1]), dtype="float64")
        truth = np.asarray([str(label) for label in labels])
        for position, label in enumerate(self.classes):
            mask = truth == label
            centroid = np.asarray(matrix[mask].mean(axis=0)).ravel()
            norm = float(np.linalg.norm(centroid)) or 1.0
            centroids[position] = centroid / norm
        self.centroids = centroids
        return self

    def decision_function(self, matrix: np.ndarray) -> np.ndarray:
        """Score every document against every class centroid (cosine similarity).

        Args:
            matrix: ``(n_documents, n_features)`` lexical weights.

        Returns:
            A ``(n_documents, n_classes)`` score matrix.
        """
        rows = np.asarray(matrix, dtype="float64")
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        normalised = rows / np.where(norms == 0.0, 1.0, norms)
        return normalised @ self.centroids.T * float(self.temperature)


class TfidfClassifier(BaseTextClassifier):
    """TF-IDF (or BM25) representation plus a linear classifier.

    Attributes:
        framework: Stack identifier archived in the model card.
    """

    framework = FRAMEWORK

    def __init__(self, **kwargs: Any) -> None:
        """Build the classifier (see :class:`~src.models.contract.BaseTextClassifier`)."""
        super().__init__(**kwargs)
        lexical = dict(self.params.get("lexical") or {})
        lexical.setdefault("mode", "tfidf")
        self._vectorizer = LexicalVectorizer.from_config(lexical)
        self._estimator: Any = None
        self._classes: list[str] = []
        self._oov_terms = 0
        self._known_terms = 0

    # ------------------------------------------------------------------ contrat ----------
    @property
    def n_features(self) -> int:
        """Size of the learned vocabulary."""
        return int(self._vectorizer.vocabulary_size) if self._vectorizer.is_fitted else 0

    @property
    def labels(self) -> list[str]:
        """Known labels, in the order of the probability columns."""
        return list(self._classes or self._labels)

    def _fit(
        self,
        documents: pd.DataFrame,
        *,
        callbacks: Sequence[Any] | None = None,
        context: Any = None,
    ) -> dict[str, float]:
        """Learn the vocabulary and the classifier on the training documents.

        Args:
            documents: Training documents (labelled frame).
            callbacks: Unused: a single-shot estimator has no epoch to report (the project
                trainer fires the hooks).
            context: Unused, same reason.

        Returns:
            Training metrics (accuracy and macro F1 on the training split).

        Raises:
            ValueError: When the label column is missing.
        """
        del callbacks, context
        target = str(self.target_name or "label")
        if target not in documents.columns:
            msg = f"Column '{target}' missing from the training frame: {sorted(documents.columns)}"
            raise ValueError(msg)
        texts = documents[self.text_column].astype(str).tolist()
        labels = documents[target].astype(str).tolist()

        self._vectorizer.fit(texts)
        matrix = self._vectorizer.transform(texts)
        self._classes = sorted(set(labels))
        self._labels = list(self._classes)
        self._estimator = self._build_estimator()
        if isinstance(self._estimator, _CentroidModel):
            self._estimator.fit(matrix, labels)
        else:
            self._estimator.fit(np.asarray(matrix, dtype="float64"), labels)
        self._count_oov_terms(texts)
        predictions = [
            self._classes[int(index)] for index in np.argmax(self._probabilities(texts), axis=1)
        ]
        metrics = classification_metrics(labels, predictions, labels=list(self._classes))
        return {
            f"train_{key}": float(value)
            for key, value in metrics.items()
            if key in {"accuracy", "macro_f1", "balanced_accuracy"}
        }

    def _predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Score raw texts.

        Args:
            texts: Texts to classify.

        Returns:
            A ``(n_texts, n_classes)`` probability matrix.
        """
        if self._estimator is None:
            msg = "TfidfClassifier has no estimator: call fit(documents) or load an artefact"
            raise RuntimeError(msg)
        return self._probabilities(list(texts))

    def _explain(self, texts: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        """Return the terms that pushed each prediction, most influential first.

        Args:
            texts: Texts to explain.
            k: Number of terms per text.

        Returns:
            One list of ``(term, weight)`` pairs per text. A centroid model explains with the
            centroid weights multiplied by the document weights (the same contribution rule, with
            no learned coefficient); a linear model uses its coefficients for the predicted class.
        """
        vocabulary = self._vocabulary()
        matrix = self._vectorizer.transform(list(texts))
        explanations: list[list[tuple[str, float]]] = []
        probabilities = self._probabilities(list(texts))
        for row, weights in enumerate(np.asarray(matrix, dtype="float64")):
            position = int(np.argmax(probabilities[row]))
            coefficients = self._coefficients_for(self._classes[position])
            contributions = weights * coefficients
            indices = np.argsort(-np.abs(contributions))[:k]
            explanations.append(
                [
                    (vocabulary[int(index)], float(contributions[int(index)]))
                    for index in indices
                    if contributions[int(index)] != 0.0
                ]
            )
        return explanations

    def _payload(self) -> dict[str, Any]:
        """Return the state persisted with the artefact."""
        return {
            "vectorizer": self._vectorizer,
            "estimator": self._estimator,
            "classes": list(self._classes),
            "oov_terms": int(self._oov_terms),
            "known_terms": int(self._known_terms),
        }

    def _restore(self, payload: Mapping[str, Any]) -> None:
        """Restore the state archived in an artefact."""
        self._vectorizer = payload.get("vectorizer") or LexicalVectorizer(
            mode=str(self.params.get("lexical", {}).get("mode", "tfidf"))
        )
        self._estimator = payload.get("estimator")
        self._classes = [str(label) for label in payload.get("classes", [])]
        self._labels = list(self._classes)
        self._oov_terms = int(payload.get("oov_terms", 0))
        self._known_terms = int(payload.get("known_terms", 0))

    def _effective_params(self) -> dict[str, Any]:
        """Return the hyper-parameters actually applied."""
        return {
            **dict(self.params),
            "vocabulary_size": self.n_features,
            "classes": list(self._classes),
        }

    def _extra_metadata(self) -> dict[str, Any]:
        """Return the facts archived next to the fit result."""
        total = self._oov_terms + self._known_terms
        return {
            "vocabulary_size": float(self.n_features),
            "n_classes": float(len(self._classes)),
            # Un terme absent du train n'a aucun poids : le taux de termes hors vocabulaire est la
            # première explication d'un rappel faible sur une classe rare.
            "oov_term_rate": float(self._oov_terms) / float(total) if total else 0.0,
            "is_transparent": 1.0,
        }

    # ------------------------------------------------------------------ interne ----------
    def _build_estimator(self) -> Any:
        """Instantiate the estimator declared by the algorithm.

        Returns:
            A scikit-learn estimator, or the nearest-centroid model of this module.

        Raises:
            ValueError: When the algorithm has no estimator here.
        """
        params = dict(self.params.get("estimator") or {})
        if self.algorithm in {"tfidf_centroid", "centroid"}:
            return _CentroidModel(
                temperature=float(params.get("temperature", 8.0)),
            )
        seed = int(self.random_state)
        if self.algorithm in {"tfidf_nb", "complement_nb", "naive_bayes"}:
            return ComplementNB(alpha=float(params.get("alpha", 0.3)))
        if self.algorithm in {"tfidf_logreg", "logistic_regression"}:
            return LogisticRegression(
                C=float(params.get("C", 4.0)),
                max_iter=int(params.get("max_iter", 400)),
                class_weight=params.get("class_weight", "balanced"),
                random_state=seed,
            )
        msg = (
            f"Unknown algorithm '{self.algorithm}' for the linear text classifier: "
            "expected one of ['tfidf_centroid', 'tfidf_logreg', 'tfidf_nb']"
        )
        raise ValueError(msg)

    def _probabilities(self, texts: Sequence[str]) -> np.ndarray:
        """Score texts with the fitted estimator, in the order of :attr:`labels`."""
        matrix = np.asarray(self._vectorizer.transform(list(texts)), dtype="float64")
        if isinstance(self._estimator, _CentroidModel):
            return softmax(self._estimator.decision_function(matrix))
        probabilities = np.asarray(self._estimator.predict_proba(matrix), dtype="float64")
        return probabilities

    def _coefficients_for(self, label: str) -> np.ndarray:
        """Return the per-term coefficients that decide one class.

        Args:
            label: Class label.

        Returns:
            A ``(n_features,)`` vector of coefficients.
        """
        if isinstance(self._estimator, _CentroidModel):
            position = self._estimator.classes.index(label)
            return np.asarray(self._estimator.centroids[position], dtype="float64")
        estimator_classes = [str(value) for value in self._estimator.classes_]
        position = estimator_classes.index(label)
        coefficients = np.asarray(self._estimator.coef_, dtype="float64")
        if coefficients.ndim == 1:  # binary case: one row of coefficients
            return (
                coefficients
                if len(estimator_classes) == 1
                else coefficients * (1 if position else -1)
            )
        return coefficients[position]

    def _vocabulary(self) -> list[str]:
        """Return the terms of the learned vocabulary, indexed by column."""
        names = self._vectorizer._vectorizer.get_feature_names_out()
        return [str(name) for name in names]

    def _count_oov_terms(self, texts: Sequence[str]) -> None:
        """Count the terms of the corpus that the vocabulary does not know."""
        vocabulary = set(self._vocabulary())
        unknown = 0
        known = 0
        for text in texts:
            for term in tokenize(text):
                if term in vocabulary:
                    known += 1
                else:
                    unknown += 1
        self._known_terms = known
        self._oov_terms = unknown


__all__ = ["FRAMEWORK", "TfidfClassifier"]
