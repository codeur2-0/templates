"""Entraînement : découpage, métriques, callbacks, et ce que le compte rendu doit contenir.

Le trainer est la pièce où une erreur coûte le plus cher : un split mal lu, une métrique calculée
sur les données d'entraînement, un seuil contractuel oublié, et le projet publie des chiffres
flatteurs. Ces tests vérifient chaque garde sur la **pile réelle** du projet — jamais sur un modèle
factice, qui validerait le test et pas le code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from src.models import build_model
from src.schemas.config import AppConfig
from src.training.callbacks import BaseCallback, CallbackContext
from src.training.metrics import (
    CLASSIFICATION_METRICS,
    classification_metrics,
    confusion_matrix_frame,
    majority_baseline,
    per_class_frame,
)
from src.training.trainer import ClassificationOutcome, ClassificationTrainer


class _SpyCallback(BaseCallback):
    """Callback de test : il enregistre les hooks reçus, sans rien modifier."""

    def __init__(self) -> None:
        """Initialise the hook log."""
        self.hooks: list[str] = []
        self.logs: list[dict[str, float]] = []

    def on_train_begin(self, context: CallbackContext) -> None:
        """Record the beginning of the run."""
        self.hooks.append("begin")

    def on_epoch_end(self, context: CallbackContext) -> None:
        """Record one epoch and the metrics it reported."""
        self.hooks.append("epoch")
        self.logs.append(dict(context.logs))

    def on_train_end(self, context: CallbackContext) -> None:
        """Record the end of the run."""
        self.hooks.append("end")


def _trainer(
    app_config: AppConfig,
    *,
    callbacks: list[BaseCallback] | None = None,
    min_primary_metric: float | None = None,
) -> ClassificationTrainer:
    """Build a trainer wired to the project configuration and to the active stack."""
    return ClassificationTrainer(
        build_model(app_config.model_dump()),
        config=app_config.train.model_dump(),
        text_column="text",
        target_column=str(app_config.data.target or "label"),
        metric_names=app_config.metrics.all_metrics,
        primary_metric=app_config.metrics.primary,
        min_primary_metric=min_primary_metric,
        callbacks=callbacks,
    )


def test_the_outcome_carries_training_and_validation_metrics(
    training_outcome: ClassificationOutcome,
) -> None:
    """Le compte rendu sépare l'entraînement de la validation, et préfixe la validation."""
    metrics = training_outcome.metrics

    assert "val_macro_f1" in metrics, "la métrique principale doit être mesurée sur la validation"
    assert any(name.startswith("train_") for name in metrics)
    assert 0.0 <= metrics["val_macro_f1"] <= 1.0
    assert training_outcome.n_train_documents > 0
    assert training_outcome.n_val_documents > 0


def test_the_validation_metrics_cover_the_contract(
    training_outcome: ClassificationOutcome,
) -> None:
    """Toutes les métriques annoncées sont publiées : aucune n'est calculée « pour plus tard »."""
    missing = [
        name for name in CLASSIFICATION_METRICS if f"val_{name}" not in training_outcome.metrics
    ]

    assert not missing, missing


def test_the_history_holds_one_entry_per_metric(training_outcome: ClassificationOutcome) -> None:
    """L'historique est publié même pour un ajustement en une passe : le schéma ne change pas."""
    history = training_outcome.history

    assert set(history) == set(training_outcome.metrics)
    assert all(len(values) == 1 for values in history.values())


def test_the_per_class_table_has_one_row_per_class(
    training_outcome: ClassificationOutcome, documents: pd.DataFrame
) -> None:
    """La table par classe est ce qui permet de voir une classe sacrifiée."""
    per_class = training_outcome.per_class

    assert len(per_class) == documents["label"].nunique()
    assert {"precision", "recall", "f1", "support"} <= set(per_class.columns)
    assert int(per_class["support"].sum()) == training_outcome.n_val_documents
    assert per_class["f1"].between(0.0, 1.0).all()


def test_the_error_table_is_bounded_and_sorted(
    app_config: AppConfig, documents: pd.DataFrame
) -> None:
    """Les erreurs archivées sont les plus confiantes : celles qu'un humain relit d'abord.

    Pour forcer des erreurs, les libellés sont **mélangés** avant l'ajustement : le modèle apprend
    une frontière qui ne correspond plus aux libellés qu'on lui présente ensuite.
    """
    frame = documents.copy()
    truth = frame["label"].astype(str).to_numpy()
    frame["label"] = np.roll(truth, 7)
    trainer = _trainer(app_config)
    trainer.run(frame)

    _, errors, metrics = trainer.evaluate(frame)

    assert not errors.empty, "un corpus aux libellés décalés doit produire des erreurs"
    assert len(errors) <= 20
    assert {"text", "label", "prediction", "confidence"} <= set(errors.columns)
    assert errors["confidence"].is_monotonic_decreasing
    assert 0.0 <= metrics["val_accuracy"] <= 1.0


def test_the_callbacks_receive_the_documented_hooks(
    app_config: AppConfig, documents: pd.DataFrame
) -> None:
    """Un ajustement déclenche ``begin``, ``epoch`` et ``end``, avec les métriques à chaque fois."""
    spy = _SpyCallback()

    _trainer(app_config, callbacks=[spy]).run(documents)

    # Le contrat est une **séquence**, pas un compte : le dresseur ouvre et ferme le run, et il
    # émet au moins un événement d'époque. Un modèle à époques (encodeur) émet les siens pendant
    # `fit`, un estimateur en un coup laisse le dresseur le faire — les deux sont valides, et le
    # dernier événement porte toujours les métriques de validation, calculées après l'ajustement.
    assert spy.hooks[0] == "begin"
    assert spy.hooks[-1] == "end"
    assert spy.hooks.count("epoch") >= 1
    assert spy.logs and "val_macro_f1" in spy.logs[-1]


def test_the_split_helper_selects_the_rows_of_a_split(
    app_config: AppConfig, documents: pd.DataFrame
) -> None:
    """Le découpage vient du corpus, pas d'un tirage à l'entraînement."""
    train = _trainer(app_config).split(documents, "train")

    assert len(train) == int((documents["split"] == "train").sum())
    assert set(train["split"]) == {"train"}


def test_the_split_helper_uses_a_frame_without_split_entirely(
    app_config: AppConfig, train_split: pd.DataFrame
) -> None:
    """Un corpus mono-split reste légitime pour une expérimentation rapide (et c'est documenté)."""
    frame = train_split.drop(columns=["split"])

    assert len(_trainer(app_config).split(frame, "train")) == len(frame)


def test_the_threshold_warning_is_logged_when_the_contract_is_missed(
    app_config: AppConfig, documents: pd.DataFrame
) -> None:
    """Un seuil contractuel manqué doit être visible dans les logs, pas seulement dans le JSON.

    Le corpus d'exemple est apprenable : pour manquer le contrat, on décale les libellés, comme
    le ferait un jeu de données mal étiqueté — c'est ce cas-là que l'avertissement doit signaler.
    """
    frame = documents.copy()
    frame["label"] = np.roll(frame["label"].astype(str).to_numpy(), 7)

    # Le journal du projet est celui de ``loguru`` (voir ``src.utils.logging``), pas celui de la
    # bibliothèque standard : ``caplog`` ne verrait rien. On branche donc un récepteur temporaire.
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        outcome = _trainer(app_config, min_primary_metric=0.99).run(frame)
    finally:
        logger.remove(sink)

    assert outcome.metrics["val_macro_f1"] < 0.99
    assert any("macro_f1" in message for message in messages)


def test_the_metrics_registry_is_consistent(documents: pd.DataFrame) -> None:
    """Les briques du registre : classes, matrice de confusion, références triviales."""
    truth = documents["label"].astype(str)
    prediction = truth.sample(frac=1.0, random_state=0).reset_index(drop=True)
    labels = sorted(truth.unique())

    metrics = classification_metrics(
        truth, prediction, labels=labels, confidences=np.full(len(truth), 0.5)
    )
    per_class = per_class_frame(truth, prediction, labels=labels)
    confusion = confusion_matrix_frame(truth, prediction, labels=labels)
    baseline = majority_baseline(truth)

    assert set(CLASSIFICATION_METRICS) <= set(metrics)
    assert confusion.to_numpy().sum() == len(documents)
    assert abs(float(per_class["support"].sum()) - len(documents)) < 1e-9
    # Un tirage au sort ne peut pas dépasser la référence « classe majoritaire » de beaucoup.
    assert metrics["accuracy"] < baseline["accuracy"] + 0.2


def test_the_calibration_metric_needs_confidences(documents: pd.DataFrame) -> None:
    """La calibration n'est publiée que si les confiances sont fournies : sinon elle mentirait."""
    labels = documents["label"].astype(str)
    without = classification_metrics(labels, labels)

    confidence = np.full(len(labels), 0.5)
    with_confidences = classification_metrics(labels, labels, confidences=confidence)

    assert "expected_calibration_error" not in without
    assert with_confidences["expected_calibration_error"] == pytest.approx(0.5)
    assert with_confidences["mean_confidence"] == pytest.approx(0.5)
