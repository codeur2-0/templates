"""Keras implementation of the model contract (functional API).

Where the TensorFlow stack of the same business case writes the optimisation loop by hand, this
module **declares** the graph and lets Keras drive it:

1. ``keras.Input`` → layers → ``keras.Model``: the network is a data structure that can be printed,
   plotted and serialised,
2. ``compile(optimizer, loss, metrics)`` binds the learning recipe to the graph,
3. ``fit`` owns the loop and calls back into :class:`EpochBridge`, which forwards one event per
   epoch to the project's callbacks (history, early stopping, quality thresholds),
4. ``save`` produces a single native ``.keras`` archive (graph + weights + compilation).

The contract, not the framework, is what the rest of the project sees: ``fit`` still returns a
:class:`~src.models.base.FitResult`, ``predict`` / ``predict_proba`` keep their shapes, and the
project's early-stopping decision is honoured through ``stop_training``.

Two points worth reading before the code:

* **probabilities, not logits.** The output layer carries the activation (``sigmoid`` / ``softmax``)
  so that Keras' native metrics (``AUC``, ``BinaryAccuracy``) read the network's output correctly;
  the TensorFlow stack deliberately keeps logits to show the other side of the trade-off.
* **class imbalance goes through ``fit(class_weight=...)``**, the native Keras mechanism, resolved
  from the training labels when the configuration says ``auto``.
"""

from __future__ import annotations

import io
import json
import random
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd

from src.models.base import BaseModel, FitResult, ModelCard, library_versions
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Marqueur écrit dans chaque artefact : permet à :meth:`load` de rejeter un fichier étranger.
CHECKPOINT_KIND = "tabular-project.keras-model/v1"

#: Extension native de Keras 3 (graphe + poids + configuration de compilation).
NATIVE_SUFFIX = ".keras"

#: Nom du membre JSON de l'archive portable (extensions étrangères à la stack).
CONFIG_MEMBER = "config.json"

#: Nom du membre NPZ de l'archive portable (poids sous forme de tableaux NumPy).
WEIGHTS_MEMBER = "weights.npz"

#: Tâches apprises par les têtes de classification.
_CLASSIFICATION = frozenset({"binary", "multiclass"})
#: Tâches apprises par les têtes de régression.
_REGRESSION = frozenset({"regression", "forecasting"})
#: Tâches non supervisées (auto-encodeur : l'erreur de reconstruction sert de score d'anomalie).
_ANOMALY = frozenset({"anomaly"})

#: Fonctions d'activation acceptées par le paramètre ``activation`` (couches cachées).
_ACTIVATIONS = frozenset({"relu", "gelu", "tanh", "elu", "selu", "silu", "linear"})

#: Optimiseurs acceptés par le paramètre ``optimizer``.
_OPTIMIZERS = frozenset({"adamw", "adam", "sgd", "rmsprop"})

#: Ordonnanceurs de taux d'apprentissage acceptés par le paramètre ``scheduler``.
_SCHEDULERS = frozenset({"cosine", "plateau", "step", "none"})

#: Traduction du nom de métrique Keras vers le nom du registre du projet.
_METRIC_ALIASES: dict[str, str] = {
    "auc": "roc_auc",
    "binary_accuracy": "accuracy",
    "sparse_categorical_accuracy": "accuracy",
    "mean_squared_error": "mse",
    "mean_absolute_error": "mae",
    "root_mean_squared_error": "rmse",
    "mse": "mse",
    "mae": "mae",
    "rmse": "rmse",
}

#: Hyper-parameters accepted by every architecture of this stack.
_COMMON_PARAMS: frozenset[str] = frozenset(
    {
        "hidden_layers",
        "dropout",
        "activation",
        "batch_norm",
        "optimizer",
        "scheduler",
        "l2",
        "momentum",
        "pos_weight",
        "class_weight",
        "step_size",
        "step_gamma",
        "plateau_factor",
        "plateau_patience",
    }
)

#: Paramètres propres à l'auto-encodeur (détection d'anomalies).
_ANOMALY_PARAMS: frozenset[str] = frozenset({"code_dim", "code_ratio"})

#: Taille des lots utilisés en inférence (borne la mémoire, sans effet sur le résultat).
INFERENCE_BATCH_SIZE = 4096

#: Espace de labels : le contrat le stocke en tableau (comme scikit-learn), les helpers acceptent
#: aussi une simple séquence (artefact relu, appel direct depuis un notebook).
LabelSpace = Sequence[Any] | np.ndarray[Any, Any] | None


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """Declaration of one architecture available in this stack.

    Attributes:
        name: Stable identifier used in the configuration and the notebooks.
        display_name: Human readable name.
        tasks: Learning tasks the architecture can serve.
        builder: Factory receiving ``input_dim``, ``output_dim`` and the resolved parameters.
        rationale: Why (and when) to pick this architecture.
        iterative: Whether the estimator trains epoch by epoch (always true here).
        defaults: Parameters added unless the configuration overrides them.
        accepted: Hyper-parameter allow-list for this architecture.
    """

    name: str
    display_name: str
    tasks: frozenset[str]
    builder: Callable[..., keras.Model]
    rationale: str = ""
    iterative: bool = True
    defaults: dict[str, Any] = field(default_factory=dict)
    accepted: frozenset[str] = _COMMON_PARAMS


# ---------------------------------------------------------------------------------------------
# Construction du graphe (API fonctionnelle)
# ---------------------------------------------------------------------------------------------
def _hidden_layers(params: Mapping[str, Any]) -> list[int]:
    """Read the hidden widths, rejecting empty or non-positive values.

    Args:
        params: Resolved parameters.

    Returns:
        The list of hidden layer widths.

    Raises:
        ValueError: When the declaration is empty or holds a non-positive width.
    """
    raw = params.get("hidden_layers") or [64, 32]
    if isinstance(raw, (int, float)):
        raw = [raw]
    widths = [int(units) for units in raw]
    if not widths or min(widths) < 1:
        msg = f"hidden_layers must hold positive widths, got {widths}"
        raise ValueError(msg)
    return widths


def _code_dim(params: Mapping[str, Any], input_dim: int) -> int:
    """Resolve the latent width of an autoencoder.

    ``code_dim`` is stored in the artefact so that reloading rebuilds the exact same graph;
    ``code_ratio`` only derives it when no explicit width is configured.

    Args:
        params: Resolved parameters.
        input_dim: Number of input features.

    Returns:
        The latent width (at least 2).
    """
    explicit = int(params.get("code_dim") or 0)
    if explicit > 0:
        return max(explicit, 1)
    ratio = float(params.get("code_ratio", 0.25) or 0.25)
    return max(2, round(int(input_dim) * ratio))


def _regularizer(params: Mapping[str, Any]) -> Any:
    """Return the L2 kernel regularizer declared in the parameters (``None`` when disabled)."""
    l2 = _numeric(params.get("l2"), 0.0)
    return keras.regularizers.L2(l2) if l2 > 0 else None


def _dense_block(
    hidden: Any,
    units: int,
    *,
    params: Mapping[str, Any],
    prefix: str,
) -> Any:
    """Apply ``Dense → [BatchNormalization] → activation → Dropout`` to a symbolic tensor.

    Args:
        hidden: Symbolic tensor coming out of the previous layer.
        units: Number of output units.
        params: Resolved parameters.
        prefix: Stable name prefix of the layers.

    Returns:
        The transformed symbolic tensor.
    """
    hidden = keras.layers.Dense(
        units, kernel_regularizer=_regularizer(params), name=f"{prefix}_dense"
    )(hidden)
    if bool(params.get("batch_norm", False)):
        # BatchNormalization stabilise l'apprentissage sur des features hétérogènes.
        hidden = keras.layers.BatchNormalization(name=f"{prefix}_norm")(hidden)
    hidden = keras.layers.Activation(
        str(params.get("activation", "relu")), name=f"{prefix}_activation"
    )(hidden)
    dropout = _numeric(params.get("dropout"), 0.0)
    if dropout > 0:
        hidden = keras.layers.Dropout(dropout, name=f"{prefix}_dropout")(hidden)
    return hidden


def _output_activation(task: str, algorithm: str) -> str:
    """Return the activation of the output layer.

    Args:
        task: Learning task.
        algorithm: Architecture name (an autoencoder always reconstructs, whatever the task).

    Returns:
        ``sigmoid`` for a binary task, ``softmax`` for multiclass, ``linear`` otherwise.
    """
    if algorithm == "autoencoder" or task in _ANOMALY or task in _REGRESSION:
        return "linear"
    if task == "multiclass":
        return "softmax"
    return "sigmoid"


def _mlp(input_dim: int, output_dim: int, params: Mapping[str, Any], *, task: str) -> keras.Model:
    """Build a plain multi-layer perceptron with the functional API.

    Args:
        input_dim: Number of input features.
        output_dim: Number of outputs.
        params: Resolved parameters.
        task: Learning task (drives the output activation).

    Returns:
        The compiled-ready (untrained) functional model.
    """
    inputs = keras.Input(shape=(input_dim,), name="features")
    hidden = inputs
    for index, units in enumerate(_hidden_layers(params)):
        hidden = _dense_block(hidden, units, params=params, prefix=f"hidden_{index}")
    outputs = keras.layers.Dense(
        output_dim,
        activation=_output_activation(task, "mlp"),
        name="output_dense",
    )(hidden)
    return keras.Model(inputs=inputs, outputs=outputs, name="mlp")


def _linear(
    input_dim: int, output_dim: int, params: Mapping[str, Any], *, task: str
) -> keras.Model:
    """Build a single dense layer (logistic / linear regression).

    Args:
        input_dim: Number of input features.
        output_dim: Number of outputs.
        params: Resolved parameters (unused: a linear model has no hidden width).
        task: Learning task (drives the output activation).

    Returns:
        The untrained functional model.
    """
    del params
    inputs = keras.Input(shape=(input_dim,), name="features")
    outputs = keras.layers.Dense(
        output_dim, activation=_output_activation(task, "linear"), name="output_dense"
    )(inputs)
    return keras.Model(inputs=inputs, outputs=outputs, name="linear")


def _residual_mlp(
    input_dim: int, output_dim: int, params: Mapping[str, Any], *, task: str
) -> keras.Model:
    """Build an MLP whose hidden blocks are wrapped in skip connections.

    The functional API expresses a residual connection with a single ``Add`` layer — no custom
    ``Layer`` subclass needed, which is precisely its advantage over the TensorFlow stack.

    Args:
        input_dim: Number of input features.
        output_dim: Number of outputs.
        params: Resolved parameters.
        task: Learning task (drives the output activation).

    Returns:
        The untrained functional model.
    """
    widths = _hidden_layers(params)
    inputs = keras.Input(shape=(input_dim,), name="features")
    hidden = _dense_block(hidden=inputs, units=widths[0], params=params, prefix="stem")
    for index in range(len(widths)):
        branch = _dense_block(hidden, widths[0], params=params, prefix=f"residual_{index}_first")
        branch = keras.layers.Dense(
            widths[0], kernel_regularizer=_regularizer(params), name=f"residual_{index}_second"
        )(branch)
        # Connexion résiduelle : le bloc apprend une correction de son entrée.
        hidden = keras.layers.Add(name=f"residual_{index}_add")([hidden, branch])
    outputs = keras.layers.Dense(
        output_dim, activation=_output_activation(task, "residual_mlp"), name="output_dense"
    )(hidden)
    return keras.Model(inputs=inputs, outputs=outputs, name="residual_mlp")


def _autoencoder(
    input_dim: int, output_dim: int, params: Mapping[str, Any], *, task: str
) -> keras.Model:
    """Build the symmetric encoder/decoder used for reconstruction-based anomaly detection.

    Args:
        input_dim: Number of input features (also the reconstruction width).
        output_dim: Ignored (the reconstruction width is ``input_dim``).
        params: Resolved parameters.
        task: Learning task.

    Returns:
        The untrained functional model.
    """
    del output_dim, task
    widths = _hidden_layers(params)
    inputs = keras.Input(shape=(input_dim,), name="features")
    hidden = inputs
    for index, units in enumerate(widths):
        hidden = _dense_block(hidden, units, params=params, prefix=f"encoder_{index}")
    code = keras.layers.Dense(_code_dim(params, input_dim), name="code_dense")(hidden)
    decoded = code
    for index, units in enumerate(reversed(widths)):
        decoded = _dense_block(decoded, units, params=params, prefix=f"decoder_{index}")
    outputs = keras.layers.Dense(input_dim, name="reconstruction_dense")(decoded)
    return keras.Model(inputs=inputs, outputs=outputs, name="autoencoder")


#: Registry of every architecture this stack can build.
ESTIMATORS: dict[str, AlgorithmSpec] = {
    spec.name: spec
    for spec in (
        AlgorithmSpec(
            name="mlp",
            display_name="MLP fonctionnel (Dense + Dropout)",
            tasks=_CLASSIFICATION | _REGRESSION | _ANOMALY,
            builder=_mlp,
            rationale=(
                "Graphe déclaré couche par couche puis compilé : `hidden_layers` fixe la capacité, "
                "`dropout` et `l2` la régularisation. L'API fonctionnelle donne un modèle "
                "inspectable (`model.summary()`), sérialisable nativement (.keras) et compatible "
                "avec les callbacks éprouvés de Keras."
            ),
            defaults={
                "hidden_layers": [64, 32],
                "dropout": 0.2,
                "activation": "relu",
                "batch_norm": False,
                "optimizer": "adamw",
                "scheduler": "none",
                "l2": 0.0,
            },
        ),
        AlgorithmSpec(
            name="linear",
            display_name="Modèle linéaire compilé",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_linear,
            rationale=(
                "Une seule couche `Dense` : régression logistique / linéaire entraînée par "
                "Keras. Baseline interprétable (les poids sont les coefficients) et référence de "
                "coût — un réseau qui ne la bat pas n'apporte rien."
            ),
            defaults={
                "dropout": 0.0,
                "batch_norm": False,
                "optimizer": "adamw",
                "scheduler": "none",
                "l2": 0.0,
            },
        ),
        AlgorithmSpec(
            name="residual_mlp",
            display_name="MLP à connexions résiduelles (Add)",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_residual_mlp,
            rationale=(
                "Les connexions résiduelles s'expriment par une couche `Add` : aucune `Layer` "
                "personnalisée n'est nécessaire, c'est l'avantage de l'API fonctionnelle. Le "
                "gradient circule quand on empile les blocs, à comparer au MLP simple à capacité "
                "égale."
            ),
            defaults={
                "hidden_layers": [64, 64],
                "dropout": 0.15,
                "activation": "relu",
                "batch_norm": True,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "l2": 0.0,
            },
        ),
        AlgorithmSpec(
            name="autoencoder",
            display_name="Auto-encodeur (erreur de reconstruction)",
            tasks=_ANOMALY,
            builder=_autoencoder,
            rationale=(
                "Encodeur/décodeur symétrique entraîné à reconstruire ses entrées : l'erreur de "
                "reconstruction sert de score d'anomalie. Non supervisé — aucune étiquette "
                "requise, le goulot (`code_dim`) force une représentation compressée du normal."
            ),
            defaults={
                "hidden_layers": [32, 16],
                "code_ratio": 0.25,
                "dropout": 0.0,
                "activation": "relu",
                "batch_norm": False,
                "optimizer": "adamw",
                "scheduler": "none",
                "l2": 0.0,
            },
            accepted=_COMMON_PARAMS | _ANOMALY_PARAMS,
        ),
    )
}


def filter_params(spec: AlgorithmSpec, params: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the hyper-parameters this architecture accepts.

    Args:
        spec: Registry entry of the architecture.
        params: Candidate parameters.

    Returns:
        The applicable subset (rejected keys are logged, never silently dropped).
    """
    kept = {str(key): value for key, value in params.items() if key in spec.accepted}
    dropped = sorted(set(params) - set(kept))
    if dropped:
        logger.debug("{} : paramètre(s) ignoré(s) {}", spec.name, dropped)
    return kept


def resolve_algorithm(algorithm: str, task: str) -> AlgorithmSpec:
    """Return the registry entry of an architecture, validating the task.

    Args:
        algorithm: Architecture identifier.
        task: Learning task.

    Returns:
        The :class:`AlgorithmSpec`.

    Raises:
        ValueError: When the architecture is unknown or does not serve ``task``.
    """
    spec = ESTIMATORS.get(str(algorithm))
    if spec is None:
        msg = (
            f"Unknown Keras architecture '{algorithm}'. Available: {sorted(ESTIMATORS)} "
            f"(for task '{task}': {available_for_task(task)})"
        )
        raise ValueError(msg)
    if task not in spec.tasks:
        msg = (
            f"Architecture '{algorithm}' does not serve task '{task}' "
            f"(it serves {sorted(spec.tasks)}). Pick one of {available_for_task(task)}."
        )
        raise ValueError(msg)
    return spec


def available_for_task(task: str) -> list[str]:
    """Return the architecture names serving a task, in registry order.

    Args:
        task: Learning task (an empty value returns every architecture).

    Returns:
        The architecture identifiers.
    """
    if not task:
        return sorted(ESTIMATORS)
    return [name for name, spec in ESTIMATORS.items() if task in spec.tasks]


# ---------------------------------------------------------------------------------------------
# Helpers numériques et contractuels
# ---------------------------------------------------------------------------------------------
def _numeric(value: Any, default: float) -> float:
    """Coerce a configuration value to ``float``, falling back on a default.

    Args:
        value: Candidate value (``"auto"`` and non-numeric entries are rejected).
        default: Value used when the candidate is unusable.

    Returns:
        The numeric value.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def seed_keras(seed: int) -> None:
    """Seed every random source Keras depends on (weights initialisation, shuffling).

    Args:
        seed: Reproducibility seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    # `set_random_seed` couvre le backend actif (TensorFlow ici) + Python + NumPy.
    keras.utils.set_random_seed(seed)


def count_parameters(network: keras.Model) -> int:
    """Count the trainable parameters of a network.

    Args:
        network: Network to inspect.

    Returns:
        The number of trainable scalars.
    """
    return int(network.count_params())


def label_list(classes: LabelSpace) -> list[Any]:
    """Return the label space as a plain list (numpy arrays have no safe truth value).

    Args:
        classes: Label space, possibly ``None``.

    Returns:
        The labels as a list.
    """
    if classes is None:
        return []
    return list(classes)


def space_positive(classes: LabelSpace) -> Any:
    """Return the label treated as the positive class of a binary task.

    Args:
        classes: Sorted label space (``None`` falls back to ``{0, 1}``).

    Returns:
        The positive label.
    """
    space = label_list(classes)
    return space[-1] if space else 1


def _label_space(labels: np.ndarray | None) -> np.ndarray | None:
    """Return the sorted label space (``None`` for an unsupervised task).

    Sorting makes the positive class deterministic — index ``-1`` — which the ranking metrics read
    as ``y_proba[:, 1]``.
    """
    if labels is None:
        return None
    return np.asarray(sorted(pd.Series(labels).unique().tolist()), dtype="object")


def _target_statistics(task: str, labels: np.ndarray | None) -> tuple[float | None, float | None]:
    """Mean and standard deviation used to normalise a continuous target.

    Args:
        task: Learning task.
        labels: Training target.

    Returns:
        ``(mean, std)`` for regression tasks, ``(None, None)`` otherwise.
    """
    if task not in _REGRESSION or labels is None or not len(labels):
        return None, None
    values = np.asarray(labels, dtype="float64")
    std = float(values.std())
    return float(values.mean()), std if std > 1e-8 else 1.0


def _encode_labels(
    labels: np.ndarray,
    task: str,
    classes: LabelSpace,
    *,
    target_mean: float | None = None,
    target_std: float | None = None,
) -> np.ndarray:
    """Encode business targets into the numeric form the compiled loss expects.

    Args:
        labels: Business target values.
        task: Learning task.
        classes: Sorted label space (classification).
        target_mean: Mean used to normalise a continuous target.
        target_std: Standard deviation used to normalise a continuous target.

    Returns:
        A numeric vector (``float`` for binary/regression, ``int`` codes for multiclass).

    Raises:
        ValueError: When a multiclass target holds a label absent from ``classes``.
    """
    if task in _REGRESSION:
        values = np.asarray(labels, dtype="float64")
        if target_mean is not None and target_std:
            values = (values - target_mean) / target_std
        return values
    space = label_list(classes)
    if task == "multiclass":
        index = {label: position for position, label in enumerate(space)}
        try:
            return np.asarray([index[label] for label in labels], dtype="int32")
        except KeyError as error:
            msg = f"Target label {error} is absent from the training label space {space}"
            raise ValueError(msg) from error
    return (np.asarray(labels) == space_positive(space)).astype("float32")


def _decode_labels(
    task: str,
    encoded: np.ndarray,
    classes: LabelSpace,
    *,
    target_mean: float | None = None,
    target_std: float | None = None,
) -> np.ndarray:
    """Turn encoded targets back into business values (for the validation metrics).

    Args:
        task: Learning task.
        encoded: Encoded target vector.
        classes: Sorted label space.
        target_mean: Mean used to normalise a continuous target.
        target_std: Standard deviation used to normalise a continuous target.

    Returns:
        The business target values.
    """
    if task in _REGRESSION:
        values = np.asarray(encoded, dtype="float64").ravel()
        if target_mean is not None and target_std:
            values = values * target_std + target_mean
        return values
    space = label_list(classes)
    if task == "multiclass":
        codes = np.asarray(encoded).ravel().astype("int")
        if not space:
            return codes
        return np.asarray([space[min(max(int(code), 0), len(space) - 1)] for code in codes])
    positive = space_positive(space)
    negative = space[0] if space else 0
    flags = np.asarray(encoded).ravel() >= 0.5
    return np.asarray([positive if flag else negative for flag in flags])


def _class_weights(
    params: Mapping[str, Any], task: str, classes: LabelSpace
) -> dict[int, float] | None:
    """Build the ``fit(class_weight=...)`` mapping from the resolved parameters.

    Args:
        params: Resolved parameters (``class_weight`` / ``pos_weight``).
        task: Learning task.
        classes: Sorted label space.

    Returns:
        The class-index → weight mapping, or ``None`` when no weighting applies.
    """
    if task not in _CLASSIFICATION:
        return None
    space = label_list(classes)
    weights = params.get("class_weight")
    if weights:
        return {index: _numeric(value, 1.0) for index, value in enumerate(weights[: len(space)])}
    pos_weight = _numeric(params.get("pos_weight"), 0.0)
    if pos_weight > 0 and task == "binary" and len(space) == 2:
        # `pos_weight` (idiome PyTorch) traduit dans le mécanisme natif de Keras.
        return {0: 1.0, 1: pos_weight}
    return None


def _loss_and_metrics(task: str) -> tuple[Any, list[Any]]:
    """Return the compiled loss and the native metrics matching the task.

    The output layer carries the activation, so the metrics read probabilities (classification) or
    values (regression) directly — no ``from_logits`` subtlety here.

    Args:
        task: Learning task.

    Returns:
        ``(loss, metrics)``.
    """
    if task == "binary":
        return keras.losses.BinaryCrossentropy(), [
            keras.metrics.AUC(name="auc"),
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
        ]
    if task == "multiclass":
        return keras.losses.SparseCategoricalCrossentropy(), [
            keras.metrics.SparseCategoricalAccuracy(name="sparse_categorical_accuracy"),
        ]
    if task in _REGRESSION:
        return keras.losses.MeanSquaredError(), [
            keras.metrics.RootMeanSquaredError(name="root_mean_squared_error"),
            keras.metrics.MeanAbsoluteError(name="mean_absolute_error"),
        ]
    # Auto-encodeur : la cible est l'entrée, l'erreur quadratique moyenne suffit.
    return keras.losses.MeanSquaredError(), [
        keras.metrics.MeanAbsoluteError(name="mean_absolute_error")
    ]


def _learning_rate_schedule(
    params: Mapping[str, Any],
    *,
    learning_rate: float,
    steps_per_epoch: int,
    epochs: int,
) -> Any:
    """Build the learning-rate schedule (counted in optimiser steps, not epochs).

    Args:
        params: Resolved parameters (``scheduler``, ``step_size``, ``step_gamma``).
        learning_rate: Initial step size.
        steps_per_epoch: Number of optimiser steps in one epoch.
        epochs: Total epoch budget.

    Returns:
        A Keras schedule, or the constant learning rate when no schedule applies.
    """
    name = str(params.get("scheduler", "none")).lower()
    total_steps = max(int(steps_per_epoch) * int(epochs), 1)
    if name == "cosine":
        return keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate, decay_steps=total_steps
        )
    if name == "step":
        return keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=max(int(params.get("step_size", 10)) * max(int(steps_per_epoch), 1), 1),
            decay_rate=float(params.get("step_gamma", 0.5)),
            staircase=True,
        )
    # `plateau` est piloté par le callback natif `ReduceLROnPlateau`, `none` garde un taux constant.
    return float(learning_rate)


def _optimizer(params: Mapping[str, Any], learning_rate: Any, weight_decay: float) -> Any:
    """Build the optimizer declared in the parameters.

    Args:
        params: Resolved parameters (``optimizer``, ``momentum``).
        learning_rate: Constant or schedule.
        weight_decay: Decoupled L2 penalty applied by AdamW.

    Returns:
        The Keras optimizer.

    Raises:
        ValueError: When the optimizer name is unknown.
    """
    name = str(params.get("optimizer", "adamw")).lower()
    if name == "adamw":
        return keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "sgd":
        return keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=_numeric(params.get("momentum"), 0.9)
        )
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    msg = f"Unknown optimizer '{name}'. Available: {sorted(_OPTIMIZERS)}"
    raise ValueError(msg)


def _current_learning_rate(optimizer: Any) -> float:
    """Read the learning rate of an optimizer (constant or scheduled)."""
    rate = getattr(optimizer, "learning_rate", None)
    if rate is None:  # pragma: no cover - défensif
        return float("nan")
    if isinstance(rate, keras.optimizers.schedules.LearningRateSchedule):
        return float(rate(optimizer.iterations))
    if callable(getattr(rate, "numpy", None)):
        return float(rate.numpy())
    return float(rate)


def _monitored_metric(train_node: Mapping[str, Any], primary: str) -> tuple[str, str]:
    """Return the monitored log key and its comparison direction.

    The project's monitor names (``val_loss``, ``val_roc_auc``) are translated into the Keras
    callback names (``val_loss``, ``val_auc``) so that the native ``EarlyStopping`` and
    ``ReduceLROnPlateau`` watch the very same quantity as the project callbacks.

    Args:
        train_node: ``train`` configuration node.
        primary: Primary metric of the project (fallback monitor).

    Returns:
        ``(monitor, mode)`` — ``mode`` is ``min`` for a loss, ``max`` for a score.
    """
    stopping = dict(train_node.get("early_stopping") or {})
    monitor = str(stopping.get("monitor") or "val_loss")
    if monitor in {"", "None"}:
        monitor = f"val_{primary}" if primary else "val_loss"
    mode = str(stopping.get("mode") or "")
    if mode not in {"min", "max"}:
        mode = "min" if monitor.endswith("loss") else "max"
    return monitor, mode


def _keras_monitor(monitor: str, task: str) -> str:
    """Translate a project monitor name into the Keras log name.

    Args:
        monitor: Project monitor (``val_loss``, ``val_roc_auc``, ``val_rmse``, ...).
        task: Learning task.

    Returns:
        The key Keras puts in its ``logs`` mapping.
    """
    if not monitor.startswith("val_"):
        return monitor
    name = monitor[len("val_") :]
    if name in {"loss", "mse", "mae"}:
        return monitor
    if task == "binary" and name in {"roc_auc", "pr_auc", "auc"}:
        return "val_auc"
    if task == "multiclass" and name in {"accuracy", "balanced_accuracy"}:
        return "val_sparse_categorical_accuracy"
    if task in _REGRESSION and name == "rmse":
        return "val_root_mean_squared_error"
    return monitor


def _score_metric(
    task: str, metric: str, *, y_true: Any, y_pred: Any, y_proba: Any
) -> float | None:
    """Compute one project metric on the validation split.

    Args:
        task: Learning task.
        metric: Metric name from the project registry.
        y_true: Business ground truth.
        y_pred: Business predictions.
        y_proba: Class probabilities (``None`` when the task has none).

    Returns:
        The metric value, or ``None`` when it is undefined for these inputs.
    """
    if not metric:
        return None
    # Import local : `losses_metrics` importe `src.models.base`, l'inverse créerait un cycle.
    from src.training.losses_metrics import MetricCalculator, MetricInputs

    try:
        calculator = MetricCalculator(task=task, metrics=[metric])
        values = calculator.evaluate(MetricInputs(y_true=y_true, y_pred=y_pred, y_proba=y_proba))
    except ValueError as error:
        logger.debug("Métrique '{}' indisponible pendant l'entraînement : {}", metric, error)
        return None
    value = values.get(metric)
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars into plain Python containers (the artefact config is written as JSON).

    Args:
        value: Value to normalise.

    Returns:
        A JSON-safe value.
    """
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _sidecar_path(destination: Path) -> Path:
    """Return the JSON metadata path accompanying a native ``.keras`` archive.

    Args:
        destination: Path ending with ``.keras``.

    Returns:
        The sibling ``*.config.json`` path.
    """
    return destination.with_name(f"{destination.name[: -len(NATIVE_SUFFIX)]}.config.json")


def _write_portable(destination: Path, config: Mapping[str, Any], network: keras.Model) -> None:
    """Write a self-contained archive (JSON configuration + serialised graph + NumPy weights).

    Used whenever the destination does not carry Keras' native extension — the contract tests save
    to ``model.joblib``, and a single file is easier to move around than an archive/sidecar pair.

    Args:
        destination: Archive path.
        config: Artefact configuration (project contract metadata).
        network: Trained functional model (serialised here with Keras' own saving API).
    """
    weights = network.get_weights()
    payload = dict(config)
    payload["network"] = keras.saving.serialize_keras_object(network)
    buffer = io.BytesIO()
    np.savez(buffer, *[np.asarray(array) for array in weights])
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(CONFIG_MEMBER, json.dumps(_json_safe(payload), indent=2))
        archive.writestr(WEIGHTS_MEMBER, buffer.getvalue())


def _read_portable(source: Path) -> tuple[dict[str, Any], keras.Model, list[np.ndarray]]:
    """Read a portable archive written by :func:`_write_portable`.

    Args:
        source: Archive path.

    Returns:
        The configuration, the rebuilt graph and its ordered weights.

    Raises:
        TypeError: When the archive is not an artefact of this project.
    """
    try:
        with zipfile.ZipFile(source) as archive:
            config = json.loads(archive.read(CONFIG_MEMBER).decode("utf-8"))
            with io.BytesIO(archive.read(WEIGHTS_MEMBER)) as buffer:
                arrays = np.load(buffer)
                weights = [arrays[name] for name in arrays.files]
    except Exception as error:
        msg = (
            f"'{source.name}' is not a Keras artefact of this project ({type(error).__name__}). "
            "Point model_file at the file written by KerasModel.save (.keras + sidecar JSON, or "
            "the portable archive)."
        )
        raise TypeError(msg) from error
    if not isinstance(config, dict) or config.get("kind") != CHECKPOINT_KIND:
        msg = f"'{source.name}' does not hold the '{CHECKPOINT_KIND}' marker."
        raise TypeError(msg)
    network = keras.saving.deserialize_keras_object(config["network"])
    return config, network, weights


def _read_sidecar(source: Path, weights: Path) -> dict[str, Any]:
    """Read the JSON metadata accompanying a native ``.keras`` archive.

    Args:
        source: Path of the sidecar JSON.
        weights: Path of the ``.keras`` archive (used in the diagnostic message).

    Returns:
        The artefact configuration.

    Raises:
        TypeError: When the sidecar is missing or is not an artefact of this project.
    """
    if not source.exists():
        msg = (
            f"Missing metadata for '{weights.name}': expected '{source.name}' next to it. The "
            "native archive holds the graph and the weights, but the project contract (features, "
            "label space, target statistics) lives in the sidecar — save both files together."
        )
        raise TypeError(msg)
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        msg = f"Unreadable model metadata '{source.name}': {error}"
        raise TypeError(msg) from error
    if not isinstance(config, dict) or config.get("kind") != CHECKPOINT_KIND:
        msg = f"'{source.name}' does not hold the '{CHECKPOINT_KIND}' marker."
        raise TypeError(msg)
    return config


class EpochBridge(keras.callbacks.Callback):
    """Bridge between Keras' epoch callbacks and the project's callbacks.

    Keras owns the loop; the project owns its side effects (history, early stopping, quality
    thresholds, progress reporting). This callback translates one Keras epoch into **one** project
    event and honours a stop request through ``model.stop_training`` — the native way to interrupt
    ``fit``.
    """

    def __init__(
        self,
        wrapper: KerasModel,
        context: Any,
        callbacks: Sequence[Any],
        *,
        primary: str,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Store the emission target.

        Args:
            wrapper: Project model (fires the project callbacks through it).
            context: Mutable project callback context.
            callbacks: Project callbacks to notify.
            primary: Primary metric name of the project.
            validation: Validation matrices, used to compute the primary metric when Keras does
                not expose it under the project's name.
        """
        super().__init__()
        self.wrapper = wrapper
        self.context = context
        self.callbacks = list(callbacks)
        self.primary = primary
        self.validation = validation

    def on_epoch_end(self, epoch: int, logs: Mapping[str, Any] | None = None) -> None:
        """Forward one Keras epoch to the project callbacks.

        Args:
            epoch: 0-based epoch index.
            logs: Keras metrics of the epoch (``loss``, ``val_loss``, metric names).
        """
        payload = {
            str(key): _numeric(value, float("nan")) for key, value in dict(logs or {}).items()
        }
        translated: dict[str, float] = {"epoch": float(epoch)}
        for key, value in payload.items():
            if not np.isfinite(value):
                continue
            translated[key] = value
            if key.startswith("val_"):
                alias = _METRIC_ALIASES.get(key[len("val_") :])
                if alias and f"val_{alias}" not in translated:
                    translated[f"val_{alias}"] = value
            else:
                alias = _METRIC_ALIASES.get(key)
                if alias and alias not in translated:
                    translated[alias] = value
        if self.primary and f"val_{self.primary}" not in translated and self.validation is not None:
            # Keras ne connaît pas le nom de la métrique primaire du projet : on la recalcule sur
            # le split de validation pour que l'early stopping projet voie la même grandeur.
            metric_value = self.wrapper.score_primary(
                self.validation[0], self.validation[1], self.primary, self.model
            )
            if metric_value is not None:
                translated[f"val_{self.primary}"] = metric_value
        stopped = self.wrapper.emit_epoch(self.context, self.callbacks, translated, epoch=epoch)
        if stopped and self.model is not None:
            # Mécanisme natif d'interruption : Keras sort proprement de la boucle d'époques.
            self.model.stop_training = True


class KerasModel(BaseModel):
    """A functional Keras network behind the project's model contract."""

    framework = "keras"
    default_model_file = "model.keras"
    epochs_based = True

    def __init__(
        self,
        *,
        algorithm: str = "mlp",
        params: Mapping[str, Any] | None = None,
        task: str = "binary",
        feature_names: Sequence[str] | None = None,
        target_name: str | None = None,
        random_state: int = 42,
        supports_proba: bool | None = None,
        config: Mapping[str, Any] | None = None,
        name: str | None = None,
    ) -> None:
        """Store the contract and resolve the architecture.

        Args:
            algorithm: Identifier present in :data:`ESTIMATORS`.
            params: Hyper-parameters (filtered against the architecture's allow-list).
            task: Learning task.
            feature_names: Features expected at inference time.
            target_name: Business target column.
            random_state: Reproducibility seed.
            supports_proba: Explicit probability support (``None`` = derived from the task).
            config: Full application configuration.
            name: Human readable name.

        Raises:
            ValueError: When the architecture is unknown or does not serve the task.
        """
        super().__init__(
            algorithm=algorithm,
            params=params,
            task=task,
            feature_names=feature_names,
            target_name=target_name,
            random_state=random_state,
            supports_proba=supports_proba,
            config=config,
            name=name,
        )
        self.spec = resolve_algorithm(algorithm, task)
        self.network_: keras.Model | None = None
        self.input_dim_: int = 0
        self.output_dim_: int = 0
        self.best_epoch_: int | None = None
        self.epochs_trained_: int = 0
        self.history_: dict[str, list[float]] = {}
        self._target_mean_: float | None = None
        self._target_std_: float | None = None

    # ------------------------------------------------------------------ construction ------
    def _resolved_params(self, labels: np.ndarray | None) -> dict[str, Any]:
        """Merge the registry defaults, the configuration and the data-driven ``auto`` values.

        Args:
            labels: Training target (used to resolve the imbalance weights).

        Returns:
            The parameters used to build and compile the network.

        Raises:
            ValueError: When the activation, the optimizer or the scheduler is unknown.
        """
        resolved: dict[str, Any] = {**self.spec.defaults, **self._effective_params()}
        activation = str(resolved.get("activation", "relu")).lower()
        if activation not in _ACTIVATIONS:
            msg = f"Unknown activation '{activation}'. Available: {sorted(_ACTIVATIONS)}"
            raise ValueError(msg)
        resolved["activation"] = activation
        if str(resolved.get("optimizer", "adamw")).lower() not in _OPTIMIZERS:
            available = sorted(_OPTIMIZERS)
            msg = f"Unknown optimizer '{resolved.get('optimizer')}'. Available: {available}"
            raise ValueError(msg)
        if str(resolved.get("scheduler", "none")).lower() not in _SCHEDULERS:
            available = sorted(_SCHEDULERS)
            msg = f"Unknown scheduler '{resolved.get('scheduler')}'. Available: {available}"
            raise ValueError(msg)
        if labels is not None and self.task in _CLASSIFICATION:
            counts = pd.Series(labels).value_counts().sort_index().to_numpy(dtype="float64")
            positive = float(counts[-1]) if counts.size else 0.0
            negative = float(counts[:-1].sum()) if counts.size > 1 else 0.0
            if str(resolved.get("class_weight")) == "auto":
                total = float(counts.sum()) or 1.0
                resolved["class_weight"] = [
                    round(total / (len(counts) * max(float(count), 1.0)), 4) for count in counts
                ]
            if str(resolved.get("pos_weight")) == "auto":
                resolved["pos_weight"] = round(max(negative / max(positive, 1.0), 1.0), 4)
        cleaned = {key: value for key, value in resolved.items() if value != "auto"}
        if self.spec.name == "autoencoder":
            # Le goulot dépend de la largeur d'entrée : figé ici pour que l'artefact reconstruise
            # exactement le même graphe.
            cleaned.setdefault("code_dim", _code_dim(cleaned, len(self.feature_names) or 1))
        return filter_params(self.spec, cleaned)

    def build_network(
        self, input_dim: int, output_dim: int, labels: np.ndarray | None = None
    ) -> keras.Model:
        """Instantiate the network declared by ``algorithm`` (used by the notebooks).

        Args:
            input_dim: Number of input features.
            output_dim: Number of outputs.
            labels: Training target, used to resolve the imbalance weights.

        Returns:
            A fresh (untrained) functional model.
        """
        params = self._resolved_params(labels)
        if self.spec.name == "autoencoder":
            params["code_dim"] = _code_dim(params, input_dim)
        return self.spec.builder(input_dim, output_dim, params, task=self.task)

    def _output_dim(self, input_dim: int, labels: np.ndarray | None) -> int:
        """Number of network outputs for the current task.

        Args:
            input_dim: Number of input features.
            labels: Training target.

        Returns:
            The output width (1 for binary/regression, ``n_classes`` for multiclass).
        """
        if self.spec.name == "autoencoder" or self.task in _ANOMALY:
            return int(input_dim)
        if self.task == "multiclass" and labels is not None:
            return max(int(pd.Series(labels).nunique()), 2)
        return 1

    # ------------------------------------------------------------------ entraînement ------
    def _fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None,
        *,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        context: Any = None,
        callbacks: Sequence[Any] | None = None,
    ) -> None:
        """Declare the graph, compile it and let ``fit`` run the epochs.

        Args:
            X: Aligned training features.
            y: Training target (``None`` for the autoencoder).
            X_val: Validation features (enables ``val_loss`` and the native callbacks).
            y_val: Validation target.
            context: Project callback context.
            callbacks: Project callbacks.

        Raises:
            ValueError: When the task is supervised but no target is provided.
        """
        self._seed_everything()
        seed_keras(self.random_state)

        matrix = self._matrix(X).astype("float32")
        labels = None if y is None else np.asarray(y).ravel()
        if labels is None and self.is_supervised:
            msg = f"Task '{self.task}' requires a target; got y=None"
            raise ValueError(msg)

        params = self._resolved_params(labels)
        if self.spec.name == "autoencoder":
            params["code_dim"] = _code_dim(params, int(matrix.shape[1]))
        self.classes_ = _label_space(labels)
        self._target_mean_, self._target_std_ = _target_statistics(self.task, labels)

        input_dim = int(matrix.shape[1])
        output_dim = self._output_dim(input_dim, labels)
        network = self.spec.builder(
            input_dim=input_dim, output_dim=output_dim, params=params, task=self.task
        )
        train_targets = self._encode_targets(labels, matrix)

        batch_size = self._batch_size()
        epochs = self._epochs()
        steps_per_epoch = max(int(np.ceil(len(matrix) / max(batch_size, 1))), 1)
        loss, metrics = _loss_and_metrics(self.task)
        schedule = _learning_rate_schedule(
            params,
            learning_rate=self._learning_rate(),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
        optimizer = _optimizer(params, schedule, self._weight_decay())
        network.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        validation: tuple[np.ndarray, np.ndarray] | None = None
        val_matrices: tuple[np.ndarray, np.ndarray] | None = None
        if X_val is not None:
            val_matrix = self._matrix(X_val).astype("float32")
            val_labels = None if y_val is None else np.asarray(y_val).ravel()
            validation = (val_matrix, self._encode_targets(val_labels, val_matrix))
            if val_labels is not None:
                val_matrices = (val_matrix, val_labels)

        monitor, mode = _monitored_metric(self._train_node(), self._primary_metric())
        keras_callbacks: list[Any] = [keras.callbacks.TerminateOnNaN()]
        stopping = dict(self._train_node().get("early_stopping") or {})
        if validation is not None and bool(stopping.get("enabled", True)):
            keras_callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor=_keras_monitor(monitor, self.task),
                    patience=int(stopping.get("patience", 5)),
                    min_delta=float(stopping.get("min_delta", 0.0)),
                    mode="max" if mode == "max" else "min",
                    # Restaure l'instantané des meilleurs poids (pas seulement un arrêt de boucle).
                    restore_best_weights=bool(stopping.get("restore_best", True)),
                    verbose=0,
                )
            )
            if str(params.get("scheduler", "none")).lower() == "plateau":
                keras_callbacks.append(
                    keras.callbacks.ReduceLROnPlateau(
                        monitor=_keras_monitor(monitor, self.task),
                        factor=_numeric(params.get("plateau_factor"), 0.5),
                        patience=int(params.get("plateau_patience", 5)),
                        mode="max" if mode == "max" else "min",
                        verbose=0,
                    )
                )
        if context is not None:
            keras_callbacks.append(
                EpochBridge(
                    self,
                    context,
                    list(callbacks or []),
                    primary=self._primary_metric(),
                    validation=val_matrices,
                )
            )

        history = network.fit(
            x=matrix,
            y=train_targets,
            validation_data=validation,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=True,
            class_weight=_class_weights(params, self.task, self.classes_),
            callbacks=keras_callbacks,
        )
        self.history_ = {
            str(key): [float(value) for value in values] for key, values in history.history.items()
        }
        self.network_ = network
        self.params = dict(params)
        self.input_dim_, self.output_dim_ = input_dim, output_dim
        self.epochs_trained_ = len(self.history_.get("loss", [])) or epochs
        self.best_epoch_ = _best_epoch(self.history_, monitor, mode)
        if context is not None:
            context.extra.update(
                {
                    "best_epoch": self.best_epoch_,
                    "epochs_trained": self.epochs_trained_,
                    "n_parameters": count_parameters(network),
                    "architecture": self.spec.name,
                    "steps_per_epoch": steps_per_epoch,
                    "keras_history": sorted(self.history_),
                }
            )
        logger.info(
            "Keras fitted | {} époque(s) | meilleure={} | paramètres={} | clés={}",
            self.epochs_trained_,
            self.best_epoch_,
            count_parameters(network),
            sorted(self.history_)[:6],
        )

    def _encode_targets(self, labels: np.ndarray | None, matrix: np.ndarray) -> np.ndarray:
        """Encode a target vector, or reuse the features for a reconstruction task.

        Args:
            labels: Business targets (``None`` for an autoencoder).
            matrix: Feature matrix (used as the reconstruction target).

        Returns:
            The array handed to ``fit``.
        """
        if labels is None or self.task in _ANOMALY or self.spec.name == "autoencoder":
            return np.ascontiguousarray(matrix, dtype="float32")
        encoded = _encode_labels(
            labels,
            self.task,
            self.classes_,
            target_mean=self._target_mean_,
            target_std=self._target_std_,
        )
        if self.task == "multiclass":
            return np.ascontiguousarray(encoded, dtype="int32")
        return np.ascontiguousarray(encoded, dtype="float32").reshape(-1, 1)

    def score_primary(
        self,
        matrix: np.ndarray,
        labels: np.ndarray,
        metric: str,
        network: keras.Model | None = None,
    ) -> float | None:
        """Score the project's primary metric on a validation split (used by the epoch bridge).

        Args:
            matrix: Raw (pre-encoded) validation features.
            labels: Business validation targets.
            metric: Primary metric name.
            network: Network being trained (defaults to the fitted one).

        Returns:
            The metric value, or ``None`` when it cannot be computed.
        """
        outputs = self._forward_matrix(matrix.astype("float32"), network)
        return _score_metric(
            self.task,
            metric,
            y_true=labels,
            y_pred=self._decode(outputs),
            y_proba=self._probabilities(outputs),
        )

    # ------------------------------------------------------------------ prédiction --------
    def _predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict one value per row (labels, values or reconstruction errors).

        Args:
            X: Aligned features.

        Returns:
            The predictions.
        """
        matrix = self._matrix(X)
        outputs = self._forward_matrix(matrix.astype("float32"))
        if self.spec.name == "autoencoder" or self.task in _ANOMALY:
            # Score d'anomalie : erreur quadratique moyenne par ligne (plus élevée = plus anormal).
            return np.mean(np.square(matrix - outputs), axis=1)
        return self._decode(outputs)

    def _predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Aligned features.

        Returns:
            A ``(n_samples, n_classes)`` array — ``(n_samples, 2)`` for a binary task.

        Raises:
            NotImplementedError: When the task exposes no probabilities.
        """
        probabilities = self._probabilities(self._forward_matrix(self._matrix(X).astype("float32")))
        if probabilities is None:
            msg = (
                f"Task '{self.task}' with architecture '{self.algorithm}' exposes no "
                "probabilities; use predict()."
            )
            raise NotImplementedError(msg)
        return probabilities

    def _forward_matrix(self, matrix: np.ndarray, network: keras.Model | None = None) -> np.ndarray:
        """Run the network over a matrix, batch by batch, in inference mode.

        Args:
            matrix: Aligned float matrix.
            network: Network to run (defaults to the fitted one; the epoch bridge passes the
                network currently being trained, which is not attached to ``self`` yet).

        Returns:
            The network outputs (probabilities, values or reconstructions).
        """
        network = network if network is not None else self._require_network()
        outputs: list[np.ndarray] = []
        for start in range(0, len(matrix), INFERENCE_BATCH_SIZE):
            chunk = np.ascontiguousarray(matrix[start : start + INFERENCE_BATCH_SIZE])
            outputs.append(np.asarray(network(chunk, training=False)))
        if not outputs:
            return np.zeros((0, max(self.output_dim_, 1)), dtype="float64")
        return np.concatenate(outputs, axis=0)

    def _decode(self, outputs: np.ndarray) -> np.ndarray:
        """Turn network outputs into business predictions.

        Args:
            outputs: Probabilities (classification) or values (regression / reconstruction).

        Returns:
            Class labels for a classification task, target values otherwise.
        """
        array = np.asarray(outputs, dtype="float64")
        if self.task == "multiclass":
            codes = np.argmax(array, axis=1)
            space = self._label_list()
            if not space:
                return codes.astype("int64")
            return np.asarray([space[min(int(code), len(space) - 1)] for code in codes])
        if self.task == "binary":
            # La couche de sortie porte le sigmoid : `outputs` est déjà une probabilité.
            scores = array.ravel()
            space = self._label_list() or [0, 1]
            positive, negative = space[-1], space[0]
            threshold = self._decision_threshold()
            return np.asarray([positive if score >= threshold else negative for score in scores])
        values = array.ravel()
        if self._target_mean_ is not None and self._target_std_:
            # Dé-normalisation : les prédictions retrouvent l'unité métier de la cible.
            values = values * float(self._target_std_) + float(self._target_mean_)
        return values

    def _probabilities(self, outputs: np.ndarray) -> np.ndarray | None:
        """Turn network outputs into class probabilities (``None`` when the task has none).

        Args:
            outputs: Raw network outputs.

        Returns:
            A ``(n_samples, n_classes)`` array for classification, ``None`` otherwise.
        """
        array = np.asarray(outputs, dtype="float64")
        if self.task == "binary":
            positive = array.ravel()
            return np.column_stack([1.0 - positive, positive])
        if self.task == "multiclass":
            return array if array.ndim > 1 else array.reshape(-1, 1)
        return None

    # ------------------------------------------------------------------ persistance -------
    def save(self, path: str | Path) -> Path:
        """Persist the model: native ``.keras`` archive + sidecar JSON, or a portable archive.

        Args:
            path: Destination file or directory. A name ending with ``.keras`` uses Keras' native
                format (graph + weights + compilation, plus a ``*.config.json`` sidecar holding the
                project contract); any other extension produces a self-contained archive.

        Returns:
            The written path.
        """
        self.check_is_fitted()
        network = self._require_network()
        destination = self._resolve_path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        config = self._artefact_config()
        if destination.name.endswith(NATIVE_SUFFIX):
            network.save(destination)
            sidecar = _sidecar_path(destination)
            sidecar.write_text(json.dumps(_json_safe(config), indent=2), encoding="utf-8")
            logger.info(
                "Model saved: {} (+ {}) | {} | {} paramètres",
                destination,
                sidecar.name,
                self.algorithm,
                count_parameters(network),
            )
            return destination
        _write_portable(destination, config, network)
        logger.info(
            "Model saved (archive portable): {} | {} | {} paramètres",
            destination,
            self.algorithm,
            count_parameters(network),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path, *, config: Mapping[str, Any] | None = None) -> BaseModel:
        """Rebuild a model from an artefact written by :meth:`save`.

        Args:
            path: Artefact path.
            config: Optional configuration refreshed into the reloaded model.

        Returns:
            The reloaded :class:`KerasModel`, ready to predict.

        Raises:
            FileNotFoundError: When the artefact does not exist.
            TypeError: When the artefact is not a model of this project.
        """
        source = Path(path)
        if not source.exists():
            msg = f"Model artefact not found: {source}"
            raise FileNotFoundError(msg)
        weights: list[np.ndarray] | None = None
        network: keras.Model
        if source.name.endswith(NATIVE_SUFFIX):
            payload = _read_sidecar(_sidecar_path(source), source)
            try:
                network = keras.saving.load_model(source)
            except Exception as error:
                msg = (
                    f"'{source.name}' could not be loaded as a Keras archive "
                    f"({type(error).__name__}). Point model_file at the file written by "
                    "KerasModel.save."
                )
                raise TypeError(msg) from error
        else:
            payload, network, weights = _read_portable(source)
        model = cls(
            algorithm=str(payload["architecture"]),
            params=dict(payload.get("params") or {}),
            task=str(payload.get("task", "binary")),
            feature_names=[str(name) for name in payload.get("feature_names") or []],
            target_name=(
                str(payload["target_name"]) if payload.get("target_name") is not None else None
            ),
            random_state=int(payload.get("random_state", 42)),
            config=config,
            name=str(payload.get("name") or payload["architecture"]),
        )
        model._restore(payload, network, weights)
        return model

    # ------------------------------------------------------------------ identité ----------
    def _repr_fields(self) -> list[str]:
        """Add the architecture identity and the training budget to the representation."""
        fields = [f"algorithm={self.algorithm or '-'}"]
        if self.network_ is not None:
            fields.append(f"parameters={count_parameters(self.network_)}")
        fields += [f"task={self.task}", f"features={len(self.feature_names)}"]
        if self.epochs_trained_:
            fields.append(f"epochs={self.epochs_trained_}")
        fields.append(f"state={self.state}")
        return fields

    def model_card(
        self,
        *,
        metrics: Mapping[str, Any] | None = None,
        artifact: str | None = None,
        notes: Iterable[str] | None = None,
    ) -> ModelCard:
        """Build the model card, enriched with the network diagnostics.

        Args:
            metrics: Metrics to archive.
            artifact: Serialised model file name.
            notes: Extra remarks.

        Returns:
            The :class:`ModelCard`.
        """
        card = super().model_card(metrics=metrics, artifact=artifact, notes=notes)
        card.library_versions = library_versions()
        extra: list[str] = []
        if self.network_ is not None:
            extra.append(f"paramètres entraînables : {count_parameters(self.network_)}")
        if self.epochs_trained_:
            extra.append(f"époques entraînées : {self.epochs_trained_}")
        if self.best_epoch_ is not None:
            extra.append(f"meilleure époque : {self.best_epoch_}")
        card.notes = [*card.notes, *extra]
        return card

    # ------------------------------------------------------------------ interne -----------
    def _artefact_config(self) -> dict[str, Any]:
        """Assemble the project-contract metadata written next to (or inside) the graph.

        The Keras graph itself is serialised by the saving API (native ``.keras`` archive, or
        ``serialize_keras_object`` inside the portable archive), never by this mapping — which must
        stay JSON-safe.
        """
        return {
            "kind": CHECKPOINT_KIND,
            "architecture": str(self.algorithm),
            "task": str(self.task),
            "target_name": self.target_name,
            "name": self.name,
            "feature_names": [str(name) for name in self.feature_names],
            "random_state": int(self.random_state),
            "params": _json_safe(self.params),
            "input_dim": int(self.input_dim_),
            "output_dim": int(self.output_dim_),
            "classes": _json_safe(self.classes_) if self.classes_ is not None else None,
            "target_mean": self._target_mean_,
            "target_std": self._target_std_,
            "best_epoch": self.best_epoch_,
            "epochs_trained": int(self.epochs_trained_),
            "history": _json_safe(self.history_),
            "keras_version": str(keras.version()),
        }

    def _restore(
        self,
        payload: Mapping[str, Any],
        network: keras.Model,
        weights: list[np.ndarray] | None = None,
    ) -> None:
        """Attach the reloaded graph, its weights and the project contract.

        Args:
            payload: Artefact configuration.
            network: Rebuilt Keras graph.
            weights: Ordered weight arrays (portable archive only).
        """
        if weights is not None:
            network.set_weights(weights)
        self.network_ = network
        self.params = dict(payload.get("params") or {})
        self.input_dim_ = int(payload.get("input_dim", 0) or 0)
        self.output_dim_ = int(payload.get("output_dim", 0) or 0)
        stored = payload.get("classes")
        self.classes_ = None if stored is None else np.asarray(stored, dtype="object")
        self._target_mean_ = payload.get("target_mean")
        self._target_std_ = payload.get("target_std")
        self.best_epoch_ = payload.get("best_epoch")
        self.epochs_trained_ = int(payload.get("epochs_trained", 0) or 0)
        self.history_ = {
            str(key): [float(value) for value in values]
            for key, values in dict(payload.get("history") or {}).items()
        }
        self._is_fitted = True
        self.fit_result_ = FitResult(
            model_name=type(self).__name__,
            algorithm=str(self.algorithm),
            epochs=self.epochs_trained_,
            history=dict(self.history_),
            extra={"restored_from_artefact": True},
        )
        logger.info(
            "Model restored | {} | entrées={} sorties={} époques={}",
            self.summary(),
            self.input_dim_,
            self.output_dim_,
            self.epochs_trained_,
        )

    def _require_network(self) -> keras.Model:
        """Return the trained network or raise a diagnostic error."""
        if self.network_ is None:
            msg = "No network is attached to this model (fit or load it first)"
            raise RuntimeError(msg)
        return self.network_

    def _label_list(self) -> list[Any]:
        """Return the training label space as a plain list."""
        return label_list(self.classes_)

    def _train_node(self) -> Mapping[str, Any]:
        """Return the ``train`` configuration node."""
        return dict(self.config.get("train") or {})

    def _epochs(self) -> int:
        """Number of epochs declared in the configuration."""
        try:
            return max(int(self._train_node().get("epochs", 1) or 1), 1)
        except (TypeError, ValueError):  # pragma: no cover - défensif
            return 1

    def _batch_size(self) -> int:
        """Batch size declared in the configuration."""
        try:
            return max(int(self._train_node().get("batch_size", 32) or 32), 1)
        except (TypeError, ValueError):  # pragma: no cover - défensif
            return 32

    def _learning_rate(self) -> float:
        """Learning rate declared in the configuration."""
        return _numeric(self._train_node().get("learning_rate"), 1e-3)

    def _weight_decay(self) -> float:
        """Decoupled weight decay declared in the configuration (applied by AdamW)."""
        return _numeric(self._train_node().get("weight_decay"), 0.0)

    def _primary_metric(self) -> str:
        """Primary metric of the project (``""`` when undeclared)."""
        return str(dict(self.config.get("metrics") or {}).get("primary", "") or "")

    def _decision_threshold(self) -> float:
        """Decision threshold applied to the positive probability of a binary task."""
        decision = dict(self.config.get("decision") or {})
        return _numeric(decision.get("threshold"), 0.5)


def _best_epoch(history: Mapping[str, Sequence[float]], monitor: str, mode: str) -> int | None:
    """Locate the best epoch of a Keras history for the monitored quantity.

    Args:
        history: Keras ``history.history`` mapping.
        monitor: Project monitor name (``val_loss``, ``val_roc_auc``, ...).
        mode: ``min`` or ``max``.

    Returns:
        The 0-based epoch index, or ``None`` when the monitor is absent.
    """
    candidates = [monitor, monitor.removeprefix("val_"), f"val_{monitor.removeprefix('val_')}"]
    for key in candidates:
        values = [float(value) for value in history.get(key, []) if np.isfinite(float(value))]
        if values:
            return int(np.argmin(values) if mode == "min" else np.argmax(values))
    return None


__all__ = [
    "CHECKPOINT_KIND",
    "CONFIG_MEMBER",
    "ESTIMATORS",
    "INFERENCE_BATCH_SIZE",
    "NATIVE_SUFFIX",
    "WEIGHTS_MEMBER",
    "AlgorithmSpec",
    "EpochBridge",
    "FitResult",
    "KerasModel",
    "ModelCard",
    "available_for_task",
    "count_parameters",
    "filter_params",
    "resolve_algorithm",
    "seed_keras",
]
