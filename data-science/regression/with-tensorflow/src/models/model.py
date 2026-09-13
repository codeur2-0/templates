"""TensorFlow implementation of the model contract.

This stack sits at the lowest level of the repository's deep-learning trio: the network is
**subclassed** (``keras.Model`` + ``call``), the blocks are hand-written layers
(:class:`DenseBlock`, :class:`ResidualBlock`), the input pipeline is a ``tf.data.Dataset`` (seeded
shuffle, batch, prefetch) and every optimisation step goes through ``tf.GradientTape`` then
``optimizer.apply_gradients`` — all of it compiled into a graph by ``tf.function``.

What this makes visible, and what the Keras stack hides behind ``compile`` / ``fit``:

* **regularisation losses are added by hand** (``network.losses`` holds the L2 penalties),
* **gradient clipping** is an explicit ``clip_by_global_norm`` before applying the gradients,
* **class imbalance** is handled with ``sample_weight`` (Keras has no ``pos_weight``),
* **logits stay logits** — ``from_logits=True`` in the loss, sigmoid/softmax only at inference,
* **learning-rate schedules count optimiser steps**, not epochs,
* **persisting a subclassed model** means writing the weights *and* the recipe that rebuilds the
  graph: ``.weights.h5`` for the weights, a sidecar JSON for the configuration (and a portable
  single-file archive for any other extension).
"""

from __future__ import annotations

import io
import json
import os
import random
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.base import BaseModel, FitResult, ModelCard, library_versions
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Marqueur écrit dans chaque artefact : permet à :meth:`load` de rejeter un fichier étranger.
CHECKPOINT_KIND = "tabular-project.tensorflow-model/v1"

#: Extension native des poids Keras 3 (``save_weights`` / ``load_weights``).
WEIGHTS_SUFFIX = ".weights.h5"

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

#: Fonctions d'activation acceptées (noms natifs Keras).
_ACTIVATIONS = frozenset({"relu", "gelu", "tanh", "elu", "selu", "silu", "linear"})

#: Optimiseurs acceptés par le paramètre ``optimizer``.
_OPTIMIZERS = frozenset({"adamw", "adam", "sgd", "rmsprop"})

#: Ordonnanceurs de taux d'apprentissage acceptés par le paramètre ``scheduler``.
_SCHEDULERS = frozenset({"cosine", "plateau", "step", "none"})

#: Hyper-parameters accepted by every architecture of this stack.
_COMMON_PARAMS: frozenset[str] = frozenset(
    {
        "hidden_layers",
        "dropout",
        "activation",
        "batch_norm",
        "optimizer",
        "scheduler",
        "clip_norm",
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


# ---------------------------------------------------------------------------------------------
# Couches personnalisées
# ---------------------------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package="tabular_project")
class DenseBlock(keras.layers.Layer):
    """``Dense → [BatchNormalization] → activation → Dropout`` as a single reusable layer.

    Writing it as a layer (instead of a ``Sequential``) keeps the name of every sub-layer stable —
    which matters because ``load_weights`` matches variables **by name**, and Keras would otherwise
    number them from a process-wide counter.
    """

    def __init__(
        self,
        units: int,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        l2: float = 0.0,
        block_name: str = "dense_block",
        **kwargs: Any,
    ) -> None:
        """Build the block and its sub-layers with explicit names.

        Args:
            units: Number of output units.
            activation: Activation name (see :data:`_ACTIVATIONS`).
            dropout: Dropout probability (``0`` disables the layer).
            batch_norm: Whether to normalise the representation before the activation.
            l2: L2 penalty applied to the kernel (surfaces in ``network.losses``).
            block_name: Stable name prefix of the sub-layers.
            **kwargs: Forwarded to :class:`keras.layers.Layer`.
        """
        super().__init__(name=block_name, **kwargs)
        self.units = int(units)
        self.activation_name = str(activation)
        self.dropout_rate = float(dropout)
        self.use_batch_norm = bool(batch_norm)
        self.l2 = float(l2)
        self.dense = keras.layers.Dense(
            self.units,
            kernel_regularizer=keras.regularizers.L2(self.l2) if self.l2 > 0 else None,
            name=f"{block_name}_dense",
        )
        self.norm: keras.layers.Layer | None = (
            keras.layers.BatchNormalization(name=f"{block_name}_norm")
            if self.use_batch_norm
            else None
        )
        self.activation_layer = keras.layers.Activation(
            self.activation_name, name=f"{block_name}_activation"
        )
        self.dropout: keras.layers.Layer | None = (
            keras.layers.Dropout(self.dropout_rate, name=f"{block_name}_dropout")
            if self.dropout_rate > 0
            else None
        )

    def call(self, features: Any, training: bool = False) -> Any:
        """Apply the block.

        Args:
            features: Batch of representations.
            training: Whether dropout and batch normalisation run in training mode.

        Returns:
            The transformed batch.
        """
        hidden = self.dense(features)
        if self.norm is not None:
            hidden = self.norm(hidden, training=training)
        hidden = self.activation_layer(hidden)
        if self.dropout is not None:
            hidden = self.dropout(hidden, training=training)
        return hidden

    def get_config(self) -> dict[str, Any]:
        """Serialise the block (required to persist a subclassed model).

        Returns:
            The constructor arguments.
        """
        return {
            **super().get_config(),
            "units": self.units,
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
            "batch_norm": self.use_batch_norm,
            "l2": self.l2,
            "block_name": self.name,
        }


@keras.utils.register_keras_serializable(package="tabular_project")
class ResidualBlock(keras.layers.Layer):
    """Two dense transformations wrapped in a skip connection.

    The block learns a *correction* of its input, which keeps gradients flowing when blocks are
    stacked — the width is constant so the skip connection is exact.
    """

    def __init__(
        self,
        units: int,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        l2: float = 0.0,
        block_name: str = "residual_block",
        **kwargs: Any,
    ) -> None:
        """Build the two inner transformations.

        Args:
            units: Number of units (identical in input and output).
            activation: Activation name.
            dropout: Dropout probability.
            batch_norm: Whether to normalise hidden representations.
            l2: L2 penalty applied to the kernels.
            block_name: Stable name prefix.
            **kwargs: Forwarded to :class:`keras.layers.Layer`.
        """
        super().__init__(name=block_name, **kwargs)
        self.units = int(units)
        self.activation_name = str(activation)
        self.dropout_rate = float(dropout)
        self.use_batch_norm = bool(batch_norm)
        self.l2 = float(l2)
        self.first = DenseBlock(
            self.units,
            activation=self.activation_name,
            dropout=self.dropout_rate,
            batch_norm=self.use_batch_norm,
            l2=self.l2,
            block_name=f"{block_name}_first",
        )
        self.second = keras.layers.Dense(
            self.units,
            kernel_regularizer=keras.regularizers.L2(self.l2) if self.l2 > 0 else None,
            name=f"{block_name}_second_dense",
        )

    def call(self, features: Any, training: bool = False) -> Any:
        """Apply the block and add its input back.

        Args:
            features: Batch of representations.
            training: Training mode flag.

        Returns:
            The corrected batch.
        """
        return features + self.second(self.first(features, training=training))

    def get_config(self) -> dict[str, Any]:
        """Serialise the block.

        Returns:
            The constructor arguments.
        """
        return {
            **super().get_config(),
            "units": self.units,
            "activation": self.activation_name,
            "dropout": self.dropout_rate,
            "batch_norm": self.use_batch_norm,
            "l2": self.l2,
            "block_name": self.name,
        }


@keras.utils.register_keras_serializable(package="tabular_project")
class TabularNet(keras.Model):
    """Subclassed network: the graph is described by ``call``, not by a layer list.

    Subclassing (versus the functional API used by the Keras stack) costs a serialisation effort —
    ``get_config`` must be written by hand — and buys full control of the forward pass.
    """

    def __init__(
        self,
        *,
        algorithm: str,
        input_dim: int,
        output_dim: int,
        params: Mapping[str, Any] | None = None,
        name: str = "tabular_net",
        **kwargs: Any,
    ) -> None:
        """Build the layers of the requested architecture.

        Args:
            algorithm: Architecture name (``mlp``, ``linear``, ``residual_mlp``, ``autoencoder``).
            input_dim: Number of input features.
            output_dim: Number of outputs (1 for binary/regression, ``n_classes`` for multiclass,
                ``input_dim`` for an autoencoder).
            params: Resolved hyper-parameters.
            name: Model name.
            **kwargs: Forwarded to :class:`keras.Model`.

        Raises:
            ValueError: When the architecture is unknown.
        """
        super().__init__(name=name, **kwargs)
        self.algorithm = str(algorithm)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.params: dict[str, Any] = dict(params or {})
        widths = _hidden_layers(self.params)
        dropout = float(self.params.get("dropout", 0.0))
        activation = str(self.params.get("activation", "relu"))
        batch_norm = bool(self.params.get("batch_norm", False))
        l2 = float(self.params.get("l2", 0.0))
        self.blocks: list[keras.layers.Layer] = []
        self.head: keras.layers.Layer | None = None
        self.decoder: list[keras.layers.Layer] = []

        if self.algorithm == "linear":
            # Une seule transformation affine : la baseline interprétable.
            self.head = keras.layers.Dense(self.output_dim, name="output_dense")
        elif self.algorithm == "mlp":
            self.blocks = [
                DenseBlock(
                    units,
                    activation=activation,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    l2=l2,
                    block_name=f"hidden_{index}",
                )
                for index, units in enumerate(widths)
            ]
            self.head = keras.layers.Dense(self.output_dim, name="output_dense")
        elif self.algorithm == "residual_mlp":
            self.blocks = [
                DenseBlock(
                    widths[0],
                    activation=activation,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    l2=l2,
                    block_name="stem",
                ),
                *[
                    ResidualBlock(
                        widths[0],
                        activation=activation,
                        dropout=dropout,
                        batch_norm=batch_norm,
                        l2=l2,
                        block_name=f"residual_{index}",
                    )
                    for index in range(len(widths))
                ],
            ]
            self.head = keras.layers.Dense(self.output_dim, name="output_dense")
        elif self.algorithm == "autoencoder":
            self.blocks = [
                DenseBlock(
                    units,
                    activation=activation,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    l2=l2,
                    block_name=f"encoder_{index}",
                )
                for index, units in enumerate(widths)
            ]
            code_dim = _code_dim(self.params, self.input_dim)
            self.head = keras.layers.Dense(code_dim, name="code_dense")
            self.decoder = [
                DenseBlock(
                    units,
                    activation=activation,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    l2=l2,
                    block_name=f"decoder_{index}",
                )
                for index, units in enumerate(reversed(widths))
            ]
            self.decoder.append(
                keras.layers.Dense(self.input_dim, name="reconstruction_dense")
            )
        else:
            msg = f"Unknown TensorFlow architecture '{algorithm}'"
            raise ValueError(msg)

    def call(self, features: Any, training: bool = False) -> Any:
        """Run the forward pass.

        Args:
            features: Batch of inputs.
            training: Whether dropout / batch normalisation run in training mode.

        Returns:
            Logits (classification), values (regression) or reconstructions (autoencoder).
        """
        hidden = features
        for block in self.blocks:
            hidden = block(hidden, training=training)
        if self.head is None:  # pragma: no cover - garde-fou (toujours construit)
            msg = "The network has no output layer"
            raise RuntimeError(msg)
        hidden = self.head(hidden)
        for block in self.decoder:
            hidden = block(hidden, training=training)
        return hidden

    def get_config(self) -> dict[str, Any]:
        """Serialise the architecture (the recipe that rebuilds the graph).

        Returns:
            The constructor arguments, JSON-safe.
        """
        return {
            **super().get_config(),
            "algorithm": self.algorithm,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "params": _json_safe(self.params),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> TabularNet:
        """Rebuild a network from :meth:`get_config`.

        Args:
            config: Serialised configuration.

        Returns:
            The rebuilt (untrained) network.
        """
        payload = dict(config)
        payload.pop("name", None)
        return cls(
            algorithm=str(payload["algorithm"]),
            input_dim=int(payload["input_dim"]),
            output_dim=int(payload["output_dim"]),
            params=dict(payload.get("params") or {}),
            name=str(config.get("name") or "tabular_net"),
        )


# ---------------------------------------------------------------------------------------------
# Registre des architectures
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


def _network(algorithm: str) -> Callable[..., TabularNet]:
    """Return the builder of an architecture (registry indirection kept for symmetry).

    Args:
        algorithm: Architecture name.

    Returns:
        A callable building a :class:`TabularNet`.
    """

    def build(input_dim: int, output_dim: int, params: Mapping[str, Any]) -> TabularNet:
        """Build the network.

        Args:
            input_dim: Number of input features.
            output_dim: Number of outputs.
            params: Resolved hyper-parameters.

        Returns:
            The untrained network.
        """
        return TabularNet(
            algorithm=algorithm, input_dim=input_dim, output_dim=output_dim, params=params
        )

    return build


def _mlp(input_dim: int, output_dim: int, params: Mapping[str, Any]) -> TabularNet:
    """Build a plain multi-layer perceptron.

    Args:
        input_dim: Number of input features.
        output_dim: Number of outputs.
        params: Resolved parameters.

    Returns:
        The untrained network.
    """
    return _network("mlp")(input_dim, output_dim, params)


def _linear(input_dim: int, output_dim: int, params: Mapping[str, Any]) -> TabularNet:
    """Build a single dense layer (logistic / linear regression trained by gradient).

    Args:
        input_dim: Number of input features.
        output_dim: Number of outputs.
        params: Resolved parameters (unused: a linear model has no hidden width).

    Returns:
        The untrained network.
    """
    del params
    return _network("linear")(input_dim, output_dim, {})


def _residual_mlp(input_dim: int, output_dim: int, params: Mapping[str, Any]) -> TabularNet:
    """Build a residual multi-layer perceptron.

    Args:
        input_dim: Number of input features.
        output_dim: Number of outputs.
        params: Resolved parameters.

    Returns:
        The untrained network.
    """
    return _network("residual_mlp")(input_dim, output_dim, params)


def _autoencoder(input_dim: int, output_dim: int, params: Mapping[str, Any]) -> TabularNet:
    """Build the autoencoder used for anomaly detection.

    Args:
        input_dim: Number of input features.
        output_dim: Ignored (the reconstruction width is ``input_dim``).
        params: Resolved parameters.

    Returns:
        The untrained network.
    """
    del output_dim
    return _network("autoencoder")(input_dim, input_dim, params)


# ---------------------------------------------------------------------------------------------
# Spécification d'algorithme (contrat partagé avec les autres stacks)
# ---------------------------------------------------------------------------------------------
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

    __slots__ = (
        "accepted",
        "builder",
        "defaults",
        "display_name",
        "iterative",
        "name",
        "rationale",
        "tasks",
    )

    def __init__(
        self,
        *,
        name: str,
        display_name: str,
        tasks: frozenset[str],
        builder: Callable[..., TabularNet],
        rationale: str = "",
        iterative: bool = True,
        defaults: Mapping[str, Any] | None = None,
        accepted: frozenset[str] = _COMMON_PARAMS,
    ) -> None:
        """Store the declaration.

        Args:
            name: Architecture identifier.
            display_name: Human readable name.
            tasks: Supported learning tasks.
            builder: Network factory.
            rationale: Pedagogical note.
            iterative: Always ``True`` for a neural network.
            defaults: Default hyper-parameters.
            accepted: Parameter allow-list.
        """
        self.name = name
        self.display_name = display_name
        self.tasks = tasks
        self.builder = builder
        self.rationale = rationale
        self.iterative = iterative
        self.defaults: dict[str, Any] = dict(defaults or {})
        self.accepted = accepted


#: Registry of every architecture this stack can build.
ESTIMATORS: dict[str, AlgorithmSpec] = {
    spec.name: spec
    for spec in (
        AlgorithmSpec(
            name="mlp",
            display_name="Réseau dense subclassé (MLP)",
            tasks=_CLASSIFICATION | _REGRESSION | _ANOMALY,
            builder=_mlp,
            rationale=(
                "Perceptron multicouche écrit comme un `keras.Model` subclassé : la passe avant "
                "est du code Python, la régularisation L2 est ajoutée explicitement à la perte et "
                "chaque pas d'optimisation passe par `GradientTape`. `hidden_layers` fixe la "
                "capacité, `dropout` et `l2` la régularisation."
            ),
            defaults={
                "hidden_layers": [64, 32],
                "dropout": 0.2,
                "activation": "relu",
                "batch_norm": False,
                "optimizer": "adamw",
                "scheduler": "none",
                "clip_norm": 5.0,
                "l2": 0.0,
            },
        ),
        AlgorithmSpec(
            name="linear",
            display_name="Modèle linéaire entraîné par gradient",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_linear,
            rationale=(
                "Une seule couche `Dense` : régression logistique / linéaire optimisée par "
                "descente de gradient. Baseline interprétable (les poids sont les coefficients) et "
                "référence de coût — un réseau qui ne la bat pas n'apporte rien."
            ),
            defaults={
                "dropout": 0.0,
                "batch_norm": False,
                "optimizer": "adamw",
                "scheduler": "none",
                "clip_norm": 5.0,
                "l2": 0.0,
            },
        ),
        AlgorithmSpec(
            name="residual_mlp",
            display_name="MLP à blocs résiduels (Layer personnalisée)",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_residual_mlp,
            rationale=(
                "`ResidualBlock` est une `keras.layers.Layer` écrite à la main : le bloc "
                "apprend une correction de son entrée, ce qui fait circuler le gradient quand on "
                "empile les "
                "couches. Montre comment exposer `call` et `get_config` sur une couche custom."
            ),
            defaults={
                "hidden_layers": [64, 64],
                "dropout": 0.15,
                "activation": "relu",
                "batch_norm": True,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "clip_norm": 5.0,
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
                "clip_norm": 5.0,
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
            f"Unknown TensorFlow architecture '{algorithm}'. Available: {sorted(ESTIMATORS)} "
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
# Déterminisme, cible, pertes et pipeline tf.data
# ---------------------------------------------------------------------------------------------
def seed_tensorflow(seed: int) -> None:
    """Seed every random source TensorFlow and Keras depend on.

    Args:
        seed: Reproducibility seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    # Déterminisme des ops : indispensable pour que deux runs à graine fixée produisent les mêmes
    # prédictions (vérifié par `test_training_is_reproducible`).
    tf.config.experimental.enable_op_determinism()


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
    """Encode business targets into the numeric form the loss expects.

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


def _sample_weights(
    labels: np.ndarray | None, task: str, params: Mapping[str, Any], classes: LabelSpace
) -> np.ndarray:
    """Per-row weights carrying the class imbalance into the loss.

    Keras has no ``pos_weight``: the equivalent is a ``sample_weight`` vector that up-weights the
    minority class (binary) or each class by its configured weight (multiclass).

    Args:
        labels: Business targets (``None`` for an autoencoder).
        task: Learning task.
        params: Resolved parameters (``pos_weight`` / ``class_weight``).
        classes: Sorted label space.

    Returns:
        A ``float32`` vector of ones when no weighting applies.
    """
    if labels is None:
        return np.ones(0, dtype="float32")
    if task == "binary":
        weight = _numeric(params.get("pos_weight"), 1.0)
        positive = space_positive(classes)
        return np.where(np.asarray(labels) == positive, weight, 1.0).astype("float32")
    if task == "multiclass":
        weights = params.get("class_weight")
        if not weights:
            return np.ones(len(labels), dtype="float32")
        space = label_list(classes)
        index = {label: _numeric(weights[position], 1.0) for position, label in enumerate(space)}
        return np.asarray([index.get(label, 1.0) for label in labels], dtype="float32")
    return np.ones(len(labels), dtype="float32")


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


def _loss_function(task: str) -> keras.losses.Loss:
    """Build the loss matching the task (logits in, no activation applied twice).

    Args:
        task: Learning task.

    Returns:
        The Keras loss.
    """
    if task == "binary":
        # `from_logits=True` : la fusion sigmoid + log est numériquement stable.
        return keras.losses.BinaryCrossentropy(from_logits=True)
    if task == "multiclass":
        return keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # Régression, prévision et reconstruction (auto-encodeur) : erreur quadratique moyenne.
    return keras.losses.MeanSquaredError()


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
    # `plateau` est piloté à la main dans la boucle d'époques (équivalent de ReduceLROnPlateau)
    # et `none` garde un taux constant.
    return float(learning_rate)


def _optimizer(params: Mapping[str, Any], learning_rate: Any, weight_decay: float) -> Any:
    """Build the optimizer declared in the parameters.

    Args:
        params: Resolved parameters (``optimizer``, ``momentum``).
        learning_rate: Constant or schedule.
        weight_decay: Decoupled L2 penalty applied by AdamW / Adam.

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
            learning_rate=learning_rate, momentum=float(params.get("momentum", 0.9))
        )
    if name == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    msg = f"Unknown optimizer '{name}'. Available: {sorted(_OPTIMIZERS)}"
    raise ValueError(msg)


def _dataset(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> Any:
    """Build the ``tf.data`` input pipeline of one epoch.

    Args:
        features: Feature matrix.
        targets: Encoded targets (the features themselves for an autoencoder).
        weights: Per-row sample weights.
        batch_size: Number of rows per optimiser step.
        seed: Shuffle seed (reproducibility).
        shuffle: Whether to shuffle the rows.

    Returns:
        A batched, prefetched dataset of ``(features, targets, weights)``.
    """
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            np.ascontiguousarray(features, dtype="float32"),
            np.ascontiguousarray(targets),
            np.ascontiguousarray(weights, dtype="float32"),
        )
    )
    if shuffle:
        # Graine explicite : le mélange est reproductible d'un run à l'autre.
        dataset = dataset.shuffle(
            buffer_size=max(len(features), 1), seed=int(seed), reshuffle_each_iteration=True
        )
    return dataset.batch(max(int(batch_size), 1)).prefetch(tf.data.AUTOTUNE)


def _compiled_train_step(
    network: TabularNet, criterion: Any, optimizer: Any, *, clip_norm: float
) -> Callable[[Any, Any, Any], Any]:
    """Compile one optimisation step **bound to a single network**.

    ``tf.function`` captures the variables of the objects it closes over: a module-level function
    shared by two successive trainings would reuse the graph (and the variables) of the first
    network and fail with *"only supports singleton tf.Variables"*. Binding the compiled step to
    the network being trained is what makes refitting and algorithm comparison work.

    The step itself shows what ``keras.Model.fit`` hides: forward pass, business loss **plus the
    regularisation losses** collected by the layers, gradients, global-norm clipping, then
    ``apply_gradients``.

    Args:
        network: Subclassed network whose variables are optimised.
        criterion: Keras loss (``from_logits=True`` for classification).
        optimizer: Keras optimizer.
        clip_norm: Global gradient-norm clipping (``0`` disables it).

    Returns:
        A callable ``(features, targets, weights) -> scalar loss`` compiled into a graph.
    """

    @tf.function(reduce_retracing=True)
    def step(features: Any, targets: Any, weights: Any) -> Any:
        """Run one batch: forward, loss, backward, apply.

        Args:
            features: Batch of inputs.
            targets: Batch of encoded targets.
            weights: Batch of sample weights.

        Returns:
            The scalar loss of the batch.
        """
        with tf.GradientTape() as tape:
            logits = network(features, training=True)
            loss = criterion(targets, logits, sample_weight=weights)
            regularisation = network.losses
            if len(regularisation) > 0:
                # Les pénalités L2 des couches ne sont PAS ajoutées automatiquement en boucle
                # custom : c'est à l'appelant de les sommer à la perte métier.
                loss = loss + tf.add_n(regularisation)
        gradients = tape.gradient(loss, network.trainable_variables)
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, network.trainable_variables, strict=True)
            if gradient is not None
        ]
        if clip_norm > 0:
            # Écrêtage de la norme globale : évite qu'un lot aberrant ne détruise les poids.
            clipped, _ = tf.clip_by_global_norm([gradient for gradient, _ in pairs], clip_norm)
            pairs = list(zip(clipped, [variable for _, variable in pairs], strict=True))
        optimizer.apply_gradients(pairs)
        return loss

    return step


def _compiled_eval_step(network: TabularNet, criterion: Any) -> Callable[[Any, Any, Any], Any]:
    """Compile the validation step (no gradient tape) bound to a single network.

    Args:
        network: Subclassed network, run in inference mode.
        criterion: Keras loss.

    Returns:
        A callable ``(features, targets, weights) -> (loss, logits)`` compiled into a graph.
    """

    @tf.function(reduce_retracing=True)
    def step(features: Any, targets: Any, weights: Any) -> Any:
        """Score one batch.

        Args:
            features: Batch of inputs.
            targets: Batch of encoded targets.
            weights: Batch of sample weights.

        Returns:
            The batch loss and the raw outputs.
        """
        logits = network(features, training=False)
        return criterion(targets, logits, sample_weight=weights), logits

    return step


def _run_epoch(train_step: Callable[[Any, Any, Any], Any], dataset: Any) -> float:
    """Run one training epoch over the ``tf.data`` pipeline.

    Args:
        train_step: Compiled optimisation step (see :func:`_compiled_train_step`).
        dataset: Batched dataset.

    Returns:
        The mean training loss of the epoch.
    """
    total = 0.0
    seen = 0
    for features, targets, weights in dataset:
        loss = train_step(features, targets, weights)
        batch = int(tf.shape(features)[0])
        total += float(loss) * batch
        seen += batch
    return total / max(seen, 1)


def _evaluate(
    eval_step: Callable[[Any, Any, Any], Any], dataset: Any
) -> tuple[float, np.ndarray, np.ndarray]:
    """Score a whole dataset (loss plus raw outputs and encoded targets).

    Args:
        eval_step: Compiled validation step (see :func:`_compiled_eval_step`).
        dataset: Batched dataset.

    Returns:
        ``(mean_loss, outputs, targets)`` as NumPy arrays.
    """
    total = 0.0
    seen = 0
    outputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for features, batch_targets, weights in dataset:
        loss, logits = eval_step(features, batch_targets, weights)
        batch = int(tf.shape(features)[0])
        total += float(loss) * batch
        seen += batch
        outputs.append(np.asarray(logits))
        targets.append(np.asarray(batch_targets))
    stacked_outputs = np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 1))
    stacked_targets = np.concatenate(targets, axis=0) if targets else np.zeros((0,))
    return total / max(seen, 1), stacked_outputs, stacked_targets


def _current_learning_rate(optimizer: Any) -> float:
    """Read the learning rate of an optimizer (constant or scheduled)."""
    rate = getattr(optimizer, "learning_rate", None)
    if rate is None:  # pragma: no cover - défensif
        return float("nan")
    if callable(getattr(rate, "numpy", None)):
        return float(rate.numpy())
    if isinstance(rate, keras.optimizers.schedules.LearningRateSchedule):
        return float(rate(optimizer.iterations))
    return float(rate)


class PlateauDecay:
    """Manual equivalent of ``ReduceLROnPlateau`` for a custom training loop.

    Keras ships this behaviour as a ``fit`` callback; a ``GradientTape`` loop has to implement it,
    which is exactly what makes the mechanism visible.
    """

    def __init__(self, *, patience: int, factor: float, mode: str, min_lr: float = 1e-6) -> None:
        """Store the decay policy.

        Args:
            patience: Epochs without improvement before decaying.
            factor: Multiplicative factor applied to the learning rate.
            mode: ``min`` (a loss improves when it decreases) or ``max``.
            min_lr: Floor below which the rate is never pushed.
        """
        self.patience = max(int(patience), 1)
        self.factor = float(factor)
        self.mode = mode if mode in {"min", "max"} else "min"
        self.min_lr = float(min_lr)
        self.best: float = float("inf") if self.mode == "min" else float("-inf")
        self.wait = 0
        self.decays = 0

    def step(self, optimizer: Any, value: float) -> None:
        """Update the tracking and decay the learning rate when the metric plateaus.

        Args:
            optimizer: Optimizer whose learning rate is adjusted.
            value: Monitored value of the epoch.
        """
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best, self.wait = float(value), 0
            return
        self.wait += 1
        if self.wait < self.patience:
            return
        rate = _current_learning_rate(optimizer)
        decayed = max(rate * self.factor, self.min_lr)
        optimizer.learning_rate.assign(decayed)
        self.wait, self.decays = 0, self.decays + 1
        logger.info(
            "Plateau sur {} : taux d'apprentissage {} -> {:.6g}", value, rate, decayed
        )


def _monitored_metric(train_node: Mapping[str, Any], primary: str) -> tuple[str, str]:
    """Return the monitored log key and its comparison direction.

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
    """Convert numpy / tensor scalars into plain Python containers.

    The artefact configuration is written as JSON: everything it holds must be a primitive or a
    container of primitives.

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
    if isinstance(value, tf.Tensor):
        return _json_safe(value.numpy())
    return value


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid over an array."""
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype="float64")))


def _softmax(values: np.ndarray) -> np.ndarray:
    """Row-wise softmax over a 2-D array."""
    array = np.asarray(values, dtype="float64")
    array = array - array.max(axis=1, keepdims=True)
    exponentials = np.exp(array)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _sidecar_path(destination: Path) -> Path:
    """Return the JSON configuration path accompanying a native weights file.

    Args:
        destination: Path ending with ``.weights.h5``.

    Returns:
        The sibling ``*.config.json`` path.
    """
    stem = destination.name[: -len(WEIGHTS_SUFFIX)]
    return destination.with_name(f"{stem}.config.json")


def _write_portable(destination: Path, config: Mapping[str, Any], network: keras.Model) -> None:
    """Write a self-contained archive (JSON configuration + NumPy weights).

    Used whenever the destination does not carry the stack's native extension — the contract tests
    save to ``model.joblib``, and a single file is easier to move around than a weights/config pair.

    Args:
        destination: Archive path.
        config: Artefact configuration.
        network: Trained network whose weights are exported.
    """
    weights = network.get_weights()
    buffer = io.BytesIO()
    np.savez(buffer, *[np.asarray(array) for array in weights])
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(CONFIG_MEMBER, json.dumps(_json_safe(config), indent=2))
        archive.writestr(WEIGHTS_MEMBER, buffer.getvalue())


def _read_portable(source: Path) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Read a portable archive written by :func:`_write_portable`.

    Args:
        source: Archive path.

    Returns:
        The configuration and the ordered weight arrays.

    Raises:
        TypeError: When the archive is not an artefact of this project.
    """
    try:
        with zipfile.ZipFile(source) as archive:
            config = json.loads(archive.read(CONFIG_MEMBER).decode("utf-8"))
            with io.BytesIO(archive.read(WEIGHTS_MEMBER)) as buffer:
                payload = np.load(buffer)
                weights = [payload[name] for name in payload.files]
    except Exception as error:
        msg = (
            f"'{source.name}' is not a TensorFlow artefact of this project "
            f"({type(error).__name__}). Point model_file at the file written by "
            "TensorFlowModel.save (.weights.h5 + sidecar JSON, or the portable archive)."
        )
        raise TypeError(msg) from error
    if not isinstance(config, dict) or config.get("kind") != CHECKPOINT_KIND:
        msg = f"'{source.name}' does not hold the '{CHECKPOINT_KIND}' marker."
        raise TypeError(msg)
    return config, weights


def _read_sidecar(source: Path, weights: Path) -> dict[str, Any]:
    """Read the JSON configuration accompanying a native weights file.

    Args:
        source: Path of the sidecar JSON.
        weights: Path of the weights file (used in the diagnostic message).

    Returns:
        The artefact configuration.

    Raises:
        TypeError: When the sidecar is missing or is not an artefact of this project.
    """
    if not source.exists():
        msg = (
            f"Missing configuration for '{weights.name}': expected '{source.name}' next to it. A "
            "subclassed Keras model cannot be rebuilt from its weights alone — save both files "
            "together (TensorFlowModel.save does it for you)."
        )
        raise TypeError(msg)
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        msg = f"Unreadable model configuration '{source.name}': {error}"
        raise TypeError(msg) from error
    if not isinstance(config, dict) or config.get("kind") != CHECKPOINT_KIND:
        msg = f"'{source.name}' does not hold the '{CHECKPOINT_KIND}' marker."
        raise TypeError(msg)
    return config


class TensorFlowModel(BaseModel):
    """A subclassed Keras network trained with ``GradientTape`` behind the project's contract."""

    framework = "tensorflow"
    default_model_file = "model.weights.h5"
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
        self.network_: TabularNet | None = None
        self.input_dim_: int = 0
        self.output_dim_: int = 0
        self.best_epoch_: int | None = None
        self.epochs_trained_: int = 0
        self._target_mean_: float | None = None
        self._target_std_: float | None = None

    # ------------------------------------------------------------------ construction ------
    def _resolved_params(self, labels: np.ndarray | None) -> dict[str, Any]:
        """Merge the registry defaults, the configuration and the data-driven ``auto`` values.

        Args:
            labels: Training target (used to resolve the imbalance weights).

        Returns:
            The parameters used to build the network, the optimizer and the loss.

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
            msg = f"Unknown optimizer '{resolved.get('optimizer')}'. Available: {sorted(_OPTIMIZERS)}"
            raise ValueError(msg)
        if str(resolved.get("scheduler", "none")).lower() not in _SCHEDULERS:
            msg = f"Unknown scheduler '{resolved.get('scheduler')}'. Available: {sorted(_SCHEDULERS)}"
            raise ValueError(msg)
        if labels is not None and self.task in _CLASSIFICATION:
            counts = pd.Series(labels).value_counts().sort_index().to_numpy(dtype="float64")
            positive = float(counts[-1]) if counts.size else 0.0
            negative = float(counts[:-1].sum()) if counts.size > 1 else 0.0
            if str(resolved.get("pos_weight")) == "auto":
                # Keras n'a pas de `pos_weight` : la valeur devient un `sample_weight`.
                resolved["pos_weight"] = round(max(negative / max(positive, 1.0), 1.0), 4)
            if str(resolved.get("class_weight")) == "auto":
                total = float(counts.sum()) or 1.0
                resolved["class_weight"] = [
                    round(total / (len(counts) * max(float(count), 1.0)), 4) for count in counts
                ]
        cleaned = {key: value for key, value in resolved.items() if value != "auto"}
        if self.spec.name == "autoencoder":
            # Le goulot dépend de la largeur d'entrée : figé ici pour que l'artefact reconstruise
            # exactement le même graphe.
            cleaned.setdefault("code_dim", _code_dim(cleaned, len(self.feature_names) or 1))
        return filter_params(self.spec, cleaned)

    def build_network(
        self, input_dim: int, output_dim: int, labels: np.ndarray | None = None
    ) -> TabularNet:
        """Instantiate the network declared by ``algorithm`` (used by the notebooks).

        Args:
            input_dim: Number of input features.
            output_dim: Number of outputs.
            labels: Training target, used to resolve the imbalance weights.

        Returns:
            A fresh (untrained) network.
        """
        params = self._resolved_params(labels)
        if self.spec.name == "autoencoder":
            params["code_dim"] = _code_dim(params, input_dim)
        return self.spec.builder(input_dim=input_dim, output_dim=output_dim, params=params)

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
        """Run the explicit epoch loop over ``tf.data`` with ``GradientTape`` steps.

        Args:
            X: Aligned training features.
            y: Training target (``None`` for the autoencoder).
            X_val: Validation features (enables ``val_loss`` and the monitored metric).
            y_val: Validation target.
            context: Project callback context.
            callbacks: Project callbacks.

        Raises:
            ValueError: When the task is supervised but no target is provided.
        """
        self._seed_everything()
        seed_tensorflow(self.random_state)

        matrix = self._matrix(X)
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
        network = self.spec.builder(input_dim=input_dim, output_dim=output_dim, params=params)

        train_targets, train_weights = self._encode_batch(labels, matrix, params)
        batch_size = self._batch_size()
        epochs = self._epochs()
        steps_per_epoch = max(int(np.ceil(len(matrix) / max(batch_size, 1))), 1)
        criterion = _loss_function(self.task)
        schedule = _learning_rate_schedule(
            params, learning_rate=self._learning_rate(), steps_per_epoch=steps_per_epoch, epochs=epochs
        )
        optimizer = _optimizer(params, schedule, self._weight_decay())
        train_dataset = _dataset(
            matrix,
            train_targets,
            train_weights,
            batch_size=batch_size,
            seed=self.random_state,
            shuffle=True,
        )
        validation = None
        if X_val is not None:
            val_matrix = self._matrix(X_val)
            val_labels = None if y_val is None else np.asarray(y_val).ravel()
            val_targets, val_weights = self._encode_batch(val_labels, val_matrix, params)
            validation = _dataset(
                val_matrix,
                val_targets,
                val_weights,
                batch_size=batch_size,
                seed=self.random_state,
                shuffle=False,
            )

        # Un appel à vide construit les variables : indispensable avant tout `get_weights`.
        network(tf.zeros((min(2, max(len(matrix), 1)), input_dim), dtype="float32"), training=False)

        monitor, mode = _monitored_metric(self._train_node(), self._primary_metric())
        clip_norm = _numeric(params.get("clip_norm"), 0.0)
        plateau: PlateauDecay | None = None
        if str(params.get("scheduler", "none")).lower() == "plateau":
            plateau = PlateauDecay(
                patience=int(params.get("plateau_patience", 5)),
                factor=float(params.get("plateau_factor", 0.5)),
                mode=mode,
            )
        stopping = dict(self._train_node().get("early_stopping") or {})
        restore_best = bool(stopping.get("restore_best"))

        train_step = _compiled_train_step(network, criterion, optimizer, clip_norm=clip_norm)
        eval_step = _compiled_eval_step(network, criterion)

        best_weights: list[np.ndarray] | None = None
        best_value = float("inf") if mode == "min" else float("-inf")
        best_epoch = -1
        epochs_trained = 0

        for epoch in range(epochs):
            train_loss = _run_epoch(train_step, train_dataset)
            logs: dict[str, float] = {
                "loss": float(train_loss),
                "lr": _current_learning_rate(optimizer),
                "epoch": float(epoch),
            }
            if validation is not None:
                val_loss, val_outputs, val_targets = _evaluate(eval_step, validation)
                logs["val_loss"] = float(val_loss)
                metric = self._primary_metric()
                if metric and self.task not in _ANOMALY:
                    decoded_true = _decode_labels(
                        self.task,
                        val_targets,
                        self.classes_,
                        target_mean=self._target_mean_,
                        target_std=self._target_std_,
                    )
                    value = _score_metric(
                        self.task,
                        metric,
                        y_true=decoded_true,
                        y_pred=self._decode(val_outputs),
                        y_proba=self._probabilities(val_outputs),
                    )
                    if value is not None:
                        logs[f"val_{metric}"] = value
                observed = float(logs.get(monitor, logs["val_loss"]))
                improved = observed < best_value if mode == "min" else observed > best_value
                if improved:
                    best_value, best_epoch = observed, epoch
                    # Instantané des meilleurs poids : `restore_best` les rechargera en fin de run.
                    best_weights = [np.asarray(array) for array in network.get_weights()]
            if plateau is not None:
                plateau.step(optimizer, float(logs.get(monitor, logs["loss"])))
            epochs_trained = epoch + 1

            if context is not None and self.emit_epoch(
                context, list(callbacks or []), logs, epoch=epoch
            ):
                logger.info(
                    "Arrêt anticipé demandé à l'époque {} | {}={:.5f}",
                    epoch,
                    monitor,
                    logs.get(monitor, float("nan")),
                )
                break

        if restore_best and best_weights is not None and best_epoch != epochs_trained - 1:
            network.set_weights(best_weights)
            logger.info("Poids de la meilleure époque restaurés | époque={}", best_epoch)

        self.network_ = network
        self.params = dict(params)
        self.input_dim_, self.output_dim_ = input_dim, output_dim
        self.best_epoch_ = best_epoch if best_epoch >= 0 else None
        self.epochs_trained_ = epochs_trained
        if context is not None:
            context.extra.update(
                {
                    "best_epoch": self.best_epoch_,
                    "epochs_trained": epochs_trained,
                    "n_parameters": count_parameters(network),
                    "architecture": self.spec.name,
                    "steps_per_epoch": steps_per_epoch,
                }
            )
        logger.info(
            "TensorFlow fitted | {} époque(s) | meilleure={} | paramètres={} | pas/époque={}",
            epochs_trained,
            best_epoch if best_epoch >= 0 else None,
            count_parameters(network),
            steps_per_epoch,
        )

    def _encode_batch(
        self, labels: np.ndarray | None, matrix: np.ndarray, params: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode a target vector (or the features themselves) and its sample weights.

        Args:
            labels: Business targets (``None`` for an autoencoder).
            matrix: Feature matrix (used as the reconstruction target).
            params: **Resolved** parameters (``self.params`` still holds ``"auto"`` before fit).

        Returns:
            ``(targets, weights)`` ready for the ``tf.data`` pipeline.
        """
        if labels is None or self.task in _ANOMALY or self.spec.name == "autoencoder":
            targets = np.ascontiguousarray(matrix, dtype="float32")
            weights = np.ones(len(matrix), dtype="float32")
            return targets, weights
        encoded = _encode_labels(
            labels, self.task, self.classes_, target_mean=self._target_mean_, target_std=self._target_std_
        )
        if self.task == "multiclass":
            targets = np.ascontiguousarray(encoded, dtype="int32")
        else:
            targets = np.ascontiguousarray(encoded, dtype="float32").reshape(-1, 1)
        weights = _sample_weights(labels, self.task, params, self.classes_)
        return targets, weights

    # ------------------------------------------------------------------ prédiction --------
    def _predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict one value per row (labels, values or reconstruction errors).

        Args:
            X: Aligned features.

        Returns:
            The predictions.
        """
        matrix = self._matrix(X)
        outputs = self._forward_matrix(matrix)
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
        probabilities = self._probabilities(self._forward_matrix(self._matrix(X)))
        if probabilities is None:
            msg = (
                f"Task '{self.task}' with architecture '{self.algorithm}' exposes no "
                "probabilities; use predict()."
            )
            raise NotImplementedError(msg)
        return probabilities

    def _forward_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Run the network over a matrix, batch by batch, in inference mode.

        Args:
            matrix: Aligned float matrix.

        Returns:
            The raw outputs (logits, values or reconstructions).
        """
        network = self._require_network()
        outputs: list[np.ndarray] = []
        for start in range(0, len(matrix), INFERENCE_BATCH_SIZE):
            chunk = tf.constant(
                np.ascontiguousarray(matrix[start : start + INFERENCE_BATCH_SIZE]),
                dtype="float32",
            )
            outputs.append(np.asarray(network(chunk, training=False)))
        if not outputs:
            return np.zeros((0, max(self.output_dim_, 1)), dtype="float64")
        return np.concatenate(outputs, axis=0)

    def _decode(self, outputs: np.ndarray) -> np.ndarray:
        """Turn raw network outputs into business predictions.

        Args:
            outputs: Logits (classification) or values (regression).

        Returns:
            Class labels for a classification task, target values otherwise.
        """
        if self.task == "multiclass":
            codes = np.argmax(np.asarray(outputs, dtype="float64"), axis=1)
            space = self._label_list()
            if not space:
                return codes.astype("int64")
            return np.asarray([space[min(int(code), len(space) - 1)] for code in codes])
        if self.task == "binary":
            scores = _sigmoid(np.asarray(outputs, dtype="float64")).ravel()
            space = self._label_list() or [0, 1]
            positive, negative = space[-1], space[0]
            threshold = self._decision_threshold()
            return np.asarray([positive if score >= threshold else negative for score in scores])
        values = np.asarray(outputs, dtype="float64").ravel()
        if self._target_mean_ is not None and self._target_std_:
            # Dé-normalisation : les prédictions retrouvent l'unité métier de la cible.
            values = values * float(self._target_std_) + float(self._target_mean_)
        return values

    def _probabilities(self, outputs: np.ndarray) -> np.ndarray | None:
        """Turn logits into probabilities (``None`` when the task has none).

        Args:
            outputs: Raw network outputs.

        Returns:
            A ``(n_samples, n_classes)`` array for classification, ``None`` otherwise.
        """
        array = np.asarray(outputs, dtype="float64")
        if self.task == "binary":
            positive = _sigmoid(array).ravel()
            return np.column_stack([1.0 - positive, positive])
        if self.task == "multiclass":
            return _softmax(array if array.ndim > 1 else array.reshape(-1, 1))
        return None

    # ------------------------------------------------------------------ persistance -------
    def save(self, path: str | Path) -> Path:
        """Persist the model: native weights + sidecar JSON, or a portable single-file archive.

        Args:
            path: Destination file or directory. A name ending with ``.weights.h5`` uses Keras'
                native format (plus a ``*.config.json`` sidecar); any other extension produces a
                self-contained archive holding both the configuration and the weights.

        Returns:
            The written path.
        """
        self.check_is_fitted()
        network = self._require_network()
        destination = self._resolve_path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name.endswith(WEIGHTS_SUFFIX):
            # Modèle subclassé : Keras ne sait pas sérialiser le graphe, d'où le sidecar JSON.
            network.save_weights(destination)
            sidecar = _sidecar_path(destination)
            sidecar.write_text(
                json.dumps(_json_safe(self._artefact_config()), indent=2), encoding="utf-8"
            )
            logger.info(
                "Model saved: {} (+ {}) | {} | {} paramètres",
                destination,
                sidecar.name,
                self.algorithm,
                count_parameters(network),
            )
            return destination
        _write_portable(destination, self._artefact_config(), network)
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

        The graph is reconstructed from the stored configuration, then the weights are attached —
        either through ``load_weights`` (native format) or ``set_weights`` (portable archive).

        Args:
            path: Artefact path.
            config: Optional configuration refreshed into the reloaded model.

        Returns:
            The reloaded :class:`TensorFlowModel`, ready to predict.

        Raises:
            FileNotFoundError: When the artefact does not exist.
            TypeError: When the artefact is not a model of this project.
        """
        source = Path(path)
        if not source.exists():
            msg = f"Model artefact not found: {source}"
            raise FileNotFoundError(msg)
        weights: list[np.ndarray] | None = None
        if source.name.endswith(WEIGHTS_SUFFIX):
            payload = _read_sidecar(_sidecar_path(source), source)
        else:
            payload, weights = _read_portable(source)
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
        model._restore(payload, weights_path=None if weights else source, weights=weights)
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
        """Assemble the configuration payload written next to (or inside) the weights."""
        network = self._require_network()
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
            "network_config": network.get_config(),
            "classes": _json_safe(self.classes_) if self.classes_ is not None else None,
            "target_mean": self._target_mean_,
            "target_std": self._target_std_,
            "best_epoch": self.best_epoch_,
            "epochs_trained": int(self.epochs_trained_),
            "keras_version": str(keras.version()),
            "tensorflow_version": str(tf.__version__),
        }

    def _restore(
        self,
        payload: Mapping[str, Any],
        *,
        weights_path: Path | None = None,
        weights: list[np.ndarray] | None = None,
    ) -> None:
        """Rebuild the graph and attach the stored weights.

        Args:
            payload: Artefact configuration.
            weights_path: Native ``.weights.h5`` file, when applicable.
            weights: Ordered weight arrays of a portable archive, when applicable.

        Raises:
            TypeError: When neither weights source is provided.
        """
        input_dim = int(payload["input_dim"])
        output_dim = int(payload["output_dim"])
        params = dict(payload.get("params") or {})
        network = self.spec.builder(input_dim=input_dim, output_dim=output_dim, params=params)
        network(tf.zeros((2, input_dim), dtype="float32"), training=False)
        if weights is not None:
            network.set_weights(weights)
        elif weights_path is not None:
            network.load_weights(weights_path)
        else:  # pragma: no cover - garde-fou
            msg = "No weights were provided to restore the model"
            raise TypeError(msg)
        self.network_ = network
        self.params = params
        self.input_dim_, self.output_dim_ = input_dim, output_dim
        stored = payload.get("classes")
        self.classes_ = None if stored is None else np.asarray(stored, dtype="object")
        self._target_mean_ = payload.get("target_mean")
        self._target_std_ = payload.get("target_std")
        self.best_epoch_ = payload.get("best_epoch")
        self.epochs_trained_ = int(payload.get("epochs_trained", 0) or 0)
        self._is_fitted = True
        self.fit_result_ = FitResult(
            model_name=type(self).__name__,
            algorithm=str(self.algorithm),
            epochs=self.epochs_trained_,
            extra={"restored_from_artefact": True},
        )
        logger.info(
            "Model restored | {} | entrées={} sorties={} époques={}",
            self.summary(),
            input_dim,
            output_dim,
            self.epochs_trained_,
        )

    def _require_network(self) -> TabularNet:
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
        try:
            return float(self._train_node().get("learning_rate", 1e-3) or 1e-3)
        except (TypeError, ValueError):  # pragma: no cover - défensif
            return 1e-3

    def _weight_decay(self) -> float:
        """Decoupled weight decay declared in the configuration (applied by AdamW)."""
        try:
            return float(self._train_node().get("weight_decay", 0.0) or 0.0)
        except (TypeError, ValueError):  # pragma: no cover - défensif
            return 0.0

    def _primary_metric(self) -> str:
        """Primary metric of the project (``""`` when undeclared)."""
        return str(dict(self.config.get("metrics") or {}).get("primary", "") or "")

    def _decision_threshold(self) -> float:
        """Decision threshold applied to the positive probability of a binary task."""
        decision = dict(self.config.get("decision") or {})
        try:
            return float(decision.get("threshold", 0.5))
        except (TypeError, ValueError):  # pragma: no cover - défensif
            return 0.5


__all__ = [
    "CHECKPOINT_KIND",
    "CONFIG_MEMBER",
    "ESTIMATORS",
    "INFERENCE_BATCH_SIZE",
    "WEIGHTS_MEMBER",
    "WEIGHTS_SUFFIX",
    "AlgorithmSpec",
    "DenseBlock",
    "FitResult",
    "ModelCard",
    "PlateauDecay",
    "ResidualBlock",
    "TabularNet",
    "TensorFlowModel",
    "available_for_task",
    "count_parameters",
    "filter_params",
    "resolve_algorithm",
    "seed_tensorflow",
]
