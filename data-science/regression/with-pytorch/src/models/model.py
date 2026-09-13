"""PyTorch implementation of the model contract.

Where scikit-learn and the boosters hide the optimisation loop, this module writes it out:
``DataLoader`` → forward pass → loss → ``backward`` → ``optimizer.step`` → scheduler. Every epoch
emits **one** event to the project callbacks, so history, logging, progress bar and quality
thresholds behave exactly like they do with the other stacks — the contract is what makes the
framework interchangeable, not the framework itself.

Design choices worth reading before the code:

* **logits, never probabilities, inside the graph.** Binary classification uses
  ``BCEWithLogitsLoss`` (numerically stable log-sum-exp) and the sigmoid is applied only at
  inference; ``pos_weight`` carries the class imbalance.
* **target normalisation for regression.** Optimising an MSE on amounts expressed in euros is
  dominated by the scale of the target: the model standardises ``y`` on the training split and
  inverse-transforms its predictions (both statistics are stored in the checkpoint).
* **determinism.** ``torch.manual_seed`` before weight initialisation, a seeded ``Generator`` for
  the shuffling, ``num_workers=0`` and ``torch.use_deterministic_algorithms``: two runs with the
  same seed produce the same predictions, which the test suite asserts.
* **checkpoint, not pickle.** ``save`` writes ``torch.save({...})`` — weights plus the metadata
  needed to rebuild the graph — and ``load`` reads it with ``weights_only=True``, so reloading an
  artefact never executes foreign code.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseModel, FitResult, ModelCard, library_versions
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Marqueur écrit dans chaque checkpoint : permet à :meth:`load` de rejeter un artefact étranger.
CHECKPOINT_KIND = "tabular-project.pytorch-checkpoint/v1"

#: Tâches apprises par les têtes de classification.
_CLASSIFICATION = frozenset({"binary", "multiclass"})
#: Tâches apprises par les têtes de régression.
_REGRESSION = frozenset({"regression", "forecasting"})
#: Tâches non supervisées (auto-encodeur : l'erreur de reconstruction sert de score d'anomalie).
_ANOMALY = frozenset({"anomaly"})

#: Fonctions d'activation acceptées par le paramètre ``activation``.
_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "leaky_relu": nn.LeakyReLU,
}

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
#: aussi une simple séquence (checkpoint relu, appel direct depuis un notebook).
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
    builder: Callable[..., nn.Module]
    rationale: str = ""
    iterative: bool = True
    defaults: dict[str, Any] = field(default_factory=dict)
    accepted: frozenset[str] = _COMMON_PARAMS


# ---------------------------------------------------------------------------------------------
# Briques de réseau
# ---------------------------------------------------------------------------------------------
def _activation(name: str) -> nn.Module:
    """Instantiate an activation function by name.

    Args:
        name: Key of :data:`_ACTIVATIONS`.

    Returns:
        The activation module.

    Raises:
        ValueError: When the activation is unknown.
    """
    factory = _ACTIVATIONS.get(str(name).lower())
    if factory is None:
        msg = f"Unknown activation '{name}'. Available: {sorted(_ACTIVATIONS)}"
        raise ValueError(msg)
    return factory()


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


def _dropout(rate: float) -> nn.Module:
    """Return a dropout layer, or an identity when the rate is zero."""
    return nn.Dropout(float(rate)) if float(rate) > 0 else nn.Identity()


def _dense_stack(
    input_dim: int,
    widths: Sequence[int],
    *,
    dropout: float,
    activation: str,
    batch_norm: bool,
) -> list[nn.Module]:
    """Build the ``Linear → [BatchNorm] → activation → Dropout`` blocks of a dense network.

    Args:
        input_dim: Width of the first block's input.
        widths: Hidden layer widths.
        dropout: Dropout probability (``0`` disables the layer).
        activation: Activation name.
        batch_norm: Whether to normalise each hidden representation.

    Returns:
        The ordered list of layers.
    """
    layers: list[nn.Module] = []
    width = int(input_dim)
    for units in widths:
        layers.append(nn.Linear(width, int(units)))
        if batch_norm:
            # BatchNorm stabilise l'apprentissage mais exige des lots de taille >= 2 à
            # l'entraînement : le dernier lot d'une époque est ignoré dans ce cas.
            layers.append(nn.BatchNorm1d(int(units)))
        layers.append(_activation(activation))
        layers.append(_dropout(dropout))
        width = int(units)
    return layers


def _mlp(input_dim: int, output_dim: int, params: Mapping[str, Any]) -> nn.Module:
    """Build a plain multi-layer perceptron.

    Args:
        input_dim: Number of input features.
        output_dim: Number of outputs (1 for binary/regression, ``n_classes`` otherwise).
        params: Resolved parameters.

    Returns:
        The untrained network.
    """
    widths = _hidden_layers(params)
    layers = _dense_stack(
        input_dim,
        widths,
        dropout=float(params.get("dropout", 0.0)),
        activation=str(params.get("activation", "relu")),
        batch_norm=bool(params.get("batch_norm", False)),
    )
    layers.append(nn.Linear(int(widths[-1]), int(output_dim)))
    return nn.Sequential(*layers)


def _linear(input_dim: int, output_dim: int, params: Mapping[str, Any]) -> nn.Module:
    """Build a single dense layer (logistic / linear regression trained by gradient).

    Args:
        input_dim: Number of input features.
        output_dim: Number of outputs.
        params: Resolved parameters (unused: a linear model has no hidden width).

    Returns:
        The untrained network.
    """
    del params
    return nn.Sequential(nn.Linear(int(input_dim), int(output_dim)))


class ResidualBlock(nn.Module):
    """Two dense layers wrapped in a skip connection.

    The residual path keeps the gradient flowing when the network is deep: the block learns a
    *correction* of its input instead of the whole transformation.
    """

    def __init__(self, width: int, *, dropout: float, activation: str, batch_norm: bool) -> None:
        """Build the block for a fixed width.

        Args:
            width: Number of units (identical in input and output, so the skip is exact).
            dropout: Dropout probability.
            activation: Activation name.
            batch_norm: Whether to normalise the hidden representation.
        """
        super().__init__()
        self.first = nn.Linear(width, width)
        self.norm: nn.Module = nn.BatchNorm1d(width) if batch_norm else nn.Identity()
        self.activation = _activation(activation)
        self.dropout = _dropout(dropout)
        self.second = nn.Linear(width, width)
        self.output_dropout = _dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the block and add its input back.

        Args:
            features: Batch of representations.

        Returns:
            The corrected representations.
        """
        hidden = self.dropout(self.activation(self.norm(self.first(features))))
        return features + self.output_dropout(self.second(hidden))


class ResidualMLP(nn.Module):
    """A stem, a stack of residual blocks and a head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_layers: Sequence[int],
        dropout: float,
        activation: str,
        batch_norm: bool,
    ) -> None:
        """Build the network.

        Args:
            input_dim: Number of input features.
            output_dim: Number of outputs.
            hidden_layers: Hidden widths; the first one sets the block width, the count sets the
                number of residual blocks.
            dropout: Dropout probability.
            activation: Activation name.
            batch_norm: Whether to normalise hidden representations.
        """
        super().__init__()
        widths = [int(units) for units in hidden_layers]
        block_width = widths[0]
        self.stem = nn.Sequential(
            *_dense_stack(
                input_dim,
                [block_width],
                dropout=dropout,
                activation=activation,
                batch_norm=batch_norm,
            )
        )
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(
                    block_width, dropout=dropout, activation=activation, batch_norm=batch_norm
                )
                for _ in widths
            ]
        )
        self.head = nn.Linear(block_width, int(output_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run stem, blocks and head.

        Args:
            features: Batch of inputs.

        Returns:
            The logits of the batch.
        """
        return self.head(self.blocks(self.stem(features)))


def _residual_mlp(input_dim: int, output_dim: int, params: Mapping[str, Any]) -> nn.Module:
    """Build a residual multi-layer perceptron.

    Args:
        input_dim: Number of input features.
        output_dim: Number of outputs.
        params: Resolved parameters.

    Returns:
        The untrained network.
    """
    return ResidualMLP(
        int(input_dim),
        int(output_dim),
        hidden_layers=_hidden_layers(params),
        dropout=float(params.get("dropout", 0.0)),
        activation=str(params.get("activation", "relu")),
        batch_norm=bool(params.get("batch_norm", False)),
    )


class Autoencoder(nn.Module):
    """Symmetric encoder/decoder used for reconstruction-based anomaly detection.

    The network is trained to reproduce its input; a sample it fails to reconstruct is far from
    the training manifold, which is exactly the anomaly signal.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_layers: Sequence[int],
        code_dim: int,
        dropout: float,
        activation: str,
        batch_norm: bool,
    ) -> None:
        """Build the encoder and its mirrored decoder.

        Args:
            input_dim: Number of input features (also the reconstruction width).
            hidden_layers: Encoder widths (the decoder mirrors them).
            code_dim: Latent width.
            dropout: Dropout probability.
            activation: Activation name.
            batch_norm: Whether to normalise hidden representations.
        """
        super().__init__()
        widths = [int(units) for units in hidden_layers]
        self.code_dim = int(code_dim)
        self.encoder = nn.Sequential(
            *_dense_stack(
                input_dim,
                widths,
                dropout=dropout,
                activation=activation,
                batch_norm=batch_norm,
            ),
            nn.Linear(int(widths[-1]), int(code_dim)),
        )
        decoder: list[nn.Module] = []
        width = int(code_dim)
        for units in reversed(widths):
            decoder.append(nn.Linear(width, units))
            if batch_norm:
                decoder.append(nn.BatchNorm1d(units))
            decoder.append(_activation(activation))
            decoder.append(_dropout(dropout))
            width = units
        decoder.append(nn.Linear(width, int(input_dim)))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Project a batch into the latent space.

        Args:
            features: Batch of inputs.

        Returns:
            The latent codes.
        """
        return self.encoder(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch.

        Args:
            features: Batch of inputs.

        Returns:
            The reconstructions.
        """
        return self.decoder(self.encode(features))


def _autoencoder(input_dim: int, output_dim: int, params: Mapping[str, Any]) -> nn.Module:
    """Build the autoencoder used for anomaly detection.

    Args:
        input_dim: Number of input features.
        output_dim: Ignored (the reconstruction width is ``input_dim``).
        params: Resolved parameters.

    Returns:
        The untrained network.
    """
    del output_dim
    return Autoencoder(
        int(input_dim),
        hidden_layers=_hidden_layers(params),
        code_dim=_code_dim(params, input_dim),
        dropout=float(params.get("dropout", 0.0)),
        activation=str(params.get("activation", "relu")),
        batch_norm=bool(params.get("batch_norm", False)),
    )


def _code_dim(params: Mapping[str, Any], input_dim: int) -> int:
    """Resolve the latent width of an autoencoder.

    ``code_dim`` is stored in the checkpoint so that reloading rebuilds the exact same graph;
    ``code_ratio`` is only used to derive it when no explicit width is configured.

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


#: Registry of every architecture this stack can build.
ESTIMATORS: dict[str, AlgorithmSpec] = {
    spec.name: spec
    for spec in (
        AlgorithmSpec(
            name="mlp",
            display_name="Perceptron multicouche (MLP)",
            tasks=_CLASSIFICATION | _REGRESSION | _ANOMALY,
            builder=_mlp,
            rationale=(
                "Empilement de couches denses avec dropout optionnel : le réseau de référence sur "
                "données tabulaires pré-traitées. `hidden_layers` fixe la capacité, `dropout` et "
                "`weight_decay` la régularisation — les deux leviers à explorer en premier."
            ),
            defaults={
                "hidden_layers": [64, 32],
                "dropout": 0.2,
                "activation": "relu",
                "batch_norm": False,
                "optimizer": "adamw",
                "scheduler": "none",
            },
        ),
        AlgorithmSpec(
            name="linear",
            display_name="Modèle linéaire entraîné par gradient",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_linear,
            rationale=(
                "Une seule couche dense : régression logistique / linéaire optimisée par descente "
                "de gradient. Baseline interprétable (les poids sont les coefficients) et "
                "référence de coût — un réseau qui ne la bat pas n'apporte rien."
            ),
            defaults={
                "dropout": 0.0,
                "batch_norm": False,
                "optimizer": "adamw",
                "scheduler": "none",
            },
        ),
        AlgorithmSpec(
            name="residual_mlp",
            display_name="MLP à blocs résiduels",
            tasks=_CLASSIFICATION | _REGRESSION,
            builder=_residual_mlp,
            rationale=(
                "Connexions résiduelles entre blocs denses : le gradient circule même quand on "
                "empile les couches, ce qui permet d'aller plus profond sans dégrader "
                "l'apprentissage. À comparer au MLP simple à capacité égale."
            ),
            defaults={
                "hidden_layers": [64, 64],
                "dropout": 0.15,
                "activation": "relu",
                "batch_norm": True,
                "optimizer": "adamw",
                "scheduler": "cosine",
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
            f"Unknown PyTorch architecture '{algorithm}'. Available: {sorted(ESTIMATORS)} "
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
# Déterminisme, cible et pertes
# ---------------------------------------------------------------------------------------------
def seed_torch(seed: int, device: torch.device) -> None:
    """Seed every random source PyTorch depends on.

    Args:
        seed: Reproducibility seed.
        device: Training device (CUDA needs its own generator seeding).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":  # pragma: no cover - exécution CPU par défaut dans ce dépôt
        torch.cuda.manual_seed_all(seed)
    # Déterminisme : indispensable pour que deux runs à graine fixée produisent les mêmes
    # prédictions (vérifié par `test_training_is_reproducible`).
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(network: nn.Module) -> int:
    """Count the trainable parameters of a network.

    Args:
        network: Network to inspect.

    Returns:
        The number of trainable scalars.
    """
    return int(
        sum(parameter.numel() for parameter in network.parameters() if parameter.requires_grad)
    )


def _resolve_device(model_node: Mapping[str, Any]) -> torch.device:
    """Resolve the training device, falling back to CPU when CUDA is unavailable."""
    requested = str(model_node.get("device", "cpu") or "cpu").lower()
    if requested in {"cuda", "gpu"} and not torch.cuda.is_available():
        logger.warning("CUDA demandé mais indisponible : repli sur CPU (déterminisme garanti)")
        return torch.device("cpu")
    if requested == "gpu":
        return torch.device("cuda")
    return torch.device(requested)


def _label_space(labels: np.ndarray | None) -> np.ndarray | None:
    """Return the sorted label space (``None`` for an unsupervised task).

    The contract stores ``classes_`` as an array (like scikit-learn); sorting makes the positive
    class deterministic — index ``-1`` — which the ranking metrics read as ``y_proba[:, 1]``.
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
    if task == "multiclass":
        space = label_list(classes)
        index = {label: position for position, label in enumerate(space)}
        try:
            return np.asarray([index[label] for label in labels], dtype="int64")
        except KeyError as error:
            msg = f"Target label {error} is absent from the training label space {space}"
            raise ValueError(msg) from error
    positive = space_positive(classes)
    return (np.asarray(labels) == positive).astype("float32")


def space_positive(classes: LabelSpace) -> Any:
    """Return the label treated as the positive class of a binary task.

    Args:
        classes: Sorted label space (``None`` falls back to ``{0, 1}``).

    Returns:
        The positive label.
    """
    space = label_list(classes)
    return space[-1] if space else 1


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
        return np.asarray([space[min(max(int(code), 0), len(space) - 1)] for code in codes])
    positive = space_positive(classes)
    negative = space[0] if space else 0
    flags = np.asarray(encoded).ravel() >= 0.5
    return np.asarray([positive if flag else negative for flag in flags])


def _loss_function(
    task: str, params: Mapping[str, Any], device: torch.device, labels: np.ndarray | None
) -> nn.Module:
    """Build the loss matching the task.

    Args:
        task: Learning task.
        params: Resolved parameters (``pos_weight`` / ``class_weight``).
        device: Training device.
        labels: Training target (used for the multiclass weight tensor).

    Returns:
        The loss module.
    """
    del labels
    if task == "binary":
        weight = params.get("pos_weight")
        pos_weight = (
            torch.tensor([float(weight)], dtype=torch.float32, device=device) if weight else None
        )
        # Logits en entrée : la fusion sigmoid + log est numériquement stable.
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if task == "multiclass":
        weights = params.get("class_weight")
        class_weight = (
            torch.tensor([float(value) for value in weights], dtype=torch.float32, device=device)
            if weights
            else None
        )
        return nn.CrossEntropyLoss(weight=class_weight)
    # Régression, prévision et reconstruction (auto-encodeur) : erreur quadratique moyenne.
    return nn.MSELoss()


def _optimizer(
    network: nn.Module, params: Mapping[str, Any], *, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    """Build the optimizer declared in the parameters.

    Args:
        network: Network whose parameters are optimised.
        params: Resolved parameters (``optimizer``, ``momentum``).
        learning_rate: Step size.
        weight_decay: L2 penalty (decoupled for AdamW).

    Returns:
        The optimizer.

    Raises:
        ValueError: When the optimizer name is unknown.
    """
    name = str(params.get("optimizer", "adamw")).lower()
    trainable = network.parameters()
    if name == "adamw":
        return torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(trainable, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            trainable,
            lr=learning_rate,
            momentum=float(params.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(trainable, lr=learning_rate, weight_decay=weight_decay)
    msg = f"Unknown optimizer '{name}'. Available: {sorted(_OPTIMIZERS)}"
    raise ValueError(msg)


def _scheduler(
    optimizer: torch.optim.Optimizer,
    params: Mapping[str, Any],
    *,
    steps_per_epoch: int,
    epochs: int,
    mode: str,
    monitor: str,
) -> Any:
    """Build the learning-rate scheduler declared in the parameters.

    ``cosine`` is stepped **per optimiser step** (it anneals over the whole budget), ``plateau``
    and ``step`` are stepped **per epoch**.

    Args:
        optimizer: Optimizer to drive.
        params: Resolved parameters (``scheduler``, ``step_size``, ``plateau_*``).
        steps_per_epoch: Number of optimiser steps in one epoch.
        epochs: Total epoch budget.
        mode: ``min`` / ``max``, used by the plateau scheduler.
        monitor: Metric watched by the plateau scheduler.

    Returns:
        A scheduler, or ``None`` when disabled.
    """
    name = str(params.get("scheduler", "none")).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(steps_per_epoch) * int(epochs), 1)
        )
    if name == "plateau":
        del monitor  # la métrique surveillée est passée à `step()` par la boucle d'époques
        plateau_mode: Literal["min", "max"] = "max" if mode == "max" else "min"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=plateau_mode,
            factor=float(params.get("plateau_factor", 0.5)),
            patience=int(params.get("plateau_patience", 5)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(int(params.get("step_size", 10)), 1),
            gamma=float(params.get("step_gamma", 0.5)),
        )
    return None


def _data_loader(
    matrix: np.ndarray,
    labels: np.ndarray | None,
    *,
    task: str,
    classes: LabelSpace,
    target_mean: float | None,
    target_std: float | None,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader[Any]:
    """Build the batch iterator of one epoch.

    Args:
        matrix: Feature matrix.
        labels: Business target (``None`` for the autoencoder).
        task: Learning task.
        classes: Sorted label space.
        target_mean: Normalisation mean of a continuous target.
        target_std: Normalisation standard deviation of a continuous target.
        batch_size: Number of rows per optimiser step.
        seed: Seed of the shuffling generator (reproducibility).
        shuffle: Whether to shuffle the rows.

    Returns:
        A ``DataLoader`` of ``(features, targets)`` tensors.
    """
    features = torch.as_tensor(np.ascontiguousarray(matrix), dtype=torch.float32)
    if labels is None:
        # Auto-encodeur : la cible est l'entrée elle-même (reconstruction).
        targets = features.clone()
    elif task in _ANOMALY:
        targets = features.clone()
    else:
        encoded = _encode_labels(
            labels, task, classes, target_mean=target_mean, target_std=target_std
        )
        dtype = torch.int64 if task == "multiclass" else torch.float32
        shaped = encoded.reshape(-1, 1) if dtype is torch.float32 else encoded
        targets = torch.as_tensor(np.ascontiguousarray(shaped), dtype=dtype)
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(features, targets),
        batch_size=max(int(batch_size), 1),
        shuffle=shuffle,
        # `num_workers=0` : chargement dans le processus principal, seul mode réellement
        # déterministe (aucun ordre de worker, aucune sérialisation inter-processus).
        num_workers=0,
        generator=generator if shuffle else None,
        drop_last=False,
    )


def _run_epoch(
    network: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Any = None,
    clip_norm: float | None = None,
    skip_singleton_batches: bool = False,
    device: torch.device | None = None,
) -> float:
    """Run one training epoch: forward, loss, backward, optimiser step.

    Args:
        network: Network in training mode.
        loader: Batch iterator.
        criterion: Loss module.
        optimizer: Optimizer stepped after each batch.
        scheduler: Scheduler stepped **per batch** (``cosine``), if any.
        clip_norm: Global gradient-norm clipping (``None`` disables it).
        skip_singleton_batches: Ignore batches of a single row (``BatchNorm1d`` needs >= 2).
        device: Device the batches are moved to.

    Returns:
        The mean training loss of the epoch.
    """
    network.train()
    total = 0.0
    seen = 0
    for features, targets in loader:
        if skip_singleton_batches and int(features.shape[0]) < 2:
            continue
        if device is not None:
            features = features.to(device)
            targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = network(features)
        loss = criterion(outputs, targets)
        loss.backward()
        if clip_norm:
            # Écrêtage de la norme globale : évite qu'un lot aberrant ne détruise les poids.
            torch.nn.utils.clip_grad_norm_(network.parameters(), float(clip_norm))
        optimizer.step()
        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler.step()
        batch = int(features.shape[0])
        total += float(loss.item()) * batch
        seen += batch
    return total / max(seen, 1)


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
    task: str,
    metric: str,
    *,
    y_true: Any,
    y_pred: Any,
    y_proba: Any,
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


def _read_checkpoint(source: Path) -> dict[str, Any]:
    """Read and validate a checkpoint written by :meth:`PyTorchModel.save`.

    Args:
        source: Artefact path.

    Returns:
        The checkpoint payload.

    Raises:
        TypeError: When the file is not a checkpoint of this project.
    """
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as error:
        # Tout artefact illisible (joblib, .keras, fichier corrompu) doit produire un diagnostic
        # exploitable plutôt qu'une erreur de désérialisation opaque.
        msg = (
            f"'{source.name}' is not a PyTorch checkpoint of this project "
            f"({type(error).__name__}). Point model_file at the artefact written by "
            "PyTorchModel.save (weights + graph metadata, read with weights_only=True)."
        )
        raise TypeError(msg) from error
    if not isinstance(payload, dict) or payload.get("kind") != CHECKPOINT_KIND:
        msg = (
            f"'{source.name}' does not hold the '{CHECKPOINT_KIND}' marker "
            f"(got {type(payload).__name__}); it was not written by PyTorchModel.save."
        )
        raise TypeError(msg)
    return payload


def _json_safe(value: Any) -> Any:
    """Convert numpy / torch scalars into plain Python containers.

    ``torch.load(weights_only=True)`` refuses arbitrary objects: everything stored in the
    checkpoint must be a tensor, a primitive or a container of those.

    Args:
        value: Value to normalise.

    Returns:
        A checkpoint-safe value.
    """
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid over an array."""
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype="float64")))


def _softmax(values: np.ndarray) -> np.ndarray:
    """Row-wise softmax over a 2-D array."""
    shifted = np.asarray(values, dtype="float64")
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


class PyTorchModel(BaseModel):
    """A neural network behind the project's model contract."""

    framework = "pytorch"
    default_model_file = "model.pt"
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
        self.device_: torch.device = _resolve_device(dict(self.config.get("model") or {}))
        self.network_: nn.Module | None = None
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
            ValueError: When the optimizer or the scheduler is unknown.
        """
        resolved: dict[str, Any] = {**self.spec.defaults, **self._effective_params()}
        if str(resolved.get("optimizer", "adamw")).lower() not in _OPTIMIZERS:
            msg = (
                f"Unknown optimizer '{resolved.get('optimizer')}'. Available: {sorted(_OPTIMIZERS)}"
            )
            raise ValueError(msg)
        if str(resolved.get("scheduler", "none")).lower() not in _SCHEDULERS:
            msg = (
                f"Unknown scheduler '{resolved.get('scheduler')}'. Available: {sorted(_SCHEDULERS)}"
            )
            raise ValueError(msg)
        if labels is not None and self.task in _CLASSIFICATION:
            counts = pd.Series(labels).value_counts().sort_index().to_numpy(dtype="float64")
            positive = float(counts[-1]) if counts.size else 0.0
            negative = float(counts[:-1].sum()) if counts.size > 1 else 0.0
            if str(resolved.get("pos_weight")) == "auto":
                # BCEWithLogitsLoss : poids de la classe positive = négatifs / positifs.
                resolved["pos_weight"] = round(max(negative / max(positive, 1.0), 1.0), 4)
            if str(resolved.get("class_weight")) == "auto":
                total = float(counts.sum()) or 1.0
                resolved["class_weight"] = [
                    round(total / (len(counts) * max(float(count), 1.0)), 4) for count in counts
                ]
        cleaned = {key: value for key, value in resolved.items() if value != "auto"}
        if self.spec.name == "autoencoder":
            # Le goulot dépend de la largeur d'entrée : figé ici pour que le checkpoint
            # reconstruise exactement le même graphe.
            cleaned.setdefault("code_dim", _code_dim(cleaned, len(self.feature_names) or 1))
        return filter_params(self.spec, cleaned)

    def build_network(
        self, input_dim: int, output_dim: int, labels: np.ndarray | None = None
    ) -> nn.Module:
        """Instantiate the network declared by ``algorithm`` (used by the notebooks).

        Args:
            input_dim: Number of input features.
            output_dim: Number of outputs.
            labels: Training target, used to resolve the imbalance weights.

        Returns:
            A fresh (untrained) network on the model's device.
        """
        params = self._resolved_params(labels)
        if self.spec.name == "autoencoder":
            params["code_dim"] = _code_dim(params, input_dim)
        return self.spec.builder(input_dim=input_dim, output_dim=output_dim, params=params).to(
            self.device_
        )

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
        """Run the explicit epoch loop (batches, forward, loss, backward, optimiser step).

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
        seed_torch(self.random_state, self.device_)

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
        network = self.spec.builder(input_dim=input_dim, output_dim=output_dim, params=params).to(
            self.device_
        )
        criterion = _loss_function(self.task, params, self.device_, labels)
        optimizer = _optimizer(
            network,
            params,
            learning_rate=self._learning_rate(),
            weight_decay=self._weight_decay(),
        )
        train_loader = _data_loader(
            matrix,
            labels,
            task=self.task,
            classes=self.classes_,
            target_mean=self._target_mean_,
            target_std=self._target_std_,
            batch_size=self._batch_size(),
            seed=self.random_state,
            shuffle=True,
        )
        validation = None
        if X_val is not None:
            validation = self._tensor_batch(
                self._matrix(X_val), None if y_val is None else np.asarray(y_val).ravel()
            )

        monitor, mode = _monitored_metric(self._train_node(), self._primary_metric())
        scheduler = _scheduler(
            optimizer,
            params,
            steps_per_epoch=len(train_loader),
            epochs=self._epochs(),
            mode=mode,
            monitor=monitor,
        )
        per_batch = isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        skip_singleton = any(isinstance(layer, nn.BatchNorm1d) for layer in network.modules())
        stopping = dict(self._train_node().get("early_stopping") or {})
        restore_best = bool(stopping.get("restore_best"))

        best_state = deepcopy(network.state_dict())
        best_value = float("inf") if mode == "min" else float("-inf")
        best_epoch = -1
        epochs_trained = 0

        for epoch in range(self._epochs()):
            train_loss = _run_epoch(
                network,
                train_loader,
                criterion,
                optimizer,
                scheduler=scheduler if per_batch else None,
                clip_norm=params.get("clip_norm"),
                skip_singleton_batches=skip_singleton,
                device=self.device_,
            )
            logs: dict[str, float] = {
                "loss": float(train_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "epoch": float(epoch),
            }
            if validation is not None:
                logs.update(self._validation_logs(network, criterion, validation))
                value = float(logs.get(monitor, logs["val_loss"]))
                improved = value < best_value if mode == "min" else value > best_value
                if improved:
                    best_value, best_epoch = value, epoch
                    best_state = deepcopy(network.state_dict())
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(float(logs.get(monitor, logs["loss"])))
            elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler.step()
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

        if restore_best and best_epoch >= 0 and best_epoch != epochs_trained - 1:
            network.load_state_dict(best_state)
            logger.info("Poids de la meilleure époque restaurés | époque={}", best_epoch)

        network.eval()
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
                    "device": str(self.device_),
                    "architecture": self.spec.name,
                }
            )
        logger.info(
            "PyTorch fitted | {} époque(s) | meilleure={} | paramètres={} | device={}",
            epochs_trained,
            best_epoch if best_epoch >= 0 else None,
            count_parameters(network),
            self.device_,
        )

    def _validation_logs(
        self,
        network: nn.Module,
        criterion: nn.Module,
        validation: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, float]:
        """Score the validation split: loss plus the project's primary metric.

        Args:
            network: Network being trained (switched to ``eval`` for the duration).
            criterion: Loss module.
            validation: Validation tensors.

        Returns:
            The ``val_*`` logs of the epoch.
        """
        features, targets = validation
        network.eval()
        with torch.no_grad():
            tensor_outputs = network(features)
            loss = criterion(tensor_outputs, targets)
        outputs = tensor_outputs.detach().cpu().numpy()
        network.train()
        logs: dict[str, float] = {"val_loss": float(loss.item())}

        metric = self._primary_metric()
        if metric and self.task not in _ANOMALY:
            decoded_true = _decode_labels(
                self.task,
                targets.detach().cpu().numpy(),
                self.classes_,
                target_mean=self._target_mean_,
                target_std=self._target_std_,
            )
            value = _score_metric(
                self.task,
                metric,
                y_true=decoded_true,
                y_pred=self._decode(outputs),
                y_proba=self._probabilities(outputs),
            )
            if value is not None:
                logs[f"val_{metric}"] = value
        return logs

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
        was_training = network.training
        network.eval()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(matrix), INFERENCE_BATCH_SIZE):
                chunk = torch.as_tensor(
                    np.ascontiguousarray(matrix[start : start + INFERENCE_BATCH_SIZE]),
                    dtype=torch.float32,
                    device=self.device_,
                )
                outputs.append(network(chunk).detach().cpu().numpy())
        if was_training:
            network.train()
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
            positive = space[-1]
            negative = space[0]
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
        """Persist a checkpoint (weights + graph metadata) with ``torch.save``.

        Args:
            path: Destination file or directory.

        Returns:
            The written path.
        """
        self.check_is_fitted()
        destination = self._resolve_path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint(), destination)
        logger.info(
            "Model saved: {} ({} | {} paramètres)",
            destination,
            self.algorithm,
            count_parameters(self._require_network()),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path, *, config: Mapping[str, Any] | None = None) -> BaseModel:
        """Rebuild a model from a checkpoint written by :meth:`save`.

        The graph is reconstructed from the stored metadata and the weights are read with
        ``weights_only=True``: no Python object of the artefact is ever executed.

        Args:
            path: Artefact path.
            config: Optional configuration refreshed into the reloaded model.

        Returns:
            The reloaded :class:`PyTorchModel`, ready to predict.

        Raises:
            FileNotFoundError: When the artefact does not exist.
            TypeError: When the artefact is not a checkpoint of this project.
        """
        source = Path(path)
        if not source.exists():
            msg = f"Model artefact not found: {source}"
            raise FileNotFoundError(msg)
        payload = _read_checkpoint(source)
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
        model._restore(payload)
        return model

    # ------------------------------------------------------------------ identité ----------
    def _repr_fields(self) -> list[str]:
        """Add the architecture identity and the training budget to the representation."""
        fields = [f"algorithm={self.algorithm or '-'}"]
        if self.network_ is not None:
            fields.append(f"parameters={count_parameters(self.network_)}")
        fields += [
            f"task={self.task}",
            f"features={len(self.feature_names)}",
            f"device={self.device_}",
        ]
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
        extra.append(f"device : {self.device_}")
        card.notes = [*card.notes, *extra]
        return card

    # ------------------------------------------------------------------ interne -----------
    def _checkpoint(self) -> dict[str, Any]:
        """Assemble the payload written by :meth:`save`."""
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
            "state": network.state_dict(),
            "classes": _json_safe(self.classes_) if self.classes_ is not None else None,
            "target_mean": self._target_mean_,
            "target_std": self._target_std_,
            "best_epoch": self.best_epoch_,
            "epochs_trained": int(self.epochs_trained_),
            # `torch.__version__` est un `TorchVersion` (sous-classe de str) refusé par
            # `weights_only=True` : conversion explicite en str.
            "torch_version": str(torch.__version__),
        }

    def _restore(self, payload: Mapping[str, Any]) -> None:
        """Rebuild the graph and attach the stored weights.

        Args:
            payload: Checkpoint read from disk.
        """
        input_dim = int(payload["input_dim"])
        output_dim = int(payload["output_dim"])
        params = dict(payload.get("params") or {})
        network = self.spec.builder(input_dim=input_dim, output_dim=output_dim, params=params).to(
            self.device_
        )
        network.load_state_dict(payload["state"])
        network.eval()
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
            extra={"restored_from_checkpoint": True, "device": str(self.device_)},
        )
        logger.info(
            "Model restored | {} | entrées={} sorties={} époques={}",
            self.summary(),
            input_dim,
            output_dim,
            self.epochs_trained_,
        )

    def _require_network(self) -> nn.Module:
        """Return the fitted network or raise a diagnostic error."""
        if self.network_ is None:
            msg = "No network is attached to this model (fit or load it first)"
            raise RuntimeError(msg)
        return self.network_

    def _tensor_batch(
        self, matrix: np.ndarray, labels: np.ndarray | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a matrix (and optional target) into device tensors.

        Args:
            matrix: Feature matrix.
            labels: Business target.

        Returns:
            The feature tensor and the encoded target tensor (the features themselves for an
            autoencoder).
        """
        features = torch.as_tensor(
            np.ascontiguousarray(matrix), dtype=torch.float32, device=self.device_
        )
        if labels is None or self.task in _ANOMALY:
            return features, features.clone()
        encoded = _encode_labels(
            labels,
            self.task,
            self.classes_,
            target_mean=self._target_mean_,
            target_std=self._target_std_,
        )
        dtype = torch.int64 if self.task == "multiclass" else torch.float32
        shaped = encoded.reshape(-1, 1) if dtype is torch.float32 else encoded
        return features, torch.as_tensor(
            np.ascontiguousarray(shaped), dtype=dtype, device=self.device_
        )

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
        """L2 penalty declared in the configuration (decoupled by AdamW)."""
        try:
            return float(self._train_node().get("weight_decay", 0.0) or 0.0)
        except (TypeError, ValueError):  # pragma: no cover - défensif
            return 0.0

    def _label_list(self) -> list[Any]:
        """Return the training label space as a plain list."""
        return label_list(self.classes_)

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
    "ESTIMATORS",
    "INFERENCE_BATCH_SIZE",
    "AlgorithmSpec",
    "Autoencoder",
    "FitResult",
    "ModelCard",
    "PyTorchModel",
    "ResidualBlock",
    "ResidualMLP",
    "available_for_task",
    "count_parameters",
    "filter_params",
    "resolve_algorithm",
    "seed_torch",
]
