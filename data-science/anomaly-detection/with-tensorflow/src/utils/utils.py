"""Small, dependency-free helpers shared across the project."""

from __future__ import annotations

import hashlib
import os
import random
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")


def set_seed(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed every random generator that may be used by the project.

    Reproducibility is a hard requirement: without an explicit seeding of Python, NumPy and
    the deep learning framework in use, results drift between two identical runs.

    Args:
        seed: Seed value.
        deterministic: Also enable deterministic algorithms for PyTorch / TensorFlow.

    Returns:
        A mapping describing which generators were seeded (useful for logging).
    """
    seeded: dict[str, Any] = {"python": True, "numpy": True, "env_pythonhashseed": seed}
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Règle de sécurité : on n'importe JAMAIS un framework lourd *seulement* pour le semer.
    # Faire cohabiter PyTorch et TensorFlow dans un même processus peut déclencher un conflit de
    # runtime OpenMP (segfault), et chaque import coûte plusieurs secondes au démarrage. On ne
    # seme donc que les frameworks déjà chargés par le projet ; un modèle qui en utilise un autre
    # pose sa propre graine dans ``fit()``.
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        seeded["torch"] = True

    if "tensorflow" in sys.modules:
        import tensorflow as tf

        tf.random.set_seed(seed)
        seeded["tensorflow"] = True

    return seeded


@contextmanager
def timer(label: str = "block") -> Iterator[dict[str, float]]:
    """Measure the wall-clock duration of a block of code.

    Args:
        label: Human readable label stored in the result mapping.

    Yields:
        A mutable mapping filled with ``label`` and ``seconds`` on exit.

    Example:
        >>> with timer("training") as elapsed:
        ...     pass
        >>> sorted(elapsed)
        ['label', 'seconds']
    """
    result: dict[str, Any] = {"label": label, "seconds": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["seconds"] = time.perf_counter() - start


def flatten_dict(
    mapping: Mapping[str, Any], parent_key: str = "", separator: str = "."
) -> dict[str, Any]:
    """Flatten a nested mapping into dotted keys.

    Args:
        mapping: Mapping to flatten.
        parent_key: Prefix used during recursion.
        separator: Key separator.

    Returns:
        The flattened mapping.

    Example:
        >>> flatten_dict({"a": {"b": 1}})
        {'a.b': 1}
    """
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_dict(value, new_key, separator))
        else:
            flat[new_key] = value
    return flat


def chunked(sequence: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield consecutive chunks of ``size`` elements.

    Args:
        sequence: Sequence to split.
        size: Maximum chunk size (strictly positive).

    Yields:
        Successive chunks.
    """
    if size <= 0:
        msg = f"Chunk size must be strictly positive, got {size}"
        raise ValueError(msg)
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Compute a short SHA-256 fingerprint of a DataFrame (shape + values).

    Used to assert that a dataset has not silently changed between two runs.

    Args:
        frame: DataFrame to fingerprint.

    Returns:
        The first 16 hexadecimal characters of the digest.
    """
    payload = (
        frame.to_numpy().tobytes() + str(frame.shape).encode() + ",".join(frame.columns).encode()
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def human_number(value: float, digits: int = 1) -> str:
    """Format a number in a compact, human readable way (``1.2M``, ``3.4k``).

    Args:
        value: Number to format.
        digits: Number of decimals kept.

    Returns:
        The formatted string.
    """
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= threshold:
            return f"{value / threshold:.{digits}f}{suffix}"
    return f"{value:.{digits}f}"


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide without raising on a zero denominator.

    Args:
        numerator: Numerator.
        denominator: Denominator.
        default: Value returned when ``denominator`` is zero.

    Returns:
        The quotient or the default value.
    """
    return numerator / denominator if denominator else default


def required_keys(mapping: Mapping[str, Any], keys: Sequence[str], context: str) -> None:
    """Assert that a mapping contains every required key.

    Args:
        mapping: Mapping to check.
        keys: Required keys.
        context: Label used in the error message.

    Raises:
        KeyError: When at least one key is missing.
    """
    missing = [key for key in keys if key not in mapping]
    if missing:
        msg = f"{context}: missing required key(s) {missing}. Available: {sorted(mapping)}"
        raise KeyError(msg)
