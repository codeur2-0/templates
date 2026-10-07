"""Uniform access to the configuration, whatever its container.

A configuration travels through the project in three shapes:

* a ``DictConfig`` produced by Hydra and validated by
  :func:`src.schemas.config.validate_config`,
* a **pydantic** object (``AppConfig`` and its nodes) once validated,
* a plain ``dict`` when it has been dumped for an artefact or a test.

Each of the three exposes a different API (``.get`` does not exist on a pydantic model, a
``DictConfig`` raises on a missing key). Reading a value directly therefore produces code that
works in a notebook and fails in a pipeline — a classic and expensive configuration bug.

The two helpers below are the single access point: :func:`as_mapping` normalises a node, and
:func:`node` extracts a sub-node, always returning a plain ``dict``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def as_mapping(config: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    """Normalise any configuration container into a plain dictionary.

    Args:
        config: Mapping, pydantic model, OmegaConf node or ``None``.

    Returns:
        A shallow plain-``dict`` copy (empty when ``config`` is ``None``).
    """
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    if hasattr(config, "to_container"):  # OmegaConf DictConfig
        return dict(config.to_container(resolve=True))
    return dict(config)


def node(config: Mapping[str, Any] | Any | None, name: str) -> dict[str, Any]:
    """Return a sub-node of the configuration as a plain dictionary.

    Args:
        config: Configuration container.
        name: Name of the sub-node (``train``, ``preprocessing``, ``predict``, ...).

    Returns:
        The sub-node as a dictionary; an empty mapping when the node is absent or not a mapping.
    """
    root = as_mapping(config)
    candidate = root.get(name)
    return (
        as_mapping(candidate)
        if isinstance(candidate, (Mapping,)) or hasattr(candidate, "model_dump")
        else {}
    )


def value(config: Mapping[str, Any] | Any | None, name: str, default: Any = None) -> Any:
    """Read one value of a configuration node with a default.

    Args:
        config: Configuration container.
        name: Key to read.
        default: Value returned when the key is absent.

    Returns:
        The value, or ``default``.
    """
    return as_mapping(config).get(name, default)


__all__ = ["as_mapping", "node", "value"]
