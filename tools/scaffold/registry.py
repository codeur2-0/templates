"""Typed access to the stack / family registries used by the scaffold engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

REGISTRY_DIR = Path(__file__).resolve().parent / "registry"


class DocLink(BaseModel):
    """A documentation reference displayed in the generated README."""

    model_config = ConfigDict(extra="forbid")

    name: str
    url: str


class StackSpec(BaseModel):
    """Metadata describing a technological stack (``with-<stack>`` folders)."""

    model_config = ConfigDict(extra="forbid")

    key: str
    display_name: str
    category: str
    template_dir: str
    epochs_based: bool = False
    supports_proba: bool = True
    dependencies: list[str] = Field(default_factory=list)
    docs: list[DocLink] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    commands: dict[str, str] = Field(default_factory=dict)


class FamilySpec(BaseModel):
    """Metadata describing a family of problems (a template bundle)."""

    model_config = ConfigDict(extra="forbid")

    key: str
    display_name: str
    modality: str
    task: str
    notebook_builder: str
    template_dir: str
    description: str
    allowed_stacks: list[str] = Field(default_factory=list)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping file.

    Args:
        path: File to read.

    Returns:
        The parsed mapping (empty dict when the file does not exist).
    """
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        msg = f"{path} must contain a YAML mapping at the top level"
        raise TypeError(msg)
    return payload


def load_stacks(registry_dir: Path | None = None) -> dict[str, StackSpec]:
    """Load every stack definition, keyed by stack name."""
    raw = _load_yaml((registry_dir or REGISTRY_DIR) / "stacks.yaml")
    return {key: StackSpec.model_validate({**value, "key": key}) for key, value in raw.items()}


def load_families(registry_dir: Path | None = None) -> dict[str, FamilySpec]:
    """Load every family definition, keyed by family name."""
    raw = _load_yaml((registry_dir or REGISTRY_DIR) / "families.yaml")
    return {key: FamilySpec.model_validate({**value, "key": key}) for key, value in raw.items()}


def get_stack(key: str, registry_dir: Path | None = None) -> StackSpec:
    """Return one stack definition, raising a helpful error when unknown."""
    stacks = load_stacks(registry_dir)
    normalised = key.removeprefix("with-")
    if normalised not in stacks:
        msg = f"Unknown stack '{key}'. Available: {sorted(stacks)}"
        raise KeyError(msg)
    return stacks[normalised]


def get_family(key: str, registry_dir: Path | None = None) -> FamilySpec:
    """Return one family definition, raising a helpful error when unknown."""
    families = load_families(registry_dir)
    if key not in families:
        msg = f"Unknown family '{key}'. Available: {sorted(families)}"
        raise KeyError(msg)
    return families[key]
