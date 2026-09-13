"""Filesystem layout of the project.

Every path manipulated by the code base is derived from :class:`ProjectPaths`. This single
source of truth avoids hard-coded relative paths, which break as soon as the working
directory changes (a very common issue with Hydra, which may relocate the process CWD).

The project root is resolved from ``__file__`` so that the code behaves identically when it
is launched with ``python -m src.main``, ``python src/main.py``, ``pytest`` or a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

#: Absolute path of the project root (the directory containing ``src/`` and ``conf/``).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Typed description of the on-disk layout.

    Attributes:
        root: Absolute path of the project root.
        data_dir: Root directory of every dataset artefact.
        artifacts_dir: Root directory of every produced artefact.
        outputs_dir: Hydra run directory (logs, composed config snapshots).
    """

    root: Path = PROJECT_ROOT
    data_dir: Path = field(default=None)  # type: ignore[assignment]
    artifacts_dir: Path = field(default=None)  # type: ignore[assignment]
    outputs_dir: Path = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Derive the conventional sub-directories when they are not provided explicitly."""
        root = Path(self.root).resolve()
        object.__setattr__(self, "root", root)
        defaults = {
            "data_dir": root / "data",
            "artifacts_dir": root / "artifacts",
            "outputs_dir": root / "outputs",
        }
        for name, value in defaults.items():
            current = getattr(self, name, None)
            if current is None:
                object.__setattr__(self, name, value)
            else:
                object.__setattr__(self, name, Path(current))

    # ---------------------------------------------------------------- factories ---------
    @classmethod
    def from_root(cls, root: Path | str | None = None) -> ProjectPaths:
        """Build the layout from a project root.

        Args:
            root: Project root directory. Defaults to the detected ``PROJECT_ROOT``.

        Returns:
            The resolved :class:`ProjectPaths` instance.
        """
        return cls(root=Path(root) if root is not None else PROJECT_ROOT)

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> ProjectPaths:
        """Build the layout from a Hydra/OmegaConf configuration mapping.

        Args:
            config: Full application config; the ``paths`` node is used when present.

        Returns:
            The resolved :class:`ProjectPaths` instance.
        """
        settings: Mapping[str, Any] = {}
        if config is not None:
            node = config.get("paths") if hasattr(config, "get") else None
            settings = dict(node) if node else {}
        root_value = settings.get("root")
        kwargs: dict[str, Path] = {}
        if root_value:
            kwargs["root"] = Path(root_value).expanduser().resolve()
        for key in ("data_dir", "artifacts_dir", "outputs_dir"):
            if settings.get(key):
                kwargs[key] = Path(settings[key]).expanduser()
        return cls(**kwargs)

    # ---------------------------------------------------------------- data --------------
    @property
    def raw_dir(self) -> Path:
        """Directory holding immutable raw data."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Directory holding cleaned / transformed data."""
        return self.data_dir / "processed"

    @property
    def external_dir(self) -> Path:
        """Directory holding third-party data."""
        return self.data_dir / "external"

    # ---------------------------------------------------------------- artefacts ---------
    @property
    def models_dir(self) -> Path:
        """Directory holding serialised models and preprocessing artefacts."""
        return self.artifacts_dir / "models"

    @property
    def reports_dir(self) -> Path:
        """Directory holding human readable evaluation reports."""
        return self.artifacts_dir / "reports"

    @property
    def figures_dir(self) -> Path:
        """Directory holding generated figures."""
        return self.artifacts_dir / "figures"

    @property
    def metrics_dir(self) -> Path:
        """Directory holding machine readable metrics (JSON)."""
        return self.artifacts_dir / "metrics"

    # ---------------------------------------------------------------- helpers -----------
    def data_file(self, name: str, fmt: str = "parquet", *, stage: str = "raw") -> Path:
        """Return the path of a dataset file for a given stage.

        Args:
            name: Dataset name without extension.
            fmt: File format (``parquet``, ``csv``, ``json``).
            stage: One of ``raw``, ``processed``, ``external``.

        Returns:
            The absolute path of the file.
        """
        directories = {
            "raw": self.raw_dir,
            "processed": self.processed_dir,
            "external": self.external_dir,
        }
        if stage not in directories:
            msg = f"Unknown data stage '{stage}'. Allowed: {sorted(directories)}"
            raise ValueError(msg)
        return directories[stage] / f"{name}.{fmt}"

    def artifact(self, name: str, *, subdir: str = "models") -> Path:
        """Return the path of an artefact file.

        Args:
            name: File name (extension included).
            subdir: Artefact sub-directory (``models``, ``reports``, ``figures``, ``metrics``).

        Returns:
            The absolute path of the artefact.
        """
        return self.artifacts_dir / subdir / name

    def ensure(self) -> ProjectPaths:
        """Create every directory of the layout.

        Returns:
            ``self``, to allow chaining.
        """
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.external_dir,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
            self.metrics_dir,
            self.outputs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict[str, str]:
        """Serialise the layout as a plain ``str -> str`` mapping (for logs / JSON)."""
        return {
            "root": str(self.root),
            "data_dir": str(self.data_dir),
            "raw_dir": str(self.raw_dir),
            "processed_dir": str(self.processed_dir),
            "external_dir": str(self.external_dir),
            "artifacts_dir": str(self.artifacts_dir),
            "models_dir": str(self.models_dir),
            "reports_dir": str(self.reports_dir),
            "figures_dir": str(self.figures_dir),
            "metrics_dir": str(self.metrics_dir),
            "outputs_dir": str(self.outputs_dir),
        }
