"""Jinja2 rendering engine: turns template layers + a manifest into a project directory."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from tools.scaffold.config import ProjectSpec
from tools.scaffold.context import build_context
from tools.scaffold.registry import FamilySpec, StackSpec

TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates"

#: Files whose template name does not start with a dot but must be written as a dotfile.
RENAME_RULES: dict[str, str] = {
    "gitignore": ".gitignore",
    "gitattributes": ".gitattributes",
    "env_example": ".env.example",
    "dockerignore": ".dockerignore",
    "flake8": ".flake8",
}

#: Template sub-directories that are only used through `{% include %}` / `{% import %}`.
PARTIAL_DIR = "_partials"

#: Files never rendered (they are not templates).
SKIP_NAMES = {"README.md.j2.disabled"}


@dataclass(slots=True)
class RenderReport:
    """Summary of a project generation, printed by the CLI."""

    project_key: str
    destination: Path
    files: list[Path] = field(default_factory=list)
    skipped_existing: list[Path] = field(default_factory=list)

    @property
    def n_files(self) -> int:
        """Number of files written."""
        return len(self.files)


def layers_for(
    spec: ProjectSpec, family: FamilySpec, stack: StackSpec, template_root: Path | None = None
) -> list[Path]:
    """Return the template layers of a project, from the most specific to the most generic.

    Five layers are composed (Jinja precedence: the first existing match wins):

    1. ``stack/<stack>``      framework specific model and trainer
    2. ``family/<family>``    synthetic data generator, business flavour
    3. ``task/<task>``        evaluator, reports, predictor, figures
    4. ``modality/<modality>``loaders, preprocessing, features, pipelines, tests
    5. ``base``               README, packaging, conf/, utils, main, scripts

    Args:
        spec: Validated project manifest.
        family: Family definition selected by the manifest.
        stack: Stack definition selected by the manifest.
        template_root: Optional override of the template root directory.

    Returns:
        The ordered list of existing layer directories (Jinja precedence: first wins).
    """
    root = template_root or TEMPLATE_ROOT
    candidates = [
        root / "stack" / stack.template_dir,
        root / "family" / family.template_dir,
        root / "task" / _task_dir(family),
        root / "modality" / spec.modality,
        root / "base",
    ]
    return [path for path in candidates if path.is_dir()]


def _task_dir(family: FamilySpec) -> str:
    """Map a family task onto its template directory name."""
    aliases = {
        "binary": "classification",
        "multiclass": "classification",
        "regression": "regression",
        "clustering": "clustering",
        "forecasting": "forecasting",
        "ranking": "ranking",
        "anomaly": "anomaly",
        "retrieval": "retrieval",
        "generation": "generation",
        "pipeline": "pipeline",
    }
    return aliases.get(family.task, family.task)


def collect_templates(layers: Iterable[Path]) -> dict[str, Path]:
    """Build the ``relative template path -> layer directory`` mapping.

    Later layers (in iteration order) override earlier ones.

    Args:
        layers: Layer directories, generic first.

    Returns:
        Mapping of template relative paths to the layer that provides them.
    """
    collected: dict[str, Path] = {}
    for layer in layers:
        for path in sorted(layer.rglob("*")):
            if not path.is_file() or path.name in SKIP_NAMES:
                continue
            collected[path.relative_to(layer).as_posix()] = layer
    return collected


def target_path(template_rel: str) -> str:
    """Convert a template relative path into the output relative path.

    Args:
        template_rel: Path of the template inside its layer.

    Returns:
        Path of the generated file inside the project directory.
    """
    path = template_rel
    if path.endswith(".j2"):
        path = path[: -len(".j2")]
    parts = path.split("/")
    parts[-1] = RENAME_RULES.get(parts[-1], parts[-1])
    return "/".join(parts)


class ProjectRenderer:
    """Render (or copy) every template of the applicable layers into a project directory.

    The renderer is intentionally stateless apart from its Jinja environment: the whole
    generation is a pure function of (manifest, registries, templates), which makes the
    repository fully reproducible.
    """

    def __init__(
        self,
        spec: ProjectSpec,
        family: FamilySpec,
        stack: StackSpec,
        template_root: Path | None = None,
    ) -> None:
        """Prepare the renderer.

        Args:
            spec: Validated manifest.
            family: Family definition.
            stack: Stack definition.
            template_root: Optional template root override (used by tests).
        """
        self.spec = spec
        self.family = family
        self.stack = stack
        self.template_root = template_root or TEMPLATE_ROOT
        self.layers = layers_for(spec, family, stack, self.template_root)
        if not self.layers:
            msg = f"No template layer found for project '{spec.key}'"
            raise FileNotFoundError(msg)
        # Jinja resolves the first match in loader order -> highest precedence first.
        self.env = Environment(
            loader=FileSystemLoader([str(layer) for layer in self.layers]),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            autoescape=False,
        )
        self.context: dict[str, Any] = build_context(spec, family, stack)

    def render(self, destination: Path, *, force: bool = True) -> RenderReport:
        """Generate the project tree.

        Args:
            destination: Target directory (created if needed).
            force: When ``False``, existing files are left untouched.

        Returns:
            A :class:`RenderReport` describing what has been written.
        """
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        report = RenderReport(project_key=self.spec.key, destination=destination)

        # Precedence: iterate generic -> specific so specific layers win.
        templates = collect_templates(reversed(self.layers))
        for template_rel, layer in sorted(templates.items()):
            if template_rel.startswith(PARTIAL_DIR) or f"/{PARTIAL_DIR}/" in template_rel:
                continue
            output_rel = target_path(template_rel)
            output_path = destination / output_rel
            if output_path.exists() and not force:
                report.skipped_existing.append(output_path)
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if template_rel.endswith(".j2"):
                content = self._render_template(template_rel)
                output_path.write_text(content, encoding="utf-8")
            else:
                shutil.copyfile(layer / template_rel, output_path)
            report.files.append(output_path)

        self._create_placeholder_dirs(destination)
        return report

    def _render_template(self, template_rel: str) -> str:
        """Render one Jinja template with the project context."""
        try:
            template = self.env.get_template(template_rel)
        except TemplateNotFound as exc:  # pragma: no cover - defensive
            msg = f"Template '{template_rel}' not found in layers {[str(p) for p in self.layers]}"
            raise FileNotFoundError(msg) from exc
        rendered = template.render(**self.context)
        if not rendered.endswith("\n"):
            rendered += "\n"
        return rendered

    def _create_placeholder_dirs(self, destination: Path) -> None:
        """Create the runtime directories kept empty in git (with ``.gitkeep``)."""
        placeholders = [
            "data/raw",
            "data/processed",
            "data/external",
            "artifacts/models",
            "artifacts/reports",
            "artifacts/figures",
            "artifacts/metrics",
            "outputs",
        ]
        placeholders.extend(self.spec.extras.get("placeholder_dirs", []))
        for relative in placeholders:
            directory = destination / relative
            directory.mkdir(parents=True, exist_ok=True)
            keep = directory / ".gitkeep"
            if not keep.exists():
                keep.write_text("", encoding="utf-8")
