"""Vérification de bout en bout d'un projet généré (harnais QA du dépôt).

Chaque projet du dépôt doit satisfaire le même contrat de qualité :

1. ``ruff check``      — aucune erreur de lint ;
2. ``ruff format``     — aucun fichier à reformater ;
3. ``mypy``            — typage vérifié sur ``src`` et ``scripts`` ;
4. ``pytest``          — suite de tests verte ;
5. ``run_notebooks``   — les 6 notebooks s'exécutent sans exception ;
6. ``pipeline``        — ``generate-data → train → evaluate → predict`` réussit.

Usage:
    # vérifier un projet
    python -m tools.verify data-science/classification/with-sklearn

    # vérifier tous les projets générés
    python -m tools.verify --all

    # vérification rapide (lint + tests, sans notebooks ni pipeline)
    python -m tools.verify --all --quick

    # réécrire les sorties des notebooks après exécution
    python -m tools.verify --all --notebooks-inplace
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Domaines contenant des projets générés.
DOMAINS = (
    "data-science",
    "data-eng",
    "ai-eng",
    "mlops",
    "analytics",
    "computer-vision",
    "nlp",
)


@dataclass(slots=True)
class StepResult:
    """Outcome of one verification step.

    Attributes:
        name: Step identifier.
        success: Whether the step passed.
        duration: Wall-clock duration, in seconds.
        detail: Last meaningful output lines (for the failure report).
    """

    name: str
    success: bool
    duration: float = 0.0
    detail: str = ""


@dataclass(slots=True)
class ProjectReport:
    """Aggregated verification report of one project.

    Attributes:
        path: Project directory (relative to the repository root).
        steps: Ordered step results.
    """

    path: str
    steps: list[StepResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every step passed."""
        return all(step.success for step in self.steps)

    @property
    def duration(self) -> float:
        """Total verification duration, in seconds."""
        return sum(step.duration for step in self.steps)


def discover_projects(only: Sequence[str] | None = None) -> list[Path]:
    """Find every generated project (a directory holding a ``pyproject.toml``).

    Args:
        only: Optional path fragments used to filter the discovery.

    Returns:
        The sorted project directories.
    """
    found: list[Path] = []
    for domain in DOMAINS:
        root = REPO_ROOT / domain
        if not root.exists():
            continue
        for candidate in root.rglob("pyproject.toml"):
            project = candidate.parent
            relative = project.relative_to(REPO_ROOT)
            if Path("tools") in relative.parents or ".venv" in relative.parts:
                continue
            found.append(project)
    if only:
        found = [path for path in found if any(fragment in str(path) for fragment in only)]
    return sorted(found)


def _run(
    command: Sequence[str],
    cwd: Path,
    *,
    timeout: int = 3600,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run a command and capture its output.

    Args:
        command: Command and arguments.
        cwd: Working directory.
        timeout: Maximum duration, in seconds.
        env: Optional environment overrides.

    Returns:
        The ``(return_code, tail_output)`` pair.
    """
    process_env = dict(os.environ)
    process_env.update(env or {})
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=process_env,
            check=False,
        )
    except FileNotFoundError as error:
        return 127, f"{error}"
    except subprocess.TimeoutExpired:
        return 124, f"timeout après {timeout}s"
    output = (completed.stdout or "") + (completed.stderr or "")
    lines = [line for line in output.splitlines() if line.strip()]
    return completed.returncode, "\n".join(lines[-12:])


def _tool(command: str) -> list[str] | None:
    """Resolve a development tool invocation (binary or ``python -m``).

    Args:
        command: Tool name (``ruff``, ``mypy``, ``pytest``).

    Returns:
        The command prefix, or ``None`` when the tool is unavailable.
    """
    executable = shutil.which(command)
    if executable:
        return [executable]
    probe = subprocess.run(
        [sys.executable, "-m", command, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        return [sys.executable, "-m", command]
    return None


def _step(name: str, command: Sequence[str] | None, cwd: Path, *, missing: str = "") -> StepResult:
    """Run one verification step.

    Args:
        name: Step name.
        command: Command to run (``None`` marks the tool as unavailable).
        cwd: Working directory.
        missing: Message used when the tool is unavailable.

    Returns:
        The :class:`StepResult`.
    """
    started = time.perf_counter()
    if command is None:
        return StepResult(
            name=name, success=False, duration=0.0, detail=missing or "outil indisponible"
        )
    code, detail = _run(command, cwd)
    return StepResult(
        name=name,
        success=code == 0,
        duration=time.perf_counter() - started,
        detail="" if code == 0 else detail,
    )


def verify_project(
    project: Path,
    *,
    quick: bool = False,
    notebooks_inplace: bool = False,
    python: str | None = None,
) -> ProjectReport:
    """Verify one generated project.

    Args:
        project: Project directory.
        quick: Skip the notebooks and the end-to-end pipeline.
        notebooks_inplace: Rewrite the notebooks with their executed outputs.
        python: Python interpreter used inside the project (defaults to ``sys.executable``).

    Returns:
        The :class:`ProjectReport`.
    """
    interpreter = python or sys.executable
    report = ProjectReport(path=str(project.relative_to(REPO_ROOT)))

    ruff = _tool("ruff")
    mypy = _tool("mypy")
    pytest_tool = _tool("pytest")

    report.steps.append(
        _step(
            "ruff check",
            [*ruff, "check", "."] if ruff else None,
            project,
            missing="ruff absent (pip install ruff)",
        )
    )
    report.steps.append(
        _step(
            "ruff format",
            [*ruff, "format", "--check", "."] if ruff else None,
            project,
            missing="ruff absent (pip install ruff)",
        )
    )
    report.steps.append(
        _step("mypy", [*mypy] if mypy else None, project, missing="mypy absent (pip install mypy)")
    )
    report.steps.append(
        _step(
            "pytest",
            [*pytest_tool, "tests", "-q", "-p", "no:cacheprovider"] if pytest_tool else None,
            project,
            missing="pytest absent (pip install pytest)",
        )
    )

    if not quick:
        notebook_arguments = [interpreter, "scripts/run_notebooks.py"]
        if notebooks_inplace:
            notebook_arguments.append("--inplace")
        runner = project / "scripts" / "run_notebooks.py"
        report.steps.append(
            _step(
                "notebooks",
                notebook_arguments if runner.exists() else None,
                project,
                missing="scripts/run_notebooks.py absent",
            )
        )
        report.steps.append(
            _step(
                "pipeline",
                [interpreter, "-m", "src.main", "mode=all"],
                project,
                missing="src/main.py absent",
            )
        )
    return report


def _print(report: ProjectReport, *, verbose: bool) -> None:
    """Print one project report.

    Args:
        report: Report to print.
        verbose: Whether failure details are printed.
    """
    status = "✓" if report.ok else "✗"
    print(f"{status} {report.path}  ({report.duration:.1f}s)")
    for step in report.steps:
        mark = "  ✓" if step.success else "  ✗"
        print(f"{mark} {step.name:<14} {step.duration:6.1f}s")
        if not step.success and verbose and step.detail:
            for line in step.detail.splitlines():
                print(f"      | {line}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command line arguments.

    Returns:
        Process exit code (``0`` when every project passed).
    """
    parser = argparse.ArgumentParser(description="Vérifie les projets générés du dépôt.")
    parser.add_argument("paths", nargs="*", help="Projets à vérifier (chemins ou fragments)")
    parser.add_argument("--all", action="store_true", help="Vérifier tous les projets découverts")
    parser.add_argument("--quick", action="store_true", help="Lint + tests uniquement")
    parser.add_argument(
        "--notebooks-inplace", action="store_true", help="Réécrire les notebooks avec leurs sorties"
    )
    parser.add_argument("--quiet", action="store_true", help="N'afficher que les échecs")
    args = parser.parse_args(list(argv) if argv is not None else None)

    fragments = list(args.paths) or None
    projects = (
        discover_projects(fragments)
        if (args.all or not fragments)
        else [Path(item) for item in fragments]
    )
    projects = [path if path.is_absolute() else REPO_ROOT / path for path in projects]
    projects = [path for path in projects if path.exists()]
    if not projects:
        print("Aucun projet trouvé.", file=sys.stderr)
        return 1

    print(f"Vérification de {len(projects)} projet(s)…\n")
    reports = [
        verify_project(project, quick=args.quick, notebooks_inplace=args.notebooks_inplace)
        for project in projects
    ]
    for report in reports:
        if args.quiet and report.ok:
            continue
        _print(report, verbose=True)

    failed = [report for report in reports if not report.ok]
    print(f"\n{len(reports) - len(failed)}/{len(reports)} projet(s) conforme(s).")
    if failed:
        for report in failed:
            broken = ", ".join(step.name for step in report.steps if not step.success)
            print(f"  ✗ {report.path} — {broken}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
