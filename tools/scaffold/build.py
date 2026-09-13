"""Command line interface of the scaffold tool.

The whole repository is *generated*: every project directory is the deterministic output of a
manifest plus the template layers. This CLI is the single entry point.

Usage:
    # lister les manifestes disponibles
    python -m tools.scaffold.build --list

    # générer un projet
    python -m tools.scaffold.build --manifest tools/scaffold/manifests/ds-classification-sklearn.yaml

    # générer tous les projets (ou un sous-ensemble)
    python -m tools.scaffold.build --all
    python -m tools.scaffold.build --all --domain data-science --stack sklearn

    # vérifier ce qui serait généré sans écrire
    python -m tools.scaffold.build --all --dry-run
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

SCAFFOLD_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCAFFOLD_DIR.parents[1]
MANIFEST_DIR = SCAFFOLD_DIR / "manifests"
DEFAULTS_DIR = SCAFFOLD_DIR / "defaults"

if str(REPO_ROOT) not in sys.path:  # pragma: no cover - dépend du mode de lancement
    sys.path.insert(0, str(REPO_ROOT))

from tools.scaffold.config import ProjectSpec, load_manifest  # noqa: E402
from tools.scaffold.engine import ProjectRenderer, RenderReport  # noqa: E402
from tools.scaffold.registry import FamilySpec, StackSpec, load_families, load_stacks  # noqa: E402


def iter_manifests(
    manifest_dir: Path | None = None,
    *,
    domain: str | None = None,
    stack: str | None = None,
    family: str | None = None,
    pattern: str | None = None,
) -> list[Path]:
    """List manifest files, optionally filtered.

    Args:
        manifest_dir: Directory to scan (defaults to ``tools/scaffold/manifests``).
        domain: Keep manifests of this domain only.
        stack: Keep manifests using this stack only.
        family: Keep manifests of this family only.
        pattern: Glob pattern applied to the file name.

    Returns:
        The sorted list of manifest paths.
    """
    directory = manifest_dir or MANIFEST_DIR
    candidates = sorted(directory.glob(pattern or "*.yaml"))
    selected: list[Path] = []
    for candidate in candidates:
        try:
            spec = load_manifest(candidate, DEFAULTS_DIR)
        except Exception:  # noqa: BLE001 - a broken manifest must not hide the others
            selected.append(candidate)
            continue
        if domain and spec.domain != domain:
            continue
        if stack and spec.stack_key != stack.removeprefix("with-"):
            continue
        if family and spec.family != family:
            continue
        selected.append(candidate)
    return selected


def format_python_files(project_dir: Path) -> bool:
    """Normalise le style du projet généré avec ruff (format + corrections sûres).

    Le code généré doit être **immédiatement conforme** au linter du projet : plutôt que de
    formater chaque template à la main, on applique ``ruff format`` puis ``ruff check --fix``
    sur la sortie. L'étape est optionnelle : si ruff n'est pas installé, la génération réussit
    quand même (un avertissement est affiché).

    Args:
        project_dir: Racine du projet généré (contient son ``pyproject.toml``).

    Returns:
        ``True`` si le formatage a été appliqué, ``False`` sinon.
    """
    import shutil
    import subprocess

    executable = shutil.which("ruff")
    prefix = [executable] if executable else [sys.executable, "-m", "ruff"]
    probe = subprocess.run(  # noqa: S603 - commande figée et contrôlée
        [*prefix, "--version"], capture_output=True, text=True, check=False, cwd=project_dir
    )
    if probe.returncode != 0:
        return False
    for arguments in (["format", "--quiet", "."], ["check", "--quiet", "--fix", "."]):
        subprocess.run(  # noqa: S603
            [*prefix, *arguments], capture_output=True, text=True, check=False, cwd=project_dir
        )
    return True


def build_project(
    manifest_path: Path,
    *,
    repo_root: Path | None = None,
    force: bool = True,
    dry_run: bool = False,
    with_notebooks: bool = True,
    with_format: bool = True,
) -> tuple[ProjectSpec, RenderReport | None]:
    """Generate one project from its manifest.

    Args:
        manifest_path: Manifest file.
        repo_root: Repository root (destination base).
        force: Overwrite existing files.
        dry_run: Resolve and validate everything without writing.
        with_notebooks: Also generate the notebooks.
        with_format: Normalise the generated Python files with ruff.

    Returns:
        The validated spec and the render report (``None`` in dry-run mode).

    Raises:
        KeyError: When the manifest references an unknown stack or family.
    """
    root = repo_root or REPO_ROOT
    spec = load_manifest(manifest_path, DEFAULTS_DIR)

    stacks: dict[str, StackSpec] = load_stacks()
    families: dict[str, FamilySpec] = load_families()
    stack_key = spec.stack_key
    if stack_key not in stacks:
        msg = f"Manifest '{manifest_path.name}': unknown stack '{spec.stack}'. Known: {sorted(stacks)}"
        raise KeyError(msg)
    if spec.family not in families:
        msg = f"Manifest '{manifest_path.name}': unknown family '{spec.family}'. Known: {sorted(families)}"
        raise KeyError(msg)

    stack = stacks[stack_key]
    family = families[spec.family]
    _check_stack_allowed(spec, family)

    destination = root / spec.relative_path
    if dry_run:
        print(f"[dry-run] {spec.key} -> {destination.relative_to(root)} (stack={stack_key}, family={spec.family})")
        return spec, None

    renderer = ProjectRenderer(spec, family, stack)
    report = renderer.render(destination, force=force)
    if with_notebooks:
        from tools.scaffold.notebooks import build_notebooks

        notebooks = build_notebooks(spec, family, stack, destination / "notebooks")
        report.files.extend(notebooks)
    if with_format and not format_python_files(destination):
        print(f"  ! {spec.key}: ruff indisponible, style non normalisé (pip install ruff)")
    return spec, report


def _check_stack_allowed(spec: ProjectSpec, family: FamilySpec) -> None:
    """Warn when a manifest uses a stack outside the family's declared list."""
    if family.allowed_stacks and spec.stack_key not in family.allowed_stacks:
        print(
            f"  ! {spec.key}: la stack '{spec.stack_key}' n'est pas déclarée pour la famille "
            f"'{spec.family}' (autorisées : {family.allowed_stacks})",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="scaffold",
        description="Génère les projets du dépôt depuis leurs manifestes déclaratifs.",
    )
    parser.add_argument("--manifest", type=Path, help="Chemin d'un manifeste YAML")
    parser.add_argument("--all", action="store_true", help="Générer tous les manifestes")
    parser.add_argument("--list", action="store_true", help="Lister les manifestes disponibles")
    parser.add_argument("--domain", help="Filtrer par domaine (data-science, nlp, ...)")
    parser.add_argument("--stack", help="Filtrer par stack (sklearn, pytorch, ...)")
    parser.add_argument("--family", help="Filtrer par famille de problèmes")
    parser.add_argument("--pattern", help="Motif glob sur le nom du manifeste")
    parser.add_argument("--dry-run", action="store_true", help="Valider sans écrire")
    parser.add_argument("--no-notebooks", action="store_true", help="Ne pas générer les notebooks")
    parser.add_argument("--no-format", action="store_true", help="Ne pas passer ruff sur la sortie")
    parser.add_argument("--keep-existing", action="store_true", help="Ne pas écraser les fichiers existants")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Racine du dépôt cible")
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifests: list[Path] = []
    if args.list:
        manifests = iter_manifests(
            domain=args.domain, stack=args.stack, family=args.family, pattern=args.pattern
        )
        print(f"{len(manifests)} manifeste(s) disponible(s) :")
        for manifest in manifests:
            try:
                spec = load_manifest(manifest, DEFAULTS_DIR)
                print(f"  {spec.key:<44} {spec.relative_path}")
            except Exception as exc:  # noqa: BLE001 - rapport d'erreur explicite
                print(f"  {manifest.name:<44} INVALIDE : {exc}")
        return 0

    if args.manifest:
        manifests = [args.manifest]
    elif args.all or args.domain or args.stack or args.family or args.pattern:
        manifests = iter_manifests(
            domain=args.domain, stack=args.stack, family=args.family, pattern=args.pattern
        )
    else:
        parser.print_help()
        return 1

    if not manifests:
        print("Aucun manifeste ne correspond aux filtres fournis.", file=sys.stderr)
        return 1

    failures: list[str] = []
    total_files = 0
    for manifest in manifests:
        try:
            spec, report = build_project(
                manifest,
                repo_root=args.repo_root,
                force=not args.keep_existing,
                dry_run=args.dry_run,
                with_notebooks=not args.no_notebooks,
                with_format=not args.no_format,
            )
            if report is not None:
                total_files += report.n_files
                print(f"  ✓ {spec.key:<44} {spec.relative_path} ({report.n_files} fichiers)")
        except Exception as exc:  # noqa: BLE001 - un manifeste cassé ne doit pas tout arrêter
            failures.append(f"{manifest.name}: {type(exc).__name__}: {exc}")
            print(f"  ✗ {manifest.name}: {exc}", file=sys.stderr)
            if args.dry_run:
                traceback.print_exc()

    print(f"\n{len(manifests) - len(failures)}/{len(manifests)} projet(s) généré(s), {total_files} fichier(s) écrit(s).")
    if failures:
        print("Échecs :", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
