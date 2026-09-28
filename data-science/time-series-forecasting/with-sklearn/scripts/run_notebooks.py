"""Exécute les notebooks du projet de bout en bout (garantie « tout est exécutable »).

Ce script est l'équivalent de ``make notebooks`` : il exécute chaque ``.ipynb`` avec
``nbclient`` dans le répertoire ``notebooks/`` (le même contexte d'exécution qu'un utilisateur),
vérifie qu'aucune cellule ne lève d'exception, et renvoie un code de retour non nul sinon.

Pourquoi ``nbclient`` et pas ``jupyter nbconvert`` ?

* ``nbclient`` est déjà une dépendance de développement (avec ``nbformat`` et ``ipykernel``),
* il expose une API Python typée, donc un rapport d'échec exploitable en CI,
* il n'impose pas l'installation complète de Jupyter sur une machine de production.

Usage:
    python scripts/run_notebooks.py                     # exécute tout, ne modifie rien
    python scripts/run_notebooks.py --inplace           # réécrit les notebooks avec leurs sorties
    python scripts/run_notebooks.py --filter 01_        # seulement les notebooks contenant "01_"
    python scripts/run_notebooks.py --timeout 1800      # délai par cellule (secondes)
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


def discover(pattern: str | None = None) -> list[Path]:
    """List the notebooks to execute.

    Args:
        pattern: Optional substring filter applied to the file name.

    Returns:
        The sorted notebook paths.

    Raises:
        FileNotFoundError: When the ``notebooks/`` directory does not exist.
    """
    if not NOTEBOOK_DIR.exists():
        msg = f"Répertoire introuvable : {NOTEBOOK_DIR}"
        raise FileNotFoundError(msg)
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if pattern:
        notebooks = [path for path in notebooks if pattern in path.name]
    return notebooks


def execute(path: Path, *, timeout: int, inplace: bool) -> tuple[bool, str, float]:
    """Execute one notebook.

    Args:
        path: Notebook to execute.
        timeout: Per-cell timeout, in seconds.
        inplace: Whether the executed notebook (with its outputs) is written back.

    Returns:
        A ``(success, message, duration_seconds)`` tuple.
    """
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
        allow_errors=False,
    )
    started = time.perf_counter()
    try:
        client.execute()
    except CellExecutionError as error:
        return (
            False,
            str(error).strip().splitlines()[-1] if str(error).strip() else "échec inconnu",
            time.perf_counter() - started,
        )
    # Un notebook cassé ne doit pas arrêter l'exécution des autres : on capture l'erreur, on la
    # rapporte dans le résumé, et la boucle continue.
    except Exception as error:
        return False, f"{type(error).__name__}: {error}", time.perf_counter() - started

    if inplace:
        nbformat.write(notebook, path)
    return True, f"{len(notebook.cells)} cellules", time.perf_counter() - started


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` when every notebook executed successfully).
    """
    parser = argparse.ArgumentParser(description="Exécute les notebooks du projet.")
    parser.add_argument(
        "--filter", help="N'exécuter que les notebooks dont le nom contient ce motif"
    )
    parser.add_argument("--timeout", type=int, default=900, help="Timeout par cellule (secondes)")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Réécrire les notebooks avec les sorties produites (défaut : aucune écriture)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        notebooks = discover(args.filter)
    except FileNotFoundError as error:
        print(f"✗ {error}", file=sys.stderr)
        return 1
    if not notebooks:
        print("Aucun notebook à exécuter.")
        return 0

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    failures: list[str] = []
    print(f"Exécution de {len(notebooks)} notebook(s) depuis {NOTEBOOK_DIR}")
    for path in notebooks:
        success, message, duration = execute(path, timeout=args.timeout, inplace=args.inplace)
        status = "✓" if success else "✗"
        print(f"  {status} {path.name:<32} {duration:6.1f}s  {message[:120]}")
        if not success:
            failures.append(path.name)

    print(
        f"\n{len(notebooks) - len(failures)}/{len(notebooks)} notebook(s) exécuté(s) avec succès."
    )
    if failures:
        print("Échecs : " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
