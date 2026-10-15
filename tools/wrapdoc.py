"""Repliage des commentaires trop longs dans les gabarits et les modules du générateur.

Les gabarits du générateur (``tools/scaffold/templates/**/*.py`` et ``*.py.j2``) sont exemptés de
``E501`` par ``pyproject.toml`` **à la génération** — mais pas les projets qu'ils produisent. Un
commentaire français de 140 colonnes dans un gabarit passe donc tous les contrôles du dépôt, puis
fait échouer ``ruff check`` du projet généré, cent fichiers plus loin. Ce script ferme cet écart :
il replie les commentaires qui dépassent la largeur cible, **avant** le build.

Trois règles, apprises à l'usage :

* seuls les commentaires (`#`) sont repliés — une docstring ou une chaîne de cellule de notebook
  n'est pas du commentaire, et une césure automatique y casserait le texte produit ;
* le repli se fait en **une seule passe** ligne par ligne : replier un fichier déjà replié ne doit
  rien changer (idempotence), et une seconde passe sur les lignes produites les fusionnerait ;
* une ligne qui contient une URL ou un mot plus long que la largeur est laissée intacte et signalée
  (``--strict`` en fait une erreur) : mieux vaut un avertissement qu'une phrase tronquée.

Usage :
    python -m tools.wrapdoc --check              # ne modifie rien, sort 1 si un repli est requis
    python -m tools.wrapdoc tools/scaffold       # replie sur place, largeur 99 par défaut
    python -m tools.wrapdoc --width 88 --strict  # autre largeur, avertissements bloquants
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

#: Largeur cible par défaut : la limite des gabarits, une colonne sous celle de ``ruff`` (100).
DEFAULT_WIDTH = 99

#: Extensions inspectées : les gabarits Python et les modules Python du générateur.
SUFFIXES = (".py", ".py.j2")

#: Un commentaire : indentation, croisillons, puis le texte. Le ``#`` initial peut être suivi d'un
#: second croisillon (``##``, décorations de gabarit) — il fait alors partie du préfixe.
COMMENT = re.compile(r"^(?P<indent>\s*)(?P<hashes>#+)(?P<space>\s*)(?P<body>\S.*)?$")

#: Motifs qu'on refuse de couper : une URL coupée ne se recolle pas.
UNBREAKABLE = re.compile(r"https?://\S+")


@dataclass
class WrapReport:
    """Result of a run, per file and in aggregate.

    Attributes:
        files_scanned: Number of files inspected.
        files_changed: Number of files whose content changed (or would change).
        lines_wrapped: Number of comment lines reflowed.
        warnings: Lines left intact because they could not be wrapped safely.
    """

    files_scanned: int = 0
    files_changed: int = 0
    lines_wrapped: int = 0
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Return a human-readable summary.

        Returns:
            Two summary lines plus one line per warning.
        """
        summary = (
            f"{self.files_scanned} fichier(s) inspecté(s) · "
            f"{self.files_changed} modifié(s) · {self.lines_wrapped} commentaire(s) replié(s)"
        )
        if not self.warnings:
            return summary
        return "\n".join([summary, *(f"  ⚠ {warning}" for warning in self.warnings)])


def wrap_comment(line: str, width: int = DEFAULT_WIDTH) -> list[str]:
    """Wrap one comment line so that no produced line exceeds ``width`` columns.

    Args:
        line: Raw line, newline included.
        width: Maximum line width.

    Returns:
        The wrapped lines (a single element when nothing had to change). A line that is not a
        comment, that is short enough, or that carries an URL is returned unchanged.
    """
    body = line.rstrip("\n")
    if len(body) <= width or UNBREAKABLE.search(body):
        return [line]
    match = COMMENT.match(body)
    if match is None or match.group("body") is None:
        return [line]
    prefix = f"{match.group('indent')}{match.group('hashes')} "
    words = match.group("body").split()
    if any(len(word) > width - len(prefix) for word in words):
        return [line]
    wrapped: list[str] = []
    current = f"{prefix}{words[0]}"
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            wrapped.append(f"{current}\n")
            current = f"{prefix}{word}"
    wrapped.append(f"{current}\n")
    return wrapped


def wrap_text(text: str, width: int = DEFAULT_WIDTH) -> tuple[str, int, list[str]]:
    """Wrap every comment line of a source text.

    Args:
        text: File content.
        width: Maximum line width.

    Returns:
        ``(wrapped text, number of lines wrapped, warnings)``. A line that is too long but cannot
        be wrapped (no break opportunity) is reported as a warning and left as is.
    """
    lines = text.splitlines(keepends=True)
    produced: list[str] = []
    counter = 0
    warnings: list[str] = []
    for index, line in enumerate(lines, start=1):
        wrapped = wrap_comment(line, width)
        if len(wrapped) == 1:
            if len(line.rstrip("\n")) > width and COMMENT.match(line.rstrip("\n")) is not None:
                warnings.append(f"ligne {index} non repliable ({len(line.rstrip())} colonnes)")
            produced.append(line)
            continue
        produced.extend(wrapped)
        counter += 1
    return "".join(produced), counter, warnings


def iter_sources(paths: list[Path]) -> list[Path]:
    """List the files to inspect, expanding directories.

    Args:
        paths: Files or directories given on the command line.

    Returns:
        The matching files, sorted, with the templates' ``*.py.j2`` included.
    """
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.name.endswith(SUFFIXES)
            )
        elif path.is_file() and path.name.endswith(SUFFIXES):
            found.append(path)
    return sorted(set(found))


def process(paths: list[Path], *, width: int, check: bool, quiet: bool) -> WrapReport:
    """Wrap the comment lines of every target file.

    Args:
        paths: Files or directories to process.
        width: Maximum line width.
        check: When ``True``, report without writing anything.
        quiet: When ``True``, keep the summary but drop the per-file lines.

    Returns:
        The aggregated :class:`WrapReport`.
    """
    report = WrapReport()
    for file in iter_sources(paths):
        report.files_scanned += 1
        original = file.read_text(encoding="utf-8")
        wrapped, counter, warnings = wrap_text(original, width)
        report.warnings.extend(f"{file}: {warning}" for warning in warnings)
        if wrapped == original:
            continue
        report.files_changed += 1
        report.lines_wrapped += counter
        if not quiet:
            print(f"  ↻ {file} ({counter} commentaire(s))")
        if not check:
            file.write_text(wrapped, encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """Entry point of the command line.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` when nothing is left to wrap (or when the writes succeeded), ``1`` in ``--check`` mode
        when at least one file would change, ``2`` when a warning is blocking.
    """
    parser = argparse.ArgumentParser(
        prog="wrapdoc",
        description="Replie les commentaires trop longs des gabarits du générateur.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("tools/scaffold")],
        help="Fichiers ou dossiers à inspecter (défaut : tools/scaffold).",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Largeur cible.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="N'écrit rien et sort en erreur si un repli est nécessaire.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Échoue aussi quand une ligne trop longue n'a pas pu être repliée.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Résumé seul, sans le détail par fichier."
    )
    args = parser.parse_args(argv)

    report = process(args.paths, width=args.width, check=args.check, quiet=args.quiet)
    print(report.render())
    if args.strict and report.warnings:
        return 2
    if args.check and report.files_changed:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - point d'entrée
    raise SystemExit(main())
