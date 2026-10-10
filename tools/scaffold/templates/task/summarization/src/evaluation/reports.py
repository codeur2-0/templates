"""Lecture d'une évaluation de résumé : la table des faits, le verdict, les références.

Un fichier de métriques JSON ne se lit pas : il faut savoir quel chiffre regarder, contre
quoi le comparer, et ce que le modèle a oublié. Le constructeur de rapport produit donc
**un document** — ``evaluation_report.md`` — qui commence par le verdict, poursuit par la
comparaison des stratégies, la couverture par type de fait, la ventilation par segment et
une sélection d'erreurs relues en clair, et finit par les limites de la mesure. Chaque
table citée est écrite séparément en CSV, donc réutilisable dans un portail ou un notebook.

Deux partis pris de rédaction :

* **les références sont publiées à côté du modèle**, jamais en note : un ROUGE de 0,42 ne veut rien
  dire seul, il vaut ce que valent le résumé vide (0,0) et la baseline extractive mesurée sur les
  mêmes lignes ;
* **les limites sont écrites noir sur blanc** : le ROUGE mesure la formulation, la couverture
  mesure la substance, et la détection d'hallucination s'arrête aux valeurs vérifiables. Un
  rapport qui ne dit pas cela laisse croire que 0,45 est « presque la référence ».
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from src.evaluation.evaluator import SummaryEvaluation
from src.utils.io import write_table, write_text
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths
from src.visualization.plots import plot_all

logger = get_logger(__name__)


@dataclass(slots=True)
class ReportBundle:
    """Paths written by one report build.

    Attributes:
        report: Markdown report.
        tables: CSV tables written for the report.
        figures: PNG figures written for the report.
        metrics: JSON metrics written for machines.
    """

    report: Path
    tables: dict[str, Path] = field(default_factory=dict)
    figures: dict[str, Path] = field(default_factory=dict)
    metrics: Path | None = None


class ReportBuilder:
    """Turn a :class:`SummaryEvaluation` into a readable report plus its tables and figures."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        paths: ProjectPaths | None = None,
        with_figures: bool = True,
    ) -> None:
        """Configure the builder.

        Args:
            config: Full application configuration (project identity, thresholds).
            paths: Project filesystem layout.
            with_figures: Whether the figures are produced (they are, unless a test says otherwise).
        """
        self.config = dict(config or {})
        self.paths = paths or ProjectPaths.from_root()
        self.with_figures = bool(with_figures)

    def build(self, result: SummaryEvaluation, *, name: str | None = None) -> ReportBundle:
        """Write the report, its tables and its figures.

        Args:
            result: Evaluation to publish.
            name: Base name of the report (defaults to ``evaluation_report``).

        Returns:
            The :class:`ReportBundle` describing what was written.
        """
        self.paths.ensure()
        stem = str(name or "evaluation_report")
        report_path = self.paths.reports_dir / f"{stem}.md"
        tables = self._tables(result)
        figures = (
            plot_all(
                per_strategy=result.per_strategy,
                fidelity=result.fidelity,
                predictions=result.predictions,
                segments=result.segments,
                figures_dir=self.paths.figures_dir,
            )
            if self.with_figures
            else {}
        )
        document = self._render(result, tables=tables, figures=figures)
        write_text(report_path, document)
        logger.info("Rapport écrit : {}", report_path)
        return ReportBundle(report=report_path, tables=tables, figures=dict(figures))

    # ------------------------------------------------------------------ tables --------------
    def _tables(self, result: SummaryEvaluation) -> dict[str, Path]:
        """Write the tables of the evaluation as CSV.

        Args:
            result: Evaluation to publish.

        Returns:
            Mapping ``table name -> path``.
        """
        candidates: dict[str, pd.DataFrame] = {
            "per_strategy": result.per_strategy,
            "segment_metrics": result.segments,
            "fidelity": result.fidelity,
            "predictions": result.predictions,
            "errors": result.errors,
        }
        written: dict[str, Path] = {}
        for name, frame in candidates.items():
            if frame is None or frame.empty:
                continue
            written[name] = write_table(frame, self.paths.reports_dir / f"{name}.csv")
        return written

    # ------------------------------------------------------------------ rendu ---------------
    def _render(
        self,
        result: SummaryEvaluation,
        *,
        tables: Mapping[str, Path],
        figures: Mapping[str, Path],
    ) -> str:
        """Render the markdown document.

        Args:
            result: Evaluation to publish.
            tables: Written tables (kept for the recap section).
            figures: Written figures.

        Returns:
            The markdown document.
        """
        project = dict(self.config.get("project", {}) or {})
        data = dict(self.config.get("data", {}) or {})
        model = dict(self.config.get("model", {}) or {})
        title = project.get("title", project.get("name", "résumé automatique"))
        lines: list[str] = [
            f"# Rapport d'évaluation — {title}",
            "",
            f"- **Projet** : `{project.get('key', 'resume-automatique')}` "
            f"({project.get('domain', 'nlp')}/{project.get('problem', 'summarization')}, "
            f"stack `{project.get('stack', 'seq2seq')}`)",
            f"- **Jeu de données** : `{data.get('dataset_name', 'intervention_reports')}` — "
            f"{result.n_documents} document(s) de test, "
            f"{len(result.strategies)} stratégie(s) mesurée(s) sur ces mêmes lignes",
            f"- **Algorithme servi** : `{model.get('algorithm', 'transformer_tiny')}`",
            f"- **Métrique contractuelle** : `{result.primary_metric}` "
            f"(seuil {result.threshold if result.threshold is not None else 'non déclaré'})",
            "",
            "## 1. Verdict",
            "",
            self._verdict_paragraph(result),
            "",
            "## 2. Métriques du modèle servi",
            "",
            self._metrics_table(result),
            "",
            "## 3. Références mesurées sur les mêmes lignes",
            "",
            self._baselines_table(result),
            "",
            "## 4. Où le résumé se trompe",
            "",
            self._fidelity_lines(result),
            "",
            "## 5. Ventilation par segment",
            "",
            self._segment_lines(result),
            "",
            "## 6. Erreurs relues",
            "",
            self._error_lines(result),
            "",
            "## 7. Ce que cette évaluation ne dit pas",
            "",
            self._limits_lines(result),
            "",
            "## 8. Reproductibilité",
            "",
            self._reproducibility_lines(result),
            "",
        ]
        if figures:
            lines.extend(["## 9. Figures", ""])
            lines.extend(self._figure_lines(figures))
            lines.append("")
        if tables:
            lines.extend(["## 10. Tables publiées", ""])
            for path in sorted(tables.values()):
                lines.append(f"- `{_relative(path, self.paths.root)}`")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------ sections ------------
    def _verdict_paragraph(self, result: SummaryEvaluation) -> str:
        """Sentence that reads the contract, margin included."""
        detail = dict(result.verdict_detail)
        if result.verdict == "indéterminé":
            reason = detail.get("reason", "aucun seuil exploitable")
            return f"**Verdict indéterminé** : {reason}."
        observed = float(detail.get("observed", 0.0))
        threshold = float(detail.get("threshold", 0.0))
        margin = float(detail.get("margin", 0.0))
        verb = "au-dessus" if margin >= 0.0 else "en dessous"
        return (
            f"**Verdict : {result.verdict.upper()}** — `{result.primary_metric}` vaut "
            f"**{observed:.4f}** sur {result.n_documents} document(s) de test, soit "
            f"{abs(margin):.4f} {verb} du seuil contractuel de {threshold:.4f}. "
            f"La couverture des faits saillants est de "
            f"{float(result.metrics.get('fact_coverage', 0.0)):.4f} et "
            f"{float(result.metrics.get('unsupported_share', 0.0)):.1%} des résumés contiennent "
            f"au moins une valeur absente du document."
        )

    def _metrics_table(self, result: SummaryEvaluation) -> str:
        """Two-column table of the published metrics, with the metric names in code style."""
        metrics = dict(result.metrics)
        ordered = [
            "rouge1_f",
            "rouge2_f",
            "rouge_l_f",
            "rouge1_p_mean",
            "rouge1_r_mean",
            "rouge_l_r_mean",
            "fact_coverage",
            "fact_precision",
            "unsupported_share",
            "unsupported_facts_mean",
            "compression_mean",
            "compression_max",
            "sentences_mean",
            "hit_max_length_share",
            "n_words_mean",
            "latency_p50_ms",
            "latency_p95_ms",
            "n_documents",
        ]
        rows = [("Métrique", "Valeur"), ("---", "---")]
        for name in ordered:
            if name in metrics:
                rows.append((f"`{name}`", _format(metrics[name])))
        for name in sorted(metrics):
            if name not in ordered and not name.startswith("covered_"):
                rows.append((f"`{name}`", _format(metrics[name])))
        return "\n".join(f"| {left} | {right} |" for left, right in rows)

    def _baselines_table(self, result: SummaryEvaluation) -> str:
        """Comparison table of the model and its references, on identical lines."""
        rows = [
            "| Référence | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Couverture | Compression |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, block in sorted(result.baselines.items()):
            rows.append(
                f"| {name} | {_format(block.get('rouge1_f', 0.0))} "
                f"| {_format(block.get('rouge2_f', 0.0))} "
                f"| {_format(block.get('rouge_l_f', 0.0))} "
                f"| {_format(block.get('fact_coverage', 0.0))} "
                f"| {_format(block.get('compression', 0.0))} |"
            )
        served = result.strategies[0] if result.strategies else "modele"
        rows.append(
            f"| **{served} (servi)** | {_format(result.metrics.get('rouge1_f', 0.0))} "
            f"| {_format(result.metrics.get('rouge2_f', 0.0))} "
            f"| {_format(result.metrics.get('rouge_l_f', 0.0))} "
            f"| {_format(result.metrics.get('fact_coverage', 0.0))} | "
            f"{_format(result.metrics.get('compression_mean', 0.0))} |"
        )
        return "\n".join(rows)

    def _fidelity_lines(self, result: SummaryEvaluation) -> str:
        """Per-fact-type coverage, with the weakest type called out."""
        frame = result.fidelity
        if frame.empty:
            return "Aucune table de fidélité disponible."
        columns = [column for column in frame.columns if column.startswith("covered_")]
        if not columns:
            return "Aucune couverture par type disponible."
        lines = ["| Type de fait | Couverture |", "| --- | --- |"]
        values = {column: float(frame[column].mean()) for column in columns}
        for column, value in sorted(values.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"| `{column.removeprefix('covered_')}` | {value:.4f} |")
        weakest = min(values, key=lambda name: values[name]).removeprefix("covered_")
        strongest = max(values, key=lambda name: values[name]).removeprefix("covered_")
        lines.extend(
            [
                "",
                f"Le type le mieux couvert est `{strongest}` "
                f"({values['covered_' + strongest]:.4f}), le moins bien couvert est "
                f"`{weakest}` ({values['covered_' + weakest]:.4f}) : c'est le "
                "premier endroit à regarder avant de raccourcir un résumé.",
            ]
        )
        return "\n".join(lines)

    def _segment_lines(self, result: SummaryEvaluation) -> str:
        """Segment table, one block per dimension, sorted by decreasing ROUGE-1."""
        frame = result.segments
        if frame.empty:
            return "Aucune ventilation par segment disponible."
        lines = [
            "| Segment | Valeur | Documents | ROUGE-1 F1 | Couverture | Compression |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for segment in sorted({str(value) for value in frame["segment"]}):
            subset = frame.loc[frame["segment"] == segment].sort_values("rouge1_f", ascending=False)
            for row in subset.itertuples(index=False):
                lines.append(
                    f"| `{row.segment}` | {row.value} | {int(row.n_documents)} "
                    f"| {_format(row.rouge1_f)} "
                    f"| {_format(getattr(row, 'fact_coverage', 0.0))} "
                    f"| {_format(getattr(row, 'compression', 0.0))} |"
                )
        return "\n".join(lines)

    def _error_lines(self, result: SummaryEvaluation) -> str:
        """Five worst documents, printed with the missing facts and the unsupported values."""
        frame = result.errors
        if frame.empty:
            return "Aucune erreur publiée (aucun document évalué)."
        lines: list[str] = []
        for row in frame.head(5).itertuples(index=False):
            lines.extend(
                [
                    f"### `{row.doc_id}` — ROUGE-1 {_format(row.rouge1_f)}, "
                    f"couverture {_format(row.fact_coverage)}",
                    "",
                    f"- **Type** : {row.intervention_type} · urgence {row.urgency} · "
                    f"{int(row.n_predicted_words)} mots produits",
                    f"- **Faits saillants manquants** : {row.missing_facts or 'aucun'}",
                    f"- **Valeurs non supportées** : {row.unsupported_values or 'aucune'}",
                    f"- **Référence** : {_excerpt(row.reference_summary)}",
                    f"- **Résumé produit** : {_excerpt(row.prediction)}",
                    "",
                ]
            )
        lines.append(
            f"Les {len(frame)} cas les plus éloignés sont publiés dans "
            "`artifacts/reports/errors.csv`, avec le type d'intervention, l'urgence "
            "et les faits manquants."
        )
        return "\n".join(lines)

    def _limits_lines(self, result: SummaryEvaluation) -> str:
        """What the measurement does not cover — written, not implied."""
        unsupported_share = float(result.metrics.get("unsupported_share", 0.0))
        return "\n".join(
            [
                "- **ROUGE mesure la formulation, pas le sens** : un résumé qui dit la même "
                "chose avec d'autres mots obtient un ROUGE bas, et un résumé qui recopie la "
                "référence en changeant un chiffre obtient un ROUGE haut.",
                "- **La couverture est mesurée sur les faits annotés** : elle compte les valeurs "
                "du document (durées, références, symptômes, actions, statuts) retrouvées dans "
                "le résumé, et ne dit rien de l'ordre ni de la causalité des phrases.",
                "- **La détection d'hallucination s'arrête aux valeurs vérifiables** : une "
                "valeur absente du document est comptée "
                f"({unsupported_share:.1%} des résumés en contiennent au moins une), mais une "
                "phrase inventée sans valeur numérique n'est pas détectable sans juge humain.",
                "- **La latence dépend de la machine** : elle est publiée pour la lecture, "
                "jamais pour le verdict, et n'entre dans aucune comparaison entre stratégies.",
                "- **Les documents sont synthétiques** : le vocabulaire est un gabarit "
                "paramétré, donc les valeurs absolues de ROUGE ne se transposent pas à un "
                "corpus réel — les écarts entre stratégies, eux, se lisent.",
            ]
        )

    def _reproducibility_lines(self, result: SummaryEvaluation) -> str:
        """Recipe of the run: seed, corpus, split sizes and strategies compared."""
        data = dict(self.config.get("data", {}) or {})
        metrics_config = dict(self.config.get("metrics", {}) or {})
        detail = dict(result.verdict_detail)
        measured = ", ".join(f"`{name}`" for name in result.strategies) or "aucune"
        references = ", ".join(f"`{name}`" for name in sorted(result.baselines)) or "aucune"
        direction = "maximiser"
        if detail.get("direction", "maximize") != "maximize":
            direction = "minimiser"
        return "\n".join(
            [
                f"- Graine : `{self.config.get('seed', 42)}` (corpus et modèles).",
                f"- Corpus : `{data.get('dataset_name', 'intervention_reports')}`, "
                f"{result.n_documents} document(s) de test, découpage écrit dans le corpus.",
                f"- Stratégies mesurées : {measured}.",
                f"- Références publiées : {references}.",
                f"- Barème : `{detail.get('metric', result.primary_metric)}` "
                f"({direction}), seuil {metrics_config.get('min_primary', 'non déclaré')}.",
                "- Deux exécutions à graine fixée produisent les mêmes résumés, donc les mêmes "
                "métriques ; la suite de tests le vérifie (`tests/test_training.py`, "
                "`tests/test_pipeline.py`).",
            ]
        )

    def _figure_lines(self, figures: Mapping[str, Path]) -> list[str]:
        """Markdown list of the figures, with their relative paths."""
        return [
            f"- `{_relative(path, self.paths.root)}` — {_FIGURE_TITLES.get(name, name)}"
            for name, path in sorted(figures.items())
        ]


#: Human titles of the figures, used in the report.
_FIGURE_TITLES: dict[str, str] = {
    "rouge_par_strategie": "ROUGE-1/2/L de chaque stratégie, mesuré sur les mêmes lignes",
    "couverture_par_type": "couverture des faits saillants par type",
    "compression_couverture": "longueur produite contre couverture des faits",
    "couts_sortie": "distribution des longueurs produites et part de résumés tronqués",
    "segments": "ROUGE-1 par segment du corpus (type d'intervention, urgence, site)",
}


def _format(value: object) -> str:
    """Format a metric for the report (four decimals for scores, integers as integers).

    Args:
        value: Metric value.

    Returns:
        The formatted string.
    """
    if isinstance(value, bool):
        return "oui" if value else "non"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.4f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def _excerpt(text: object, *, limit: int = 280) -> str:
    """Cut a text for the report, on a word boundary.

    Args:
        text: Text to shorten.
        limit: Maximum number of characters.

    Returns:
        The excerpt, suffixed with an ellipsis when it was cut.
    """
    value = " ".join(str(text).split())
    if len(value) <= limit:
        return value
    return value[:limit].rsplit(" ", 1)[0] + "…"


def _relative(path: Path, root: Path) -> str:
    """Render a path relative to the project root when possible.

    Args:
        path: Absolute path.
        root: Project root.

    Returns:
        The relative path, or the absolute one when it lies outside the project.
    """
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def report_sections() -> Sequence[str]:
    """Titles of the report sections (used by the tests and the README).

    Returns:
        The section titles, in order.
    """
    return (
        "1. Verdict",
        "2. Métriques du modèle servi",
        "3. Références mesurées sur les mêmes lignes",
        "4. Où le résumé se trompe",
        "5. Ventilation par segment",
        "6. Erreurs relues",
        "7. Ce que cette évaluation ne dit pas",
        "8. Reproductibilité",
    )


__all__ = ["ReportBuilder", "ReportBundle", "report_sections"]
