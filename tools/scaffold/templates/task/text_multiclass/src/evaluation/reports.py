"""Rapport d'évaluation lisible : verdict, ventilation, erreurs, recommandations.

Le rapport est écrit en français et en Markdown parce qu'il est lu par des humains — et parce
qu'un chiffre sans phrase qui l'interprète ne se discute pas. Il est reconstruit à partir du
*payload* d'évaluation, jamais d'un log : ce que le rapport affiche est exactement ce que
``evaluation_metrics.json`` contient, et réciproquement.

Trois règles d'honnêteté y sont appliquées :

* le **verdict** compare la métrique principale au seuil contractuel, et il est *indéterminé*
  quand la métrique n'a pas pu être calculée (plutôt que « conforme » par défaut) ;
* les **références triviales** (classe majoritaire, tirage stratifié) sont publiées dans le même
  tableau que le modèle : un F1 macro de 0,72 ne vaut rien sans le 0,27 d'un modèle constant ;
* les **limites** sont nommées avec leur chiffre (classe la plus faible, erreur de calibration,
  écart entre styles) : une limite sans mesure est une opinion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.evaluator import EvaluationResult
from src.utils.io import write_table, write_text
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths
from src.visualization.plots import plot_calibration, plot_confusion, plot_lengths, plot_per_class

logger = get_logger(__name__)

#: Metrics rendered in the main table, in reading order with their French label.
MAIN_METRICS: tuple[tuple[str, str], ...] = (
    ("accuracy", "Exactitude globale"),
    ("macro_f1", "F1 macro (métrique principale)"),
    ("weighted_f1", "F1 pondérée par les effectifs"),
    ("balanced_accuracy", "Exactitude équilibrée (rappel moyen par classe)"),
    ("macro_precision", "Précision macro"),
    ("macro_recall", "Rappel macro"),
    ("cohen_kappa", "Kappa de Cohen (accord au-delà du hasard)"),
    ("expected_calibration_error", "Erreur de calibration attendue (ECE)"),
    ("latency_p50_ms", "Latence médiane par prédiction (ms)"),
    ("latency_p95_ms", "Latence p95 par prédiction (ms)"),
)


@dataclass
class ReportBundle:
    """What the report builder produced.

    Attributes:
        report_path: The Markdown report.
        summary: One-line summary printed by the pipeline.
        artifacts: Every artefact written (report, tables, figures).
        verdict: ``conforme``, ``non conforme`` or ``indéterminé``.
    """

    report_path: Path
    summary: str
    artifacts: list[Path] = field(default_factory=list)
    verdict: str = "indéterminé"


class ReportBuilder:
    """Render the evaluation report, its tables and its figures."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        paths: ProjectPaths | None = None,
    ) -> None:
        """Configure the builder.

        Args:
            config: Full application configuration (project identity, model, metrics).
            paths: Project filesystem layout.
        """
        self.config = dict(config or {})
        self.paths = paths or ProjectPaths.from_root()

    # ------------------------------------------------------------------ entrée ------------
    def build(
        self,
        result: EvaluationResult,
        *,
        model_summary: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        documents: pd.DataFrame | None = None,
    ) -> ReportBundle:
        """Write the report, the tables and the figures.

        Args:
            result: Evaluation payload.
            model_summary: Description of the evaluated model (name, framework, algorithm, ...).
            metadata: Generation metadata (archived next to the corpus).
            documents: Labelled corpus, used by the length figure.

        Returns:
            The :class:`ReportBundle`.

        Raises:
            ValueError: When the evaluation payload holds no metric.
        """
        if not result.metrics:
            msg = "The evaluation payload holds no metric: nothing to report"
            raise ValueError(msg)
        tables = self._tables(result)
        figures = self._figures(result, documents)
        report_path = write_text(
            self.paths.reports_dir / str(self._report_name()),
            self._render(result, tables, figures, dict(model_summary or {}), dict(metadata or {})),
        )
        bundle = ReportBundle(
            report_path=report_path,
            summary=self._summary(result),
            artifacts=[report_path, *tables.values(), *figures.values()],
            verdict=result.verdict,
        )
        logger.info("Report written: {} ({})", report_path, result.verdict)
        return bundle

    # ------------------------------------------------------------------ tables ------------
    def _tables(self, result: EvaluationResult) -> dict[str, Path]:
        """Write the tabular artefacts of the report."""
        written: dict[str, Path] = {}
        for name, frame in (
            ("per_class", result.per_class),
            ("segment_metrics", result.segments),
            ("top_errors", result.top_errors),
            ("predictions", result.predictions.drop(columns=["text"], errors="ignore")),
        ):
            if frame is None or frame.empty:
                continue
            written[name] = write_table(frame, self.paths.reports_dir / f"{name}.csv")
        if not result.confusion.empty:
            written["confusion_matrix"] = write_table(
                result.confusion.reset_index(names="expected_label"),
                self.paths.reports_dir / "confusion_matrix.csv",
            )
        return written

    def _figures(
        self, result: EvaluationResult, documents: pd.DataFrame | None
    ) -> dict[str, Path]:
        """Draw the diagnostic figures the data can support."""
        written: dict[str, Path] = {}
        if not result.confusion.empty:
            written["confusion"] = plot_confusion(
                result.confusion, self.paths.figures_dir / "confusion_matrix.png"
            )
        if not result.per_class.empty:
            written["per_class"] = plot_per_class(
                result.per_class, self.paths.figures_dir / "f1_per_class.png"
            )
        if not result.predictions.empty:
            written["calibration"] = plot_calibration(
                result.predictions, self.paths.figures_dir / "calibration.png"
            )
        if documents is not None and not documents.empty and "label" in documents.columns:
            written["lengths"] = plot_lengths(documents, self.paths.figures_dir / "lengths.png")
        return written

    # ------------------------------------------------------------------ rendu -------------
    def _render(
        self,
        result: EvaluationResult,
        tables: Mapping[str, Path],
        figures: Mapping[str, Path],
        model_summary: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> str:
        """Render the Markdown report."""
        project = dict(self.config.get("project") or {})
        algorithm = model_summary.get("algorithm", "-")
        framework = model_summary.get("framework", "-")
        dimensions = model_summary.get("n_features", 0)
        lines: list[str] = [
            f"# Rapport d'évaluation — {project.get('title', 'classifieur de texte')}",
            "",
            f"- **Projet** : `{project.get('key', 'projet')}` "
            f"({project.get('domain', '-')}/{project.get('problem', '-')}, famille "
            f"`{project.get('family', '-')}`, stack `{project.get('stack', '-')}`)",
            f"- **Modèle** : {model_summary.get('name', '-')} ({algorithm}, "
            f"framework `{framework}`, {dimensions} dimensions)",
            f"- **Corpus évalué** : {result.n_documents} tickets de test, "
            f"{result.n_classes} classes",
            f"- **Généré le** : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## 1. Verdict",
            "",
            self._verdict_paragraph(result),
            "",
            "## 2. Métriques",
            "",
            self._metrics_table(result),
            "",
            "## 3. Références triviales",
            "",
            self._baselines_table(result),
            "",
            "## 4. Ventilation par classe",
            "",
            self._per_class_table(result),
            "",
        ]
        lines.extend(self._confusion_lines(result, figures))
        lines.extend(self._segment_lines(result))
        lines.extend(self._error_lines(result))
        lines.extend(self._figure_lines(figures))
        lines.extend(self._reading_lines(result))
        lines.extend(self._reproducibility_lines(metadata))
        return "\n".join(lines).rstrip() + "\n"

    def _verdict_paragraph(self, result: EvaluationResult) -> str:
        """State the contractual verdict with its numbers."""
        observed = result.metrics.get(result.primary_metric)
        if observed is None or result.threshold is None or result.verdict == "indéterminé":
            return (
                f"**Verdict : {result.verdict}** — la métrique principale "
                f"`{result.primary_metric}` ou son seuil contractuel n'est pas disponible : "
                "un verdict ne s'invente pas, le rapport se contente des chiffres mesurés."
            )
        state = "**conforme**" if result.verdict == "conforme" else "**non conforme**"
        value = float(observed)
        # ``mypy`` a raison : sans le garde ci-dessus, ``observed`` resterait ``float | None``.
        threshold = float(result.threshold)
        margin = value - threshold
        return (
            f"Le projet est {state} au contrat : `{result.primary_metric} = {value:.4f}` "
            f"pour un seuil de non-régression de {threshold:.4f} (marge {margin:+.4f}). "
            "Le seuil est un garde-fou, pas un objectif : la lecture utile est le tableau qui suit."
        )

    def _metrics_table(self, result: EvaluationResult) -> str:
        """Render the main metrics table."""
        rows = ["| Indicateur | Valeur |", "| --- | --- |"]
        for key, label in MAIN_METRICS:
            if key in result.metrics:
                rows.append(f"| {label} | **{result.metrics[key]:.4f}** |")
        return "\n".join(rows)

    def _baselines_table(self, result: EvaluationResult) -> str:
        """Render the trivial references table."""
        if not result.baselines:
            return "_Aucune référence triviale calculée._"
        rows = ["| Référence | Exactitude | F1 macro |", "| --- | --- | --- |"]
        for name, values in sorted(result.baselines.items()):
            rows.append(
                f"| `{name}` | {values.get('accuracy', float('nan')):.4f} | "
                f"{values.get('macro_f1', float('nan')):.4f} |"
            )
        model_accuracy = result.metrics.get("accuracy")
        model_f1 = result.metrics.get("macro_f1")
        if model_accuracy is not None and model_f1 is not None:
            rows.append(f"| **modèle** | **{model_accuracy:.4f}** | **{model_f1:.4f}** |")
        return "\n".join(rows)

    def _per_class_table(self, result: EvaluationResult) -> str:
        """Render the per-class table, weakest F1 first."""
        if result.per_class.empty:
            return "_Aucune ventilation par classe disponible._"
        frame = result.per_class.sort_values("f1")
        rows = [
            "| Catégorie | Effectif | Précision | Rappel | F1 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for _, row in frame.iterrows():
            rows.append(
                f"| `{row['class']}` | {int(row['support'])} | {row['precision']:.3f} | "
                f"{row['recall']:.3f} | **{row['f1']:.3f}** |"
            )
        return "\n".join(rows)

    def _confusion_lines(self, result: EvaluationResult, figures: Mapping[str, Path]) -> list[str]:
        """Render the confusion section, with the worst confusion named."""
        if result.confusion.empty:
            return []
        matrix = result.confusion.to_numpy(dtype="float64")
        labels = [str(label) for label in result.confusion.columns]
        lines = ["", "## 5. Matrice de confusion", ""]
        if "confusion" in figures:
            lines.append(f"![Matrice de confusion]({_relative(figures['confusion'], self.paths.root)})")
            lines.append("")
        off_diagonal = matrix.copy()
        for position in range(min(matrix.shape)):
            off_diagonal[position, position] = 0.0
        if off_diagonal.max() > 0:
            row, column = (int(index) for index in divmod(int(off_diagonal.argmax()), matrix.shape[1]))
            lines.append(
                f"Confusion dominante : **{int(off_diagonal[row, column])}** tickets étiquetés "
                f"`{labels[row]}` classés `{labels[column]}` — c'est la paire à traiter en premier "
                "(désambiguïsation de vocabulaire ou règle métier, pas nécessairement un modèle "
                "plus gros)."
            )
        return lines

    def _segment_lines(self, result: EvaluationResult) -> list[str]:
        """Render the segment table."""
        if result.segments.empty:
            return []
        rows = [
            "",
            "## 6. Ventilation par segment",
            "",
            "| Segment | Tickets | Exactitude | F1 macro | ECE |",
            "| --- | --- | --- | --- | --- |",
        ]
        for _, row in result.segments.iterrows():
            rows.append(
                f"| `{row['segment']}` | {int(row['n_documents'])} | {row['accuracy']:.3f} | "
                f"{row['macro_f1']:.3f} | {row['expected_calibration_error']:.3f} |"
            )
        return rows

    def _error_lines(self, result: EvaluationResult) -> list[str]:
        """Render the most confident mistakes."""
        if result.top_errors.empty:
            return ["", "## 7. Erreurs", "", "Aucune erreur : le modèle classe parfaitement le test."]
        rows = [
            "",
            "## 7. Erreurs les plus confiantes",
            "",
            "| Ticket | Référence | Prédiction | Confiance | Extrait |",
            "| --- | --- | --- | --- | --- |",
        ]
        for _, row in result.top_errors.head(10).iterrows():
            preview = str(row.get("text_preview", "")).replace("|", "/")
            rows.append(
                f"| `{row['doc_id']}` | `{row['expected_label']}` | `{row['prediction']}` | "
                f"{row['confidence']:.3f} | {preview[:90]} |"
            )
        return rows

    def _figure_lines(self, figures: Mapping[str, Path]) -> list[str]:
        """Reference the figures with a relative path."""
        if not figures:
            return []
        lines = ["", "## 8. Figures", ""]
        captions = {
            "per_class": "F1 par catégorie, effectif annoté",
            "calibration": "Courbe de calibration (confiance annoncée vs exactitude observée)",
            "lengths": "Longueur des textes par catégorie",
        }
        for name, path in sorted(figures.items()):
            if name == "confusion":
                continue
            lines.append(f"![{captions.get(name, name)}]({_relative(path, self.paths.root)})")
            lines.append("")
        return lines

    def _reading_lines(self, result: EvaluationResult) -> list[str]:
        """Write the honest reading of the run, with the numbers that support it."""
        lines = ["", "## 9. Lecture honnête", ""]
        if not result.per_class.empty:
            weakest = result.per_class.sort_values("f1").iloc[0]
            strongest = result.per_class.sort_values("f1").iloc[-1]
            lines.append(
                f"- La catégorie la plus faible est `{weakest['class']}` "
                f"(F1 {weakest['f1']:.3f} sur {int(weakest['support'])} tickets) contre "
                f"{strongest['f1']:.3f} pour `{strongest['class']}` : l'écart se lit dans la "
                "matrice de confusion, pas dans la moyenne."
            )
        error = result.metrics.get("expected_calibration_error")
        if error is not None:
            lines.append(
                f"- L'erreur de calibration vaut {error:.3f} : le modèle annonce une confiance "
                "qui s'écarte de son exactitude réelle, ce qui interdit de router "
                "automatiquement sur la seule confiance."
            )
        styles = result.segments[result.segments["segment"].str.startswith("style=")]
        if len(styles) >= 2:
            best = styles.sort_values("macro_f1").iloc[-1]
            worst = styles.sort_values("macro_f1").iloc[0]
            lines.append(
                f"- L'écart entre styles rédactionnels va de {worst['macro_f1']:.3f} "
                f"(`{worst['segment']}`) à {best['macro_f1']:.3f} (`{best['segment']}`) : "
                "un modèle de vocabulaire perd là où le ticket ne réutilise pas les mots "
                "de sa catégorie."
            )
        if result.baselines.get("classe_majoritaire"):
            floor = result.baselines["classe_majoritaire"].get("accuracy", 0.0)
            gain = result.metrics.get("accuracy", 0.0) - floor
            lines.append(
                f"- Le gain sur un modèle constant (classe majoritaire, "
                f"{floor:.3f} d'exactitude) est de {gain:+.3f} : c'est la vraie mesure "
                "de ce que le projet apporte."
            )
        return lines

    def _reproducibility_lines(self, metadata: Mapping[str, Any]) -> list[str]:
        """Close the report with the exact commands of the run."""
        lines = [
            "",
            "## 10. Reproductibilité",
            "",
            "```bash",
            "make all        # generate-data -> train -> evaluate -> predict",
            "make verify     # lint, types, tests, notebooks, pipeline",
            "```",
            "",
        ]
        if metadata:
            lines.append(
                "Le corpus est synthétique et sa recette est archivée avec lui "
                "(`data/raw/generation_metadata.json`) : "
                f"{int(metadata.get('n_documents', 0))} tickets, graine "
                f"{metadata.get('seed', '-')}, parts de styles "
                f"{float(metadata.get('share_paraphrase', 0.0)):.0%} paraphrase et "
                f"{float(metadata.get('share_bruite', 0.0)):.0%} bruité."
            )
        return lines

    def _report_name(self) -> str:
        """Resolve the report file name declared by the stack."""
        train = dict(self.config.get("train") or {})
        artifacts = dict(train.get("artifacts") or {})
        return str(artifacts.get("report_file", "evaluation_report.md"))

    def _summary(self, result: EvaluationResult) -> str:
        """One-line summary printed by the pipeline."""
        observed = result.metrics.get(result.primary_metric)
        rendered = "n/a" if observed is None else f"{float(observed):.4f}"
        return (
            f"{result.n_documents} tickets évalués | {result.primary_metric}={rendered} | "
            f"exactitude={result.metrics.get('accuracy', 0.0):.4f} | "
            f"verdict {result.verdict}"
        )


def _relative(path: Path, root: Path) -> str:
    """Render a path relative to the project root (links must stay portable)."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:  # pragma: no cover - defensive: figures always live in the project
        return path.as_posix()


__all__ = ["MAIN_METRICS", "ReportBuilder", "ReportBundle"]
