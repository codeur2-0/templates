"""Rapport d'évaluation lisible : verdict, ventilation, erreurs, confiance, surfaces réservées.

Le rapport est écrit en français et en Markdown parce qu'il est lu par des humains — et parce
qu'un
chiffre sans phrase qui l'interprète ne se discute pas. Il est reconstruit à partir du *payload*
d'évaluation, jamais d'un log : ce que le rapport affiche est exactement ce que
``evaluation_metrics.json`` contient, et réciproquement.

Quatre règles d'honnêteté y sont appliquées :

* le **verdict** compare la métrique principale au seuil contractuel, et il est *indéterminé*
quand la
  métrique n'a pas pu être calculée (plutôt que « conforme » par défaut) ;
* les **références** sont publiées dans le même tableau que le modèle : le plancher trivial (0,0
pour
  un système qui n'annote rien) et la **couche de règles**, mesurée sur les mêmes lignes — un F1
  de
  0,92 ne vaut rien sans le 0,87 de la référence explicable qui l'accompagne ;
* la **confiance** est jugée sur pièces : la précision observée par niveau est tabulée, et une
  confiance de 0,9 qui ne vaut que 0,6 en précision est publiée comme telle ;
* les **limites** sont nommées avec leur chiffre (type le plus faible, erreurs de bornes, écart
entre
  surfaces réservées et surfaces vues à l'entraînement) : une limite sans mesure est une opinion.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.evaluator import SEGMENT_COLUMNS, EvaluationResult
from src.utils.io import write_table, write_text
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths
from src.visualization.plots import (
    plot_confidence,
    plot_entity_counts,
    plot_holdout,
    plot_lengths,
    plot_per_label,
)

logger = get_logger(__name__)

#: Metrics rendered in the main table, in reading order with their French label.
MAIN_METRICS: tuple[tuple[str, str], ...] = (
    ("entity_f1", "F1 au niveau entité, micro (métrique principale)"),
    ("entity_precision", "Précision au niveau entité"),
    ("entity_recall", "Rappel au niveau entité"),
    ("macro_f1", "F1 macro (les cinq types comptent pareil)"),
    ("partial_f1", "F1 partielle (bornes tolérées, même type)"),
    ("type_accuracy", "Exactitude du type sur les spans alignés"),
    ("boundary_accuracy", "Exactitude des bornes sur les spans alignés"),
    ("holdout_recall", "Rappel sur les surfaces réservées aux splits d'évaluation"),
    ("seen_surface_recall", "Rappel sur les surfaces déjà vues à l'entraînement"),
    ("mean_confidence", "Confiance moyenne publiée avec les mentions"),
    ("confidence_gap", "Écart de confiance sur les mentions correctes"),
    ("n_predicted_entities", "Mentions publiées"),
    ("share_from_rules", "Part des mentions corroborées par une règle"),
    ("latency_p50_ms", "Latence médiane par message (ms, artefact chaud)"),
    ("latency_p95_ms", "Latence p95 par message (ms, artefact chaud)"),
)

#: French labels of the mistake kinds produced by the error analysis.
ERROR_KINDS: dict[str, str] = {
    "inventee": "mention inventée (aucune référence, aucun recouvrement)",
    "bornes": "bornes décalées (bon type, recouvrement partiel)",
    "manquee": "mention manquée (référence non couverte)",
}


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
        spans: pd.DataFrame | None = None,
    ) -> ReportBundle:
        """Write the report, the tables and the figures.

        Args:
            result: Evaluation payload.
            model_summary: Description of the evaluated extractor (name, framework, algorithm,
            ...).
            metadata: Generation metadata (archived next to the corpus).
            documents: Full message table, used by the corpus figures.
            spans: Full annotation table, used by the corpus figures.

        Returns:
            The :class:`ReportBundle`.

        Raises:
            ValueError: When the evaluation payload holds no metric.
        """
        if not result.metrics:
            msg = "The evaluation payload holds no metric: nothing to report"
            raise ValueError(msg)
        tables = self._tables(result)
        figures = self._figures(result, documents, spans)
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
            ("per_label", result.per_label),
            ("segment_metrics", result.segments),
            ("confidence", result.confidence),
            ("holdout", result.holdout),
            ("errors", result.errors),
            (
                "predictions",
                result.predictions.drop(columns=["text"], errors="ignore")
                if not result.predictions.empty
                else result.predictions,
            ),
        ):
            if frame is None or frame.empty:
                continue
            written[name] = write_table(frame, self.paths.reports_dir / f"{name}.csv")
        return written

    def _figures(
        self,
        result: EvaluationResult,
        documents: pd.DataFrame | None,
        spans: pd.DataFrame | None,
    ) -> dict[str, Path]:
        """Draw the diagnostic figures the data can support."""
        written: dict[str, Path] = {}
        if not result.per_label.empty:
            written["per_label"] = plot_per_label(
                result.per_label, self.paths.figures_dir / "f1_per_label.png"
            )
        if not result.confidence.empty:
            written["confidence"] = plot_confidence(
                result.confidence, self.paths.figures_dir / "confidence.png"
            )
        if not result.holdout.empty:
            written["holdout"] = plot_holdout(
                result.holdout, self.paths.figures_dir / "holdout.png"
            )
        distribution = _distribution(documents, spans)
        if not distribution.empty:
            written["counts"] = plot_entity_counts(
                distribution, self.paths.figures_dir / "entity_counts.png"
            )
        if documents is not None and not documents.empty and "n_tokens" in documents.columns:
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
        labels = ", ".join(f"`{label}`" for label in model_summary.get("labels") or [])
        title = str(project.get("title") or "extraction d'entités")
        lines: list[str] = [
            f"# Rapport d'évaluation — {title}",
            "",
            f"- **Projet** : `{project.get('key', 'projet')}` "
            f"({project.get('domain', '-')}/{project.get('problem', '-')}, famille "
            f"`{project.get('family', '-')}`, stack `{project.get('stack', '-')}`)",
            f"- **Modèle** : {model_summary.get('name', '-')} "
            f"(`{model_summary.get('algorithm', '-')}`, framework "
            f"`{model_summary.get('framework', '-')}`, état `{model_summary.get('state', '-')}`)",
            f"- **Entités extraites** : {labels or '-'}",
            f"- **Corpus évalué** : {result.n_documents} messages de test, "
            f"{result.n_entities} entités de référence",
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
            "## 3. Références mesurées sur le même split",
            "",
            self._baselines_table(result),
            "",
            "## 4. Ventilation par type d'entité",
            "",
            self._per_label_table(result),
            "",
            "## 5. Erreurs",
            "",
            self._error_lines(result),
            "",
            "## 6. Confiance publiée",
            "",
            self._confidence_lines(result, figures),
            "",
            "## 7. Surfaces réservées et surfaces vues à l'entraînement",
            "",
            self._holdout_lines(result),
            "",
            "## 8. Ventilation par segment",
            "",
            self._segment_lines(result),
            "",
        ]
        lines.extend(self._figure_lines(figures))
        lines.extend(self._reading_lines(result))
        lines.extend(self._reproducibility_lines(metadata, tables))
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
            "Le seuil est un garde-fou, pas un objectif : la lecture utile est la comparaison "
            "avec la couche de règles du tableau suivant."
        )

    def _metrics_table(self, result: EvaluationResult) -> str:
        """Render the main metrics table."""
        rows = ["| Indicateur | Valeur |", "| --- | --- |"]
        for key, label in MAIN_METRICS:
            if key in result.metrics:
                rows.append(f"| {label} | **{result.metrics[key]:.4f}** |")
        return "\n".join(rows)

    def _baselines_table(self, result: EvaluationResult) -> str:
        """Render the measured references table (trivial floor and rule layer)."""
        if not result.baselines:
            return "_Aucune référence mesurée : le rapport ne peut pas situer le modèle._"
        rows = [
            "| Référence | F1 micro | F1 macro | Rappel | Précision | Mentions publiées |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, values in sorted(result.baselines.items()):
            rows.append(
                f"| `{name}` | {values.get('entity_f1', 0.0):.4f} | "
                f"{values.get('macro_f1', 0.0):.4f} | {values.get('entity_recall', 0.0):.4f} | "
                f"{values.get('entity_precision', 0.0):.4f} | "
                f"{values.get('n_predicted_entities', 0.0):.0f} |"
            )
        rows.append(
            f"| **modèle** | **{result.metrics.get('entity_f1', 0.0):.4f}** | "
            f"**{result.metrics.get('macro_f1', 0.0):.4f}** | "
            f"{result.metrics.get('entity_recall', 0.0):.4f} | "
            f"{result.metrics.get('entity_precision', 0.0):.4f} | "
            f"{result.metrics.get('n_predicted_entities', 0.0):.0f} |"
        )
        lines = [*rows]
        if "regles" in result.baselines:
            gain = result.metrics.get("entity_f1", 0.0) - result.baselines["regles"].get(
                "entity_f1", 0.0
            )
            lines.append("")
            lines.append(
                f"Gain du modèle sur la couche de règles : **{gain:+.4f}** de F1 micro. "
                "La couche de règles est apprise sur le **train** uniquement puis mesurée sur le "
                "test : elle ne peut pas recopier les surfaces réservées, et le modèle non plus — "
                "c'est l'écart qui est publié, pas une intention."
            )
        return "\n".join(lines)

    def _per_label_table(self, result: EvaluationResult) -> str:
        """Render the per-label table, weakest F1 first."""
        if result.per_label.empty:
            return "_Aucune ventilation par type disponible._"
        frame = result.per_label[result.per_label["label"] != "micro"].sort_values("f1")
        rows = [
            "| Type | Support | Précision | Rappel | F1 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for _, row in frame.iterrows():
            rows.append(
                f"| `{row['label']}` | {int(row['support'])} | {row['precision']:.3f} | "
                f"{row['recall']:.3f} | **{row['f1']:.3f}** |"
            )
        return "\n".join(rows)

    def _error_lines(self, result: EvaluationResult) -> str:
        """Render the mistakes, counted by kind then listed."""
        if result.errors.empty:
            return "Aucune erreur : toutes les références du test sont retrouvées."
        counts = result.errors["kind"].astype(str).value_counts().to_dict()
        lines = [
            "| Nature | Nombre (échantillon publié) | Définition |",
            "| --- | --- | --- |",
        ]
        for kind, label in ERROR_KINDS.items():
            lines.append(f"| `{kind}` | {int(counts.get(kind, 0))} | {label} |")
        lines.extend(
            [
                "",
                "Les erreurs les plus instructives, avec la phrase qui les entoure :",
                "",
                "| Message | Nature | Type | Surface de référence | Surface prédite | Contexte |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for _, row in result.errors.head(12).iterrows():
            lines.append(
                f"| `{row['msg_id']}` | `{row['kind']}` | `{row['label']}` | "
                f"{str(row.get('surface', ''))[:40] or '-'} | "
                f"{str(row.get('predicted_surface', '') or '-')[:40]} | "
                f"{str(row.get('context', '-'))[:70]} |"
            )
        return "\n".join(lines)

    def _confidence_lines(self, result: EvaluationResult, figures: Mapping[str, Path]) -> str:
        """Render the confidence table, with the sentence its numbers support."""
        if result.confidence.empty:
            return "_Aucune mention publiée : la confiance ne peut pas être mesurée._"
        rows = [
            "| Niveau de corroboration | Mentions | Correctes | Précision observée |",
            "| --- | --- | --- | --- |",
        ]
        for _, row in result.confidence.iterrows():
            precision = "n/a" if int(row["n_mentions"]) == 0 else f"{row['precision']:.3f}"
            rows.append(
                f"| {row['bucket']} | {int(row['n_mentions'])} | {int(row['n_correct'])} | "
                f"{precision} |"
            )
        if "confidence" in figures:
            rows.extend(
                [
                    "",
                    f"![Précision observée par niveau de confiance]"
                    f"({_relative(figures['confidence'], self.paths.root)})",
                ]
            )
        gap = result.metrics.get("confidence_gap")
        if gap is not None:
            rows.extend(
                [
                    "",
                    f"Confiance moyenne des mentions correctes : écart de **{gap:.3f}** à 1,0. "
                    "Une confiance qui ne vaut que ce qu'elle annonce interdit de router "
                    "automatiquement sur le seul niveau le plus haut ; les tranches basses se "
                    "relisent.",
                ]
            )
        return "\n".join(rows)

    def _holdout_lines(self, result: EvaluationResult) -> str:
        """Render the reserved-surface comparison."""
        if result.holdout.empty:
            return "_Le corpus ne déclare pas de surfaces réservées : rien à comparer._"
        rows = ["| Groupe | Entités | Retrouvées | Rappel |", "| --- | --- | --- | --- |"]
        for _, row in result.holdout.iterrows():
            label = (
                "surfaces réservées aux splits d'évaluation"
                if row["group"] == "reservee"
                else "surfaces vues à l'entraînement"
            )
            rows.append(
                f"| {label} | {int(row['n_entities'])} | {int(row['matched'])} | "
                f"**{row['recall']:.3f}** |"
            )
        reserved = result.metrics.get("holdout_recall")
        seen = result.metrics.get("seen_surface_recall")
        if reserved is not None and seen is not None:
            rows.extend(
                [
                    "",
                    f"Écart de rappel entre les surfaces déjà vues ({seen:.3f}) et les surfaces "
                    f"réservées ({reserved:.3f}) : **{seen - reserved:+.3f}**. Cet écart mesure ce "
                    "que le système a appris de la *forme* d'un nom par rapport à ce qu'il "
                    "recopie d'une liste — c'est la seule mesure qui distingue les deux.",
                ]
            )
        return "\n".join(rows)

    def _segment_lines(self, result: EvaluationResult) -> str:
        """Render the segment table."""
        if result.segments.empty:
            return "_Aucun segment déclaré par le corpus._"
        rows = [
            "| Dimension | Valeur | Messages | Entités | F1 micro | F1 partielle | Bornes |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, row in result.segments.iterrows():
            rows.append(
                f"| `{row['segment']}` | `{row['value']}` | {int(row['n_documents'])} | "
                f"{int(row['n_entities'])} | {row['entity_f1']:.3f} | {row['partial_f1']:.3f} | "
                f"{int(row['boundary_errors'])} |"
            )
        columns = ", ".join(f"`{name}`" for name in SEGMENT_COLUMNS)
        rows.extend(
            [
                "",
                f"Dimensions publiées : {columns}. Un score global qui cache un effondrement sur "
                "les messages abrégés n'est pas un résultat ; la table le montre.",
            ]
        )
        return "\n".join(rows)

    def _figure_lines(self, figures: Mapping[str, Path]) -> list[str]:
        """Reference the figures with a relative path."""
        if not figures:
            return []
        captions = {
            "per_label": "F1 par type d'entité, avec son support",
            "counts": "Corpus annoté : mentions par type et par split",
            "lengths": "Longueur des messages par style rédactionnel",
            "holdout": "Rappel sur les surfaces réservées, à côté des surfaces déjà vues",
        }
        lines = ["## 9. Figures", ""]
        for name, path in sorted(figures.items()):
            if name == "confidence":
                continue
            lines.append(f"![{captions.get(name, name)}]({_relative(path, self.paths.root)})")
            lines.append("")
        return lines

    def _reading_lines(self, result: EvaluationResult) -> list[str]:
        """Write the honest reading of the run, with the numbers that support it."""
        lines = ["## 10. Lecture honnête", ""]
        labels = result.per_label[result.per_label["label"] != "micro"]
        if not labels.empty:
            ranked = labels.sort_values("f1")
            weakest, strongest = ranked.iloc[0], ranked.iloc[-1]
            lines.append(
                f"- Le type le plus faible est `{weakest['label']}` "
                f"(F1 {weakest['f1']:.3f} sur {int(weakest['support'])} entités) contre "
                f"{strongest['f1']:.3f} pour `{strongest['label']}` : la F1 macro existe "
                "précisément pour rendre cet écart visible, la F1 micro le noierait."
            )
        boundary = result.metrics.get("boundary_accuracy")
        partial = result.metrics.get("partial_f1")
        strict = result.metrics.get("entity_f1")
        if boundary is not None and partial is not None and strict is not None:
            lines.append(
                f"- Bornes exactes : {boundary:.3f} d'exactitude, et la F1 partielle "
                f"({partial:.3f}) dépasse la F1 stricte ({strict:.3f}) de {partial - strict:+.3f}. "
                "L'écart chiffre ce que l'exigence de bornes coûte au système."
            )
        reserved = result.metrics.get("holdout_recall")
        seen = result.metrics.get("seen_surface_recall")
        if reserved is not None and seen is not None:
            lines.append(
                f"- Rappel de {seen:.3f} sur les surfaces vues à l'entraînement contre "
                f"{reserved:.3f} sur les surfaces réservées : le corpus a été construit pour que "
                "cet écart existe, et il est publié plutôt que contourné."
            )
        gap = result.metrics.get("confidence_gap")
        if gap is not None:
            lines.append(
                f"- L'écart de confiance sur les mentions correctes vaut {gap:.3f} : la confiance "
                "publiée est un **niveau de corroboration** par les règles, pas une probabilité du "
                "modèle, et le rapport ne lui prête pas d'autre sens."
            )
        styles = result.segments[result.segments["segment"] == "style"]
        if len(styles) >= 2:
            best = styles.sort_values("entity_f1").iloc[-1]
            worst = styles.sort_values("entity_f1").iloc[0]
            lines.append(
                f"- L'écart entre styles rédactionnels va de {worst['entity_f1']:.3f} "
                f"(`{worst['value']}`) à {best['entity_f1']:.3f} (`{best['value']}`) : l'écart "
                "entre styles rédactionnels est mesuré, jamais supposé. "
                "abrégés (« cmd 1234 », « 89.90 EUR ») sont plus difficiles que les messages "
                "rédigés, et c'est mesuré plutôt que supposé."
            )
        return lines

    def _reproducibility_lines(
        self, metadata: Mapping[str, Any], tables: Mapping[str, Path]
    ) -> list[str]:
        """Close the report with the exact commands of the run."""
        lines = [
            "## 11. Reproductibilité",
            "",
            "```bash",
            "make all        # generate-data -> train -> evaluate -> predict",
            "make verify     # lint, types, tests, notebooks, pipeline",
            "```",
            "",
        ]
        listed = ", ".join(f"`{name}.csv`" for name in sorted(tables))
        if listed:
            lines.append(f"Tables publiées à côté de ce rapport : {listed}.")
            lines.append("")
        if metadata:
            holdout = metadata.get("holdout") or {}
            share = float(holdout.get("share", 0.0))
            lines.append(
                "Le corpus est synthétique et sa recette est archivée avec lui "
                "(`data/raw/generation_metadata.json`) : "
                f"{int(metadata.get('n_documents', 0))} messages, "
                f"{int(metadata.get('n_entities', 0))} mentions, "
                f"graine `{metadata.get('seed', '-')}`, "
                f"{share:.1%} des mentions portant une surface réservée aux splits d'évaluation."
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
            f"{result.n_documents} messages évalués | {result.n_entities} entités | "
            f"{result.primary_metric}={rendered} | "
            f"macro_f1={result.metrics.get('macro_f1', 0.0):.4f} | verdict {result.verdict}"
        )


def _distribution(documents: pd.DataFrame | None, spans: pd.DataFrame | None) -> pd.DataFrame:
    """Count the mentions per type and per split from the two corpus tables.

    Args:
        documents: Message table (``msg_id``, ``split``).
        spans: Annotation table (``msg_id``, ``label``).

    Returns:
        One row per (split, label) with the number of mentions, or an empty frame when the corpus
        is
        unavailable (a report can be rebuilt from the evaluation payload alone).
    """
    if documents is None or spans is None:
        return pd.DataFrame()
    if documents.empty or spans.empty or "split" not in documents.columns:
        return pd.DataFrame()
    split_of = dict(zip(documents["msg_id"], documents["split"], strict=True))
    annotated = spans.assign(split=spans["msg_id"].map(split_of))
    return (
        annotated.groupby(["split", "label"], observed=True)
        .size()
        .rename("n_mentions")
        .reset_index()
    )


def _relative(path: Path, root: Path) -> str:
    """Render a path relative to the project root (links must stay portable)."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:  # pragma: no cover - defensive: figures always live in the project
        return path.as_posix()


__all__ = ["ERROR_KINDS", "MAIN_METRICS", "ReportBuilder", "ReportBundle"]
