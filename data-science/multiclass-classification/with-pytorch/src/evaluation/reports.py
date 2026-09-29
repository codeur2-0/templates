"""Evaluation reports: turn a multiclass :class:`EvaluationResult` into readable artefacts.

Artefacts produced:

* ``artifacts/reports/evaluation_report.md`` — the document a reviewer reads: verdict, the model
  against its references, per-class quality, confusions, decision policies, expert review,
  calibration, errors by segment, structural limits and recommendations;
* ``artifacts/metrics/classification_report.json`` — the machine readable counterpart;
* ``artifacts/reports/{per_class,decision_policies,abstention_curve,top_errors}.csv`` — the tables,
  for a dashboard or a spreadsheet.

The report mixes **computed** statements (numbers and recommendations derived from them) with
**documented** guidance (the business recommendations and confusion notes of the use case), so it
stays useful when the metrics move.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.decision import DiagnosisSettings
from src.evaluation.evaluator import EvaluationResult
from src.utils.io import write_json, write_text
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths
from src.visualization.plots import MulticlassPlots

logger = get_logger(__name__)

#: Titre du cas d'usage, utilisé dans l'en-tête du rapport.
USE_CASE_TITLE = "Diagnostic du mode de défaillance machine avec PyTorch"

#: Cible métier.
TARGET_NAME = "failure_mode"

#: Recommandations métier documentées pour ce cas d'usage.
BUSINESS_RECOMMENDATIONS: tuple[str, ...] = (
    "Router sur la décision **à coût minimal** plutôt que sur l'argmax : une panne réelle "
    "classée fausse alarme coûte bien plus cher qu'une équipe dérangée pour rien.",
    "Envoyer en revue experte les alarmes dont la confiance est sous le seuil : quelques "
    "pourcents d'alarmes relues suppriment une part importante des erreurs coûteuses.",
    "Ajouter les features physiques (écart de température, puissance, effort) comme recettes "
    "déclarées : c'est le principal levier de performance, bien avant l'hyperparamétrage.",
    "Instrumenter les machines sans capteur de vibration : c'est la seule donnée manquante du "
    "parc, et elle discrimine usure et surcharge.",
    "Surveiller la dérive de la répartition des modes et de `tool_wear_min` : un changement de "
    "politique d'outillage déplace toute la distribution.",
    "Ne pas promettre de diagnostic sur `random_failure` : documenter le plafond et traiter "
    "ces pannes par la maintenance préventive, pas par le modèle.",
)


class ReportBuilder:
    """Build the Markdown report, its JSON counterpart, the CSV tables and the figures."""

    def __init__(
        self, paths: ProjectPaths | None = None, *, config: Mapping[str, Any] | None = None
    ) -> None:
        """Store the output layout and the configuration.

        Args:
            paths: Project layout (figures and reports directories).
            config: Root configuration mapping (``diagnosis`` block, seed, artefact names).
        """
        self.paths = paths or ProjectPaths.from_root()
        self.config: dict[str, Any] = dict(config or {})
        self.plots = MulticlassPlots(self.paths.figures_dir)
        diagnosis = dict(self.config.get("diagnosis") or {})
        self.confusion_notes: dict[frozenset[str], str] = {
            frozenset(part.strip() for part in str(key).split("/")): str(note)
            for key, note in dict(diagnosis.get("confusion_notes") or {}).items()
        }

    # ------------------------------------------------------------------ public API ------
    def build(
        self,
        result: EvaluationResult,
        *,
        model: Any = None,
        thresholds: pd.DataFrame | None = None,
        report_name: str | None = None,
    ) -> dict[str, Path]:
        """Generate every report artefact.

        Args:
            result: Evaluation result to document.
            model: Optional model, used to document its identity.
            thresholds: Unused in multiclass (no binary threshold); kept for API symmetry.
            report_name: Output Markdown file name.

        Returns:
            Mapping of artefact kind (``report``, ``json``, tables, figure names) to path.
        """
        del thresholds
        self.paths.ensure()
        figures = self.plots.save_all(result)
        markdown = self.render_markdown(result, figures=figures, model=model)

        report_file = report_name or self._config_artifact("report_file", "evaluation_report.md")
        written: dict[str, Path] = {
            "report": write_text(self.paths.reports_dir / report_file, markdown)
        }
        written["json"] = write_json(
            self.paths.metrics_dir / "classification_report.json",
            {
                **result.to_dict(),
                "recommendations": self.recommendations(result),
                "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                "figures": {name: _relative(path, self.paths) for name, path in figures.items()},
            },
        )
        tables = {
            "per_class": result.per_class,
            "decision_policies": result.decision,
            "abstention_curve": result.abstention,
            "top_errors": result.errors,
        }
        for name, frame in tables.items():
            if frame is not None and not frame.empty:
                destination = self.paths.reports_dir / f"{name}.csv"
                frame.to_csv(destination, index=False)
                written[name] = destination
        written.update(figures)
        logger.info("Rapport généré : {}", written["report"])
        return written

    def recommendations(self, result: EvaluationResult) -> list[str]:
        """Return the recommendations: computed ones first, then the documented ones.

        Args:
            result: Evaluation result.

        Returns:
            The ordered recommendation list.
        """
        return [*self._dynamic_recommendations(result), *BUSINESS_RECOMMENDATIONS]

    # ------------------------------------------------------------------ rendering -------
    def render_markdown(
        self,
        result: EvaluationResult,
        *,
        figures: Mapping[str, Path] | None = None,
        model: Any = None,
    ) -> str:
        """Render the Markdown report.

        Args:
            result: Evaluation result.
            figures: Mapping of figure name to path (for relative links).
            model: Optional model, used for the identity section.

        Returns:
            The Markdown document.
        """
        figures = figures or {}
        model_summary = (
            getattr(model, "summary", lambda: "-")()
            if model is not None
            else str(result.extras.get("model", "-"))
        )
        verdict = "conforme" if result.is_compliant else "non conforme"
        objectives_met = result.extras.get("objectives_met", "-")
        primary_value = _fmt(result.primary_value)
        latency = _fmt(result.extras.get("latency_ms_per_1000"))
        stamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines: list[str] = [
            f"# Rapport d'évaluation — {USE_CASE_TITLE}",
            "",
            f"*Généré le {stamp} · split `{result.split}` · {result.n_samples:,} alarmes · "
            f"cible `{TARGET_NAME}`*",
            "",
            "## 1. Synthèse",
            "",
            "| Indicateur | Valeur |",
            "| --- | --- |",
            f"| Verdict | **{verdict}** ({objectives_met} objectifs atteints) |",
            f"| Métrique principale (`{result.primary_metric}`) | **{primary_value}** |",
            f"| Taux d'erreur de diagnostic (argmax) | {_fmt(result.error_rate, percent=True)} |",
            f"| Erreur de calibration (ECE) | {_fmt(result.ece)} |",
            f"| Classes | {', '.join(f'`{label}`' for label in result.labels)} |",
            f"| Modèle | {model_summary} |",
            "",
            self._position_sentence(result),
            "",
            "## 2. Verdict par objectif",
            "",
            _table(result.verdict),
            "",
            "## 3. Le modèle face à ses références",
            "",
            "Le même split de test est rejoué par chaque référence : la classe majoritaire "
            "(plancher), le routage actuel par code automate (la règle à battre) et le plafond "
            "oracle publié par le générateur (le meilleur score atteignable, bruit compris).",
            "",
            _table(result.references),
            "",
            _figure_link(figures.get("references"), self.paths, "Références"),
            "",
            "## 4. Métriques globales",
            "",
            "| Métrique | Valeur |",
            "| --- | --- |",
            *[f"| `{name}` | {_fmt(value)} |" for name, value in sorted(result.metrics.items())],
            "",
            "## 5. Qualité par mode",
            "",
            _table(result.per_class),
            "",
            _figure_link(figures.get("per_class_metrics"), self.paths, "Qualité par mode"),
            "",
            *self._structural_lines(result),
            "## 6. Confusions",
            "",
            _matrix_table(result.confusion_matrix, result.labels),
            "",
            _figure_link(figures.get("confusion_matrix"), self.paths, "Matrice de confusion"),
            "",
            "### Paires les plus confondues",
            "",
            _table(result.confusions.head(6)),
            "",
            *self._confusion_readings(result),
            "## 7. Décision : argmax ou coût minimal",
            "",
            "L'argmax répond à *quel mode est le plus probable* ; le métier demande *quelle "
            "action coûte le moins*. La décision à coût minimal choisit, pour chaque alarme, le "
            "mode dont le coût attendu (probabilités x matrice de coûts) est le plus faible.",
            "",
            _table(result.decision),
            "",
            _figure_link(figures.get("decision_costs"), self.paths, "Coût des politiques"),
            "",
            "### Matrice de coûts (EUR par alarme, vraie classe en ligne, décision en colonne)",
            "",
            self._cost_matrix_table(result),
            "",
            "## 8. Revue experte des alarmes incertaines",
            "",
            _table(result.abstention),
            "",
            _figure_link(figures.get("abstention_tradeoff"), self.paths, "Revue experte"),
            "",
            self._abstention_sentence(result),
            "",
            "## 9. Calibration",
            "",
            _table(result.calibration),
            "",
            _figure_link(figures.get("reliability"), self.paths, "Diagramme de fiabilité"),
            "",
            self._calibration_sentence(result),
            "",
            "## 10. Erreurs",
            "",
            *self._segment_lines(result),
            "### Erreurs les plus confiantes",
            "",
            "Un diagnostic **confiant et faux** est le plus dangereux : il ne part pas en revue.",
            "",
            _table(result.errors.head(10), max_columns=9),
            "",
            "## 11. Recommandations",
            "",
            *[
                f"{index}. {item}"
                for index, item in enumerate(self.recommendations(result), start=1)
            ],
            "",
            "## Figures",
            "",
            *[
                f"- {name} : {_figure_link(path, self.paths, name)}"
                for name, path in sorted(figures.items())
            ],
            "",
            "## Reproductibilité",
            "",
            "| Élément | Valeur |",
            "| --- | --- |",
            f"| Tâche | `{result.task}` |",
            f"| Features | {result.extras.get('n_features', '-')} |",
            f"| Répartition des classes (test) | {result.extras.get('class_balance', {})} |",
            f"| Seed | `{self.config.get('seed', '-')}` |",
            f"| Latence (ms pour 1 000 alarmes) | {latency} |",
            "",
            "> Rapport généré automatiquement par `src/evaluation/reports.py`. Toute métrique est "
            "recalculable avec `python scripts/evaluate.py`.",
            "",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ sections --------
    def _position_sentence(self, result: EvaluationResult) -> str:
        """Place the model between the current rule and the oracle ceiling."""
        references = (
            result.references.set_index("référence")
            if not result.references.empty
            else pd.DataFrame()
        )
        rule = "routage actuel (code automate)"
        oracle = "plafond oracle (générateur)"
        if rule not in references.index or oracle not in references.index:
            return (
                "_Références incomplètes : la position du modèle entre la règle et le plafond "
                "n'est pas calculable._"
            )
        rule_f1 = float(references.loc[rule, "f1_macro"])
        oracle_f1 = float(references.loc[oracle, "f1_macro"])
        model_f1 = (
            result.primary_value
            if result.primary_metric == "f1_macro"
            else float(references.loc["modèle", "f1_macro"])
        )
        if not np.isfinite(oracle_f1 - rule_f1) or oracle_f1 <= rule_f1:
            return ""
        share = (model_f1 - rule_f1) / (oracle_f1 - rule_f1)
        return (
            f"Le modèle parcourt **{share:.0%}** du chemin entre la règle actuelle (macro-F1 "
            f"{rule_f1:.3f}) et le plafond atteignable ({oracle_f1:.3f}) : c'est cette part, "
            "et non le macro-F1 absolu, qui mesure ce que le modèle apporte."
        )

    def _structural_lines(self, result: EvaluationResult) -> list[str]:
        """Explain the classes that carry no observable signal."""
        per_class = result.per_class
        if per_class.empty or "structural" not in per_class or not per_class["structural"].any():
            return []
        lines = ["### Modes sans signal observable", ""]
        reference = result.extras.get("generation_reference", {}) or {}
        oracle_recall = dict(reference.get("recall_per_class_oracle") or {})
        for _, row in per_class[per_class["structural"]].iterrows():
            oracle = oracle_recall.get(str(row["class"]))
            oracle_text = f" (oracle : {float(oracle):.3f})" if oracle is not None else ""
            recall = float(row["recall"])
            lines.append(
                f"- `{row['class']}` : rappel {recall:.3f}{oracle_text}. Aucun capteur n'annonce "
                "ce mode : il est exclu de l'objectif de rappel par mode et relève de la "
                "maintenance préventive, pas du diagnostic."
            )
        return [*lines, ""]

    def _confusion_readings(self, result: EvaluationResult) -> list[str]:
        """Attach the documented business reading to the confused pairs, when one exists."""
        if result.confusions.empty or not self.confusion_notes:
            return []
        lines: list[str] = []
        seen: set[frozenset[str]] = set()
        for _, row in result.confusions.head(6).iterrows():
            pair = frozenset({str(row["classe réelle"]), str(row["classe prédite"])})
            note = self.confusion_notes.get(pair)
            if note and pair not in seen:
                seen.add(pair)
                lines.append(f"- **{row['classe réelle']} ↔ {row['classe prédite']}** : {note}")
        return ["### Lecture métier", "", *lines, ""] if lines else []

    def _cost_matrix_table(self, result: EvaluationResult) -> str:
        """Render the cost matrix actually used by the decision."""
        try:
            settings = DiagnosisSettings.resolve(self.config or None, classes=result.labels)
        except ValueError as exc:
            return f"_Matrice indisponible : {exc}_"
        matrix = settings.cost_matrix(result.labels)
        frame = pd.DataFrame(matrix, index=result.labels, columns=result.labels)
        return _matrix_table(frame.to_numpy(), result.labels, integer=False)

    def _abstention_sentence(self, result: EvaluationResult) -> str:
        """Comment the review threshold against the cheapest threshold of the curve."""
        table = result.abstention
        if table.empty:
            return ""
        current = float(result.extras.get("review_threshold", float("nan")))
        best = table.loc[table["cost_per_alarm"].idxmin()]
        at_current = table.loc[(table["threshold"] - current).abs().idxmin()]
        reviewed_now = 1.0 - float(at_current["coverage"])
        cost_now = float(at_current["cost_per_alarm"])
        reviewed_best = 1.0 - float(best["coverage"])
        cost_best = float(best["cost_per_alarm"])
        return (
            f"Au seuil configuré ({current:.2f}), {reviewed_now:.1%} des alarmes partent en revue "
            f"pour un coût de {cost_now:.0f} EUR par alarme ; le seuil le moins coûteux de la "
            f"courbe est {float(best['threshold']):.2f} ({cost_best:.0f} EUR, "
            f"{reviewed_best:.1%} de revue)."
        )

    def _calibration_sentence(self, result: EvaluationResult) -> str:
        """Comment the expected calibration error."""
        ece = result.ece
        if not np.isfinite(ece):
            return ""
        if ece <= 0.05:
            return (
                f"ECE = {ece:.3f} : les probabilités sont utilisables telles quelles pour la "
                "décision à coût minimal."
            )
        return (
            f"ECE = {ece:.3f} : les probabilités s'écartent de la fréquence observée. Avant de "
            "les multiplier par des euros, les recalibrer (`CalibratedClassifierCV`, méthode "
            "isotonic) sur le split de validation."
        )

    def _segment_lines(self, result: EvaluationResult) -> list[str]:
        """Error rate by categorical segment of the prediction frame."""
        frame = result.predictions
        if frame.empty:
            return []
        segments = [
            column
            for column in frame.columns
            if column
            not in {"y_true", "y_pred", "decision", "second_choice", "is_error", "needs_review"}
            and not column.startswith("proba_")
            and frame[column].dtype == object
            and 1 < frame[column].nunique() <= 12
        ][:3]
        lines: list[str] = []
        for column in segments:
            table = (
                frame.groupby(frame[column].astype(str))
                .agg(
                    alarmes=("is_error", "size"),
                    taux_erreur=("is_error", "mean"),
                    confiance=("confidence", "mean"),
                )
                .sort_values("taux_erreur", ascending=False)
                .reset_index()
            )
            lines += [f"### Erreurs par `{column}`", "", _table(table), ""]
        return lines

    def _dynamic_recommendations(self, result: EvaluationResult) -> list[str]:
        """Metric-driven recommendations (actionable next steps)."""
        recommendations: list[str] = []
        decision = (
            result.decision.set_index("politique") if not result.decision.empty else pd.DataFrame()
        )
        if {"argmax", "coût minimal"} <= set(decision.index):
            saving = float(decision.loc["argmax", "cost_per_alarm"]) - float(
                decision.loc["coût minimal", "cost_per_alarm"]
            )
            if saving > 0:
                recommendations.append(
                    f"Router sur la décision à coût minimal plutôt que sur l'argmax : "
                    f"{saving:.0f} EUR économisés par alarme sur le test, au prix de quelques "
                    "déplacements préventifs."
                )
        if not result.abstention.empty:
            best = result.abstention.loc[result.abstention["cost_per_alarm"].idxmin()]
            current = float(result.extras.get("review_threshold", float("nan")))
            if np.isfinite(current) and abs(float(best["threshold"]) - current) >= 0.05:
                target = float(best["threshold"])
                recommendations.append(
                    f"Déplacer le seuil de revue experte de {current:.2f} à {target:.2f} : c'est "
                    "le point le moins coûteux de la courbe d'abstention (à confirmer sur la "
                    "validation)."
                )
        if np.isfinite(result.ece) and result.ece > 0.05:
            recommendations.append(
                "Recalibrer les probabilités (isotonic sur la validation) : la décision à coût "
                "minimal suppose des probabilités fidèles."
            )
        per_class = result.per_class
        if not per_class.empty:
            candidates = (
                per_class[~per_class["structural"]] if "structural" in per_class else per_class
            )
            if not candidates.empty:
                worst = candidates.loc[candidates["recall"].idxmin()]
                worst_recall = float(worst["recall"])
                recommendations.append(
                    f"Le mode `{worst['class']}` est le moins bien rappelé ({worst_recall:.3f}) : "
                    "c'est là qu'une feature supplémentaire ou des données étiquetées rapportent "
                    "le plus."
                )
        if not result.confusions.empty:
            top = result.confusions.iloc[0]
            pair = f"`{top['classe réelle']}` → `{top['classe prédite']}`"
            recommendations.append(
                f"La confusion la plus fréquente est {pair} ({int(top['effectif'])} alarmes) : "
                "l'instrumenter en priorité (feature dédiée, revue ciblée)."
            )
        if not result.feature_importance.empty:
            share = float(
                result.feature_importance.head(3)["importance"].sum()
                / max(float(result.feature_importance["importance"].abs().sum()), 1e-9)
            )
            if share > 0.75:
                recommendations.append(
                    f"{share:.0%} de l'importance tient sur trois features : surveiller leur "
                    "dérive et leur disponibilité en production."
                )
        return recommendations

    # ------------------------------------------------------------------ helpers ---------
    def _config_artifact(self, key: str, default: str) -> str:
        """Read an artefact file name from the configuration."""
        artifacts = (self.config.get("train") or {}).get("artifacts") or {}
        return str(artifacts.get(key, default))


def _fmt(value: Any, *, percent: bool = False) -> str:
    """Format a numeric value for the report."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.1%}" if percent else f"{number:.4f}"


def _relative(path: Path | None, root: Path | ProjectPaths) -> str:
    """Return a project-relative path (absolute when the file is outside the project)."""
    if path is None:
        return ""
    base = Path(str(getattr(root, "root", root)))
    try:
        return str(Path(path).relative_to(base)).replace("\\", "/")
    except ValueError:
        return str(path)


def _figure_link(path: Path | None, paths: ProjectPaths, label: str) -> str:
    """Render a Markdown image link relative to the report directory (or an empty string).

    Le rapport vit dans ``artifacts/reports`` et les figures dans ``artifacts/figures`` : le lien
    doit partir du rapport, sinon aucune visionneuse Markdown n'affiche l'image.
    """
    if path is None:
        return ""
    relative = os.path.relpath(Path(path), start=paths.reports_dir).replace("\\", "/")
    return f"![{label}]({relative})"


def _table(
    frame: pd.DataFrame | None, *, max_columns: int = 12, float_format: str = "{:.4f}"
) -> str:
    """Render a DataFrame as a Markdown table (truncated for readability)."""
    if frame is None or frame.empty:
        return "_Aucune donnée._"
    trimmed = frame.iloc[:, :max_columns].copy()
    for column in trimmed.columns:
        if pd.api.types.is_float_dtype(trimmed[column]):
            trimmed[column] = trimmed[column].map(
                lambda value: float_format.format(value) if pd.notna(value) else ""
            )
    header = "| " + " | ".join(str(column) for column in trimmed.columns) + " |"
    separator = "| " + " | ".join("---" for _ in trimmed.columns) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in trimmed.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _matrix_table(matrix: np.ndarray | None, labels: Sequence[str], *, integer: bool = True) -> str:
    """Render a square matrix as a Markdown table."""
    if matrix is None:
        return "_Aucune matrice._"
    array = np.asarray(matrix)
    header = "| réel \\ prédit | " + " | ".join(str(label) for label in labels) + " |"
    separator = "| " + " | ".join(["---"] * (len(labels) + 1)) + " |"
    rows = [
        f"| **{label}** | "
        + " | ".join(
            str(int(value)) if integer else f"{float(value):.0f}" for value in array[index]
        )
        + " |"
        for index, label in enumerate(labels)
    ]
    return "\n".join([header, separator, *rows])
