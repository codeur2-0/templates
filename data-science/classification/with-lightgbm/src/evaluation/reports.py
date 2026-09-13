"""Evaluation reports: turn an :class:`EvaluationResult` into readable artefacts.

Two artefacts are produced:

* ``artifacts/reports/evaluation_report.md`` — the document a reviewer reads: metrics,
  per-class diagnostics, confusion matrix, error analysis, hypotheses and recommendations,
* ``artifacts/metrics/classification_report.json`` — the machine readable counterpart.

The report mixes **computed** statements (numbers, dynamic recommendations derived from the
metrics) with **documented** guidance (the business recommendations of this use case), so
that it stays useful even when the metrics move.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.evaluator import EvaluationResult
from src.utils.io import write_json, write_text
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths
from src.visualization.plots import ClassificationPlots

logger = get_logger(__name__)

#: Titre du cas d'usage, utilisé dans l'en-tête du rapport.
USE_CASE_TITLE = "Prédiction d'attrition client (churn télécom) avec LightGBM"

#: Cible métier.
TARGET_NAME = "churned"

#: Recommandations métier documentées pour ce cas d'usage.
BUSINESS_RECOMMENDATIONS: tuple[str, ...] = (
    "Piloter le modèle sur le rappel de la classe churn et le coût par alerte, pas sur "
    "l'accuracy globale.",
    "Choisir le seuil de décision avec la table d'arbitrage précision/rappel/volume (voir le "
    "rapport) plutôt qu'avec 0.5 par défaut.",
    "Recalibrer les probabilités (isotonic ou Platt) avant tout usage en revenu attendu ou en "
    "scoring CRM.",
    "Ajouter des features temporelles (évolution de la consommation, récence du dernier "
    "ticket) : c'est le levier de performance principal.",
    "Mettre en place un suivi de dérive sur `contract_type`, `monthly_charges` et "
    "`satisfaction_score` (voir mlops/model-monitoring).",
    "Documenter les variables proxy (moyen de paiement) pour éviter un biais commercial "
    "involontaire.",
)


class ReportBuilder:
    """Build the Markdown report, its JSON counterpart and the figures."""

    def __init__(
        self, paths: ProjectPaths | None = None, *, config: Mapping[str, Any] | None = None
    ) -> None:
        """Store the output layout and the configuration.

        Args:
            paths: Project layout (figures and reports directories).
            config: Root configuration mapping, used to document the run.
        """
        self.paths = paths or ProjectPaths.from_root()
        self.config: dict[str, Any] = dict(config or {})
        self.plots = ClassificationPlots(self.paths.figures_dir)

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
            thresholds: Optional threshold analysis table.
            report_name: Output Markdown file name.

        Returns:
            Mapping of artefact kind (``report``, ``json``, figure names) to path.
        """
        self.paths.ensure()
        figures = self.plots.save_all(result, thresholds=thresholds)
        markdown = self.render_markdown(result, figures=figures, thresholds=thresholds, model=model)

        report_path = self.paths.reports_dir / (
            report_name or str(self._config_artifact("report_file", "evaluation_report.md"))
        )
        written: dict[str, Path] = {"report": write_text(report_path, markdown)}

        json_path = self.paths.metrics_dir / "classification_report.json"
        written["json"] = write_json(
            json_path,
            {
                **result.to_dict(),
                "recommendations": self.recommendations(result),
                "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                "figures": {
                    name: str(path.relative_to(self.paths.root)) for name, path in figures.items()
                },
            },
        )

        errors_path = self.paths.reports_dir / "top_errors.csv"
        if not result.errors.empty:
            written["errors"] = result.errors.to_csv(errors_path, index=False) and errors_path
        written.update(figures)
        logger.info("Rapport généré : {}", written["report"])
        return written

    def recommendations(self, result: EvaluationResult) -> list[str]:
        """Return the recommendations: dynamic (metrics-driven) then business (documented).

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
        thresholds: pd.DataFrame | None = None,
        model: Any = None,
    ) -> str:
        """Render the Markdown report.

        Args:
            result: Evaluation result.
            figures: Mapping of figure name to path (for relative links).
            thresholds: Optional threshold table.
            model: Optional model, used for the identity section.

        Returns:
            The Markdown document.
        """
        figures = figures or {}
        # Les cellules du tableau sont calculées avant la liste : chaque ligne de template
        # reste ainsi sous la limite de longueur (ruff E501) et le code reste lisible.
        primary_value = _fmt(result.primary_value)
        error_rate = _fmt(result.error_rate, percent=True)
        classes_cell = ", ".join(f"`{label}`" for label in result.labels) or "-"
        model_summary = (
            getattr(model, "summary", lambda: "-")()
            if model is not None
            else str(result.extras.get("model", "-"))
        )
        lines: list[str] = [
            f"# Rapport d'évaluation — {USE_CASE_TITLE}",
            "",
            f"*Généré le {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
            f"split `{result.split}` · {result.n_samples:,} observations · cible `{TARGET_NAME}`*",
            "",
            "## 1. Synthèse",
            "",
            "| Indicateur | Valeur |",
            "| --- | --- |",
            f"| Métrique principale (`{result.primary_metric}`) | **{primary_value}** |",
            f"| Taux d'erreur | {error_rate} |",
            f"| Nombre d'erreurs analysées | {len(result.errors)} |",
            f"| Classes | {classes_cell} |",
            f"| Modèle | {model_summary} |",
            "",
            "## 2. Métriques",
            "",
            "| Métrique | Valeur |",
            "| --- | --- |",
        ]
        lines.extend(
            f"| `{name}` | {_fmt(value)} |" for name, value in sorted(result.metrics.items())
        )
        lines += ["", "### Détail par classe", "", _dataframe_to_markdown(result.per_class), ""]

        if result.confusion_matrix is not None:
            lines += [
                "## 3. Matrice de confusion",
                "",
                _matrix_to_markdown(result.confusion_matrix, result.labels),
                "",
                _figure_link(figures.get("confusion_matrix"), self.paths, "Matrice de confusion"),
                "",
                _read_confusion(result),
                "",
            ]

        lines += [
            "## 4. Analyse d'erreurs",
            "",
            f"{len(result.errors)} erreurs parmi {result.n_samples:,} observations "
            f"({_fmt(result.error_rate, percent=True)}). Les erreurs les plus coûteuses sont "
            "celles où le modèle est **confiant et faux** : elles sont listées ci-dessous, "
            "triées par confiance décroissante.",
            "",
            _dataframe_to_markdown(result.errors.head(12), max_columns=8),
            "",
            "### Hypothèses sur les causes",
            "",
        ]
        lines.extend(f"- {hypothesis}" for hypothesis in self._error_hypotheses(result))
        lines += ["", "## 5. Recommandations", ""]
        lines.extend(
            f"{index}. {item}" for index, item in enumerate(self.recommendations(result), start=1)
        )

        if thresholds is not None and not thresholds.empty:
            lines += [
                "",
                "## 6. Arbitrage du seuil de décision",
                "",
                _dataframe_to_markdown(thresholds, max_columns=8),
                "",
                _figure_link(
                    figures.get("threshold_tradeoff"),
                    self.paths,
                    "Arbitrage précision/rappel/volume",
                ),
                "",
            ]

        lines += [
            "",
            "## Figures",
            "",
        ]
        lines.extend(
            f"- {name}: {_figure_link(path, self.paths, name)}"
            for name, path in sorted(figures.items())
        )
        lines += [
            "",
            "## Reproductibilité",
            "",
            "| Élément | Valeur |",
            "| --- | --- |",
            f"| Tâche | `{result.task}` |",
            f"| Features | {result.extras.get('n_features', '-')} |",
            f"| Balance des classes | {result.extras.get('class_balance', {})} |",
            f"| Seed | `{self.config.get('seed', '-')}` |",
            "",
            "> Rapport généré automatiquement par `src/evaluation/reports.py`. Toute métrique est "
            "recalculable avec `python scripts/evaluate.py`.",
            "",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ insights --------
    def _error_hypotheses(self, result: EvaluationResult) -> list[str]:
        """Derive plausible error causes from the diagnostics.

        Args:
            result: Evaluation result.

        Returns:
            The hypothesis list.
        """
        hypotheses: list[str] = []
        per_class = result.per_class
        if not per_class.empty:
            worst_recall = per_class.loc[per_class["recall"].idxmin()]
            worst_precision = per_class.loc[per_class["precision"].idxmin()]
            hypotheses.append(
                f"La classe `{worst_recall['class']}` est la moins bien rappelée "
                f"(recall={worst_recall['recall']:.3f}) : le modèle la confond avec une "
                "autre classe."
            )
            hypotheses.append(
                f"La classe `{worst_precision['class']}` génère le plus de faux positifs "
                f"(precision={worst_precision['precision']:.3f}) : le seuil de décision ou les "
                "features disponibles ne permettent pas de la distinguer finement."
            )

        if not result.feature_importance.empty:
            top = result.feature_importance.head(3)["feature"].tolist()
            hypotheses.append(
                f"La décision repose surtout sur {top} : si ces variables sont bruitées ou "
                "indisponibles en production, la performance chutera fortement."
            )

        if not result.errors.empty and "confidence" in result.errors.columns:
            confident = float((result.errors["confidence"] >= 0.75).mean())
            if confident > 0.2:
                hypotheses.append(
                    f"{confident:.0%} des erreurs sont faites avec une confiance ≥ 0.75 : "
                    "le modèle est **mal calibré** sur ces zones, un recalibrage (isotonic/Platt) "
                    "ou un rejet automatique (seuil d'abstention) est à prévoir."
                )
        curves = result.curves.get("calibration") or {}
        if curves.get("mean_predicted") and curves.get("fraction_positive"):
            gap = float(
                np.mean(
                    np.abs(
                        np.asarray(curves["mean_predicted"])
                        - np.asarray(curves["fraction_positive"])
                    )
                )
            )
            if gap > 0.05:
                hypotheses.append(
                    f"Écart moyen de calibration de {gap:.3f} : les probabilités ne sont pas "
                    "utilisables telles quelles pour un calcul de revenu attendu."
                )
        hypotheses.append(
            "Une partie des erreurs est irréductible : le générateur injecte un bruit "
            "volontaire, la performance maximale atteignable est donc inférieure à 1.0."
        )
        return hypotheses

    def _dynamic_recommendations(self, result: EvaluationResult) -> list[str]:
        """Metric-driven recommendations (actionable next steps)."""
        recommendations: list[str] = []
        metrics = result.metrics
        per_class = result.per_class

        positive = _positive_class_row(per_class)
        if positive is not None:
            if float(positive["recall"]) < 0.55:
                recommendations.append(
                    "Rappel de la classe positive faible : abaisser le seuil de décision "
                    "(voir l'analyse de seuil) ou pondérer les classes à l'entraînement."
                )
            if float(positive["precision"]) < 0.55:
                recommendations.append(
                    "Précision de la classe positive faible : ajouter des features discriminantes "
                    "ou exiger un double contrôle humain sur les alertes."
                )
        if np.isfinite(metrics.get("roc_auc", float("nan"))) and metrics.get("roc_auc", 0) < 0.65:
            recommendations.append(
                "AUC modérée : le signal est faible. Priorité à la donnée (nouvelles sources, "
                "historique plus long) plutôt qu'à l'hyperparamétrage."
            )
        if result.curves.get("calibration"):
            means = np.asarray(result.curves["calibration"]["mean_predicted"])
            observed = np.asarray(result.curves["calibration"]["fraction_positive"])
            if len(means) and float(np.abs(means - observed).max()) > 0.15:
                recommendations.append(
                    "Calibration insuffisante pour un usage probabiliste : envelopper le modèle "
                    "dans un `CalibratedClassifierCV` (sigmoïde ou isotonic)."
                )
        if not result.feature_importance.empty:
            share = float(
                result.feature_importance.head(3)["importance"].sum()
                / max(float(result.feature_importance["importance"].sum()), 1e-9)
            )
            if share > 0.75:
                recommendations.append(
                    f"Concentration excessive de l'importance ({share:.0%} sur 3 features) : "
                    "risque de dérive. Surveiller ces variables en production."
                )
        recommendations.append(
            "Comparer cette exécution aux autres stacks du même cas d'usage "
            "(mêmes données, mêmes métriques) avant de figer un choix technique."
        )
        return recommendations

    # ------------------------------------------------------------------ helpers ---------
    def _config_artifact(self, key: str, default: str) -> str:
        """Read an artefact file name from the configuration."""
        artifacts = (self.config.get("train") or {}).get("artifacts") or {}
        return str(artifacts.get(key, default))


def _positive_class_row(per_class: pd.DataFrame) -> pd.Series | None:
    """Return the row of the positive class (label ``1`` when present)."""
    if per_class.empty:
        return None
    for label in ("1", "1.0", "yes", "true", "churn"):
        match = per_class[per_class["class"].astype(str).str.lower() == label]
        if not match.empty:
            return match.iloc[0]
    return per_class.iloc[-1] if len(per_class) == 2 else None


def _read_confusion(result: EvaluationResult) -> str:
    """Summarise the confusion matrix in plain French."""
    matrix = result.confusion_matrix
    if matrix is None or matrix.size != 4:
        return (
            "Matrice multi-classes : les erreurs se répartissent sur plusieurs paires de classes."
        )
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in np.asarray(matrix).ravel()
    )
    total = max(true_negative + false_positive + false_negative + true_positive, 1)
    # Une table (libellé, effectif, lecture métier) plutôt que quatre f-strings :
    # le rendu reste aligné et chaque ligne du template tient sous 100 caractères.
    rows = (
        ("Vrais négatifs", true_negative, "clients conservés correctement identifiés"),
        ("Vrais positifs", true_positive, "départs détectés"),
        ("Faux positifs", false_positive, "coût : action de rétention inutile"),
        ("Faux négatifs", false_negative, "coût : client perdu sans action"),
    )
    return "\n".join(
        f"- **{label}** : {count} ({count / total:.1%}) — {comment}."
        for label, count, comment in rows
    )


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
    """Return a report-relative path.

    Args:
        path: File to express relatively.
        root: Project root (a ``Path`` or a :class:`ProjectPaths` instance).

    Returns:
        The relative path, or the absolute one when the file is outside the project.
    """
    if path is None:
        return ""
    # ``root`` peut être un ``Path`` ou un ``ProjectPaths`` : on normalise en chaîne pour
    # rester compatible avec les deux (et avec mypy).
    base = str(getattr(root, "root", root))
    try:
        return str(Path(path).relative_to(Path(base)))
    except ValueError:
        return str(path)


def _figure_link(path: Path | None, root: Path | ProjectPaths, label: str) -> str:
    """Render a Markdown image link (or an empty string)."""
    relative = _relative(path, root)
    return f"![{label}]({relative})" if relative else ""


def _dataframe_to_markdown(
    frame: pd.DataFrame | None, *, max_columns: int = 10, float_format: str = "{:.4f}"
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


def _matrix_to_markdown(matrix: np.ndarray | None, labels: Sequence[str]) -> str:
    """Render a confusion matrix as a Markdown table."""
    if matrix is None:
        return "_Aucune matrice._"
    array = np.asarray(matrix)
    header = "| réel \\ prédit | " + " | ".join(str(label) for label in labels) + " |"
    separator = "| " + " | ".join(["---"] * (len(labels) + 1)) + " |"
    rows = [
        f"| **{label}** | " + " | ".join(str(int(value)) for value in array[index]) + " |"
        for index, label in enumerate(labels)
    ]
    return "\n".join([header, separator, *rows])
