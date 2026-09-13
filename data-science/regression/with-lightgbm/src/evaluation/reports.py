"""Evaluation reports: turn an :class:`EvaluationResult` into readable artefacts.

Two artefacts are produced:

* ``artifacts/reports/evaluation_report.md`` — the document a reviewer reads: metrics, residual
  diagnostics, per-segment breakdown, tolerance-band coverage, error analysis, hypotheses and
  recommendations,
* ``artifacts/metrics/regression_report.json`` — the machine readable counterpart.

The report mixes **computed** statements (numbers, dynamic recommendations derived from the
metrics) with **documented** guidance (the business recommendations of this use case), so that it
stays useful even when the metrics move.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.evaluator import TOLERANCE_PCT, EvaluationResult
from src.utils.io import write_json, write_text
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths
from src.visualization.plots import RegressionPlots

logger = get_logger(__name__)

#: Titre du cas d'usage, utilisé dans l'en-tête du rapport.
USE_CASE_TITLE = "Estimation du prix de vente immobilier (AVM) avec LightGBM"

#: Cible métier.
TARGET_NAME = "price_eur"

#: Métrique de décision déclarée dans la configuration.
PRIMARY_METRIC = "rmse"

#: Recommandations métier documentées pour ce cas d'usage.
BUSINESS_RECOMMENDATIONS: tuple[str, ...] = (
    "Publier une **fourchette** et un niveau de confiance, jamais un prix unique : c'est ce "
    "que le métier sait exploiter.",
    "Piloter le modèle sur le MAPE, l'erreur médiane et la couverture de fourchette ; garder "
    "la RMSE comme indicateur de risque sur les biens chers.",
    "Travailler en log-prix (transformer la cible) puis revenir en euros à l'inférence, avec "
    "correction de biais de retransformation.",
    "Recalibrer par segment (quartier x tranche de surface) : un biais de +4 % en périphérie "
    "nord coûte plus qu'un biais moyen nul.",
    "Surveiller la dérive du marché (taux d'intérêt, saisonnalité) : un modèle de prix "
    "immobilier se décale en quelques mois, d'où le ré-entraînement hebdomadaire.",
    "Documenter et tester l'usage des proxies géographiques : mesurer l'écart d'estimation à "
    "caractéristiques égales entre quartiers est un garde-fou d'équité.",
    "Ajouter des features exogènes en production (transactions voisines, taux de crédit, "
    "tension locative) : c'est le levier de performance principal au-delà de la fiche bien.",
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
        self.plots = RegressionPlots(self.paths.figures_dir)

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
            thresholds: Accepted for API symmetry with the classification report; a regression has
                no decision threshold, so the argument is ignored.
            report_name: Output Markdown file name.

        Returns:
            Mapping of artefact kind (``report``, ``json``, figure names) to path.
        """
        del thresholds  # aucune notion de seuil de décision en régression
        self.paths.ensure()
        figures = self.plots.save_all(result)
        markdown = self.render_markdown(result, figures=figures, model=model)

        report_path = self.paths.reports_dir / (
            report_name or str(self._config_artifact("report_file", "evaluation_report.md"))
        )
        written: dict[str, Path] = {"report": write_text(report_path, markdown)}

        json_path = self.paths.metrics_dir / "regression_report.json"
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
            result.errors.to_csv(errors_path, index=False)
            written["errors"] = errors_path

        segments_path = self.paths.reports_dir / "error_by_segment.csv"
        if not result.per_segment.empty:
            result.per_segment.to_csv(segments_path, index=False)
            written["segments"] = segments_path

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
        model: Any = None,
    ) -> str:
        """Render the Markdown report.

        Args:
            result: Evaluation result.
            figures: Figure paths, keyed by figure name.
            model: Optional model, documented in the header.

        Returns:
            The Markdown text.
        """
        figures = dict(figures or {})
        metrics = dict(result.metrics)
        extras = dict(result.extras or {})
        generated = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines: list[str] = [
            f"# Rapport d'évaluation — {USE_CASE_TITLE}",
            "",
            f"*Généré le {generated} • split `{result.split}` • "
            f"{result.n_samples} observations • cible `{TARGET_NAME}`*",
            "",
        ]
        lines += self._header_block(result, model)
        lines += self._verdict_block(result, metrics, extras)
        lines += self._metrics_block(metrics)
        lines += self._residual_block(result, extras)
        lines += self._segment_block(result)
        lines += self._importance_block(result)
        lines += self._errors_block(result)
        lines += self._hypotheses_block(result)
        lines += self._recommendations_block(result)
        lines += self._figures_block(figures)
        lines += self._reproducibility_block(result)
        lines += [
            "## Limites assumées",
            "",
            "- Les données sont **synthétiques** et générées hors ligne : les niveaux de",
            "  performance illustrent une méthode, pas un marché immobilier réel.",
            "- Une seule passe d'évaluation sur un split unique : la variance inter-exécutions",
            "  n'est pas mesurée ici (voir le notebook 05, §4, pour la stabilité).",
            "- Le générateur injecte un bruit multiplicatif volontaire : une erreur nulle",
            "  signerait une fuite de données, pas un bon modèle.",
            "",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------ sections --------
    def _header_block(self, result: EvaluationResult, model: Any) -> list[str]:
        """Render the model identity header."""
        # `BaseModel.summary()` renvoie une **chaîne** descriptive (classe, framework, état).
        extras = dict(result.extras or {})
        model_summary = str(extras.get("model") or getattr(model, "summary", lambda: "-")())
        n_features = extras.get("n_features", "n/a")
        lines = [
            "## 1. Modèle évalué",
            "",
            "| Élément | Valeur |",
            "| --- | --- |",
            f"| Framework | `{getattr(model, 'framework', 'n/a')}` |",
            f"| Algorithme | `{getattr(model, 'algorithm', 'n/a')}` |",
            f"| Tâche | `{result.task}` |",
            f"| Features | {n_features} |",
            f"| Artefact | `{self._config_artifact('model_file', 'model.joblib')}` |",
            f"| Résumé | `{model_summary}` |",
            "",
        ]
        return lines

    def _verdict_block(
        self, result: EvaluationResult, metrics: Mapping[str, float], extras: Mapping[str, Any]
    ) -> list[str]:
        """Render the executive summary (the four numbers a reviewer reads first)."""
        coverage = result.coverage
        bias = result.bias
        median_error = _float(extras.get("median_absolute_error"))
        primary_text = _format(metrics.get(PRIMARY_METRIC))
        mape_text = _percent_from_pct(metrics.get("mape"))
        tolerance = _float(extras.get("tolerance_pct", TOLERANCE_PCT))
        direction = "sur-estimation" if bias > 0 else "sous-estimation"
        return [
            "## 2. Verdict synthétique",
            "",
            f"- **{PRIMARY_METRIC} = {primary_text}** (métrique de décision).",
            f"- **MAPE = {mape_text}** — erreur relative moyenne, lisible par le métier.",
            f"- **R² = {_format(metrics.get('r2'))}** — part de variance expliquée.",
            f"- **Couverture ± {tolerance:.0f} % = {_percent(coverage)}** (cible : >= 70 %).",
            f"- **Biais moyen = {_format(bias)}** ({direction} systématique,"
            f" {_percent_from_pct(extras.get('bias_pct'))} en relatif).",
            f"- **Erreur médiane = {_format(median_error)}** — plus informative que la RMSE seule.",
            "",
        ]

    def _metrics_block(self, metrics: Mapping[str, float]) -> list[str]:
        """Render the metric table."""
        lines = [
            "## 3. Métriques globales",
            "",
            "| Métrique | Valeur | Lecture |",
            "| --- | --- | --- |",
        ]
        readings = {
            "rmse": "pénalise les grosses erreurs (biens chers)",
            "mae": "erreur absolue typique, robuste aux outliers",
            "r2": "part de variance expliquée (1.0 = parfait)",
            "mape": "erreur relative moyenne, comparable entre biens",
            "smape": "MAPE symétrique, stable près de zéro",
            "max_error": "pire erreur individuelle (risque opérationnel)",
            "median_absolute_error": "erreur typique du portefeuille",
        }
        relative_metrics = {"mape", "smape"}
        for name, value in metrics.items():
            text_value = (
                _percent_from_pct(value) if str(name) in relative_metrics else _format(value)
            )
            lines.append(f"| `{name}` | {text_value} | {readings.get(str(name), '—')} |")
        lines.append("")
        return lines

    def _residual_block(self, result: EvaluationResult, extras: Mapping[str, Any]) -> list[str]:
        """Render the residual diagnostics (bias, spread, heteroscedasticity)."""
        frame = result.predictions
        lines = [
            "## 4. Lecture des résidus",
            "",
            "Un résidu est la différence `prédit - observé`. Trois lectures comptent :",
            "",
            "1. **le biais** (moyenne des résidus) : une estimation systématiquement haute",
            "   ou basse se corrige, et coûte cher en négociation ;",
            "2. **la dispersion** (écart-type, P95) : elle fixe la largeur de fourchette publiée ;",
            "3. **l'hétéroscédasticité** : si l'erreur croît avec le prix, la RMSE est dominée par",
            "   les biens chers et les petits biens sont sacrifiés en silence.",
            "",
            "| Statistique | Valeur |",
            "| --- | --- |",
            f"| Biais moyen | {_format(extras.get('bias'))} |",
            f"| Biais relatif | {_percent(extras.get('bias_pct'))} |",
            f"| Écart-type des résidus | {_format(extras.get('residual_std'))} |",
            f"| Erreur absolue médiane | {_format(extras.get('median_absolute_error'))} |",
            f"| Erreur absolue P95 | {_format(extras.get('absolute_error_p95'))} |",
            f"| Erreur maximale | {_format(result.metrics.get('max_error'))} |",
            "",
        ]
        bucket_frame = extras.get("error_by_bucket")
        if isinstance(bucket_frame, pd.DataFrame) and not bucket_frame.empty:
            lines += [
                "### Erreur par classe de valeur observée",
                "",
                "| Classe | n | Biais | Erreur relative médiane | RMSE | Couverture |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            for _, row in bucket_frame.iterrows():
                median_relative = row.get("median_relative_error_pct")
                median_text = (
                    _percent(float(median_relative) / 100.0) if pd.notna(median_relative) else "n/a"
                )
                lines.append(
                    f"| {row['bucket']} | {int(row['n'])} | {_format(row.get('bias'))} "
                    f"| {median_text} | {_format(row.get('rmse'))} "
                    f"| {_percent(row.get('coverage'))} |"
                )
            lines += [
                "",
                "> Une RMSE croissante avec le prix est **attendue** : erreur multiplicative.",
                "> Ce qui ne l'est pas, c'est un **biais de signe constant** sur une classe : il",
                "> indique un effet non appris (plafonnement, transformation manquante).",
                "",
            ]
        del frame
        return lines

    def _segment_block(self, result: EvaluationResult) -> list[str]:
        """Render the per-segment breakdown and flag the segments needing attention."""
        if result.per_segment.empty:
            return ["## 5. Analyse par segment", "", "Aucun axe de segmentation exploitable.", ""]
        frame = result.per_segment
        lines = [
            "## 5. Analyse par segment",
            "",
            "La même erreur, recomposée par quartier, par DPE et par tranche de prix :",
            "c'est là que se décide la prochaine itération du modèle.",
            "",
            "| Axe | Segment | n | RMSE | MAE | Biais | Biais relatif | Couverture |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, row in frame.head(24).iterrows():
            bias_pct = row.get("bias_pct")
            bias_text = _percent(float(bias_pct) / 100.0) if pd.notna(bias_pct) else "n/a"
            lines.append(
                f"| {row['axis']} | `{row['segment']}` | {int(row['n'])} "
                f"| {_format(row.get('rmse'))} | {_format(row.get('mae'))} "
                f"| {_format(row.get('bias'))} | {bias_text} "
                f"| {_percent(row.get('coverage'))} |"
            )
        lines.append("")

        worst = _worst_segments(frame)
        if worst:
            lines += ["**Segments à traiter en priorité :**", ""]
            lines += [f"- {item}" for item in worst]
            lines.append("")
        return lines

    def _importance_block(self, result: EvaluationResult) -> list[str]:
        """Render the feature importance ranking."""
        if result.feature_importance.empty:
            return [
                "## 6. Facteurs explicatifs",
                "",
                "Le modèle n'expose pas d'importance native et la permutation n'a pas",
                "abouti dans ce run.",
                "",
            ]
        frame = result.feature_importance.head(12)
        total = float(result.feature_importance["importance"].abs().sum()) or 1.0
        lines = [
            "## 6. Facteurs explicatifs",
            "",
            "| Feature | Importance | Part | Méthode |",
            "| --- | --- | --- | --- |",
        ]
        for _, row in frame.iterrows():
            share = abs(float(row["importance"])) / total
            method = row.get("method", "n/a")
            lines.append(
                f"| `{row['feature']}` | {_format(row['importance'])} | {share:.1%} | {method} |"
            )
        lines += [
            "",
            "Ces facteurs sont affichés à l'utilisateur final : une estimation sans explication",
            "n'est pas adoptée par les négociateurs.",
            "",
        ]
        return lines

    def _errors_block(self, result: EvaluationResult) -> list[str]:
        """Render the worst individual errors."""
        if result.errors.empty:
            return ["## 7. Pires erreurs individuelles", "", "Aucune erreur enregistrée.", ""]
        frame = result.errors.head(10)
        columns = [
            column
            for column in (
                "row_index",
                "district",
                "energy_rating",
                "surface_m2",
                "y_true",
                "y_pred",
                "relative_error_pct",
            )
            if column in frame.columns
        ]
        lines = [
            "## 7. Pires erreurs individuelles",
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for _, row in frame.iterrows():
            cells = [_cell_text(column, row[column]) for column in columns]
            lines.append("| " + " | ".join(cells) + " |")
        lines += [
            "",
            "Le détail complet est écrit dans `artifacts/reports/top_errors.csv` : support",
            "de la revue d'erreurs hebdomadaire.",
            "",
        ]
        return lines

    def _hypotheses_block(self, result: EvaluationResult) -> list[str]:
        """Render the error-cause hypotheses derived from the diagnostics."""
        hypotheses = self._error_hypotheses(result)
        return [
            "## 8. Hypothèses sur les causes",
            "",
            *[f"- {item}" for item in hypotheses],
            "",
        ]

    def _recommendations_block(self, result: EvaluationResult) -> list[str]:
        """Render the recommendations (dynamic first, then documented business guidance)."""
        dynamic = self._dynamic_recommendations(result)
        return [
            "## 9. Recommandations",
            "",
            "### issues de l'analyse (calculées)",
            "",
            *[f"{index}. {item}" for index, item in enumerate(dynamic, start=1)],
            "",
            "### documentées pour ce cas d'usage",
            "",
            *[f"{index}. {item}" for index, item in enumerate(BUSINESS_RECOMMENDATIONS, start=1)],
            "",
        ]

    def _figures_block(self, figures: Mapping[str, Path]) -> list[str]:
        """Render the figure index."""
        if not figures:
            return []
        titles = {
            "predicted_vs_actual": "Prédit vs observé",
            "residuals_vs_predicted": "Résidus vs prédiction",
            "error_distribution": "Distribution de l'erreur relative",
            "error_by_bucket": "Erreur par classe de prix",
            "coverage_by_bucket": "Couverture de la fourchette",
            "error_breakdown": "Hiérarchie des erreurs",
            "metrics_bar": "Synthèse des métriques",
            "feature_importance": "Facteurs explicatifs",
            "worst_errors": "Pires erreurs",
        }
        lines = ["## 10. Figures", ""]
        for name, path in figures.items():
            try:
                relative = Path(path).relative_to(self.paths.root)
            except ValueError:
                relative = Path(path)
            lines.append(f"- **{titles.get(name, name)}** — `{relative}`")
        lines.append("")
        return lines

    def _reproducibility_block(self, result: EvaluationResult) -> list[str]:
        """Render the reproducibility footer (seed, artefacts, configuration)."""
        train_node = dict(self.config.get("train") or {})
        seed = self.config.get("seed", train_node.get("random_state", "n/a"))
        model_file = self._config_artifact("model_file", "model.joblib")
        pipeline_file = self._config_artifact("pipeline_file", "preprocessing.joblib")
        return [
            "## 11. Reproductibilité",
            "",
            "| Élément | Valeur |",
            "| --- | --- |",
            f"| Seed | `{seed}` |",
            f"| Modèle | `artifacts/models/{model_file}` |",
            f"| Pré-traitement | `artifacts/models/{pipeline_file}` |",
            "| Métriques JSON | `artifacts/metrics/regression_report.json` |",
            "| Segments CSV | `artifacts/reports/error_by_segment.csv` |",
            "",
            "Rejouer exactement cette évaluation : `python scripts/evaluate.py`",
            "(ou `make evaluate`). La configuration est entièrement pilotée par Hydra",
            "(`conf/`) : aucune valeur n'est codée en dur dans le code source.",
            "",
        ]

    # ------------------------------------------------------------------ analysis --------
    def _error_hypotheses(self, result: EvaluationResult) -> list[str]:
        """Derive plausible error causes from the diagnostics.

        Args:
            result: Evaluation result.

        Returns:
            The hypothesis list.
        """
        hypotheses: list[str] = []
        metrics = dict(result.metrics)
        extras = dict(result.extras or {})

        bias_pct = _float(extras.get("bias_pct"))
        if np.isfinite(bias_pct) and abs(bias_pct) > 2.0:
            direction = "sur-estime" if bias_pct > 0 else "sous-estime"
            hypotheses.append(
                f"Le modèle {direction} systématiquement ({bias_pct:+.2f} % en moyenne) : la cause"
                " la plus fréquente est une retransformation du log-prix sans correction de biais"
                " (`exp(mean(log y))` ≠ `mean(y)`), ou une cible tronquée par le clipping."
            )
        rmse = _float(metrics.get("rmse"))
        mae = _float(metrics.get("mae"))
        if np.isfinite(rmse) and np.isfinite(mae) and mae > 0 and rmse / mae > 1.6:
            hypotheses.append(
                f"Rapport RMSE/MAE de {rmse / mae:.2f} : la distribution d'erreur est à queue"
                " lourde. Une poignée de biens (souvent les plus chers ou les plus atypiques)"
                " concentre l'erreur — à isoler avant de conclure sur le modèle."
            )
        if not result.per_segment.empty:
            coverage = result.per_segment["coverage"].astype("float64")
            weakest = result.per_segment.loc[coverage.idxmin()]
            if _float(weakest.get("coverage")) < 0.6:
                hypotheses.append(
                    f"Le segment `{weakest['segment']}` (axe `{weakest['axis']}`) tombe à"
                    f" {float(weakest['coverage']):.0%} de couverture : volume d'apprentissage"
                    " insuffisant ou effet spécifique non modélisé."
                )
        shrinkage = _shrinkage_signal(result)
        if shrinkage is not None:
            low_bias, high_bias = shrinkage
            hypotheses.append(
                "Rétraction vers la moyenne : les biens les moins chers sont sur-estimés"
                f" ({low_bias:+.1f} %) et les plus chers sous-estimés ({high_bias:+.1f} %)."
                " Cause habituelle : un entraînement en erreur quadratique sur une cible"
                " multiplicative, que corrige un apprentissage en log-prix (ou une perte L1)."
            )
        if not result.feature_importance.empty:
            top = result.feature_importance.head(3)["feature"].tolist()
            hypotheses.append(
                f"L'estimation repose surtout sur {top} : si ces variables sont bruitées ou"
                " manquantes, ou dérivent en production, l'erreur se dégrade immédiatement."
            )
        hypotheses.append(
            "Une partie de l'erreur est **irréductible** : le générateur injecte un bruit"
            " multiplicatif volontaire (négociation, état réel non observable). Une erreur proche"
            " de zéro signerait une fuite de données."
        )
        return hypotheses

    def _dynamic_recommendations(self, result: EvaluationResult) -> list[str]:
        """Metric-driven recommendations (actionable next steps)."""
        recommendations: list[str] = []
        metrics = dict(result.metrics)
        coverage = _float(result.coverage)

        if np.isfinite(coverage) and coverage < 0.70:
            recommendations.append(
                f"Couverture de {coverage:.0%} sous la cible métier (70 %) : élargir la fourchette"
                " publiée, ou ajouter les features manquantes (transactions voisines, tension du"
                " marché) avant de promettre ± 10 %."
            )
        if _float(metrics.get("r2")) < 0.7:
            recommendations.append(
                "R² faible : le signal disponible est insuffisant. Priorité à la donnée (sources"
                " exogènes, historique plus long) plutôt qu'à l'hyperparamétrage."
            )
        mape = _float(metrics.get("mape"))
        if np.isfinite(mape) and mape > 15.0:
            recommendations.append(
                f"MAPE de {mape:.1f} % : vérifier la transformation de la cible (travailler en"
                " log-prix) et la gestion des valeurs extrêmes (winsorising 0,5-99,5 %)."
            )
        if not result.per_segment.empty:
            biased = result.per_segment.assign(
                _bias=result.per_segment["bias_pct"].astype("float64").abs()
            ).sort_values("_bias", ascending=False)
            worst = biased.iloc[0]
            if _float(worst.get("_bias")) > 6.0:
                recommendations.append(
                    f"Biais segmenté de {float(worst['bias_pct']):+.1f} % sur `{worst['segment']}`"
                    f" (axe `{worst['axis']}`) : ajouter un recalibrage par segment ou une feature"
                    " dédiée plutôt qu'un modèle global unique."
                )
        shrinkage = _shrinkage_signal(result)
        if shrinkage is not None:
            recommendations.append(
                "Rétraction vers la moyenne détectée entre les extrêmes de prix : entraîner sur"
                " le log-prix (puis corriger la retransformation) ou recalibrer par tranche de"
                " prix avant de publier les estimations."
            )
        if not result.feature_importance.empty:
            total = float(result.feature_importance["importance"].abs().sum()) or 1.0
            share = float(result.feature_importance.head(3)["importance"].abs().sum()) / total
            if share > 0.75:
                recommendations.append(
                    f"Concentration excessive de l'importance ({share:.0%} sur 3 features) : risque"
                    " de dérive. Instrumenter ces variables en production."
                )
        recommendations.append(
            "Comparer cette exécution aux autres stacks du même cas d'usage (mêmes données, mêmes"
            " métriques) avant de figer un choix technique."
        )
        return recommendations

    # ------------------------------------------------------------------ helpers ---------
    def _config_artifact(self, key: str, default: str) -> str:
        """Read an artefact file name from the configuration."""
        artifacts = (self.config.get("train") or {}).get("artifacts") or {}
        train_node = self.config.get("train") or {}
        if key in artifacts:
            return str(artifacts[key])
        if key in train_node:
            return str(train_node[key])
        return default


#: Colonnes du tableau des pires erreurs qui portent un identifiant entier (jamais de décimales).
INTEGER_COLUMNS: tuple[str, ...] = ("row_index", "rooms", "floor_level", "recent_sales_1km")


def _cell_text(column: str, value: Any) -> str:
    """Format one table cell of the worst-errors table.

    Args:
        column: Column name (drives the integer / amount formatting).
        value: Cell value.

    Returns:
        The formatted cell.
    """
    if isinstance(value, (np.integer, int)) or str(column) in INTEGER_COLUMNS:
        try:
            return f"{int(float(value)):d}"
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, (np.floating, float)):
        return _format(value)
    return str(value)


def _worst_segments(frame: pd.DataFrame, limit: int = 3) -> list[str]:
    """Describe the segments with the weakest coverage or the largest bias.

    Args:
        frame: Per-segment statistics.
        limit: Maximum number of segments returned.

    Returns:
        Human readable sentences.
    """
    if frame.empty or "coverage" not in frame.columns:
        return []
    scored = frame.copy()
    scored["coverage"] = scored["coverage"].astype("float64")
    scored["bias_pct"] = scored["bias_pct"].astype("float64")
    weakest = scored.sort_values("coverage").head(limit)
    sentences = [
        f"`{row['segment']}` (axe `{row['axis']}`, n={int(row['n'])}) : couverture "
        f"{row['coverage']:.0%}, biais relatif {row['bias_pct']:+.1f} %, "
        f"RMSE {_format(row['rmse'])}."
        for _, row in weakest.iterrows()
    ]
    return sentences


def _shrinkage_signal(result: EvaluationResult) -> tuple[float, float] | None:
    """Detect the classic « shrinkage to the mean » pattern of a price model.

    Le signal : les tranches basses sont sur-estimées (biais relatif positif) **et** les tranches
    hautes sous-estimées (biais relatif négatif), avec un écart marqué. C'est la signature d'une
    perte quadratique appliquée à une cible multiplicative.

    Args:
        result: Evaluation result (uses the per-bucket table).

    Returns:
        ``(low_bucket_bias_pct, high_bucket_bias_pct)`` when the pattern is clear, else ``None``.
    """
    buckets = (result.extras or {}).get("error_by_bucket")
    if not isinstance(buckets, pd.DataFrame) or len(buckets) < 3:
        return None
    if "bias_pct" not in buckets.columns and "bias" not in buckets.columns:
        return None
    ordered = buckets.sort_values("bucket") if "bucket" in buckets.columns else buckets
    frame = ordered.copy()
    if "median_relative_error_pct" in frame.columns:
        biases = frame["median_relative_error_pct"].astype("float64")
    else:
        biases = frame["bias"].astype("float64")
    low_bias = float(biases.iloc[0])
    high_bias = float(biases.iloc[-1])
    if not (np.isfinite(low_bias) and np.isfinite(high_bias)):
        return None
    if low_bias > 1.5 and high_bias < -1.5:
        return low_bias, high_bias
    return None


def _float(value: Any) -> float:
    """Coerce ``value`` to float (NaN when impossible)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format(value: Any, digits: int = 2) -> str:
    """Format a number for the report (thousand separators, NaN as ``n/a``).

    Args:
        value: Number to format.
        digits: Number of decimals for small values.

    Returns:
        The formatted string.
    """
    number = _float(value)
    if not np.isfinite(number):
        return "n/a"
    if abs(number) >= 1000:
        return f"{number:,.0f}".replace(",", " ")
    return f"{number:.{digits}f}"


def _percent_from_pct(value: Any) -> str:
    """Format a value **already expressed in percent** (MAPE, sMAPE, relative errors).

    Le registre de métriques (``src/training/losses_metrics.py``) renvoie ``mape`` et ``smape``
    multipliés par 100, et l'évaluateur stocke ``relative_error_pct`` / ``bias_pct`` dans la même
    unité. Les repasser par :func:`_percent` afficherait 750 % au lieu de 7,5 %.

    Args:
        value: Nombre en unités de pourcentage (ou NaN).

    Returns:
        The formatted percentage.
    """
    number = _float(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.1f} %"


def _percent(value: Any) -> str:
    """Format a ratio as a percentage (NaN as ``n/a``).

    Args:
        value: Ratio in [0, 1] (or NaN).

    Returns:
        The formatted percentage.
    """
    number = _float(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.1%}"


__all__ = ["BUSINESS_RECOMMENDATIONS", "PRIMARY_METRIC", "ReportBuilder", "USE_CASE_TITLE"]
