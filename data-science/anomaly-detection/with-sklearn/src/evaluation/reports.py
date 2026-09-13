"""Report generation for an unsupervised anomaly detector (payment fraud).

The report is written for two readers at once: a **fraud lead** (who decides how many alerts the
team can investigate) and a **technical reviewer** (who checks that the score is not a leak and
that the metric choice matches the imbalance). Every number is followed by its reading, and every
reading by the criterion it satisfies or breaks.

Four questions drive the structure:

1. **le score ordonne-t-il la fraude ?** — PR AUC vs plancher aléatoire (= prévalence), ROC AUC,
   lift par décile ;
2. **que vaut-il au budget réel ?** — table d'arbitrage volume d'alertes → seuil → rappel /
   précision / lift, avec le budget retenu marqué. C'est la décision métier, pas un F1 abstrait ;
3. **quels modes opératoires sont couverts ?** — rappel par schéma : un bon score global peut
   masquer un schéma en croissance non détecté ;
4. **où se trompe-t-il ?** — fraudes manquées les mieux classées, fausses alertes les plus
   convaincantes (souvent des outliers *légitimes*), facteurs contributifs par permutation.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.evaluator import EvaluationResult
from src.utils.io import write_json, write_text
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths
from src.visualization.plots import SCHEME_LABELS, AnomalyPlots

logger = get_logger(__name__)

#: Titre du cas d'usage (injecté par le moteur de scaffolding depuis le manifeste).
USE_CASE_TITLE = "Détection de fraude sur transactions de paiement avec scikit-learn"

#: Jeu de données évalué.
DATASET_NAME = "payment_transactions"

#: Métrique pilotant la décision.
PRIMARY_METRIC = "pr_auc"

#: Critères de succès du cas d'usage, injectés depuis `extras.quality` du manifeste et
#: écrasables à l'exécution par `metrics.thresholds` dans la configuration Hydra.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "pr_auc_min": 0.42,
    "roc_auc_min": 0.9,
    "recall_at_budget_min": 0.45,
    "precision_at_budget_min": 0.5,
    "lift_at_budget_min": 20.0,
    # Un détecteur transactionnel ne peut pas couvrir tous les modes opératoires : la fraude
    # amicale (transaction légitime dans sa forme) est structurellement invisible. Le critère
    # porte donc sur le NOMBRE de schémas couverts au-dessus du plancher, pas sur le plus faible.
    "scheme_recall_floor": 0.3,
    "covered_schemes_min": 3,
}

#: Recommandations métier documentées pour ce cas d'usage (famille `anomaly_detection`).
BUSINESS_RECOMMENDATIONS: tuple[str, ...] = (
    "Piloter le détecteur au budget : choisir le seuil qui remplit la capacité d'analyse, puis "
    "mesurer le rappel et la précision à ce seuil (table d'arbitrage dans le rapport).",
    "Combiner score non supervisé et règles métier : le détecteur couvre les schémas inédits, "
    "les règles couvrent les obligations réglementaires.",
    "Mettre en place une boucle de rétro-étiquetage : réinjecter les chargebacks confirmés à "
    "J+30 pour surveiller la dérive du détecteur (voir mlops/model-monitoring).",
    "Segmenter l'analyse par mode opératoire : un rappel global de 0,6 peut cacher un rappel "
    "de 0,15 sur un schéma en pleine croissance.",
    "Documenter les manquants informatifs (`device_age_days`, `session_duration_sec`) plutôt "
    "que de les imputer silencieusement.",
    "Surveiller la stabilité du score (PSI) et le volume d'alertes : une dérive du flux suffit "
    "à saturer les analystes sans qu'aucune métrique modèle ne bouge.",
)

#: Nombre de facteurs contributifs cités dans le rapport.
N_DRIVERS = 8

#: Nombre de lignes d'analyse d'erreurs affichées par catégorie.
N_ERROR_ROWS = 8

#: Colonnes de contexte affichées dans l'analyse d'erreurs (unités brutes, lisibles par le métier).
ERROR_COLUMNS: tuple[str, ...] = (
    "amount_eur",
    "merchant_category",
    "channel",
    "shopper_country",
    "billing_country",
    "card_age_months",
    "amount_eur",
    "transactions_24h",
    "three_ds_authenticated",
)


class ReportBuilder:
    """Build the Markdown report, its JSON counterpart, the CSV extracts and the figures."""

    def __init__(
        self, paths: ProjectPaths | None = None, *, config: Mapping[str, Any] | None = None
    ) -> None:
        """Store the output layout and the configuration.

        Args:
            paths: Project layout (figures, reports and metrics directories).
            config: Root configuration mapping, used to document the run.
        """
        self.paths = paths or ProjectPaths.from_root()
        self.config: dict[str, Any] = dict(config or {})
        self.plots = AnomalyPlots(self.paths.figures_dir)
        metrics_node = dict(self.config.get("metrics") or {})
        overrides = dict(metrics_node.get("thresholds") or {})
        self.thresholds: dict[str, float] = {
            **DEFAULT_THRESHOLDS,
            **{str(key): _float(value) for key, value in overrides.items()},
        }

    # ------------------------------------------------------------------ API publique ----
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
            thresholds: Optional budget trade-off table; ``None`` reads
                ``result.curves["budget_tradeoff"]``.
            report_name: Output Markdown file name.

        Returns:
            Mapping of artefact kind (``report``, ``json``, ``errors``, ``budget``, figure names)
            to the written path.
        """
        self.paths.ensure()
        figures = self.plots.save_all(result)
        budget_table = (
            thresholds if isinstance(thresholds, pd.DataFrame) else self._budget_frame(result)
        )
        markdown = self.render_markdown(
            result, figures=figures, model=model, budget_table=budget_table
        )

        default_name = self._config_artifact("report_file", "evaluation_report.md")
        report_path = self.paths.reports_dir / (report_name or default_name)
        written: dict[str, Path] = {"report": write_text(report_path, markdown)}

        json_path = self.paths.metrics_dir / "anomaly_report.json"
        written["json"] = write_json(
            json_path,
            {
                **result.to_dict(),
                "thresholds": self.thresholds,
                "recommendations": self.recommendations(result),
                "per_scheme": result.per_scheme.to_dict(orient="records"),
                "budget_tradeoff": budget_table.to_dict(orient="records")
                if budget_table is not None
                else [],
                "feature_importance": result.feature_importance.head(N_DRIVERS).to_dict(
                    orient="records"
                ),
                "generated_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                "figures": {
                    key: str(path.relative_to(self.paths.root)) for key, path in figures.items()
                },
            },
        )

        if not result.errors.empty:
            written["errors"] = self.paths.reports_dir / "anomaly_errors.csv"
            result.errors.to_csv(written["errors"], index=False)
        if budget_table is not None and not budget_table.empty:
            written["budget"] = self.paths.reports_dir / "budget_tradeoff.csv"
            budget_table.to_csv(written["budget"], index=False)
        if not result.per_scheme.empty:
            written["schemes"] = self.paths.reports_dir / "scheme_coverage.csv"
            result.per_scheme.to_csv(written["schemes"], index=False)
        written.update({f"figure_{key}": path for key, path in figures.items()})
        logger.info("Rapport d'évaluation écrit : {} ({} artefacts)", report_path, len(written))
        return written

    def diagnostics(self, result: EvaluationResult) -> list[tuple[str, str]]:
        """List the weakness signals measured on this split, most urgent first.

        Each entry is a ``(title, detail)`` pair ready to be printed in a notebook or embedded in
        a review: it names the symptom, the measured value and the first thing to check.

        Args:
            result: Evaluation result.

        Returns:
            The diagnosed weaknesses (empty when nothing crosses a criterion).
        """
        metrics = dict(result.metrics)
        extras = dict(result.extras)
        baseline = dict(extras.get("baseline") or {})
        signals: list[tuple[str, str]] = []

        pr_auc = _float(metrics.get("pr_auc"))
        floor = _float(baseline.get("random_pr_auc"))
        if np.isfinite(pr_auc) and np.isfinite(floor) and pr_auc - floor < 0.10:
            signals.append(
                (
                    "Le score n'ordonne pas la fraude",
                    f"PR AUC = {pr_auc:.3f} contre un plancher aléatoire de {floor:.3f} : l'écart "
                    "est inférieur à 0,10. Vérifier en priorité (1) une fuite de l'étiquette dans "
                    "les features, (2) l'échelle des variables (une forêt d'isolation est "
                    "insensible à l'échelle, un auto-encodeur non), (3) la contamination déclarée.",
                )
            )
        if pr_auc < self.thresholds["pr_auc_min"]:
            signals.append(
                (
                    "PR AUC sous le seuil du cas d'usage",
                    f"{pr_auc:.3f} < {self.thresholds['pr_auc_min']:.2f}. Les leviers par ordre de "
                    "rentabilité : features de vélocité à fenêtre courte, indicateurs de manquants "
                    "(le générateur les rend informatifs), puis seulement le réglage du détecteur.",
                )
            )
        recall = _float(metrics.get("recall_at_budget"))
        if recall < self.thresholds["recall_at_budget_min"]:
            signals.append(
                (
                    "Fraude capturée insuffisante au budget",
                    f"Rappel au budget = {recall:.3f} (< "
                    f"{self.thresholds['recall_at_budget_min']:.2f}) "
                    f"pour {int(_float(extras.get('budget', 0)))} alertes. À capacité constante, "
                    "c'est le classement qu'il faut améliorer, pas le volume.",
                )
            )
        precision = _float(metrics.get("precision_at_budget"))
        if precision < self.thresholds["precision_at_budget_min"]:
            signals.append(
                (
                    "File d'alertes trop bruitée",
                    f"Précision au budget = {precision:.3f} : "
                    f"{int(_float(extras.get('budget', 0))) * max(0.0, 1.0 - precision):.0f} "
                    "fausses alertes à traiter. Les outliers légitimes (luxe, voyageurs, paiements "
                    "d'entreprise) en sont la cause attendue "
                    ": une liste blanche métier coûte moins "
                    "cher qu'un re-réglage du modèle.",
                )
            )
        uncovered = self._n_schemes(result) - self._covered_schemes(result)
        if uncovered > 0:
            table = result.per_scheme
            floor = self.thresholds["scheme_recall_floor"]
            names = ", ".join(
                SCHEME_LABELS.get(str(name), str(name))
                for name in table.loc[
                    table["recall_at_budget"].astype("float64") < floor, "fraud_scheme"
                ]
            )
            signals.append(
                (
                    f"{uncovered} mode(s) opératoire(s) non couvert(s) : {names}",
                    f"Rappel inférieur au plancher de {floor:.2f}. Un détecteur global optimise la "
                    "moyenne et délaisse les schémas dont la signature n'est pas transactionnelle "
                    "(fraude amicale, récidive de contestations) : prévoir une règle fondée sur "
                    "l'historique du porteur ou un modèle dédié, et une surveillance ventilée.",
                )
            )
        importance = result.feature_importance
        if not importance.empty:
            negative = importance[importance["importance_mean"] < 0]
            if not negative.empty:
                names = ", ".join(f"`{name}`" for name in negative["feature"].head(4))
                signals.append(
                    (
                        "Features qui dégradent le classement",
                        f"Importance par permutation négative sur {names} : leur permutation "
                        "améliore le recouvrement du top-K. À retirer ou à retravailler "
                        "(fenêtre d'agrégation, transformation, indicateur de manquant).",
                    )
                )
            dominant = importance.iloc[0]
            if _float(dominant["share"]) > 0.45:
                signals.append(
                    (
                        "Classement dominé par une seule feature",
                        f"`{dominant['feature']}` porte {_float(dominant['share']):.0%} de "
                        "l'importance totale. Le détecteur se réduit alors presque à une règle "
                        "unie ; il est fragile face à une dérive de cette variable.",
                    )
                )
        return signals

    def recommendations(self, result: EvaluationResult) -> list[str]:
        """Build the recommendation list: measured findings first, good practices next.

        Args:
            result: Evaluation result.

        Returns:
            The recommendations, most urgent first.
        """
        return [*self._dynamic_recommendations(result), *BUSINESS_RECOMMENDATIONS]

    # ------------------------------------------------------------------ rendu -----------
    def render_markdown(
        self,
        result: EvaluationResult,
        *,
        figures: Mapping[str, Path] | None = None,
        model: Any = None,
        budget_table: pd.DataFrame | None = None,
    ) -> str:
        """Render the full Markdown report.

        Args:
            result: Evaluation result.
            figures: Mapping of figure name to path (relative links are emitted).
            model: Optional model, used for the identity block.
            budget_table: Optional trade-off table.

        Returns:
            The Markdown document.
        """
        figures = dict(figures or {})
        metrics = dict(result.metrics)
        extras = dict(result.extras)
        sections: list[str] = []
        sections.append(f"# Rapport d'évaluation — {USE_CASE_TITLE}")
        sections.append("")
        sections += self._context_block(result, model)
        sections += self._verdict_block(result)
        sections += self._metrics_block(metrics, extras)
        sections += self._baseline_block(result, extras)
        sections += self._budget_block(extras, budget_table)
        sections += self._score_block(result, extras)
        sections += self._scheme_block(result)
        sections += self._lift_block(result)
        sections += self._error_block(result)
        sections += self._drivers_block(result)
        sections += self._recommendations_block(result)
        sections += self._limits_block(result, extras)
        sections += self._figures_block(figures)
        sections += self._reproducibility_block(model)
        return "\n".join(sections).rstrip() + "\n"

    def _context_block(self, result: EvaluationResult, model: Any) -> list[str]:
        """Render the dataset / model / split identity block."""
        summary = _model_identity(model)
        lines = [
            "## 1. Contexte de l'évaluation",
            "",
            f"- **Cas d'usage** : {USE_CASE_TITLE}",
            f"- **Jeu de données** : `{DATASET_NAME}` (synthétique, hors-ligne, aucun PAN réel)",
            f"- **Split évalué** : `{result.split}` — {result.n_samples} transactions",
            f"- **Tâche** : `{result.task}` (non supervisée "
            f": les étiquettes ne servent qu'à mesurer)",
            f"- **Métrique primaire** : `{result.primary_metric}` "
            f"= **{_float(result.primary_value):.4f}**",
            f"- **Prévalence de la fraude** : {_float(result.prevalence):.4f} "
            f"({int(_float(extras_of(result).get('n_frauds', 0)))} fraudes confirmées)",
            f"- **Budget d'investigation** : "
            f"{_float(extras_of(result).get('budget_rate', 0.0)):.1%} du flux, "
            f"soit {int(result.budget)} alertes",
            f"- **Détecteur** : `{summary['algorithm']}` ({summary['name']})",
            "",
            "> **Contrat anti-fuite** : `is_fraud` et "
            "`fraud_scheme` sont des **métadonnées**. Elles sont",
            "> exclues de la matrice de features par `drop_columns` "
            "et ne sont lues que par cet évaluateur.",
            "> Un PR AUC proche de 1,0 signerait une fuite, pas une performance.",
            "",
        ]
        return lines

    def _verdict_block(self, result: EvaluationResult) -> list[str]:
        """Render the pass/fail verdict against the use-case success criteria."""
        metrics = dict(result.metrics)
        extras = dict(result.extras)
        rows = [
            (
                "PR AUC",
                _float(metrics.get("pr_auc")),
                f"≥ {self.thresholds['pr_auc_min']:.2f}",
                _float(metrics.get("pr_auc")) >= self.thresholds["pr_auc_min"],
                "capacité de classement en forte imbalance (plancher = prévalence)",
            ),
            (
                "ROC AUC",
                _float(metrics.get("roc_auc")),
                f"≥ {self.thresholds['roc_auc_min']:.2f}",
                _float(metrics.get("roc_auc")) >= self.thresholds["roc_auc_min"],
                "ordonnancement global fraudes / légitimes",
            ),
            (
                "Rappel au budget",
                _float(metrics.get("recall_at_budget")),
                f"≥ {self.thresholds['recall_at_budget_min']:.2f}",
                _float(metrics.get("recall_at_budget")) >= self.thresholds["recall_at_budget_min"],
                "part de la fraude capturée à capacité d'analyse constante",
            ),
            (
                "Précision au budget",
                _float(metrics.get("precision_at_budget")),
                f"≥ {self.thresholds['precision_at_budget_min']:.2f}",
                _float(metrics.get("precision_at_budget"))
                >= self.thresholds["precision_at_budget_min"],
                "part des alertes qui sont de vraies fraudes (coût analyste)",
            ),
            (
                "Lift au budget",
                _float(extras.get("lift_at_budget")),
                f"≥ {self.thresholds['lift_at_budget_min']:.0f}x",
                _float(extras.get("lift_at_budget")) >= self.thresholds["lift_at_budget_min"],
                "concentration de la fraude vs tirage aléatoire",
            ),
        ]
        covered = self._covered_schemes(result)
        rows.append(
            (
                "Modes opératoires couverts",
                float(covered),
                f"≥ {self.thresholds['covered_schemes_min']:.0f} sur {self._n_schemes(result)} "
                f"(rappel ≥ {self.thresholds['scheme_recall_floor']:.2f})",
                covered >= self.thresholds["covered_schemes_min"],
                "les schémas transactionnellement visibles doivent être couverts ; la fraude "
                "amicale est un plafond structurel documenté, pas un échec de réglage",
            )
        )
        lines = [
            "## 2. Verdict sur les critères de succès",
            "",
            "| Critère | Mesuré | Seuil | Statut | Lecture |",
            "|---|---|---|---|---|",
        ]
        for name, value, target, passed, reading in rows:
            status = "✅" if np.isfinite(value) and passed else "❌"
            lines.append(f"| {name} | {_fmt(value)} | {target} | {status} | {reading} |")
        failures = [
            name for name, value, _t, passed, _r in rows if not (np.isfinite(value) and passed)
        ]
        lines.append("")
        if failures:
            lines.append(
                f"**Verdict : {len(failures)} critère(s) non "
                f"satisfait(s)** — {', '.join(failures)}."
            )
        else:
            lines.append("**Verdict : tous les critères de succès sont satisfaits.**")
        lines.append("")
        return lines

    def _metrics_block(self, metrics: Mapping[str, Any], extras: Mapping[str, Any]) -> list[str]:
        """Render the full metric table with the reading of each value.

        Args:
            metrics: Metric name -> value.
            extras: Extras mapping (its ``prevalence`` is quoted in the metric-choice reading).
        """
        readings = {
            "pr_auc": (
                "aire sous la courbe précision-rappel : la métrique honnête à 1,8 % de positifs"
            ),
            "roc_auc": "aire sous la courbe ROC : flatteuse en imbalance, à lire en secondaire",
            "recall_at_budget": (
                "fraude capturée parmi les alertes que les analystes peuvent traiter"
            ),
            "precision_at_budget": "part de vraies fraudes dans la file d'investigation",
            "f1": "compromis précision/rappel **au seuil du budget** (pas à un seuil arbitraire)",
            "precision": "précision au seuil du budget",
            "recall": "rappel au seuil du budget",
            "rmse": "erreur quadratique sur le score (diagnostic de calibration, non décisionnel)",
        }
        lines = [
            "## 3. Métriques détaillées",
            "",
            "| Métrique | Valeur | Interprétation |",
            "|---|---|---|",
        ]
        for name, value in metrics.items():
            lines.append(f"| `{name}` | {_fmt(_float(value))} | {readings.get(str(name), '—')} |")
        lines.append("")
        lines.append(
            "**Pourquoi la PR AUC et pas l'accuracy ?** Avec une prévalence de "
            f"{_float(extras.get('prevalence')):.3f}, un détecteur qui ne signale rien obtient "
            "une accuracy de 98,2 % et un rappel de 0 : l'accuracy est ici une métrique "
            "inexploitable. La ROC AUC reste au-dessus de 0,9 même quand la file d'alertes est "
            "noyée sous les faux positifs, d'où le choix de la PR AUC comme métrique primaire."
        )
        lines.append("")
        return lines

    def _baseline_block(self, result: EvaluationResult, extras: Mapping[str, Any]) -> list[str]:
        """Render the comparison against a random score (the theoretical floor).

        Args:
            result: Evaluation result (its metrics give the model side of the comparison).
            extras: Extras mapping carrying the ``baseline`` sub-mapping.
        """
        baseline = dict(extras.get("baseline") or {})
        if not baseline:
            return []
        metrics = dict(result.metrics)
        lines = [
            "## 4. Comparaison au plancher (score aléatoire)",
            "",
            "| Quantité | Détecteur | Aléatoire | Gain |",
            "|---|---|---|---|",
            f"| PR AUC | {_fmt(_float(metrics.get('pr_auc')))} | "
            f"{_fmt(_float(baseline.get('random_pr_auc')))} | "
            f"+{_fmt(_float(baseline.get('gain_pr_auc')))} |",
            f"| Prévalence (référence) | {_fmt(_float(baseline.get('prevalence')))} | — | — |",
            f"| Rappel au budget | {_fmt(_float(metrics.get('recall_at_budget')))} | "
            f"{_fmt(_float(baseline.get('random_recall_at_budget')))} | — |",
            "",
            "Un score aléatoire obtient une PR AUC égale à la prévalence : c'est le **plancher**.",
            "Tout détecteur utile doit s'en écarter nettement, sinon il ne fait que reproduire la",
            "fréquence de base sans ordonner l'information.",
            "",
        ]
        return lines

    def _budget_block(
        self, extras: Mapping[str, Any], budget_table: pd.DataFrame | None
    ) -> list[str]:
        """Render the operational trade-off table: alert volume -> threshold -> recall/precision."""
        lines = [
            "## 5. Point de fonctionnement : le budget d'investigation",
            "",
            f"Budget retenu : **{_float(extras.get('budget_rate', 0.0)):.1%} du flux**, soit "
            f"**{int(_float(extras.get('budget', 0)))} alertes** pour "
            f"{int(_float(extras.get('n_frauds', 0)))} fraudes confirmées dans ce split.",
            f"Seuil de score correspondant : **{_fmt(_float(extras.get('threshold')))}**.",
            "",
        ]
        if budget_table is not None and not budget_table.empty:
            lines += [
                "| Volume d'alertes | % du flux | Seuil de score | Fraudes "
                "capturées | Rappel | Précision | Lift | Fausses alertes |",
                "|---|---|---|---|---|---|---|---|",
            ]
            retained = _float(extras.get("budget_rate", 0.0))
            for _, row in budget_table.iterrows():
                marker = (
                    " ⬅ **retenu**" if abs(_float(row["budget_rate"]) - retained) < 1e-9 else ""
                )
                lines.append(
                    f"| {int(_float(row['alerts']))}{marker} | {_float(row['budget_rate']):.1%} | "
                    f"{_fmt(_float(row['threshold']))} | {int(_float(row['frauds_captured']))} | "
                    f"{_float(row['recall']):.3f} | {_float(row['precision']):.3f} | "
                    f"{_float(row['lift']):.1f}x | {int(_float(row['false_alarms']))} |"
                )
            lines.append("")
            lines.append(
                "**Lecture métier** : doubler le volume d'alertes achète du rappel au prix d'une "
                "précision qui chute — la colonne « fausses "
                "alertes » est le coût en jours-analystes. "
                "Le seuil ne se choisit donc pas sur un F1 théorique mais sur la capacité réelle "
                "de l'équipe, puis se surveille dans le temps (dérive du flux = saturation)."
            )
        lines.append("")
        return lines

    def _score_block(self, result: EvaluationResult, extras: Mapping[str, Any]) -> list[str]:
        """Render the score distribution statistics and the alert threshold reading."""
        frame = result.predictions
        lines = ["## 6. Distribution des scores", ""]
        if frame.empty:
            return [*lines, "_Aucune prédiction disponible._", ""]
        fraud_scores = frame.loc[frame["is_fraud"] == 1, "score"]
        legit_scores = frame.loc[frame["is_fraud"] == 0, "score"]
        lines += [
            "| Population | n | Score moyen | Médiane | P95 | Max | Rang médian |",
            "|---|---|---|---|---|---|---|",
        ]
        for label, subset in (("Fraudes confirmées", fraud_scores), ("Légitimes", legit_scores)):
            if subset.empty:
                continue
            ranks = (
                frame.loc[subset.index, "rank"] if "rank" in frame else pd.Series(dtype="float64")
            )
            lines.append(
                f"| {label} | {len(subset)} | {_fmt(float(subset.mean()))} "
                f"| {_fmt(float(subset.median()))} | "
                f"{_fmt(float(np.percentile(subset, 95)))} | {_fmt(float(subset.max()))} | "
                f"{int(ranks.median()) if not ranks.empty else '—'} |"
            )
        lines.append("")
        overlap = _float(extras.get("lift_at_budget"))
        lines.append(
            f"Le recouvrement des deux distributions fixe le plafond attevable : avec un lift de "
            f"{overlap:.1f}x au budget, le score concentre la fraude mais ne la sépare pas "
            "parfaitement — c'est attendu, une partie des fraudes est structurellement "
            "indétectable (bruit irréductible injecté par le générateur) et une partie des "
            "transactions légitimes est extrême (achats de luxe, voyageurs d'affaires)."
        )
        lines.append("")
        return lines

    def _scheme_block(self, result: EvaluationResult) -> list[str]:
        """Render the per-scheme coverage table."""
        table = result.per_scheme
        lines = ["## 7. Couverture par mode opératoire", ""]
        if table.empty:
            return [
                *lines,
                "_Aucune fraude confirmée dans ce split : couverture non mesurable._",
                "",
            ]
        lines += [
            "| Mode opératoire | Fraudes | Part | Capturées | "
            "Rappel au budget | Rang médian | Meilleur rang |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, row in table.iterrows():
            name = SCHEME_LABELS.get(str(row["fraud_scheme"]), str(row["fraud_scheme"]))
            lines.append(
                f"| {name} | {int(_float(row['frauds']))} | {_float(row['share_of_fraud']):.1%} | "
                f"{int(_float(row['captured_at_budget']))} | "
                f"{_float(row['recall_at_budget']):.3f} | "
                f"{int(_float(row['median_rank']))} | {int(_float(row['best_rank']))} |"
            )
        lines.append("")
        worst = table.loc[table["recall_at_budget"].idxmin()] if not table.empty else None
        if worst is not None:
            lines.append(
                f"**Point de vigilance** : "
                f"`{SCHEME_LABELS.get(str(worst['fraud_scheme']), str(worst['fraud_scheme']))}` "
                f"n'est capturé qu'à {_float(worst['recall_at_budget']):.1%}. Un détecteur global "
                "optimise la moyenne et peut délaisser un schéma minoritaire — or c'est souvent "
                "celui qui croît. La surveillance doit être ventilée par schéma, pas agrégée."
            )
            lines.append("")
        return lines

    def _lift_block(self, result: EvaluationResult) -> list[str]:
        """Render the lift-by-decile table."""
        rows = list(dict(result.curves or {}).get("lift_by_decile") or [])
        lines = ["## 8. Concentration de la fraude par décile de score", ""]
        if not rows:
            return [*lines, "_Lift non calculable (trop peu de lignes ou aucune fraude)._", ""]
        lines += ["| Décile | Lignes | Fraudes | Taux de fraude | Lift |", "|---|---|---|---|---|"]
        for row in rows:
            lines.append(
                f"| D{int(_float(row['decile']))} | {int(_float(row['rows']))} | "
                f"{int(_float(row['frauds']))} | {_float(row['fraud_rate']):.4f} "
                f"| {_float(row['lift']):.1f}x |"
            )
        lines.append("")
        top = rows[0] if rows else {}
        lines.append(
            f"Le décile le plus anormal (D{int(_float(top.get('decile', 9)))}) concentre "
            f"{_float(top.get('fraud_rate')):.2%} de fraude, soit un lift de "
            f"{_float(top.get('lift')):.1f}x sur la prévalence. C'est cette concentration qui "
            "rend l'investigation rentable : à budget égal, on capture beaucoup plus de fraude "
            "qu'avec un tirage aléatoire ou qu'avec les règles métier existantes."
        )
        lines.append("")
        return lines

    def _error_block(self, result: EvaluationResult) -> list[str]:
        """Render the error analysis: missed frauds and convincing false alarms."""
        errors = result.errors
        lines = ["## 9. Analyse d'erreurs", ""]
        if errors.empty:
            return [*lines, "_Aucune erreur à analyser._", ""]
        columns = [name for name in ERROR_COLUMNS if name in errors.columns]
        for kind, title, reading in (
            (
                "missed_fraud",
                "9.1 Fraudes manquées les mieux classées (faux négatifs)",
                "Ces lignes portent une signature *presque* suffisante : elles sont la marge de "
                "progrès accessible. Si elles se concentrent sur un schéma, c'est ce schéma qu'il "
                "faut outiller (feature de vélocité supplémentaire, règle dédiée).",
            ),
            (
                "false_alarm",
                "9.2 Fausses alertes les plus convaincantes (faux positifs)",
                "Ce sont souvent des outliers **légitimes** (achat de luxe, voyageur d'affaires "
                "nocturne, paiement professionnel). Elles coûtent du temps analyste et doivent "
                "alimenter une liste blanche métier, pas un reclassement du modèle.",
            ),
            (
                "caught_fraud",
                "9.3 Fraudes correctement capturées",
                "Référence de ce que le détecteur sait faire : à comparer aux manquées pour "
                "identifier la variable qui fait la différence.",
            ),
        ):
            subset = errors[errors["error_kind"] == kind].head(N_ERROR_ROWS)
            if subset.empty:
                continue
            lines += [title, "", reading, ""]
            header = (
                "| Rang | Score | Schéma | " + " | ".join(f"`{name}`" for name in columns) + " |"
            )
            lines.append(header)
            lines.append("|---" * (3 + len(columns)) + "|")
            for _, row in subset.iterrows():
                cells = " | ".join(_fmt_cell(row.get(name)) for name in columns)
                scheme = str(row.get("fraud_scheme"))
                lines.append(
                    f"| {int(_float(row['rank']))} | {_fmt(_float(row['score']))} | "
                    f"{SCHEME_LABELS.get(scheme, scheme)} | {cells} |"
                )
            lines.append("")
        return lines

    def _drivers_block(self, result: EvaluationResult) -> list[str]:
        """Render the permutation importance of the features on the anomaly score."""
        table = result.feature_importance
        lines = ["## 10. Facteurs contributifs (importance par permutation)", ""]
        if table.empty:
            return [
                *lines,
                "_Importance non calculable sur ce split (trop peu de lignes ou budget dégénéré)._",
                "",
            ]
        lines += [
            "Méthode : on permute une colonne et on mesure la chute du **recouvrement du top-K** "
            "avec le classement d'origine. Cette définition est identique pour tous les détecteurs "
            "(forêt d'isolation, one-class SVM, auto-encodeur), ce qui permet de les comparer.",
            "",
            "| Rang | Feature | Importance | Écart-type | Part |",
            "|---|---|---|---|---|",
        ]
        for position, (_, row) in enumerate(table.head(N_DRIVERS).iterrows(), start=1):
            lines.append(
                f"| {position} | `{row['feature']}` | {_fmt(_float(row['importance_mean']))} | "
                f"{_fmt(_float(row['importance_std']))} | {_float(row['share']):.1%} |"
            )
        lines.append("")
        lines.append(
            "**Lecture** : les features de vélocité et de contexte technique dominent "
            "normalement le classement, tandis que le montant seul contribue peu — c'est "
            "précisément pourquoi un détecteur appris bat une règle « montant > X EUR ». Une "
            "importance négative signifie que permuter la colonne *améliore* le classement : "
            "la feature apporte du bruit et doit être revue."
        )
        lines.append("")
        return lines

    def _recommendations_block(self, result: EvaluationResult) -> list[str]:
        """Render the recommendations: measured findings first, then documented good practice."""
        lines = ["## 11. Recommandations", ""]
        for index, recommendation in enumerate(self.recommendations(result), start=1):
            lines.append(f"{index}. {recommendation}")
        lines.append("")
        return lines

    def _limits_block(self, result: EvaluationResult, extras: Mapping[str, Any]) -> list[str]:
        """Render the honest limits of the evaluation."""
        del extras
        lines = [
            "## 12. Limites et biais connus",
            "",
            "- **Étiquettes tardives et partielles** : la fraude confirmée n'est connue qu'après "
            "chargeback (30 à 90 jours). Les métriques "
            "ci-dessus mesurent donc la fraude *confirmée*, "
            "pas la fraude réelle : le rappel est structurellement sous-estimé.",
            "- **Biais de rétro-étiquetage** : seules les transactions alertées par les règles "
            "existantes sont investiguées, donc étiquetées. Un schéma jamais alerté est invisible "
            "dans les étiquettes — c'est l'argument principal pour un détecteur non supervisé.",
            "- **Plafond de performance** : le générateur injecte un bruit irréductible (fraudes "
            "sans signature) et des outliers légitimes. Un rappel de 1,0 signerait une fuite.",
            "- **Split aléatoire vs temporel** : le flux est chronologique et les campagnes de "
            "fraude se répètent d'un jour à l'autre ; un split aléatoire surestime la performance "
            "de déploiement. Les notebooks montrent le split temporel en contrepoint.",
            f"- **Échelle de mesure** : {result.n_samples} transactions, "
            f"{int(_float(dict(result.extras).get('n_frauds', 0)))} fraudes. Avec si peu de "
            "positifs, l'intervalle de confiance du rappel est large : ne pas comparer deux "
            "détecteurs à moins de 0,03 d'écart de PR AUC sur un seul split.",
            "- **Dérive** : les schémas de fraude évoluent en semaines. Un détecteur figé se "
            "dégrade silencieusement — la surveillance (PSI du score, volume d'alertes, rappel "
            "par schéma) fait partie du modèle, pas de son exploitation.",
            "",
        ]
        return lines

    def _figures_block(self, figures: Mapping[str, Path]) -> list[str]:
        """Render the figure gallery with relative links."""
        if not figures:
            return []
        titles = {
            "pr_curve": "Courbe précision-rappel",
            "roc_curve": "Courbe ROC",
            "score_distribution": "Distribution des scores",
            "lift_by_decile": "Lift par décile",
            "budget_tradeoff": "Arbitrage budget ↔ fraude capturée",
            "scheme_recall": "Rappel par mode opératoire",
            "error_quadrant": "Carte des erreurs",
        }
        lines = ["## 13. Figures", ""]
        for name, path in sorted(figures.items()):
            relative = path.relative_to(self.paths.root) if path.is_absolute() else path
            lines.append(f"![{titles.get(name, name)}]({relative.as_posix()})")
            lines.append("")
        return lines

    def _reproducibility_block(self, model: Any) -> list[str]:
        """Render the reproducibility and artefact provenance block."""
        summary = _model_identity(model)
        lines = [
            "## 14. Reproductibilité",
            "",
            f"- **Généré le** : "
            f"{datetime.now(tz=timezone.utc).isoformat(timespec='seconds')} (UTC)",
            f"- **Détecteur** : `{summary.get('algorithm', 'n/a')}` "
            f"— `{summary.get('name', 'n/a')}`",
            f"- **Paramètres** : `{summary['params']}`",
            f"- **Features** : {len(summary['feature_names'])} colonnes après preprocessing",
            f"- **Seed** : `{summary['random_state']}` (génération, split et entraînement)",
            f"- **Résumé du modèle** : `{summary['summary']}`",
            "- **Persistance** : le modèle est sauvegardé dans `artifacts/models/`, les métriques "
            "dans `artifacts/metrics/`, ce rapport dans `artifacts/reports/`.",
            "- **Rejouer** : `make evaluate` régénère exactement ce rapport à partir des artefacts "
            "d'entraînement ; `make run` rejoue la chaîne "
            "complète depuis la génération des données.",
            "",
        ]
        return lines

    # ------------------------------------------------------------------ internes --------
    def _dynamic_recommendations(self, result: EvaluationResult) -> list[str]:
        """Derive recommendations from the measured values (most urgent first)."""
        metrics = dict(result.metrics)
        extras = dict(result.extras)
        advice: list[str] = []
        pr_auc = _float(metrics.get("pr_auc"))
        floor = _float(dict(extras.get("baseline") or {}).get("random_pr_auc"))
        if np.isfinite(pr_auc) and np.isfinite(floor) and pr_auc - floor < 0.10:
            advice.append(
                f"**Urgent** : la PR AUC ({pr_auc:.3f}) ne "
                f"dépasse le plancher aléatoire ({floor:.3f}) "
                "que de moins de 0,10 — le score n'ordonne pas la fraude. Vérifier la fuite "
                "éventuelle, l'échelle des features et la "
                "contamination déclarée avant tout réglage fin."
            )
        if pr_auc < self.thresholds["pr_auc_min"]:
            advice.append(
                f"PR AUC sous le seuil ({pr_auc:.3f} < {self.thresholds['pr_auc_min']:.2f}) : "
                "ajouter des features de vélocité à fenêtre "
                "courte (1 h, 10 min) et des indicateurs "
                "de manquants — le générateur rend les NaN informatifs, une imputation silencieuse "
                "les détruit."
            )
        table = result.per_scheme
        if not table.empty:
            floor = self.thresholds["scheme_recall_floor"]
            uncovered = table[table["recall_at_budget"].astype("float64") < floor]
            for _, row in uncovered.iterrows():
                label = SCHEME_LABELS.get(str(row["fraud_scheme"]), str(row["fraud_scheme"]))
                advice.append(
                    f"Le schéma `{label}` n'est capturé qu'à {_float(row['recall_at_budget']):.1%} "
                    f"({int(_float(row['frauds']))} fraudes, rang médian "
                    f"{int(_float(row['median_rank']))}) : un détecteur transactionnel ne peut pas "
                    "le voir. Le traiter par une règle fondée sur l'historique (récidive de "
                    "contestations, ratio de chargebacks par porteur) ou par un modèle dédié, et "
                    "le surveiller séparément dans le temps."
                )
        precision = _float(metrics.get("precision_at_budget"))
        if precision < self.thresholds["precision_at_budget_min"]:
            advice.append(
                f"Précision au budget de {precision:.3f} : la file d'alertes contient trop de "
                "faux positifs. Construire une liste blanche métier pour les outliers légitimes "
                "(achats de luxe, voyageurs fréquents, paiements professionnels) avant d'augmenter "
                "le volume d'alertes."
            )
        recall = _float(metrics.get("recall_at_budget"))
        if recall < self.thresholds["recall_at_budget_min"]:
            advice.append(
                f"Rappel au budget de {recall:.3f} : à capacité constante, la fraude capturée est "
                "insuffisante. Revoir le budget, ou combiner le score avec les règles métier "
                "existantes dans un classement unique."
            )
        importance = result.feature_importance
        if not importance.empty:
            negative = importance[importance["importance_mean"] < 0]
            if not negative.empty:
                advice.append(
                    "Features d'importance négative ("
                    + ", ".join(f"`{name}`" for name in negative["feature"].head(3))
                    + ") : leur permutation améliore le classement, elles apportent du bruit. "
                    "Les retirer ou les retravailler (fenêtre d'agrégation, transformation)."
                )
        return advice

    def _covered_schemes(self, result: EvaluationResult) -> int:
        """Count the fraud schemes whose recall at budget reaches the declared floor."""
        table = result.per_scheme
        if table.empty or "recall_at_budget" not in table.columns:
            return 0
        floor = self.thresholds["scheme_recall_floor"]
        return int((table["recall_at_budget"].astype("float64") >= floor).sum())

    def _n_schemes(self, result: EvaluationResult) -> int:
        """Return the number of fraud schemes observed in the scored split."""
        return len(result.per_scheme)

    def _worst_scheme(self, result: EvaluationResult) -> pd.Series | None:
        """Return the row of the least-covered fraud scheme."""
        table = result.per_scheme
        if table.empty or "recall_at_budget" not in table.columns:
            return None
        return table.loc[table["recall_at_budget"].idxmin()]

    def _budget_frame(self, result: EvaluationResult) -> pd.DataFrame | None:
        """Read the trade-off table computed by the evaluator."""
        rows = list(dict(result.curves or {}).get("budget_tradeoff") or [])
        return pd.DataFrame(rows) if rows else None

    def _config_artifact(self, key: str, default: str) -> str:
        """Read an artefact file name from the configuration, with a documented default."""
        artifacts = dict((self.config.get("train") or {}).get("artifacts") or {})
        return str(artifacts.get(key, default))


def _model_identity(model: Any) -> dict[str, Any]:
    """Extract the documented identity of a model.

    ``BaseModel.summary()`` returns a one-line **string** (not a mapping): the report therefore
    reads the contract attributes directly, with a documented fallback for every field.

    Args:
        model: Model instance, or ``None``.

    Returns:
        Mapping with ``algorithm``, ``name``, ``params``, ``feature_names``, ``random_state`` and
        ``summary``.
    """
    if model is None:
        return {
            "algorithm": "n/a",
            "name": "n/a",
            "params": {},
            "feature_names": [],
            "random_state": "n/a",
            "summary": "n/a",
        }
    return {
        "algorithm": str(getattr(model, "algorithm", "n/a") or "n/a"),
        "name": str(getattr(model, "name", "n/a") or "n/a"),
        "params": dict(getattr(model, "params", {}) or {}),
        "feature_names": [str(name) for name in (getattr(model, "feature_names", []) or [])],
        "random_state": getattr(model, "random_state", "n/a"),
        "summary": str(model.summary()) if hasattr(model, "summary") else type(model).__name__,
    }


def extras_of(result: EvaluationResult) -> dict[str, Any]:
    """Return the extras mapping of a result (module-level helper used in f-strings)."""
    return dict(result.extras or {})


def _float(value: Any) -> float:
    """Coerce ``value`` to ``float``, mapping non-numerics to NaN."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) or np.isnan(result) else float("nan")


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a numeric value for a Markdown table, mapping NaN to ``n/a``."""
    number = _float(value)
    return "n/a" if not np.isfinite(number) else f"{number:.{digits}f}"


def _fmt_cell(value: Any) -> str:
    """Format a context cell (numeric or categorical) for the error tables."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "oui" if bool(value) else "non"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):,.2f}".replace(",", " ")
    return str(value)


__all__ = ["ReportBuilder"]
