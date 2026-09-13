"""Évaluation d'un détecteur d'anomalies non supervisé (fraude sur paiements).

Un détecteur de fraude n'a **pas de vérité terrain exploitable à l'entraînement** : le
chargeback qui confirme la fraude arrive 30 à 90 jours après la transaction. Le modèle est donc
entraîné sans étiquette, et les étiquettes ne servent qu'à *mesurer* la performance — c'est
exactement le rôle de la colonne de métadonnées ``is_fraud`` produite par le générateur.

L'évaluation répond à trois questions, dans cet ordre :

1. **le score ordonne-t-il la fraude ?** — PR AUC (métrique primaire, car avec ~1,8 % de
   positifs la ROC AUC reste flatteuse même quand la file d'alertes est noyée), ROC AUC,
   comparaison explicite au **plancher** qu'est un score aléatoire (PR AUC = prévalence) ;
2. **que vaut-il au budget réel ?** — les analystes ne peuvent investiguer que ~2 % du flux.
   On calcule donc le rappel, la précision et le *lift* au budget, ainsi que la table
   d'arbitrage complète (volume d'alertes → seuil → précision → rappel → fraudes capturées),
   parce qu'un F1 à seuil arbitraire n'a aucun sens opérationnel ;
3. **où se trompe-t-il ?** — ventilation par mode opératoire (un rappel global de 0,6 peut
   cacher un rappel de 0,1 sur un schéma en croissance), faux positifs les plus convaincants
   (souvent des outliers *légitimes* injectés par le générateur), fraudes manquées les mieux
   classées, et facteurs contributifs par permutation du score.

Toutes les fonctions sont déterministes à seed fixée et n'écrivent rien : la persistance est
prise en charge par le pipeline d'évaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.models.base import BaseModel
from src.training.losses_metrics import MetricCalculator, MetricInputs
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Colonne de diagnostic produite par le générateur : métadonnée, **jamais** une feature.
LABEL_COLUMN = "is_fraud"

#: Colonne décrivant le mode opératoire de la fraude confirmée (diagnostic et explication).
SCHEME_COLUMN = "fraud_scheme"

#: Part du flux transmise aux analystes : la contrainte réelle n'est pas un seuil mais un volume.
DEFAULT_BUDGET_RATE = 0.02

#: Volumes d'alertes explorés par la table d'arbitrage (parts du flux).
BUDGET_GRID: tuple[float, ...] = (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)

#: Nombre de lignes conservées pour l'analyse d'erreurs (fraudes manquées, faux positifs).
TOP_K_ERRORS = 25

#: Nombre maximal de points conservés pour les courbes (JSON des rapports et figures).
MAX_CURVE_POINTS = 240

#: Nombre de répétitions de l'importance par permutation (compromis stabilité / coût).
PERMUTATION_REPEATS = 6


@dataclass(slots=True)
class EvaluationResult:
    """Structured outcome of an anomaly-detection evaluation.

    Attributes:
        task: Learning task (``anomaly``).
        split: Split that was scored (``test``, ``val``, ...).
        primary_metric: Name of the metric driving the decision (typically ``pr_auc``).
        metrics: Metric name -> value.
        predictions: Row-level frame (``score``, ``rank``, ``percentile``, ``flagged``,
            ``is_fraud``, ``fraud_scheme``, plus the context columns).
        labels: Names of the diagnostic columns carried in ``predictions``.
        per_scheme: One row per fraud scheme (count, recall at budget, mean score, mean rank).
        errors: Worst mistakes: missed frauds ranked highest and false alarms ranked highest.
        curves: Down-sampled points of the PR curve, the ROC curve, the score distributions, the
            lift by decile and the budget trade-off table.
        feature_importance: Permutation importance of each feature on the anomaly score.
        extras: Free-form payload (budget, prevalence, threshold, baseline comparison,
            calibration of the score into a probability-like quantity).
        n_samples: Number of scored rows.
    """

    task: str
    split: str = "test"
    primary_metric: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    labels: list[str] = field(default_factory=list)
    per_scheme: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    curves: dict[str, Any] = field(default_factory=dict)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    extras: dict[str, Any] = field(default_factory=dict)
    n_samples: int = 0

    @property
    def primary_value(self) -> float:
        """Value of the metric driving the decision."""
        return _to_float(self.metrics.get(self.primary_metric, float("nan")))

    @property
    def prevalence(self) -> float:
        """Share of confirmed frauds in the scored population."""
        return _to_float(self.extras.get("prevalence", float("nan")))

    @property
    def budget(self) -> int:
        """Number of transactions that can be investigated (the operational constraint)."""
        return int(_to_float(self.extras.get("budget", 0.0)))

    @property
    def threshold(self) -> float:
        """Score above which a transaction enters the investigation queue."""
        return _to_float(self.extras.get("threshold", float("nan")))

    @property
    def lift_at_budget(self) -> float:
        """Precision at budget divided by the prevalence (1.0 = random ranking)."""
        return _to_float(self.extras.get("lift_at_budget", float("nan")))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the evaluation."""
        return {
            "task": self.task,
            "split": self.split,
            "primary_metric": self.primary_metric,
            "primary_value": self.primary_value,
            "metrics": {key: _to_float(value) for key, value in self.metrics.items()},
            "n_samples": int(self.n_samples),
            "prevalence": self.prevalence,
            "budget": self.budget,
            "threshold": self.threshold,
            "lift_at_budget": self.lift_at_budget,
            "baseline": dict(self.extras.get("baseline", {})),
        }


class Evaluator:
    """Score an unsupervised anomaly detector against confirmed fraud labels.

    The labels are read from the *context* frame (metadata column), never from the feature
    matrix, which keeps the no-leakage contract auditable.

    Example:
        >>> evaluator = Evaluator(model, metrics_config={"task": "anomaly", "primary": "pr_auc"})
        >>> result = evaluator.evaluate(X_test, context=test_frame)
        >>> result.primary_metric
        'pr_auc'
    """

    def __init__(
        self,
        model: BaseModel,
        *,
        metrics_config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        task: str | None = None,
        top_k_errors: int = TOP_K_ERRORS,
        budget_rate: float | None = None,
        context_columns: Sequence[str] | None = None,
    ) -> None:
        """Inject the model and the metric policy.

        Args:
            model: Fitted model implementing :class:`BaseModel`; ``predict`` must return a
                continuous anomaly score (higher = more anomalous).
            metrics_config: ``metrics`` configuration node (primary / secondary / task / extra).
            paths: Project layout, used to persist metrics.
            task: Task override (defaults to ``model.task``).
            top_k_errors: Number of rows kept per error category.
            budget_rate: Share of the flow that can be investigated. ``None`` reads
                ``metrics.extra.budget_rate``, then falls back to :data:`DEFAULT_BUDGET_RATE`.
            context_columns: Columns copied from the enriched context into the diagnostic frame.
        """
        self.model = model
        self.paths = paths or ProjectPaths.from_root()
        self.config: dict[str, Any] = dict(metrics_config or {})
        self.task = str(task or self.config.get("task") or model.task)
        self.top_k_errors = int(top_k_errors)
        self.context_columns: tuple[str, ...] = tuple(str(name) for name in (context_columns or ()))
        extra_node = dict(self.config.get("extra") or {})
        rate = budget_rate if budget_rate is not None else extra_node.get("budget_rate")
        self.budget_rate = float(rate) if rate is not None else DEFAULT_BUDGET_RATE
        self.primary_metric = str(self.config.get("primary", "pr_auc"))
        self.metric_names = [
            self.primary_metric,
            *[str(name) for name in self.config.get("secondary", []) or []],
        ]

    @classmethod
    def from_config(
        cls, model: BaseModel, config: Mapping[str, Any], paths: ProjectPaths | None = None
    ) -> Evaluator:
        """Build an evaluator from a root configuration mapping.

        Args:
            model: Fitted model.
            config: Root configuration.
            paths: Optional project layout.

        Returns:
            The configured evaluator.
        """
        metrics_node = dict(config.get("metrics") or {})
        data_node = dict(config.get("data") or {})
        context_columns = tuple(
            str(data_node[key])
            for key in ("id_column", "group_column", "time_column")
            if data_node.get(key)
        )
        return cls(
            model,
            metrics_config=metrics_node,
            paths=paths,
            task=metrics_node.get("task"),
            context_columns=context_columns or None,
        )

    # ------------------------------------------------------------------ évaluation ------
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series | Sequence[Any] | None = None,
        *,
        split: str = "test",
        context: pd.DataFrame | None = None,
    ) -> EvaluationResult:
        """Score the detector on a feature matrix.

        Args:
            X: Preprocessed features (the space in which the detector was fitted).
            y: Ignored — the detector is trained without labels. Accepted for signature symmetry
                with the supervised evaluators. If a series *is* passed it is used as the label,
                which makes the evaluator reusable on a held-out labelled set.
            split: Split name, used in logs and artefacts.
            context: Raw/enriched rows aligned with ``X``; carries ``is_fraud``, ``fraud_scheme``
                and the identifier/timestamp columns.

        Returns:
            The :class:`EvaluationResult`.

        Raises:
            RuntimeError: When the model is not fitted.
            ValueError: When no fraud label can be located (neither ``y`` nor ``context``).
        """
        self.model.check_is_fitted()
        scores = self._scores(X)
        labels = self._labels(y, context, n=len(scores))
        schemes = self._schemes(context, n=len(scores))

        budget = round(self.budget_rate * len(scores))
        budget = max(1, min(budget, len(scores)))
        flagged = np.zeros(len(scores), dtype=int)
        flagged[np.argsort(-scores, kind="stable")[:budget]] = 1
        prevalence = float(np.mean(labels)) if len(labels) else float("nan")

        frame = self._row_frame(scores, flagged, labels, schemes, context, X)
        metrics = self._compute_metrics(scores, labels, flagged, budget)
        baseline = self._baseline_comparison(scores, labels, budget)
        per_scheme = self._scheme_breakdown(frame, budget)
        errors = self._error_analysis(frame)
        curves = {
            "precision_recall": self._pr_curve_points(scores, labels),
            "roc": self._roc_curve_points(scores, labels),
            "score_distribution": self._score_distribution(scores, labels),
            "lift_by_decile": self._lift_by_decile(scores, labels),
            "budget_tradeoff": self.budget_table(scores, labels).to_dict(orient="records"),
        }
        importance = self._feature_importance(X, scores)

        extras: dict[str, Any] = {
            "budget": budget,
            "budget_rate": self.budget_rate,
            "prevalence": prevalence,
            "n_frauds": int(np.sum(labels)),
            "threshold": _to_float(np.min(scores[flagged == 1])) if budget else float("nan"),
            "score_min": _to_float(np.min(scores)),
            "score_max": _to_float(np.max(scores)),
            "score_mean": _to_float(np.mean(scores)),
            "lift_at_budget": _to_float(
                metrics.get("precision_at_budget", float("nan")) / prevalence
            )
            if prevalence
            else float("nan"),
            "baseline": baseline,
            "n_flagged": int(np.sum(flagged)),
        }
        result = EvaluationResult(
            task=self.task,
            split=split,
            primary_metric=self.primary_metric,
            metrics=metrics,
            predictions=frame,
            labels=[LABEL_COLUMN, SCHEME_COLUMN],
            per_scheme=per_scheme,
            errors=errors,
            curves=curves,
            feature_importance=importance,
            extras=extras,
            n_samples=len(scores),
        )
        logger.info(
            "Évaluation {} | {}={:.4f} | rappel@budget={:.3f} "
            "| précision@budget={:.3f} | lift={:.1f}x",
            split,
            self.primary_metric,
            result.primary_value,
            _to_float(metrics.get("recall_at_budget", float("nan"))),
            _to_float(metrics.get("precision_at_budget", float("nan"))),
            _to_float(extras["lift_at_budget"]),
        )
        return result

    def budget_table(self, scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        """Build the operational trade-off table: alert volume -> threshold -> recall/precision.

        This is the table the fraud lead actually reads: it converts a model score into a
        staffing decision.

        Args:
            scores: Anomaly scores (higher = more anomalous).
            labels: Confirmed fraud labels (0/1).

        Returns:
            One row per explored budget share.
        """
        n = len(scores)
        order = np.argsort(-scores, kind="stable")
        sorted_labels = np.asarray(labels)[order]
        sorted_scores = np.asarray(scores, dtype="float64")[order]
        cumulative_frauds = np.cumsum(sorted_labels)
        total_frauds = max(int(np.sum(labels)), 1)
        prevalence = float(np.mean(labels)) if n else float("nan")
        rows: list[dict[str, Any]] = []
        for rate in BUDGET_GRID:
            volume = max(1, min(round(rate * n), n))
            captured = int(cumulative_frauds[volume - 1])
            precision = captured / volume
            rows.append(
                {
                    "budget_rate": rate,
                    "alerts": volume,
                    "threshold": float(sorted_scores[volume - 1]),
                    "frauds_captured": captured,
                    "recall": captured / total_frauds,
                    "precision": precision,
                    "lift": (precision / prevalence) if prevalence else float("nan"),
                    "false_alarms": volume - captured,
                }
            )
        return pd.DataFrame(rows).round(4)

    def compare_to_baseline(self, X: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
        """Compare the detector against a random score (the theoretical floor).

        Args:
            X: Feature matrix (only its length is used).
            labels: Confirmed fraud labels.

        Returns:
            Mapping with ``pr_auc`` / ``roc_auc`` for both the model and the random baseline.
        """
        scores = self._scores(X)
        rng = np.random.default_rng(0)
        random_scores = rng.random(len(scores))
        return {
            "model_pr_auc": float(average_precision_score(labels, scores)),
            "random_pr_auc": float(average_precision_score(labels, random_scores)),
            "model_roc_auc": float(roc_auc_score(labels, scores)),
            "random_roc_auc": float(roc_auc_score(labels, random_scores)),
        }

    def save_metrics(self, result: EvaluationResult, path: Any = None) -> Any:
        """Persist the metric summary as JSON.

        Args:
            result: Evaluation result.
            path: Destination file; ``None`` uses ``artifacts/metrics/evaluation_metrics.json``.

        Returns:
            The written path.
        """
        from src.utils.io import write_json

        destination = path or (self.paths.metrics_dir / "evaluation_metrics.json")
        payload = {
            **result.to_dict(),
            "extras": _sanitise(result.extras),
            "per_scheme": result.per_scheme.to_dict(orient="records"),
        }
        return write_json(destination, payload)

    # ------------------------------------------------------------------ internes --------
    def _scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return the continuous anomaly score produced by the model."""
        raw = np.asarray(self.model.predict(X), dtype="float64").ravel()
        if np.all(np.isin(raw, (-1.0, 1.0))):
            # Un détecteur qui ne rend que +/-1 ne peut pas être classé : le rappel au budget
            # devient indéterminé. On avertit explicitement plutôt que de produire un chiffre faux.
            logger.warning(
                "Le modèle rend une étiquette binaire et non un score continu : "
                "le classement au budget sera arbitraire. Utilisez `score_samples`/erreur de "
                "reconstruction plutôt que `predict`."
            )
        return raw

    def _labels(
        self, y: pd.Series | Sequence[Any] | None, context: pd.DataFrame | None, *, n: int
    ) -> np.ndarray:
        """Locate the fraud labels: explicit ``y`` first, otherwise the context metadata."""
        if y is not None and np.asarray(y).size:
            return np.asarray(y).astype(int).ravel()
        if context is not None and LABEL_COLUMN in context.columns:
            labels = context[LABEL_COLUMN].astype("float64").to_numpy()
            if len(labels) != n:
                msg = f"Le contexte fournit {len(labels)} étiquettes pour {n} lignes scorées"
                raise ValueError(msg)
            return np.nan_to_num(labels, nan=0.0).astype(int)
        msg = (
            f"Aucune étiquette '{LABEL_COLUMN}' disponible : passez `y` ou un `context` contenant "
            "cette colonne de métadonnées pour évaluer un détecteur d'anomalies."
        )
        raise ValueError(msg)

    def _schemes(self, context: pd.DataFrame | None, *, n: int) -> np.ndarray:
        """Read the fraud scheme metadata (``"legitimate"`` when absent)."""
        if context is not None and SCHEME_COLUMN in context.columns:
            values = context[SCHEME_COLUMN].astype("string").to_numpy()
            return np.where(pd.isna(values), "legitimate", values.astype(object))
        return np.full(n, "legitimate", dtype=object)

    def _row_frame(
        self,
        scores: np.ndarray,
        flagged: np.ndarray,
        labels: np.ndarray,
        schemes: np.ndarray,
        context: pd.DataFrame | None,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assemble the row-level diagnostic frame."""
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(len(scores), dtype="int64")
        ranks[order] = np.arange(1, len(scores) + 1)
        frame = pd.DataFrame(
            {
                "score": scores,
                "rank": ranks,
                "percentile": 100.0 * (1.0 - (ranks - 1) / max(len(scores), 1)),
                "flagged": flagged,
                LABEL_COLUMN: labels,
                SCHEME_COLUMN: schemes,
            }
        )
        frame["outcome"] = np.select(
            [
                (frame["flagged"] == 1) & (frame[LABEL_COLUMN] == 1),
                (frame["flagged"] == 1) & (frame[LABEL_COLUMN] == 0),
                (frame["flagged"] == 0) & (frame[LABEL_COLUMN] == 1),
            ],
            ["true_positive", "false_positive", "false_negative"],
            default="true_negative",
        )
        if context is not None:
            for column in self.context_columns:
                if column in context.columns and column not in frame:
                    frame[column] = np.asarray(context[column].to_numpy())[: len(frame)]
        frame.index = X.index if hasattr(X, "index") else frame.index
        return frame

    def _compute_metrics(
        self, scores: np.ndarray, labels: np.ndarray, flagged: np.ndarray, budget: int
    ) -> dict[str, float]:
        """Compute every configured metric through the shared metric calculator."""
        calculator = MetricCalculator(
            task=self.task, metrics=self.metric_names, extra={"budget": budget}
        )
        inputs = MetricInputs(y_true=labels, y_pred=flagged, y_proba=scores)
        values = calculator.evaluate(inputs)
        # Garde-fou : sans aucun positif, les AUC sont indéfinies et doivent le rester visibles.
        if int(np.sum(labels)) == 0:
            logger.warning("Aucune fraude confirmée dans ce split : PR AUC et ROC AUC indéfinies.")
        return {str(key): _to_float(value) for key, value in values.items()}

    def _baseline_comparison(
        self, scores: np.ndarray, labels: np.ndarray, budget: int
    ) -> dict[str, float]:
        """Measure the gap between the detector and a random score (floor = prevalence)."""
        if int(np.sum(labels)) == 0:
            return {"random_pr_auc": float("nan"), "gain_pr_auc": float("nan")}
        rng = np.random.default_rng(0)
        random_scores = rng.random(len(scores))
        random_pr = float(average_precision_score(labels, random_scores))
        model_pr = float(average_precision_score(labels, scores))
        order = np.argsort(-random_scores, kind="stable")[:budget]
        random_recall = float(np.sum(np.asarray(labels)[order]) / max(int(np.sum(labels)), 1))
        return {
            "random_pr_auc": random_pr,
            "model_pr_auc": model_pr,
            "gain_pr_auc": model_pr - random_pr,
            "random_recall_at_budget": random_recall,
            "prevalence": float(np.mean(labels)),
        }

    def _scheme_breakdown(self, frame: pd.DataFrame, budget: int) -> pd.DataFrame:
        """Break the performance down by fraud scheme (the key operational diagnostic)."""
        if frame.empty:
            return pd.DataFrame()
        frauds = frame[frame[LABEL_COLUMN] == 1]
        if frauds.empty:
            return pd.DataFrame()
        del budget
        rows: list[dict[str, Any]] = []
        for scheme, group in frauds.groupby(SCHEME_COLUMN, observed=True):
            captured = int((group["flagged"] == 1).sum())
            rows.append(
                {
                    "fraud_scheme": str(scheme),
                    "frauds": len(group),
                    "captured_at_budget": captured,
                    "recall_at_budget": captured / max(len(group), 1),
                    "mean_score": float(group["score"].mean()),
                    "median_rank": float(group["rank"].median()),
                    "best_rank": int(group["rank"].min()),
                    "worst_rank": int(group["rank"].max()),
                }
            )
        table = pd.DataFrame(rows).sort_values("frauds", ascending=False).reset_index(drop=True)
        table["share_of_fraud"] = (table["frauds"] / max(len(frauds), 1)).round(4)
        return table.round(4)

    def _error_analysis(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Extract the most instructive mistakes: near-miss frauds and convincing false alarms."""
        if frame.empty:
            return pd.DataFrame()
        missed = frame[frame["outcome"] == "false_negative"].nsmallest(self.top_k_errors, "rank")
        alarms = frame[frame["outcome"] == "false_positive"].nsmallest(self.top_k_errors, "rank")
        caught = frame[frame["outcome"] == "true_positive"].nsmallest(self.top_k_errors, "rank")
        blocks: list[pd.DataFrame] = []
        error_blocks = ((missed, "missed_fraud"), (alarms, "false_alarm"), (caught, "caught_fraud"))
        for block, kind in error_blocks:
            if block.empty:
                continue
            copy = block.copy()
            copy["error_kind"] = kind
            blocks.append(copy)
        if not blocks:
            return pd.DataFrame()
        return pd.concat(blocks, ignore_index=False).reset_index(names="row_index")

    def _pr_curve_points(self, scores: np.ndarray, labels: np.ndarray) -> list[dict[str, float]]:
        """Down-sample the precision-recall curve for reports and figures."""
        if int(np.sum(labels)) == 0:
            return []
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        return _curve_points(
            {"recall": recall[:-1], "precision": precision[:-1], "threshold": thresholds}
        )

    def _roc_curve_points(self, scores: np.ndarray, labels: np.ndarray) -> list[dict[str, float]]:
        """Down-sample the ROC curve for reports and figures."""
        if len(np.unique(labels)) < 2:
            return []
        fpr, tpr, thresholds = roc_curve(labels, scores)
        return _curve_points({"fpr": fpr, "tpr": tpr, "threshold": thresholds})

    def _score_distribution(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> dict[str, list[dict[str, float]]]:
        """Histogram the scores separately for frauds and legitimate transactions."""
        return {
            "fraud": _histogram_points(scores[labels == 1]) if np.any(labels == 1) else [],
            "legitimate": _histogram_points(scores[labels == 0]) if np.any(labels == 0) else [],
        }

    def _lift_by_decile(self, scores: np.ndarray, labels: np.ndarray) -> list[dict[str, float]]:
        """Compute the fraud rate and lift per score decile (most anomalous first)."""
        if len(scores) < 10 or int(np.sum(labels)) == 0:
            return []
        deciles = pd.qcut(pd.Series(scores), q=10, labels=False, duplicates="drop")
        frame = pd.DataFrame({"decile": np.asarray(deciles), "label": labels})
        grouped = frame.groupby("decile", observed=True)["label"].agg(["mean", "sum", "size"])
        prevalence = float(np.mean(labels))
        rows: list[dict[str, float]] = []
        # Le décile 9 (scores les plus élevés) est présenté en premier : c'est la file d'alertes.
        for decile in sorted(grouped.index, reverse=True):
            rate = float(grouped.loc[decile, "mean"])
            rows.append(
                {
                    "decile": int(decile),
                    "fraud_rate": rate,
                    "frauds": int(grouped.loc[decile, "sum"]),
                    "rows": int(grouped.loc[decile, "size"]),
                    "lift": (rate / prevalence) if prevalence else float("nan"),
                }
            )
        return rows

    def _feature_importance(self, X: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
        """Measure which features drive the anomaly score, by permutation.

        Permutation is used instead of model-native importances so that the analysis works for
        every detector (isolation forest, one-class SVM, autoencoder) with the same definition:
        how much does shuffling a column change the *ranking* of the most anomalous rows?

        Args:
            X: Feature matrix used for scoring.
            scores: Anomaly scores, used to build the ranking target.

        Returns:
            One row per feature: mean importance, standard deviation and share of the total.
        """
        matrix = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        if matrix.shape[1] == 0 or matrix.shape[0] < 30:
            return pd.DataFrame()
        # Cible de permutation : le rang induit par le score (0/1 sur le top 2 %).
        budget = max(1, min(round(self.budget_rate * len(scores)), len(scores)))
        target = np.zeros(len(scores), dtype=int)
        target[np.argsort(-scores, kind="stable")[:budget]] = 1
        if int(target.sum()) == 0 or int(target.sum()) == len(target):
            return pd.DataFrame()
        try:
            outcome = permutation_importance(
                _ScoreModel(self.model),
                matrix,
                target,
                n_repeats=PERMUTATION_REPEATS,
                random_state=0,
                scoring=_top_k_overlap,
                n_jobs=1,
            )
        except Exception as error:
            logger.warning("Importance par permutation indisponible : {}", error)
            return pd.DataFrame()
        names = list(matrix.columns)
        table = pd.DataFrame(
            {
                "feature": names,
                "importance_mean": np.asarray(outcome.importances_mean, dtype="float64"),
                "importance_std": np.asarray(outcome.importances_std, dtype="float64"),
            }
        )
        total = float(np.abs(table["importance_mean"]).sum())
        table["share"] = (np.abs(table["importance_mean"]) / total) if total else 0.0
        return table.sort_values("importance_mean", ascending=False).reset_index(drop=True).round(5)


class _ScoreModel:
    """Thin adapter exposing ``predict`` as the score used by :func:`permutation_importance`."""

    def __init__(self, model: BaseModel) -> None:
        """Wrap the detector.

        Args:
            model: Fitted anomaly detector.
        """
        self.model = model

    def fit(self, X: pd.DataFrame, y: Any = None) -> _ScoreModel:
        """No-op required by :func:`sklearn.inspection.permutation_importance`.

        The detector is already fitted: permuting features must not retrain anything, otherwise
        the importance would measure the *learner*'s sensitivity instead of the score's.

        Args:
            X: Feature matrix (ignored).
            y: Target (ignored).

        Returns:
            ``self``.
        """
        del X, y
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the anomaly scores.

        Args:
            X: Feature matrix.

        Returns:
            A 1-D array of scores.
        """
        return np.asarray(self.model.predict(X), dtype="float64").ravel()


def _top_k_overlap(estimator: _ScoreModel, X: pd.DataFrame, y: np.ndarray) -> float:
    """Scorer for permutation importance: overlap between the flagged top-k and the target.

    Args:
        estimator: Score model adapter.
        X: Permuted feature matrix.
        y: Target flag vector (top-k of the unpermuted score).

    Returns:
        The share of the target rows still flagged after permutation.
    """
    scores = estimator.predict(X)
    budget = int(np.sum(y))
    flagged = np.zeros(len(scores), dtype=int)
    flagged[np.argsort(-scores, kind="stable")[: max(budget, 1)]] = 1
    return float(np.sum(flagged & np.asarray(y).astype(int)) / max(budget, 1))


# --------------------------------------------------------------------------- utilitaires ---
def _to_float(value: Any) -> float:
    """Coerce a value to float, mapping non-numerics to NaN."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _as_matrix(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return a float matrix from a frame or an array."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype="float64")
    return np.asarray(X, dtype="float64")


def _curve_points(columns: Mapping[str, np.ndarray]) -> list[dict[str, float]]:
    """Down-sample a curve to at most :data:`MAX_CURVE_POINTS` rows, endpoints preserved."""
    length = min(len(array) for array in columns.values())
    if length == 0:
        return []
    if length <= MAX_CURVE_POINTS:
        indices = np.arange(length)
    else:
        indices = np.unique(
            np.concatenate(
                [
                    np.linspace(0, length - 1, MAX_CURVE_POINTS).astype(int),
                    np.array([0, length - 1]),
                ]
            )
        )
    return [
        {str(key): float(np.asarray(values)[index]) for key, values in columns.items()}
        for index in indices
    ]


def _histogram_points(values: np.ndarray, bins: int = 40) -> list[dict[str, float]]:
    """Histogram a score vector into ``{bin_center, density}`` points."""
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if array.size == 0:
        return []
    counts, edges = np.histogram(array, bins=min(bins, max(4, array.size // 5)))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = float(counts.sum()) or 1.0
    return [
        {"bin_center": float(center), "density": float(count / total)}
        for center, count in zip(centers, counts, strict=True)
    ]


def _sanitise(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Make a mapping JSON-safe (numpy scalars -> Python floats/ints)."""
    clean: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            clean[str(key)] = _sanitise(value)
        elif isinstance(value, (np.floating, float)):
            clean[str(key)] = float(value)
        elif isinstance(value, (np.integer, int)):
            clean[str(key)] = int(value)
        elif isinstance(value, (np.bool_, bool)):
            clean[str(key)] = bool(value)
        elif isinstance(value, np.ndarray):
            clean[str(key)] = value.tolist()
        else:
            clean[str(key)] = value
    return clean
