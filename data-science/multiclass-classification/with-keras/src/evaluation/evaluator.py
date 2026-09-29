"""Evaluation of a multiclass diagnosis model.

A multiclass evaluation that stops at a global accuracy answers none of the questions a
maintenance manager asks. This evaluator answers five of them, in this order:

1. **Is the model better than what we already have?** It is scored against three references on
   the same test rows: the majority class (floor), the current routing rule (``alarm_code`` →
   class, the business reference) and the oracle ceiling published by the data generator.
2. **For which classes?** Per-class precision / recall / F1 / one-vs-rest AUC, a confusion matrix
   ordered by the business class order, and the confused pairs ranked by volume.
3. **What does it cost?** The ``argmax`` decision is compared with the minimum-cost decision
   under the cost matrix of :mod:`src.evaluation.decision`, and with the current rule.
4. **When should a human decide?** The coverage / accuracy / cost trade-off of sending
   low-confidence alarms to an expert.
5. **Can the probabilities be trusted?** Top-label reliability and expected calibration error,
   because the minimum-cost decision multiplies probabilities by euros.

Everything ends in a **verdict**: each contractual objective with its measured value, its
threshold and its status. Nothing is printed; the structured :class:`EvaluationResult` feeds
the report, the metrics JSON and the notebooks alike.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.evaluation.decision import (
    DiagnosisSettings,
    minimum_cost_decision,
    realised_costs,
)
from src.models.base import BaseModel
from src.training.losses_metrics import MetricCalculator, MetricInputs
from src.utils.io import write_json
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: File written by the data generator, holding the oracle ceiling and the references.
GENERATION_METADATA = "generation_metadata.json"

#: Confidence thresholds explored by the abstention analysis.
ABSTENTION_GRID: tuple[float, ...] = tuple(
    float(value) for value in np.round(np.arange(0.30, 0.951, 0.05), 2)
)

#: Number of equal-width bins of the reliability diagram.
CALIBRATION_BINS = 10

#: Rows scored to measure the inference latency (the business contract is per 1 000 alarms).
LATENCY_BATCH = 1000


@dataclass(slots=True)
class EvaluationResult:
    """Structured outcome of a multiclass evaluation.

    The first fields mirror the other tasks of the repository (same names), so the shared
    pipelines, reports and notebooks read every task the same way.

    Attributes:
        task: Learning task (``multiclass``).
        split: Split that was scored.
        primary_metric: Name of the decision metric.
        metrics: Metric name -> value (global, computed with the shared registry).
        predictions: Row-level frame: truth, prediction, decision, confidence, probabilities.
        labels: Class labels in the business order.
        confusion_matrix: Raw confusion matrix (rows = truth, columns = prediction).
        classification_report: Per-class report as a dict (JSON friendly).
        per_class: Per-class precision / recall / F1 / support / one-vs-rest AUC.
        confusions: Confused pairs, ranked by volume.
        references: Model versus majority class, current rule and oracle ceiling.
        decision: Cost and error profile of each decision policy.
        abstention: Coverage / accuracy / cost when low-confidence alarms go to an expert.
        calibration: Reliability table of the top-label confidence.
        verdict: Contractual objectives with measured value, threshold and status.
        errors: Worst misdiagnosed alarms (confident and wrong first).
        curves: One-vs-rest ROC points per class.
        feature_importance: Feature -> importance (native or permutation).
        extras: Free-form payload (model summary, class balance, generation references, ...).
        n_samples: Number of scored rows.
    """

    task: str
    split: str = "test"
    primary_metric: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    labels: list[str] = field(default_factory=list)
    confusion_matrix: np.ndarray | None = None
    classification_report: dict[str, Any] = field(default_factory=dict)
    per_class: pd.DataFrame = field(default_factory=pd.DataFrame)
    confusions: pd.DataFrame = field(default_factory=pd.DataFrame)
    references: pd.DataFrame = field(default_factory=pd.DataFrame)
    decision: pd.DataFrame = field(default_factory=pd.DataFrame)
    abstention: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    verdict: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    curves: dict[str, Any] = field(default_factory=dict)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    extras: dict[str, Any] = field(default_factory=dict)
    n_samples: int = 0

    @property
    def primary_value(self) -> float:
        """Value of the primary metric (NaN when unavailable)."""
        return float(self.metrics.get(self.primary_metric, float("nan")))

    @property
    def error_rate(self) -> float:
        """Share of misdiagnosed alarms (``argmax`` prediction)."""
        if self.predictions.empty or "is_error" not in self.predictions.columns:
            return float("nan")
        return float(self.predictions["is_error"].mean())

    @property
    def ece(self) -> float:
        """Expected calibration error of the top-label confidence."""
        return float(self.extras.get("ece", float("nan")))

    @property
    def is_compliant(self) -> bool:
        """``True`` when every evaluable objective of the verdict is met."""
        if self.verdict.empty:
            return False
        evaluable = self.verdict[self.verdict["status"] != "n/a"]
        return bool(len(evaluable)) and bool((evaluable["status"] == "atteint").all())

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result (JSON friendly, heavy frames reduced to records)."""

        def records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
            if frame is None or frame.empty:
                return []
            subset = frame if limit is None else frame.head(limit)
            return json.loads(subset.to_json(orient="records", force_ascii=False))

        return {
            "task": self.task,
            "split": self.split,
            "n_samples": self.n_samples,
            "primary_metric": self.primary_metric,
            "primary_value": _to_float(self.primary_value),
            "metrics": {key: _to_float(value) for key, value in self.metrics.items()},
            "labels": list(self.labels),
            "confusion_matrix": None
            if self.confusion_matrix is None
            else np.asarray(self.confusion_matrix).tolist(),
            "classification_report": self.classification_report,
            "per_class": records(self.per_class),
            "confusions": records(self.confusions, 10),
            "references": records(self.references),
            "decision": records(self.decision),
            "abstention": records(self.abstention),
            "calibration": records(self.calibration),
            "verdict": records(self.verdict),
            "is_compliant": self.is_compliant,
            "error_rate": _to_float(self.error_rate),
            "n_errors": len(self.errors),
            "feature_importance_top": records(self.feature_importance, 15),
            "extras": self.extras,
        }


class Evaluator:
    """Score a fitted multiclass model and produce every diagnosis diagnostic."""

    def __init__(
        self,
        model: BaseModel,
        *,
        metrics_config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        task: str | None = None,
        top_k_errors: int = 25,
        config: Mapping[str, Any] | None = None,
        settings: DiagnosisSettings | None = None,
    ) -> None:
        """Inject the model, the metric policy and the decision settings.

        Args:
            model: Fitted model implementing :class:`BaseModel`.
            metrics_config: ``metrics`` configuration node (primary / secondary / task).
            paths: Project layout, used to persist metrics and read the generation metadata.
            task: Task override (defaults to ``model.task``).
            top_k_errors: Number of worst errors kept for the error analysis.
            config: Root configuration mapping, used to resolve the ``diagnosis`` block and the
                identifier column.
            settings: Pre-resolved decision settings (take precedence over ``config``).
        """
        self.model = model
        self.paths = paths or ProjectPaths.from_root()
        self.config: dict[str, Any] = dict(metrics_config or {})
        self.root_config: dict[str, Any] = dict(config or {})
        self.task = task or self.config.get("task") or model.task
        self.top_k_errors = int(top_k_errors)
        self._settings = settings
        self.primary_metric = str(self.config.get("primary", "f1_macro"))
        self.metric_names = [
            self.primary_metric,
            *[str(name) for name in self.config.get("secondary", []) or []],
        ]
        data_node = dict(self.root_config.get("data") or {})
        self.id_column = data_node.get("id_column")
        self.target_column = data_node.get("target")

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
        return cls(
            model,
            metrics_config=metrics_node,
            paths=paths,
            task=metrics_node.get("task"),
            config=config,
        )

    @property
    def settings(self) -> DiagnosisSettings:
        """Decision settings, resolved lazily against the classes of the fitted model."""
        if self._settings is None:
            self._settings = DiagnosisSettings.resolve(
                self.root_config or None, classes=self._model_classes()
            )
        return self._settings

    # ------------------------------------------------------------------ evaluation ------
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series | Sequence[Any] | None = None,
        *,
        split: str = "test",
        context: pd.DataFrame | None = None,
    ) -> EvaluationResult:
        """Score the model on a feature matrix.

        Args:
            X: Preprocessed features.
            y: Ground truth labels (required).
            split: Split name, used in logs and artefacts.
            context: Raw/enriched rows aligned with ``X`` (identifier, alarm code, raw sensors):
                required for the routing reference and a readable error analysis.

        Returns:
            The :class:`EvaluationResult`.

        Raises:
            RuntimeError: When the model is not fitted.
            ValueError: When the labels are missing or contain a class the model never saw.
        """
        self.model.check_is_fitted()
        if y is None:
            msg = "Multiclass evaluation requires ground truth labels"
            raise ValueError(msg)
        truth = pd.Series(np.asarray(y)).astype(str).reset_index(drop=True)
        classes = self._model_classes()
        unseen = sorted(set(truth) - set(classes))
        if unseen:
            msg = (
                f"Ground truth contains class(es) {unseen} unknown to the model {classes}: "
                "retrain on data covering every class before evaluating"
            )
            raise ValueError(msg)
        settings = self.settings
        labels = settings.ordered(classes)

        probabilities = self._probabilities(X, classes, labels)
        predictions = labels_from(probabilities, labels)
        decided = minimum_cost_decision(probabilities, labels, settings)
        context_frame = _aligned_context(context, len(truth))

        metrics = self._compute_metrics(truth, predictions, probabilities, labels)
        matrix = confusion_matrix(truth, predictions, labels=labels)
        report, per_class, curves = self._per_class(truth, predictions, probabilities, labels)
        predictions_frame = self._predictions_frame(
            truth, predictions, decided, probabilities, labels, context_frame
        )
        reference = self.generation_reference()
        references = self._references(
            truth, predictions, probabilities, labels, context_frame, reference
        )
        decision = self._decision_policies(truth, probabilities, labels, context_frame)
        abstention = self._abstention(truth, probabilities, labels)
        calibration, ece = _reliability(truth.to_numpy(), probabilities, labels)
        latency = self._latency_ms_per_1000(X)
        verdict = self._verdict(metrics, per_class, references, decision, ece, latency, labels)

        result = EvaluationResult(
            task=self.task,
            split=split,
            primary_metric=self.primary_metric,
            metrics=metrics,
            predictions=predictions_frame,
            labels=labels,
            confusion_matrix=matrix,
            classification_report=report,
            per_class=per_class,
            confusions=_confused_pairs(matrix, labels),
            references=references,
            decision=decision,
            abstention=abstention,
            calibration=calibration,
            verdict=verdict,
            errors=self._error_analysis(predictions_frame, context_frame),
            curves=curves,
            feature_importance=self.feature_importance(X, y=truth),
            extras={
                "model": self.model.summary(),
                "n_features": int(X.shape[1]),
                "supports_proba": bool(getattr(self.model, "supports_proba", False)),
                "class_balance": {
                    str(key): round(float(value), 4)
                    for key, value in truth.value_counts(normalize=True).items()
                },
                "ece": _to_float(ece),
                "latency_ms_per_1000": _to_float(latency),
                "review_threshold": settings.review_threshold,
                "generation_reference": reference,
            },
            n_samples=len(X),
        )
        met = int((verdict["status"] == "atteint").sum()) if not verdict.empty else 0
        evaluable = int((verdict["status"] != "n/a").sum()) if not verdict.empty else 0
        result.extras["objectives_met"] = f"{met}/{evaluable}"
        result.extras["is_compliant"] = result.is_compliant
        logger.info(
            "Evaluation on '{}' | n={} {}={:.4f} error_rate={:.3f} ece={:.3f} objectives={}",
            split,
            result.n_samples,
            self.primary_metric,
            result.primary_value,
            result.error_rate,
            ece,
            result.extras["objectives_met"],
        )
        return result

    def compare_to_baseline(
        self, X: pd.DataFrame, y: pd.Series | Sequence[Any], *, baseline: str = "dummy"
    ) -> dict[str, float]:
        """Score a trivial baseline on the same data, to prove the model adds value.

        Args:
            X: Preprocessed features (unused by the baselines, kept for API symmetry).
            y: Ground truth.
            baseline: ``dummy`` (most frequent class) or ``random`` (uniform draw).

        Returns:
            The baseline metrics, prefixed with ``baseline_``.
        """
        from sklearn.dummy import DummyClassifier

        truth = np.asarray(y).astype(str)
        estimator = DummyClassifier(
            strategy="most_frequent" if baseline == "dummy" else "uniform", random_state=0
        )
        estimator.fit(X, truth)
        calculator = MetricCalculator(task=self.task, metrics=self.metric_names)
        values = calculator.evaluate(
            MetricInputs(
                y_true=truth,
                y_pred=estimator.predict(X),
                y_proba=estimator.predict_proba(X),
                X=X,
                extra={"classes": list(estimator.classes_)},
            )
        )
        baseline_metrics = {f"baseline_{name}": _to_float(value) for name, value in values.items()}
        logger.info("Baseline '{}' | {}", baseline, baseline_metrics)
        return baseline_metrics

    def feature_importance(
        self, X: pd.DataFrame, *, y: pd.Series | Sequence[Any] | None = None
    ) -> pd.DataFrame:
        """Rank the features, natively when the model exposes it, by permutation otherwise.

        Args:
            X: Features used for permutation importance.
            y: Ground truth, required by permutation importance.

        Returns:
            A DataFrame ``feature | importance | method`` sorted by descending importance.
        """
        estimator = getattr(self.model, "estimator_", None) or getattr(self.model, "model_", None)
        native = getattr(estimator, "feature_importances_", None)
        if native is None:
            coefficients = getattr(estimator, "coef_", None)
            if coefficients is not None:
                array = np.abs(np.asarray(coefficients, dtype="float64"))
                native = array.mean(axis=0) if array.ndim > 1 else array.ravel()
        if native is not None and len(native) == X.shape[1]:
            frame = pd.DataFrame(
                {
                    "feature": list(X.columns),
                    "importance": np.asarray(native, dtype="float64"),
                    "method": "native",
                }
            )
            return frame.sort_values("importance", ascending=False).reset_index(drop=True)
        if y is None:
            return pd.DataFrame()
        try:
            from sklearn.inspection import permutation_importance
        except ImportError:  # pragma: no cover
            return pd.DataFrame()
        sample_X, sample_y = X.reset_index(drop=True), np.asarray(y).astype(str)
        if len(sample_X) > 1500:
            positions = np.random.default_rng(0).choice(len(sample_X), size=1500, replace=False)
            sample_X, sample_y = sample_X.iloc[positions], sample_y[positions]
        try:
            scoring = permutation_importance(
                _MacroF1Adapter(self.model), sample_X, sample_y, n_repeats=3, random_state=0
            )
        except Exception as exc:
            logger.warning("Permutation importance unavailable: {}", exc)
            return pd.DataFrame()
        frame = pd.DataFrame(
            {
                "feature": list(sample_X.columns),
                "importance": np.asarray(scoring.importances_mean, dtype="float64"),
                "method": "permutation (macro-F1)",
            }
        )
        return frame.sort_values("importance", ascending=False).reset_index(drop=True)

    def generation_reference(self) -> dict[str, Any]:
        """Read the references published by the data generator (oracle ceiling, rule, floor).

        Returns:
            The published references (flat keys), empty when the metadata file is absent.
        """
        for path in (
            self.paths.raw_dir / GENERATION_METADATA,
            self.paths.data_dir / GENERATION_METADATA,
        ):
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("Generation metadata unreadable at {}: {}", path, exc)
                continue
            reference = {
                key: value
                for key, value in payload.items()
                if key.endswith(("_oracle", "_majority", "_alarm_code")) or key == "majority_class"
            }
            if reference:
                reference["source"] = str(path)
            return reference
        return {}

    def save_metrics(self, result: EvaluationResult, path: str | Path | None = None) -> Path:
        """Persist the evaluation as JSON.

        Args:
            result: Evaluation result.
            path: Destination file (defaults to ``artifacts/metrics/evaluation_metrics.json``).

        Returns:
            The written path.
        """
        destination = Path(path) if path else self.paths.metrics_dir / "evaluation_metrics.json"
        return write_json(destination, result.to_dict())

    # ------------------------------------------------------------------ internals -------
    def _model_classes(self) -> list[str]:
        """Classes of the fitted model, as strings, in the model's own column order."""
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            msg = "The model exposes no `classes_`: it cannot be evaluated as a multiclass model"
            raise ValueError(msg)
        return [str(value) for value in np.asarray(classes).tolist()]

    def _probabilities(
        self, X: pd.DataFrame, classes: Sequence[str], labels: Sequence[str]
    ) -> np.ndarray:
        """Return the class probabilities, columns reordered in the business order."""
        if not getattr(self.model, "supports_proba", False):
            msg = "The diagnosis evaluation needs class probabilities (supports_proba=False)"
            raise ValueError(msg)
        raw = np.asarray(self.model.predict_proba(X), dtype="float64")
        if raw.ndim != 2 or raw.shape[1] != len(classes):
            msg = f"predict_proba returned shape {raw.shape}, expected (n, {len(classes)})"
            raise ValueError(msg)
        position = {label: index for index, label in enumerate(classes)}
        return raw[:, [position[label] for label in labels]]

    def _compute_metrics(
        self,
        truth: pd.Series,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        labels: Sequence[str],
    ) -> dict[str, float]:
        """Compute the configured metrics through the shared registry."""
        calculator = MetricCalculator(task=self.task, metrics=self.metric_names)
        values = calculator.evaluate(
            MetricInputs(
                y_true=truth.to_numpy(),
                y_pred=predictions,
                y_proba=probabilities,
                extra={"classes": list(labels)},
            )
        )
        return {name: _to_float(value) for name, value in values.items()}

    def _per_class(
        self,
        truth: pd.Series,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        labels: Sequence[str],
    ) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
        """Per-class report, table and one-vs-rest ROC points."""
        from sklearn.metrics import classification_report

        report = classification_report(
            truth, predictions, labels=list(labels), output_dict=True, zero_division=0
        )
        rows: list[dict[str, Any]] = []
        roc: dict[str, Any] = {}
        truth_values = truth.to_numpy()
        for column, label in enumerate(labels):
            entry = report.get(str(label), {})
            positives = truth_values == label
            auc = float("nan")
            if 0 < int(positives.sum()) < len(positives):
                scores = probabilities[:, column]
                auc = float(roc_auc_score(positives, scores))
                fpr, tpr, _ = roc_curve(positives, scores)
                roc[str(label)] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc}
            predicted = int((predictions == label).sum())
            support = int(positives.sum())
            rows.append(
                {
                    "class": str(label),
                    "precision": round(float(entry.get("precision", 0.0)), 4),
                    "recall": round(float(entry.get("recall", 0.0)), 4),
                    "f1": round(float(entry.get("f1-score", 0.0)), 4),
                    "auc_ovr": round(auc, 4) if np.isfinite(auc) else float("nan"),
                    "support": support,
                    "predicted": predicted,
                    "team": self.settings.team_for(label),
                    "structural": label in set(self.settings.structural_classes),
                }
            )
        return report, pd.DataFrame(rows), {"roc_ovr": roc}

    def _predictions_frame(
        self,
        truth: pd.Series,
        predictions: np.ndarray,
        decided: np.ndarray,
        probabilities: np.ndarray,
        labels: Sequence[str],
        context: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Assemble the row-level frame on which every diagnostic is built."""
        ordered = np.sort(probabilities, axis=1)
        second = (
            np.asarray(labels, dtype=object)[np.argsort(probabilities, axis=1)[:, -2]]
            if len(labels) > 1
            else predictions
        )
        frame = pd.DataFrame(
            {
                "y_true": truth.to_numpy(),
                "y_pred": predictions,
                "decision": decided,
                "second_choice": second,
                "confidence": ordered[:, -1],
                "margin": ordered[:, -1] - (ordered[:, -2] if len(labels) > 1 else 0.0),
            }
        )
        for column, label in enumerate(labels):
            frame[f"proba_{label}"] = probabilities[:, column]
        frame["is_error"] = (frame["y_true"] != frame["y_pred"]).astype(int)
        frame["needs_review"] = frame["confidence"] < self.settings.review_threshold
        if context is not None:
            keep = [
                column
                for column in (self.id_column, self.settings.alarm_code_column)
                if column and column in context.columns
            ]
            keep += [
                column
                for column in context.columns
                if column not in keep
                and column != self.target_column
                and not pd.api.types.is_numeric_dtype(context[column])
                and not pd.api.types.is_datetime64_any_dtype(context[column])
            ][:4]
            if keep:
                frame = pd.concat([context.loc[:, keep].reset_index(drop=True), frame], axis=1)
        return frame

    def _references(
        self,
        truth: pd.Series,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        labels: Sequence[str],
        context: pd.DataFrame | None,
        reference: Mapping[str, Any],
    ) -> pd.DataFrame:
        """Score the model against the majority floor, the current rule and the oracle ceiling."""
        settings = self.settings
        truth_values = truth.to_numpy()
        majority_label = str(reference.get("majority_class") or truth.value_counts().idxmax())
        scorers: list[tuple[str, np.ndarray]] = [
            ("modèle", predictions),
            ("classe majoritaire", np.full(len(truth_values), majority_label, dtype=object)),
        ]
        codes = self._alarm_codes(context)
        if codes is not None:
            scorers.append(("routage actuel (code automate)", settings.route_alarm_codes(codes)))

        rows = [
            {
                "référence": name,
                **_decision_profile(truth_values, predicted, labels, settings),
                "source": "test",
            }
            for name, predicted in scorers
        ]
        if "f1_macro_ceiling_oracle" in reference:
            rows.append(
                {
                    "référence": "plafond oracle (générateur)",
                    "f1_macro": _to_float(reference.get("f1_macro_ceiling_oracle")),
                    "accuracy": _to_float(reference.get("accuracy_ceiling_oracle")),
                    "balanced_accuracy": float("nan"),
                    "missed_failure_rate": float("nan"),
                    "cost_per_alarm": float("nan"),
                    "source": "jeu complet",
                }
            )
        return pd.DataFrame(rows)

    def _decision_policies(
        self,
        truth: pd.Series,
        probabilities: np.ndarray,
        labels: Sequence[str],
        context: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Cost and error profile of each decision policy on the same alarms."""
        settings = self.settings
        truth_values = truth.to_numpy()
        argmax = labels_from(probabilities, labels)
        minimum_cost = minimum_cost_decision(probabilities, labels, settings)
        reviewed = probabilities.max(axis=1) < settings.review_threshold
        policies: list[tuple[str, np.ndarray, np.ndarray | None]] = [
            ("argmax", argmax, None),
            ("coût minimal", minimum_cost, None),
            (
                f"coût minimal + revue (confiance < {settings.review_threshold:.2f})",
                minimum_cost,
                reviewed,
            ),
        ]
        codes = self._alarm_codes(context)
        if codes is not None:
            policies.append(
                ("routage actuel (code automate)", settings.route_alarm_codes(codes), None)
            )
        rows = []
        for name, decided, review in policies:
            profile = _decision_profile(truth_values, decided, labels, settings, reviewed=review)
            profile["review_share"] = 0.0 if review is None else round(float(np.mean(review)), 4)
            rows.append({"politique": name, **profile})
        return pd.DataFrame(rows)

    def _abstention(
        self, truth: pd.Series, probabilities: np.ndarray, labels: Sequence[str]
    ) -> pd.DataFrame:
        """Coverage / accuracy / cost curve when low-confidence alarms are sent to an expert."""
        settings = self.settings
        truth_values = truth.to_numpy()
        confidence = probabilities.max(axis=1)
        predicted = labels_from(probabilities, labels)
        decided = minimum_cost_decision(probabilities, labels, settings)
        rows = []
        for threshold in sorted({*ABSTENTION_GRID, round(settings.review_threshold, 2)}):
            reviewed = confidence < threshold
            automated = ~reviewed
            costs = realised_costs(truth_values, decided, labels, settings, reviewed=reviewed)
            rows.append(
                {
                    "threshold": float(threshold),
                    "coverage": round(float(automated.mean()), 4),
                    "accuracy_automated": round(
                        float((predicted[automated] == truth_values[automated]).mean()), 4
                    )
                    if automated.any()
                    else float("nan"),
                    "cost_per_alarm": round(float(costs.mean()), 2),
                    "reviewed": int(reviewed.sum()),
                }
            )
        return pd.DataFrame(rows)

    def _latency_ms_per_1000(self, X: pd.DataFrame) -> float:
        """Median time to score a batch of 1 000 alarms (probabilities included)."""
        if X.empty:
            return float("nan")
        batch = X.sample(n=LATENCY_BATCH, replace=True, random_state=0)
        timings = []
        for _ in range(3):
            started = time.perf_counter()
            self.model.predict_proba(batch)
            timings.append((time.perf_counter() - started) * 1000.0)
        return float(np.median(timings))

    def _verdict(
        self,
        metrics: Mapping[str, float],
        per_class: pd.DataFrame,
        references: pd.DataFrame,
        decision: pd.DataFrame,
        ece: float,
        latency: float,
        labels: Sequence[str],
    ) -> pd.DataFrame:
        """Evaluate every contractual objective of the ``diagnosis`` block."""
        objectives = self.settings.objectives
        by_reference = references.set_index("référence") if not references.empty else pd.DataFrame()
        by_policy = decision.set_index("politique") if not decision.empty else pd.DataFrame()
        rule_name = "routage actuel (code automate)"
        f1_macro = float(metrics.get("f1_macro", float("nan")))
        rule_f1 = (
            float(by_reference.loc[rule_name, "f1_macro"])
            if rule_name in by_reference.index
            else float("nan")
        )
        predictable = self.settings.predictable_classes(labels)
        recalls = (
            per_class.set_index("class").loc[predictable, "recall"]
            if not per_class.empty
            else pd.Series()
        )
        worst_recall = float(recalls.min()) if len(recalls) else float("nan")
        minimum = by_policy.loc["coût minimal"] if "coût minimal" in by_policy.index else None
        missed = float(minimum["missed_failure_rate"]) if minimum is not None else float("nan")
        cost_model = float(minimum["cost_per_alarm"]) if minimum is not None else float("nan")
        cost_rule = (
            float(by_policy.loc[rule_name, "cost_per_alarm"])
            if rule_name in by_policy.index
            else float("nan")
        )
        reduction = (
            1.0 - cost_model / cost_rule
            if np.isfinite(cost_rule) and cost_rule > 0
            else float("nan")
        )
        worst_class = str(recalls.idxmin()) if len(recalls) else "-"

        checks = [
            ("macro-F1", f1_macro, objectives["f1_macro_min"], "≥", "f1_macro"),
            (
                "gain de macro-F1 sur le routage actuel",
                f1_macro - rule_f1,
                objectives["gain_vs_alarm_code_min"],
                "≥",
                "points",
            ),
            (
                f"rappel minimal des modes prédictibles ({worst_class})",
                worst_recall,
                objectives["per_class_recall_min"],
                "≥",
                "recall",
            ),
            (
                "pannes réelles classées sans panne (décision à coût minimal)",
                missed,
                objectives["missed_failure_rate_max"],
                "≤",
                "part",
            ),
            ("erreur de calibration top-label (ECE)", ece, objectives["ece_max"], "≤", "ece"),
            (
                "baisse du coût par alarme face au routage actuel",
                reduction,
                objectives["cost_reduction_vs_alarm_code_min"],
                "≥",
                "part",
            ),
            (
                "latence pour 1 000 alarmes (ms)",
                latency,
                objectives["latency_ms_per_1000_max"],
                "≤",
                "ms",
            ),
        ]
        rows = []
        for name, value, threshold, direction, unit in checks:
            if not np.isfinite(value):
                status = "n/a"
            elif direction == "≥":
                status = "atteint" if value >= threshold else "non atteint"
            else:
                status = "atteint" if value <= threshold else "non atteint"
            rows.append(
                {
                    "objectif": name,
                    "valeur": round(float(value), 4) if np.isfinite(value) else float("nan"),
                    "seuil": f"{direction} {threshold:g}",
                    "unité": unit,
                    "status": status,
                }
            )
        return pd.DataFrame(rows)

    def _error_analysis(
        self, predictions: pd.DataFrame, context: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Rank the misdiagnosed alarms: confident and wrong first (the most damaging ones)."""
        if predictions.empty:
            return pd.DataFrame()
        errors = predictions.loc[predictions["is_error"] == 1].copy()
        if errors.empty:
            return errors
        errors = errors.sort_values("confidence", ascending=False).head(self.top_k_errors)
        if context is not None:
            extra = [
                column
                for column in context.columns
                if column not in errors.columns and column != self.target_column
            ]
            errors = pd.concat(
                [errors.reset_index(), context.loc[errors.index, extra].reset_index(drop=True)],
                axis=1,
            ).drop(columns=["index"])
        return errors.reset_index(drop=True)

    def _alarm_codes(self, context: pd.DataFrame | None) -> np.ndarray | None:
        """Return the current routing signal of every row, when the context carries it."""
        column = self.settings.alarm_code_column
        if (
            context is None
            or not column
            or column not in context.columns
            or not self.settings.alarm_code_rule
        ):
            return None
        return context[column].astype(str).to_numpy()


class _MacroF1Adapter:
    """Minimal scikit-learn facade scoring the macro-F1 (used by permutation importance)."""

    def __init__(self, model: BaseModel) -> None:
        """Wrap a :class:`BaseModel`.

        Args:
            model: Fitted model.
        """
        self.model = model

    def fit(self, X: pd.DataFrame, y: Any = None, **fit_params: Any) -> _MacroF1Adapter:
        """Return ``self`` (the wrapped model is already fitted).

        Args:
            X: Features (unused).
            y: Ground truth (unused).
            **fit_params: Ignored.

        Returns:
            ``self``.
        """
        return self

    def score(self, X: pd.DataFrame, y: Any = None) -> float:
        """Return the macro-F1 of the wrapped model.

        Args:
            X: Features.
            y: Ground truth.

        Returns:
            The macro-F1.
        """
        from sklearn.metrics import f1_score

        predictions = np.asarray(self.model.predict(X)).astype(str)
        return float(
            f1_score(np.asarray(y).astype(str), predictions, average="macro", zero_division=0)
        )


def labels_from(probabilities: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """Return the ``argmax`` label of every row.

    Args:
        probabilities: ``(n, K)`` probabilities, columns aligned with ``labels``.
        labels: Class labels.

    Returns:
        The most probable label of every row.
    """
    return np.asarray(list(labels), dtype=object)[np.asarray(probabilities).argmax(axis=1)]


def _decision_profile(
    truth: np.ndarray,
    decided: np.ndarray,
    labels: Sequence[str],
    settings: DiagnosisSettings,
    *,
    reviewed: np.ndarray | None = None,
) -> dict[str, float]:
    """Quality and cost profile of one set of decisions."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    decided = np.asarray(decided).astype(str)
    effective = decided if reviewed is None else np.where(reviewed, truth, decided)
    no_failure = settings.no_failure_class
    real = truth != no_failure if no_failure is not None else np.ones(len(truth), dtype=bool)
    missed = (
        float(np.mean(effective[real] == no_failure))
        if no_failure is not None and real.any()
        else float("nan")
    )
    costs = realised_costs(truth, decided, labels, settings, reviewed=reviewed)
    return {
        "f1_macro": round(
            float(
                f1_score(truth, effective, labels=list(labels), average="macro", zero_division=0)
            ),
            4,
        ),
        "accuracy": round(float(accuracy_score(truth, effective)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(truth, effective)), 4),
        "missed_failure_rate": round(missed, 4) if np.isfinite(missed) else float("nan"),
        "cost_per_alarm": round(float(costs.mean()), 2),
    }


def _confused_pairs(matrix: np.ndarray, labels: Sequence[str], *, top: int = 10) -> pd.DataFrame:
    """Rank the off-diagonal cells of the confusion matrix."""
    counts = np.asarray(matrix)
    support = counts.sum(axis=1)
    rows = [
        {
            "classe réelle": labels[row],
            "classe prédite": labels[column],
            "effectif": int(counts[row, column]),
            "part de la classe réelle": round(float(counts[row, column] / support[row]), 4)
            if support[row]
            else 0.0,
        }
        for row in range(len(labels))
        for column in range(len(labels))
        if row != column and counts[row, column] > 0
    ]
    if not rows:
        return pd.DataFrame(
            columns=["classe réelle", "classe prédite", "effectif", "part de la classe réelle"]
        )
    return (
        pd.DataFrame(rows).sort_values("effectif", ascending=False).head(top).reset_index(drop=True)
    )


def _reliability(
    truth: np.ndarray, probabilities: np.ndarray, labels: Sequence[str]
) -> tuple[pd.DataFrame, float]:
    """Top-label reliability table and expected calibration error.

    The top-label confidence (probability of the predicted class) is binned in equal-width bins;
    within each bin the mean confidence is compared with the observed accuracy. The ECE is the
    support-weighted mean of the gaps: 0 for a perfectly calibrated model.
    """
    confidence = probabilities.max(axis=1)
    correct = labels_from(probabilities, labels) == truth
    edges = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, CALIBRATION_BINS - 1)
    rows = []
    ece = 0.0
    for index in range(CALIBRATION_BINS):
        mask = bins == index
        if not mask.any():
            continue
        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        ece += float(mask.mean()) * abs(accuracy - mean_confidence)
        rows.append(
            {
                "bin": f"[{edges[index]:.1f}, {edges[index + 1]:.1f}]",
                "confidence": round(mean_confidence, 4),
                "accuracy": round(accuracy, 4),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows), float(ece)


def _aligned_context(context: pd.DataFrame | None, length: int) -> pd.DataFrame | None:
    """Return the context reset on a positional index, or ``None`` when it is not aligned."""
    if context is None or len(context) != length:
        if context is not None:
            logger.warning("Context ignored: {} rows for {} predictions", len(context), length)
        return None
    return context.reset_index(drop=True)


def _to_float(value: Any) -> float:
    """Coerce a metric value to ``float`` (NaN when impossible)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")
