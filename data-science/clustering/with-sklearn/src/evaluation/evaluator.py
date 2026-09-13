"""Évaluation d'une segmentation non supervisée (clustering).

Un clustering n'a pas de vérité terrain en production : on ne peut donc pas « mesurer l'erreur ».
L'évaluation repose sur trois familles de critères, toutes produites ici :

1. **critères internes** — compacité et séparation calculées sur la matrice de features
   (silhouette, Calinski-Harabasz, Davies-Bouldin, inertie, nombre de groupes) ;
2. **critères de robustesse métier** — taille minimale des groupes (un micro-groupe n'est pas
   exploitable par le CRM), stabilité des affectations entre graines (ARI), et confiance
   individuelle (distance au centroïde, écart au second groupe le plus proche) ;
3. **critères externes de diagnostic** — accord avec la structure latente injectée par le
   générateur (ARI, NMI, V-measure) et **validité externe** : les groupes doivent séparer un
   comportement observé *après* coup (churn à 90 jours), sans jamais l'avoir utilisé.

Les colonnes de diagnostic (``latent_segment``, ``churned_next_90d``) sont des métadonnées :
elles sont exclues de la matrice de features par ``drop_columns`` et ne servent qu'à juger la
segmentation produite.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
    v_measure_score,
)

from src.models.base import BaseModel
from src.training.losses_metrics import MetricCalculator, MetricInputs
from src.utils.io import write_json
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: En dessous de cette part de population, un groupe n'est pas actionnable par le CRM.
MIN_CLUSTER_SHARE = 0.03

#: Colonnes de diagnostic produites par le générateur (métadonnées, jamais des features).
LATENT_COLUMN = "latent_segment"
OUTCOME_COLUMN = "churned_next_90d"

#: Valeurs de k explorées pour la courbe de sélection (silhouette / inertie / Davies-Bouldin).
K_RANGE: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)

#: Graines utilisées pour mesurer la stabilité des affectations.
STABILITY_SEEDS: tuple[int, ...] = (7, 19, 123)

#: Nombre de points conservés pour les nuages PCA (JSON des rapports et figures).
MAX_SCATTER_POINTS = 1200


@dataclass(slots=True)
class EvaluationResult:
    """Structured outcome of a clustering evaluation.

    Attributes:
        task: Learning task (``clustering``).
        split: Split that was scored (``test``, ``val``, ...).
        primary_metric: Name of the metric driving the decision (typically ``silhouette``).
        metrics: Metric name -> value.
        predictions: Row-level frame (``cluster``, ``distance_to_centroid``, ``silhouette``,
            ``membership``, ``second_best``, ``confidence`` + diagnostic columns from context).
        labels: Raw columns used to build the cluster profiles (kept for contract symmetry with
            the supervised results, where this holds the class names).
        per_segment: One row per cluster (size, share, mean raw features, churn rate, ARI
            contribution, distance statistics).
        errors: Hardest-to-assign rows (lowest silhouette / closest to a second cluster).
        curves: Down-sampled points of the k-selection curve, silhouette histogram and PCA map.
        feature_importance: Centroid coordinates in standardised space, i.e. what *defines* each
            cluster (one row per cluster, one column per feature, plus the top drivers).
        extras: Free-form payload (profiles, external validity, stability, degeneracy flags).
        n_samples: Number of scored rows.
    """

    task: str
    split: str = "test"
    primary_metric: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    labels: list[str] = field(default_factory=list)
    per_segment: pd.DataFrame = field(default_factory=pd.DataFrame)
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
    def n_clusters(self) -> int:
        """Number of non-empty clusters produced by the model."""
        return int(_to_float(self.metrics.get("n_clusters", float("nan"))))

    @property
    def min_cluster_share(self) -> float:
        """Share of the smallest cluster (degeneracy indicator)."""
        if self.per_segment.empty or "share" not in self.per_segment.columns:
            return float("nan")
        return float(self.per_segment["share"].min())

    @property
    def ari_latent(self) -> float:
        """Agreement with the latent structure injected by the generator (diagnostic)."""
        return _to_float(self.extras.get("external_validity", {}).get("ari_latent", float("nan")))

    @property
    def churn_spread(self) -> float:
        """Gap between the healthiest and the riskiest cluster (external validity)."""
        return _to_float(self.extras.get("external_validity", {}).get("churn_spread", float("nan")))

    @property
    def stability_ari(self) -> float:
        """Mean ARI between the reference assignment and re-fits on other seeds."""
        return _to_float(self.extras.get("stability", {}).get("mean_ari", float("nan")))

    @property
    def degenerate_clusters(self) -> list[int]:
        """Cluster identifiers smaller than :data:`MIN_CLUSTER_SHARE`."""
        flagged = self.extras.get("degenerate_clusters", [])
        return [int(value) for value in flagged]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the evaluation."""
        return {
            "task": self.task,
            "split": self.split,
            "primary_metric": self.primary_metric,
            "primary_value": self.primary_value,
            "metrics": {key: _to_float(value) for key, value in self.metrics.items()},
            "n_samples": int(self.n_samples),
            "n_clusters": self.n_clusters,
            "min_cluster_share": self.min_cluster_share,
            "ari_latent": self.ari_latent,
            "stability_ari": self.stability_ari,
            "churn_spread": self.churn_spread,
            "degenerate_clusters": self.degenerate_clusters,
        }


class Evaluator:
    """Score a clustering model: internal indices, robustness and external diagnostics.

    Example:
        >>> evaluator = Evaluator(model, metrics_config={"task": "clustering",
        ...                                             "primary": "silhouette"})
        >>> result = evaluator.evaluate(X_test, context=test_frame)
        >>> result.primary_metric
        'silhouette'
    """

    def __init__(
        self,
        model: BaseModel,
        *,
        metrics_config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        task: str | None = None,
        top_k_errors: int = 25,
        context_columns: Sequence[str] | None = None,
        k_range: Sequence[int] = K_RANGE,
        min_cluster_share: float = MIN_CLUSTER_SHARE,
        stability_seeds: Sequence[int] = STABILITY_SEEDS,
    ) -> None:
        """Inject the model and the metric policy.

        Args:
            model: Fitted model implementing :class:`BaseModel`.
            metrics_config: ``metrics`` configuration node (primary / secondary / task).
            paths: Project layout, used to persist metrics.
            task: Task override (defaults to ``model.task``).
            top_k_errors: Number of hardest-to-assign rows kept for the analysis.
            context_columns: Columns copied from the enriched context into the diagnostic frame
                (identifier, group, timestamp).
            k_range: Values of k explored by the selection curve.
            min_cluster_share: Share below which a cluster is flagged as degenerate.
            stability_seeds: Seeds used to re-fit the model for the stability analysis.
        """
        self.model = model
        self.paths = paths or ProjectPaths.from_root()
        self.config: dict[str, Any] = dict(metrics_config or {})
        self.task = task or self.config.get("task") or model.task
        self.top_k_errors = int(top_k_errors)
        self.context_columns: tuple[str, ...] = tuple(str(name) for name in (context_columns or ()))
        self.k_range = tuple(int(value) for value in k_range)
        self.min_cluster_share = float(min_cluster_share)
        self.stability_seeds = tuple(int(seed) for seed in stability_seeds)
        self.primary_metric = str(self.config.get("primary", "silhouette"))
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
        metrics_node = config.get("metrics") or {}
        data_node = config.get("data") or {}
        # Les colonnes de contexte utiles au profilage sont déclarées dans la configuration
        # (identifiant, groupe, date) : aucun nom de colonne n'est codé en dur.
        context_columns = tuple(
            str(data_node[key])
            for key in ("id_column", "group_column", "time_column")
            if data_node.get(key)
        )
        return cls(
            model,
            metrics_config=dict(metrics_node),
            paths=paths,
            task=metrics_node.get("task"),
            context_columns=context_columns or None,
        )

    # ------------------------------------------------------------------ evaluation ------
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series | Sequence[Any] | None = None,
        *,
        split: str = "test",
        context: pd.DataFrame | None = None,
    ) -> EvaluationResult:
        """Score the clustering on a feature matrix.

        Args:
            X: Preprocessed features (the space in which distances are measured).
            y: Ignored — an unsupervised model has no ground truth. Accepted for signature
                symmetry with the supervised evaluators.
            split: Split name, used in logs and artefacts.
            context: Optional raw/enriched rows aligned with ``X`` (identifier, raw values,
                diagnostic metadata), used for the cluster profiles.

        Returns:
            The :class:`EvaluationResult`.

        Raises:
            RuntimeError: When the model is not fitted.
        """
        del y  # non supervisé : aucune vérité terrain
        self.model.check_is_fitted()

        assignments = np.asarray(self.model.predict(X)).ravel()
        features = _as_matrix(X)
        if len(np.unique(assignments)) < 2:
            logger.warning(
                "Un seul groupe détecté : les indices de silhouette sont indéfinis. "
                "Vérifiez le paramètre `n_clusters` et l'échelle des features."
            )

        membership = self._membership(X)
        centroids = self._centroids(features, assignments)
        distances = self._centroid_distances(features, assignments, centroids)
        row_silhouette = self._silhouette_samples(features, assignments)

        frame = self._row_frame(
            assignments, distances, row_silhouette, membership, context, features
        )
        metrics = self._compute_metrics(features, assignments)
        profile_columns = self._profile_columns(context)
        per_cluster = self._cluster_profiles(frame, assignments, context, profile_columns)
        hardest = self._hardest_assignments(frame)
        centroids_table = self._centroid_table(X, centroids, assignments)
        curves = {
            "k_selection": self.k_selection(features).to_dict(orient="records"),
            "silhouette_histogram": _histogram_points(row_silhouette),
            "distance_histogram": _histogram_points(distances),
            "pca_map": self._pca_map(features, assignments),
        }
        external = self._external_validity(assignments, context)
        stability = self.stability(X, assignments)
        degenerate = (
            [
                int(row["cluster"])
                for _, row in per_cluster.iterrows()
                if _to_float(row.get("share", float("nan"))) < self.min_cluster_share
            ]
            if not per_cluster.empty
            else []
        )

        extras: dict[str, Any] = {
            "model": self.model.summary(),
            "n_features": int(features.shape[1]),
            "profile_columns": list(profile_columns),
            "min_cluster_share_threshold": self.min_cluster_share,
            "degenerate_clusters": degenerate,
            "external_validity": external,
            "stability": stability,
            "confidence": _confidence_summary(frame),
            "centroid_inertia": float(np.mean(np.square(distances)))
            if len(distances)
            else float("nan"),
        }
        extras.update(external)

        result = EvaluationResult(
            task=self.task,
            split=split,
            primary_metric=self.primary_metric,
            metrics=metrics,
            predictions=frame,
            labels=list(profile_columns),
            per_segment=per_cluster,
            errors=hardest,
            curves=curves,
            feature_importance=centroids_table,
            extras=extras,
            n_samples=len(X),
        )
        logger.info(
            "Evaluation on '{}' | n={} groupes={} {}={} silhouette_min_cluster={:.1%}",
            split,
            result.n_samples,
            result.n_clusters,
            self.primary_metric,
            round(result.primary_value, 4),
            result.min_cluster_share,
        )
        return result

    def compare_to_baseline(
        self, X: pd.DataFrame, *, baseline: str = "random_assignment"
    ) -> dict[str, float]:
        """Score a trivial assignment on the same data, to prove the structure is real.

        A random partition of the same size has a silhouette close to zero: it is the floor any
        honest segmentation must beat. ``single_cluster`` (everyone in one group) is degenerate and
        reported as ``NaN`` on purpose.

        Args:
            X: Preprocessed features.
            baseline: ``random_assignment`` (uniform draw) or ``single_cluster``.

        Returns:
            The baseline metrics, prefixed with ``baseline_``.
        """
        features = _as_matrix(X)
        n_clusters = max(len(np.unique(np.asarray(self.model.predict(X)).ravel())), 2)
        if baseline == "single_cluster":
            labels = np.zeros(len(features), dtype=int)
            values = {"silhouette": float("nan"), "n_clusters": 1.0}
        else:
            rng = np.random.default_rng(0)
            labels = rng.integers(0, n_clusters, size=len(features))
            calculator = MetricCalculator(task=self.task, metrics=self.metric_names)
            values = calculator.evaluate(MetricInputs(y_true=None, y_pred=labels, X=features))
        baseline_metrics = {f"baseline_{name}": _to_float(value) for name, value in values.items()}
        logger.info("Baseline '{}' | {}", baseline, baseline_metrics)
        return baseline_metrics

    def k_selection(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Explore several values of k with KMeans and score each partition.

        The number of clusters is a **choice**, not a hyper-parameter with a single optimum: the
        elbow of inertia, the silhouette peak, Davies-Bouldin and the smallest cluster size rarely
        agree. Producing the whole table makes the trade-off explicit and reproducible.

        Args:
            X: Preprocessed features.

        Returns:
            One row per k (``k``, ``inertia``, ``silhouette``, ``calinski_harabasz``,
            ``davies_bouldin``, ``min_cluster_share``).
        """
        features = _as_matrix(X)
        rows: list[dict[str, float]] = []
        for k in self.k_range:
            if k >= len(features):
                break
            estimator = KMeans(n_clusters=k, n_init=10, random_state=self.model.random_state)
            labels = estimator.fit_predict(features)
            shares = pd.Series(labels).value_counts(normalize=True)
            rows.append(
                {
                    "k": float(k),
                    "inertia": float(estimator.inertia_),
                    "silhouette": _safe_silhouette(features, labels),
                    "calinski_harabasz": _safe_calinski(features, labels),
                    "davies_bouldin": _safe_davies(features, labels),
                    "min_cluster_share": float(shares.min()) if len(shares) else float("nan"),
                }
            )
        table = pd.DataFrame(rows)
        if not table.empty:
            logger.info(
                "Sélection de k | meilleur k par silhouette={} (silhouette={:.4f})",
                int(table.loc[table["silhouette"].idxmax(), "k"]),
                float(table["silhouette"].max()),
            )
        return table

    def stability(
        self, X: pd.DataFrame | np.ndarray, reference: np.ndarray | None = None
    ) -> dict[str, Any]:
        """Measure how much the assignment moves when the random seed changes.

        A segmentation that reshuffles customers between two runs cannot drive a CRM campaign:
        the same customer would receive contradictory messages. The Adjusted Rand Index between the
        reference assignment and re-fits on other seeds quantifies that risk.

        Args:
            X: Preprocessed features.
            reference: Reference assignment (the fitted model's); recomputed when ``None``.

        Returns:
            Mapping with ``mean_ari``, ``min_ari``, ``per_seed`` and the number of re-fits.
        """
        features = _as_matrix(X)
        base = (
            np.asarray(reference).ravel()
            if reference is not None
            else np.asarray(self.model.predict(X)).ravel()
        )
        per_seed: dict[str, float] = {}
        n_clusters = len(np.unique(base))
        for seed in self.stability_seeds:
            try:
                estimator = KMeans(n_clusters=max(n_clusters, 2), n_init=10, random_state=seed)
                labels = estimator.fit_predict(features)
                per_seed[str(seed)] = float(adjusted_rand_score(base, labels))
            except ValueError as error:  # k incompatible avec la taille de l'échantillon
                logger.debug("Stabilité non mesurable pour la graine {} : {}", seed, error)
        values = list(per_seed.values())
        summary = {
            "mean_ari": float(np.mean(values)) if values else float("nan"),
            "min_ari": float(np.min(values)) if values else float("nan"),
            "per_seed": per_seed,
            "n_refits": len(values),
            "reference_seed": int(self.model.random_state),
        }
        logger.info(
            "Stabilité des affectations | ARI moyen={:.3f} min={:.3f} sur {} ré-entraînements",
            summary["mean_ari"],
            summary["min_ari"],
            summary["n_refits"],
        )
        return summary

    def save_metrics(self, result: EvaluationResult, path: str | Path | None = None) -> Path:
        """Persist the evaluation metrics as JSON.

        Args:
            result: Evaluation outcome.
            path: Destination (defaults to ``artifacts/metrics/clustering_metrics.json``).

        Returns:
            The written path.
        """
        destination = (
            Path(path) if path is not None else (self.paths.metrics_dir / "clustering_metrics.json")
        )
        payload = result.to_dict()
        payload["extras"] = {
            key: value
            for key, value in result.extras.items()
            if key in {"external_validity", "stability", "confidence", "degenerate_clusters"}
        }
        return write_json(destination, payload)

    # ------------------------------------------------------------------ internals -------
    def _compute_metrics(self, features: np.ndarray, assignments: np.ndarray) -> dict[str, float]:
        """Compute the configured metrics through the shared registry."""
        calculator = MetricCalculator(task=self.task, metrics=self.metric_names)
        values = calculator.evaluate(MetricInputs(y_true=None, y_pred=assignments, X=features))
        return {name: _to_float(value) for name, value in values.items()}

    def _membership(self, X: pd.DataFrame) -> np.ndarray | None:
        """Return per-cluster membership probabilities when the model exposes them.

        Only mixture models (GaussianMixture) produce probabilities; KMeans assigns hard labels.
        The probability of the assigned cluster is a natural confidence measure, and the gap to the
        second best probability flags borderline customers.

        Args:
            X: Preprocessed features.

        Returns:
            A ``(n_samples, n_clusters)`` array, or ``None`` when unavailable.
        """
        if not bool(getattr(self.model, "supports_proba", False)):
            return None
        try:
            probabilities = np.asarray(self.model.predict_proba(X), dtype="float64")
        except (NotImplementedError, AttributeError, ValueError) as error:
            logger.debug("Probabilités d'appartenance indisponibles : {}", error)
            return None
        return probabilities if probabilities.ndim == 2 else None

    def _centroids(self, features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Return one centre per cluster, in the preprocessed feature space.

        Args:
            features: Feature matrix.
            assignments: Cluster label of each row.

        Returns:
            A ``(n_clusters, n_features)`` array of centres (model centroids when available,
            empirical group means otherwise — e.g. for algorithms without centroids).
        """
        estimator = getattr(self.model, "estimator_", None)
        for attribute in ("cluster_centers_", "means_"):
            centres = getattr(estimator, attribute, None)
            if centres is not None:
                array = np.asarray(centres, dtype="float64")
                if array.shape[1] == features.shape[1]:
                    return array
        frame = pd.DataFrame(features)
        frame["__cluster__"] = assignments
        grouped = frame.groupby("__cluster__", observed=True).mean(numeric_only=True)
        return grouped.to_numpy(dtype="float64")

    @staticmethod
    def _centroid_distances(
        features: np.ndarray, assignments: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """Euclidean distance of each row to the centre of its own cluster.

        Args:
            features: Feature matrix (standardised space).
            assignments: Cluster label of each row.
            centroids: Centre of each cluster, indexed by label.

        Returns:
            A ``(n_samples,)`` array of distances.
        """
        if centroids.size == 0:
            return np.full(len(features), np.nan)
        index = {
            int(label): position for position, label in enumerate(sorted(np.unique(assignments)))
        }
        positions = np.array([index.get(int(label), 0) for label in assignments])
        centres = centroids[np.clip(positions, 0, len(centroids) - 1)]
        return np.linalg.norm(features - centres, axis=1)

    @staticmethod
    def _silhouette_samples(features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Per-row silhouette (how well each customer fits its cluster).

        Args:
            features: Feature matrix.
            assignments: Cluster label of each row.

        Returns:
            A ``(n_samples,)`` array, all ``NaN`` when fewer than two clusters exist.
        """
        if len(np.unique(assignments)) < 2 or len(features) < 3:
            return np.full(len(features), np.nan)
        try:
            return np.asarray(silhouette_samples(features, assignments), dtype="float64")
        except ValueError as error:
            logger.debug("Silhouette par individu indisponible : {}", error)
            return np.full(len(features), np.nan)

    def _row_frame(
        self,
        assignments: np.ndarray,
        distances: np.ndarray,
        row_silhouette: np.ndarray,
        membership: np.ndarray | None,
        context: pd.DataFrame | None,
        features: np.ndarray,
    ) -> pd.DataFrame:
        """Assemble the row-level diagnostic frame.

        Args:
            assignments: Cluster label of each row.
            distances: Distance to the assigned centre.
            row_silhouette: Per-row silhouette.
            membership: Membership probabilities when available.
            context: Enriched raw rows (identifier, raw values, diagnostics).
            features: Feature matrix (used for the distance to the second-best cluster).

        Returns:
            A frame with one row per customer.
        """
        frame = pd.DataFrame(
            {
                "cluster": np.asarray(assignments, dtype="int64"),
                "distance_to_centroid": np.asarray(distances, dtype="float64"),
                "silhouette": np.asarray(row_silhouette, dtype="float64"),
            }
        )
        frame["confidence"] = _confidence_from_silhouette(frame["silhouette"].to_numpy())
        if membership is not None and membership.shape[0] == len(frame):
            ordered = np.sort(membership, axis=1)
            frame["membership"] = ordered[:, -1]
            frame["second_best"] = ordered[:, -2] if ordered.shape[1] > 1 else np.nan
            frame["membership_gap"] = frame["membership"] - frame["second_best"]
        else:
            frame["second_best_distance"] = _second_best_distance(features, assignments)
        if context is not None and len(context) == len(frame):
            kept = [
                name
                for name in dict.fromkeys(
                    [
                        *self.context_columns,
                        LATENT_COLUMN,
                        OUTCOME_COLUMN,
                        *self._raw_columns(context),
                    ]
                )
                if name in context.columns
            ]
            for name in kept:
                frame[name] = np.asarray(context[name].to_numpy(), dtype=object)
        return frame

    @staticmethod
    def _raw_columns(context: pd.DataFrame) -> list[str]:
        """Raw numeric/categorical columns worth copying into the diagnostic frame."""
        selected: list[str] = []
        for name in context.columns:
            if name in {LATENT_COLUMN, OUTCOME_COLUMN}:
                continue
            series = context[name]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_string_dtype(series):
                selected.append(str(name))
            if len(selected) >= 12:
                break
        return selected

    def _profile_columns(self, context: pd.DataFrame | None) -> list[str]:
        """Raw columns used to profile the clusters (readable units, not standardised features).

        Args:
            context: Enriched raw rows.

        Returns:
            The column names available for profiling.
        """
        if context is None:
            return []
        columns = [
            str(name)
            for name in context.columns
            if name not in {LATENT_COLUMN, OUTCOME_COLUMN, *self.context_columns}
            and (
                pd.api.types.is_numeric_dtype(context[name])
                or pd.api.types.is_string_dtype(context[name])
            )
            and not pd.api.types.is_datetime64_any_dtype(context[name])
        ]
        return columns

    def _cluster_profiles(
        self,
        frame: pd.DataFrame,
        assignments: np.ndarray,
        context: pd.DataFrame | None,
        profile_columns: Sequence[str],
    ) -> pd.DataFrame:
        """Build one profile row per cluster.

        Args:
            frame: Row-level diagnostic frame.
            assignments: Cluster label of each row.
            context: Enriched raw rows (raw values, diagnostics).
            profile_columns: Raw columns to summarise.

        Returns:
            A frame indexed by cluster with size, share, mean raw values, distance statistics,
            churn rate and the dominant latent profile.
        """
        labels = pd.Series(np.asarray(assignments, dtype="int64"), name="cluster")
        total = max(len(labels), 1)
        rows: list[dict[str, Any]] = []
        source = context.reset_index(drop=True) if context is not None else None
        for cluster, group in labels.groupby(labels, observed=True):
            positions = group.index.to_numpy()
            row: dict[str, Any] = {
                "cluster": int(cluster),
                "size": len(positions),
                "share": float(len(positions)) / float(total),
                "mean_distance": float(frame.loc[positions, "distance_to_centroid"].mean()),
                "mean_silhouette": float(frame.loc[positions, "silhouette"].mean()),
            }
            if source is not None:
                subset = source.iloc[positions]
                for name in profile_columns:
                    if name not in subset.columns:
                        continue
                    series = subset[name]
                    if pd.api.types.is_numeric_dtype(series):
                        row[f"mean_{name}"] = float(series.mean())
                    elif pd.api.types.is_string_dtype(series):
                        counts = series.value_counts(normalize=True)
                        if len(counts):
                            row[f"top_{name}"] = str(counts.index[0])
                            row[f"top_{name}_share"] = float(counts.iloc[0])
                if OUTCOME_COLUMN in subset.columns:
                    row["churn_rate"] = float(
                        pd.to_numeric(subset[OUTCOME_COLUMN], errors="coerce").mean()
                    )
                if LATENT_COLUMN in subset.columns:
                    counts = subset[LATENT_COLUMN].value_counts(normalize=True)
                    if len(counts):
                        row["dominant_latent"] = str(counts.index[0])
                        row["latent_purity"] = float(counts.iloc[0])
            rows.append(row)
        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values("size", ascending=False).reset_index(drop=True)
        return table

    def _external_validity(
        self, assignments: np.ndarray, context: pd.DataFrame | None
    ) -> dict[str, Any]:
        """Compare the discovered groups with the diagnostic metadata.

        Two distinct questions are answered here:

        * **agreement with la structure latente** (ARI, NMI, V-measure) : le générateur connaît les
          profils injectés, on vérifie que l'algorithme les retrouve. En production cette colonne
          n'existe pas — c'est un outil pédagogique ;
        * **validité externe** : les groupes séparent-ils un comportement observé après coup
          (churn à 90 jours) ? Une segmentation dont les groupes ont tous le même taux de churn
          n'apporte rien au métier, même avec une belle silhouette.

        Args:
            assignments: Cluster label of each row.
            context: Enriched raw rows.

        Returns:
            Mapping of diagnostic values (``NaN`` when the column is absent).
        """
        result: dict[str, Any] = {
            "ari_latent": float("nan"),
            "nmi_latent": float("nan"),
            "v_measure_latent": float("nan"),
            "churn_by_cluster": {},
            "churn_spread": float("nan"),
            "churn_best": float("nan"),
            "churn_worst": float("nan"),
        }
        if context is None:
            return result
        frame = context.reset_index(drop=True)
        labels = pd.Series(np.asarray(assignments, dtype="int64"))
        if LATENT_COLUMN in frame.columns:
            latent = frame[LATENT_COLUMN].astype("string")
            mask = latent.notna().to_numpy()
            if mask.sum() > 10 and labels.nunique() > 1:
                latent_codes = pd.factorize(latent[mask])[0]
                observed = labels[mask].to_numpy()
                result["ari_latent"] = float(adjusted_rand_score(latent_codes, observed))
                result["nmi_latent"] = float(normalized_mutual_info_score(latent_codes, observed))
                result["v_measure_latent"] = float(v_measure_score(latent_codes, observed))
        if OUTCOME_COLUMN in frame.columns:
            outcome = pd.to_numeric(frame[OUTCOME_COLUMN], errors="coerce")
            grouped = outcome.groupby(labels).mean().dropna()
            if len(grouped) > 1:
                result["churn_by_cluster"] = {
                    str(int(key)): round(float(value), 4) for key, value in grouped.items()
                }
                result["churn_best"] = float(grouped.min())
                result["churn_worst"] = float(grouped.max())
                result["churn_spread"] = float(grouped.max() - grouped.min())
        return result

    def _hardest_assignments(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the customers whose assignment is the least reliable.

        Args:
            frame: Row-level diagnostic frame.

        Returns:
            The ``top_k_errors`` rows with the lowest silhouette (borderline customers).
        """
        if frame.empty or "silhouette" not in frame.columns:
            return frame.head(0)
        ranked = frame.sort_values("silhouette", ascending=True, na_position="last")
        return ranked.head(self.top_k_errors).reset_index(drop=True)

    @staticmethod
    def _centroid_table(
        X: pd.DataFrame, centroids: np.ndarray, assignments: np.ndarray
    ) -> pd.DataFrame:
        """Express each centroid in the standardised feature space.

        Because the preprocessing standardises the features, a centroid coordinate **is** an effect
        size: ``+1.4`` on ``discount_share`` means "1.4 standard deviations above the average
        customer". Sorting by absolute value gives the drivers that define each cluster.

        Args:
            X: Preprocessed features (column names).
            centroids: Centre of each cluster.
            assignments: Cluster label of each row.

        Returns:
            A frame with one row per (cluster, feature) pair plus the rank of the driver.
        """
        if centroids.size == 0:
            return pd.DataFrame()
        names = [str(name) for name in X.columns][: centroids.shape[1]]
        unique = sorted({int(label) for label in assignments})
        records: list[dict[str, Any]] = []
        for position, cluster in enumerate(unique):
            if position >= len(centroids):
                break
            for index, name in enumerate(names):
                value = float(centroids[position][index])
                records.append({"cluster": cluster, "feature": name, "centroid_z": value})
        table = pd.DataFrame(records)
        if table.empty:
            return table
        table["abs_z"] = table["centroid_z"].abs()
        table["rank"] = (
            table.groupby("cluster", observed=True)["abs_z"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        return table.sort_values(["cluster", "rank"]).reset_index(drop=True)

    @staticmethod
    def _pca_map(features: np.ndarray, assignments: np.ndarray) -> dict[str, Any]:
        """Project the customers on their first two principal components (for the report figures).

        Args:
            features: Preprocessed feature matrix.
            assignments: Cluster label of each row.

        Returns:
            Mapping with ``x``, ``y``, ``cluster`` (down-sampled) and the explained variance.
        """
        from sklearn.decomposition import PCA

        if len(features) < 10 or features.shape[1] < 2:
            return {"x": [], "y": [], "cluster": [], "explained_variance": []}
        sample_size = min(len(features), MAX_SCATTER_POINTS)
        rng = np.random.default_rng(0)
        positions = (
            np.arange(len(features))
            if sample_size == len(features)
            else rng.choice(len(features), size=sample_size, replace=False)
        )
        try:
            pca = PCA(n_components=2, random_state=0)
            projected = pca.fit_transform(features)
        except ValueError as error:
            logger.debug("Projection PCA indisponible : {}", error)
            return {"x": [], "y": [], "cluster": [], "explained_variance": []}
        return {
            "x": [round(float(value), 4) for value in projected[positions, 0]],
            "y": [round(float(value), 4) for value in projected[positions, 1]],
            "cluster": [int(value) for value in np.asarray(assignments)[positions]],
            "explained_variance": [
                round(float(value), 4) for value in pca.explained_variance_ratio_
            ],
        }


# -------------------------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------------------------
def _as_matrix(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert a feature frame into a float matrix."""
    if isinstance(X, np.ndarray):
        return np.asarray(X, dtype="float64")
    return X.to_numpy(dtype="float64")


def _second_best_distance(features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    """Distance of each row to the centre of the *closest other* cluster.

    Args:
        features: Preprocessed feature matrix.
        assignments: Cluster label of each row.

    Returns:
        A ``(n_samples,)`` array (``NaN`` with fewer than two clusters).
    """
    frame = pd.DataFrame(features)
    frame["__cluster__"] = assignments
    unique = sorted({int(label) for label in assignments})
    if len(unique) < 2:
        return np.full(len(features), np.nan)
    centres = frame.groupby("__cluster__", observed=True).mean(numeric_only=True).to_numpy()
    labels = np.array([unique.index(int(label)) for label in assignments])
    deltas = features[:, None, :] - centres[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    distances[np.arange(len(features)), labels] = np.inf
    return distances.min(axis=1)


def _confidence_from_silhouette(values: np.ndarray) -> np.ndarray:
    """Map a per-row silhouette onto a three-level confidence label.

    Args:
        values: Per-row silhouette.

    Returns:
        An array of labels (``élevée``, ``moyenne``, ``faible``).
    """
    array = np.asarray(values, dtype="float64")
    labels = np.where(array >= 0.25, "élevée", np.where(array >= 0.0, "moyenne", "faible"))
    return np.where(np.isnan(array), "indéterminée", labels)


def _confidence_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Share of customers per confidence level."""
    if frame.empty or "confidence" not in frame.columns:
        return {}
    shares = frame["confidence"].value_counts(normalize=True)
    return {str(key): round(float(value), 4) for key, value in shares.items()}


def _histogram_points(values: np.ndarray, bins: int = 40) -> dict[str, Any]:
    """Down-sample a distribution into histogram points (JSON friendly)."""
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(array, bins=min(bins, max(int(np.sqrt(array.size)), 5)))
    return {
        "edges": [round(float(edge), 4) for edge in edges],
        "counts": [int(count) for count in counts],
    }


def _safe_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette that degrades to ``NaN`` instead of raising."""
    if len(np.unique(labels)) < 2 or len(features) < 3:
        return float("nan")
    try:
        return float(silhouette_score(features, labels))
    except ValueError:
        return float("nan")


def _safe_calinski(features: np.ndarray, labels: np.ndarray) -> float:
    """Calinski-Harabasz index that degrades to ``NaN`` instead of raising."""
    if len(np.unique(labels)) < 2 or len(features) < 3:
        return float("nan")
    try:
        return float(calinski_harabasz_score(features, labels))
    except ValueError:
        return float("nan")


def _safe_davies(features: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin index that degrades to ``NaN`` instead of raising."""
    if len(np.unique(labels)) < 2 or len(features) < 3:
        return float("nan")
    try:
        return float(davies_bouldin_score(features, labels))
    except ValueError:
        return float("nan")


def _to_float(value: Any) -> float:
    """Coerce a metric value to ``float`` (``NaN`` when not numeric)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


__all__ = ["K_RANGE", "MIN_CLUSTER_SHARE", "EvaluationResult", "Evaluator"]
