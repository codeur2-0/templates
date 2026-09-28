"""Inference: assign new customers to the trained segmentation.

The predictor is the *production facing* object of the project. It:

1. loads the artefacts written by training (model, preprocessing, feature builder),
2. validates the incoming payload with ``InferenceDataSchema`` (fail fast, explicit message),
3. rebuilds the exact same features and transformations as during training,
4. returns a business-readable frame: **cluster, segment name, recommended action, membership
   probability, distance to the centre, margin to the second-best cluster, confidence level** and
   the drivers that justify the assignment.

Assigning a cluster identifier alone would be a product mistake: the CRM needs a readable segment
name, an action, and above all a **confidence level**. Borderline customers (small margin between
the two closest clusters) must fall back to a generic campaign rather than receive a message tuned
for a profile they do not really belong to.

Nothing here knows how the segmentation was trained: swapping KMeans for a Gaussian mixture only
changes ``src/models/model.py`` (probabilities become available and are used automatically).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.generators import DEFAULT_DATASET_NAME, SyntheticDataGenerator
from src.data.loaders import InferenceDataLoader
from src.data.schemas import InferenceDataSchema
from src.features.build_features import FeatureBuilder
from src.models import load_model
from src.models.base import BaseModel
from src.preprocessing.pipelines import PreprocessingPipeline
from src.utils.io import load_pickle, write_table
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Identifiant métier, conservé dans les affectations pour la traçabilité.
ID_COLUMN = "customer_id"

#: Nombre de drivers cités dans la justification d'une affectation.
N_DRIVERS = 2

#: Seuils de confiance (marge relative entre le centre le plus proche et le suivant).
CONFIDENCE_BOUNDS: tuple[tuple[float, str], ...] = (
    (0.60, "élevée"),
    (0.20, "moyenne"),
    (-np.inf, "faible"),
)

#: Seuil de probabilité d'appartenance au-dessus duquel la confiance est « élevée » (modèles
#: probabilistes uniquement : KMeans n'a pas de probabilité).
MEMBERSHIP_HIGH = 0.75

#: Colonnes de contexte conservées dans le fichier d'affectations.
CONTEXT_COLUMNS: tuple[str, ...] = (
    "customer_id",
    "tenure_months",
    "recency_days",
    "orders_12m",
    "revenue_12m_eur",
    "avg_basket_eur",
    "distinct_categories_12m",
)


class Predictor:
    """Batch and single-record cluster assignment, with validation, confidence and drivers."""

    def __init__(
        self,
        *,
        model: BaseModel,
        preprocessing: PreprocessingPipeline,
        feature_builder: FeatureBuilder | None = None,
        config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        segment_labels: Mapping[str, Mapping[str, str]] | None = None,
        review_confidence: str = "faible",
    ) -> None:
        """Inject the trained artefacts.

        Args:
            model: Fitted clustering model.
            preprocessing: Fitted preprocessing pipeline.
            feature_builder: Fitted feature builder (``None`` when no derived feature is used).
            config: Root configuration mapping (used for paths, seed and options).
            paths: Project layout.
            segment_labels: Optional business naming —
                ``{cluster_id: {"name": ..., "action": ...}}``.
                declared in ``conf/config.yaml`` (node ``segmentation.labels``). Without it, a
                data-driven name is derived from the cluster centroid.
            review_confidence: Confidence level below which the customer is routed to a manual
                review instead of an automated campaign.

        Raises:
            ValueError: When ``review_confidence`` is not a known level.
        """
        known = {level for _, level in CONFIDENCE_BOUNDS}
        if review_confidence not in known:
            msg = f"review_confidence doit valoir {sorted(known)}, reçu {review_confidence!r}"
            raise ValueError(msg)
        self.model = model
        self.preprocessing = preprocessing
        self.feature_builder = feature_builder or FeatureBuilder([])
        self.config: dict[str, Any] = dict(config or {})
        self.paths = paths or ProjectPaths.from_config(self.config)
        self.segment_labels: dict[str, dict[str, str]] = {
            str(key): dict(value) for key, value in dict(segment_labels or {}).items()
        }
        self.review_confidence = review_confidence
        #: Centroïdes dans l'espace pré-traité (une ligne par cluster), calculés à la première
        #: affectation ; ils servent aux distances, à la marge et aux drivers.
        self._centroids: np.ndarray | None = None
        self._feature_names: tuple[str, ...] = ()
        #: Correspondance identifiant de cluster -> ligne de :attr:`_centroids`. Les estimateurs
        #: sklearn numérotent leurs centres de 0 à k-1 ; le repli « moyennes par groupe » conserve
        #: l'ordre des identifiants présents dans le lot. Cette table rend les deux cas explicites.
        self._label_rows: dict[int, int] = {}

    # ------------------------------------------------------------------ factories -------
    @classmethod
    def from_config(cls, config: Any, paths: ProjectPaths | None = None) -> Predictor:
        """Build a predictor from a validated application configuration.

        Args:
            config: ``AppConfig`` instance (or mapping).
            paths: Optional project layout.

        Returns:
            The predictor, with every artefact loaded from disk.

        Raises:
            FileNotFoundError: When an artefact is missing (training never ran).
        """
        payload: Mapping[str, Any] = (
            config.model_dump() if hasattr(config, "model_dump") else dict(config or {})
        )
        layout = paths or ProjectPaths.from_config(payload)
        artifacts = (payload.get("train") or {}).get("artifacts") or {}
        # Le nommage métier des segments vit dans le noeud `segmentation` de `conf/config.yaml`
        # (injecté par le manifeste) ; le noeud standard `predict` sert de repli. Rien n'est codé
        # en dur dans le code source.
        segmentation_node = dict(payload.get("segmentation") or {})
        predict_node = {**dict(payload.get("predict") or {}), **segmentation_node}
        labels_node = dict(predict_node.get("labels") or {})

        model_path = layout.models_dir / str(artifacts.get("model_file", "model.joblib"))
        pipeline_path = layout.models_dir / str(
            artifacts.get("pipeline_file", "preprocessing.joblib")
        )
        builder_path = layout.models_dir / str(
            artifacts.get("feature_builder_file", "feature_builder.joblib")
        )

        for required in (model_path, pipeline_path):
            if not required.exists():
                msg = (
                    f"Artefact manquant : {required}. Entraînez d'abord le modèle avec "
                    "`python scripts/train.py` (ou `make train`)."
                )
                raise FileNotFoundError(msg)

        model = load_model(model_path, config=config)
        preprocessing = PreprocessingPipeline.load(pipeline_path)
        feature_builder = load_pickle(builder_path) if builder_path.exists() else FeatureBuilder([])

        logger.info(
            "Predictor ready | model={} features={} segments_nommés={}",
            model.summary(),
            len(preprocessing.feature_names_out),
            len(labels_node),
        )
        return cls(
            model=model,
            preprocessing=preprocessing,
            feature_builder=feature_builder,
            config=payload,
            paths=layout,
            segment_labels=labels_node,
            review_confidence=str(predict_node.get("review_confidence") or "faible"),
        )

    # ------------------------------------------------------------------ inputs ----------
    def load_inputs(self, path: str | Path) -> pd.DataFrame:
        """Load and validate an inference file.

        Args:
            path: Parquet / CSV / JSON file.

        Returns:
            The validated frame.
        """
        loader = InferenceDataLoader(
            self.paths,
            dataset_name=str((self.config.get("data") or {}).get("dataset_name", "inference")),
            validate=bool(
                ((self.config.get("data") or {}).get("validation") or {}).get("inference", True)
            ),
        )
        return loader.load_from(path)

    def sample_inputs(self, n_samples: int = 5, *, seed: int | None = None) -> pd.DataFrame:
        """Generate a synthetic inference payload (demo without preparing a file).

        Args:
            n_samples: Number of records.
            seed: Optional seed override.

        Returns:
            A frame without the diagnostic metadata (a real scoring request has no ground truth).
        """
        data_node = dict(self.config.get("data") or {})
        generator = SyntheticDataGenerator(
            n_samples=max(int(n_samples) * 4, 50),
            # ``config.get`` renvoie ``Any | None`` : on explicite le repli pour satisfaire mypy.
            seed=int(
                seed
                if seed is not None
                else (self.config.get("seed") or data_node.get("seed") or 42)
            ),
            dataset_name=str(data_node.get("dataset_name", DEFAULT_DATASET_NAME)),
        )
        frame = generator.sample(max(int(n_samples), 1), with_target=False)
        return InferenceDataSchema.validate(frame, lazy=False)

    # ------------------------------------------------------------------ prediction ------
    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering and preprocessing to a validated payload.

        Args:
            frame: Raw records (validated).

        Returns:
            The model-ready matrix.
        """
        enriched = self.feature_builder.transform(frame)
        matrix = self.preprocessing.transform(enriched)
        logger.debug("Prepared {} record(s) -> {} feature(s)", len(matrix), matrix.shape[1])
        return matrix

    def predict(self, frame: pd.DataFrame, *, validate: bool = True) -> pd.DataFrame:
        """Assign a batch of customers to the segmentation.

        Args:
            frame: Raw records.
            validate: Enforce ``InferenceDataSchema`` before predicting.

        Returns:
            A frame with the context columns, ``cluster``, ``segment_name``,
            ``recommended_action``, ``distance_to_centroid``, ``margin``, ``confidence``,
            ``review_required`` and ``assignment_reason`` (plus ``membership`` when the model
            produces probabilities).

        Raises:
            RuntimeError: When the model is not fitted.
        """
        self.model.check_is_fitted()
        payload = InferenceDataSchema.validate(frame, lazy=False) if validate else frame.copy()

        matrix = self.prepare(payload)
        features = matrix.to_numpy(dtype="float64")
        assignments = np.asarray(self.model.predict(matrix)).ravel().astype("int64")
        centroids = self._resolve_centroids(features, assignments)
        membership = self._membership(matrix)

        distances, margins = self._distances(features, assignments, centroids)
        output = self._context_frame(payload).reset_index(drop=True)
        output["cluster"] = assignments
        if membership is not None:
            output["membership"] = membership
        output["distance_to_centroid"] = distances
        output["margin"] = margins
        output["confidence"] = self._confidence(distances, margins, membership)
        output["segment_name"] = [self._segment_name(int(value)) for value in assignments]
        output["recommended_action"] = [self._segment_action(int(value)) for value in assignments]
        output["review_required"] = output["confidence"].isin(
            _levels_at_or_below(self.review_confidence)
        )
        output["assignment_reason"] = [
            self._assignment_reason(row, centroids) for _, row in output.iterrows()
        ]

        logger.info(
            "Affectations | rows={} groupes={} confiance_faible={:.1%} revue_manuelle={}",
            len(output),
            int(output["cluster"].nunique()),
            float((output["confidence"] == "faible").mean()) if len(output) else float("nan"),
            int(output["review_required"].sum()),
        )
        return output

    def predict_one(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Assign a single customer and return a JSON-serialisable payload.

        Args:
            record: Mapping of column name to value.

        Returns:
            The assignment, readable by an API consumer.
        """
        frame = pd.DataFrame([dict(record)])
        output = self.predict(frame)
        row = output.iloc[0]
        return {key: _jsonable(value) for key, value in row.items()}

    def save(self, predictions: pd.DataFrame, path: str | Path | None = None) -> Path:
        """Persist the assignments.

        Args:
            predictions: Frame produced by :meth:`predict`.
            path: Destination (defaults to ``artifacts/reports/segment_assignments.csv``).

        Returns:
            The written path.
        """
        destination = (
            Path(path) if path is not None else (self.paths.reports_dir / "segment_assignments.csv")
        )
        return write_table(predictions, destination)

    # ------------------------------------------------------------------ internals -------
    def _resolve_centroids(self, features: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        """Return one centre per cluster, in the preprocessed feature space.

        The model's own centroids are used when available (KMeans, GaussianMixture). Otherwise the
        centres are estimated from the payload itself, which is documented in the log: an
        algorithm without centroids (hierarchical, DBSCAN) can still be scored, at the cost of a
        payload-dependent margin.

        Args:
            features: Feature matrix of the payload.
            assignments: Cluster label of each row.

        Returns:
            A ``(n_clusters, n_features)`` array of centres.
        """
        if self._centroids is not None:
            return self._centroids
        estimator = getattr(self.model, "estimator_", None)
        for attribute in ("cluster_centers_", "means_"):
            centres = getattr(estimator, attribute, None)
            if centres is not None:
                array = np.asarray(centres, dtype="float64")
                if array.ndim == 2 and array.shape[1] == features.shape[1]:
                    self._centroids = array
                    self._label_rows = {row: row for row in range(len(array))}
                    self._feature_names = tuple(
                        str(name) for name in self.preprocessing.feature_names_out
                    )
                    return array
        logger.warning(
            "L'estimateur n'expose pas de centroïdes : les distances sont estimées sur le lot "
            "courant ({} ligne(s)). La marge de confiance reste indicative.",
            len(features),
        )
        frame = pd.DataFrame(features)
        frame["__cluster__"] = assignments
        grouped = frame.groupby("__cluster__", observed=True).mean(numeric_only=True)
        self._centroids = grouped.to_numpy(dtype="float64")
        self._label_rows = {
            int(label): row for row, label in enumerate(sorted(grouped.index.tolist()))
        }
        self._feature_names = tuple(str(name) for name in grouped.columns)
        return self._centroids

    def _membership(self, matrix: pd.DataFrame) -> np.ndarray | None:
        """Probability of the assigned cluster, when the model produces probabilities."""
        if not bool(getattr(self.model, "supports_proba", False)):
            return None
        try:
            probabilities = np.asarray(self.model.predict_proba(matrix), dtype="float64")
            assignments = np.asarray(self.model.predict(matrix)).ravel().astype("int64")
        except (NotImplementedError, AttributeError, ValueError) as error:
            logger.debug("Probabilités d'appartenance indisponibles : {}", error)
            return None
        if probabilities.ndim != 2 or probabilities.shape[0] != len(assignments):
            return None
        columns = np.clip(assignments, 0, probabilities.shape[1] - 1)
        return probabilities[np.arange(len(assignments)), columns]

    def _distances(
        self, features: np.ndarray, assignments: np.ndarray, centroids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Distance to the assigned centre and margin to the closest other centre.

        Args:
            features: Feature matrix.
            assignments: Cluster label of each row.
            centroids: One centre per cluster (indexed by sorted label).

        Returns:
            The ``(distance, margin)`` pair. The margin is positive when the assignment is clear.
        """
        if centroids.size == 0:
            empty = np.full(len(features), np.nan)
            return empty, empty
        mapping = self._label_rows or {row: row for row in range(len(centroids))}
        positions = np.array([mapping.get(int(label), 0) for label in assignments])
        deltas = features[:, None, :] - centroids[None, :, :]
        all_distances = np.linalg.norm(deltas, axis=2)
        own = all_distances[np.arange(len(features)), positions]
        masked = all_distances.copy()
        masked[np.arange(len(features)), positions] = np.inf
        nearest_other = (
            masked.min(axis=1) if masked.shape[1] > 1 else np.full(len(features), np.inf)
        )
        return own, np.asarray(nearest_other, dtype="float64") - own

    def _confidence(
        self,
        distances: np.ndarray,
        margins: np.ndarray,
        membership: np.ndarray | None,
    ) -> list[str]:
        """Map the assignment geometry onto a three-level confidence label.

        The relative margin (``margin / distance``) says how far a customer sits from the frontier
        of the second-best cluster; a mixture model's membership probability says the same thing in
        probability space. Both are combined: the highest of the two readings wins, so a
        probabilistic model is never penalised by geometry alone.

        Args:
            distances: Distance to the assigned centre.
            margins: Margin to the closest other centre.
            membership: Probability of the assigned cluster (``None`` for hard assignments).

        Returns:
            One confidence label per row.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(distances > 0, margins / np.maximum(distances, 1e-9), margins)
        labels: list[str] = []
        for position, value in enumerate(np.nan_to_num(ratio, nan=-np.inf)):
            level = "faible"
            for threshold, name in CONFIDENCE_BOUNDS:
                if value >= threshold:
                    level = name
                    break
            if membership is not None and float(membership[position]) >= MEMBERSHIP_HIGH:
                level = "élevée"
            labels.append(level)
        return labels

    def _segment_name(self, cluster: int) -> str:
        """Business name of a cluster (configured, or derived from its centroid).

        Args:
            cluster: Cluster identifier.

        Returns:
            A readable segment name.
        """
        configured = self.segment_labels.get(str(cluster), {})
        name = configured.get("name")
        if name:
            return str(name)
        drivers = self._top_drivers(cluster)
        if drivers:
            signature = " · ".join(f"{feature} {sign}" for feature, _, sign in drivers)
            return f"segment {cluster} — {signature}"
        return f"segment {cluster}"

    def _segment_action(self, cluster: int) -> str:
        """CRM action attached to a cluster (configured, or explicitly unset).

        Args:
            cluster: Cluster identifier.

        Returns:
            The recommended action.
        """
        configured = self.segment_labels.get(str(cluster), {})
        return str(configured.get("action") or "à qualifier avec le CRM")

    def _top_drivers(self, cluster: int) -> list[tuple[str, float, str]]:
        """Features that define a cluster the most, in standardised space.

        Args:
            cluster: Cluster identifier.

        Returns:
            Up to :data:`N_DRIVERS` ``(feature, z, sign)`` tuples, sorted by ``|z|``.
        """
        centroids = self._centroids
        if centroids is None or not self._feature_names:
            return []
        row = self._label_rows.get(int(cluster))
        if row is None or row >= len(centroids):
            return []
        values = np.asarray(centroids[row], dtype="float64")
        order = np.argsort(-np.abs(values))[:N_DRIVERS]
        return [
            (
                self._feature_names[int(position)],
                float(values[int(position)]),
                "+" if values[int(position)] >= 0 else "-",
            )
            for position in order
            if np.isfinite(values[int(position)])
        ]

    def _assignment_reason(self, row: pd.Series, centroids: np.ndarray) -> str:
        """One-sentence justification of an assignment, readable by the CRM.

        Args:
            row: Prediction row (cluster, distance, margin, confidence, optional membership).
            centroids: Cluster centres (used for the drivers).

        Returns:
            The justification text.
        """
        del centroids  # les drivers passent par :meth:`_top_drivers`
        cluster = int(row["cluster"])
        distance = _float(row.get("distance_to_centroid"))
        margin = _float(row.get("margin"))
        confidence = str(row.get("confidence", "indéterminée"))
        membership = _float(row.get("membership"))
        parts = [f"segment {cluster} · distance {distance:,.2f} · marge {margin:,.2f}"]
        if np.isfinite(membership):
            parts.append(f"appartenance {membership:.0%}")
        parts.append(f"confiance {confidence}")
        drivers = self._top_drivers(cluster)
        if drivers:
            signature = ", ".join(
                f"{feature} {sign}{abs(value):,.1f} ET" for feature, value, sign in drivers
            )
            parts.append(f"profil : {signature}")
        return " • ".join(parts)

    def _context_frame(self, payload: pd.DataFrame) -> pd.DataFrame:
        """Keep the business-readable columns next to the assignment."""
        kept = [name for name in CONTEXT_COLUMNS if name in payload.columns]
        return payload[kept].copy() if kept else payload.head(0).copy()


def _levels_at_or_below(level: str) -> list[str]:
    """Confidence levels considered as weak as (or weaker than) ``level``.

    Args:
        level: Threshold level declared in the configuration.

    Returns:
        The levels routing a customer to a manual review.
    """
    ordered = [name for _, name in CONFIDENCE_BOUNDS]
    if level not in ordered:
        return [ordered[-1]]
    return ordered[ordered.index(level) :]


def _jsonable(value: Any) -> Any:
    """Coerce a numpy/pandas scalar into a JSON-serialisable value."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _float(value: Any) -> float:
    """Coerce a value to ``float`` (``NaN`` when not numeric)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


__all__ = ["Predictor"]
