"""Inference: estimate the target value of new records with the trained artefacts.

The predictor is the *production facing* object of the project. It:

1. loads the artefacts written by training (model, preprocessing, feature builder),
2. validates the incoming payload with ``InferenceDataSchema`` (fail fast, explicit message),
3. rebuilds the exact same features and transformations as during training,
4. returns a business-readable frame: **estimate, published range, unit price, confidence level**
   and the main drivers of the estimate.

Publishing a single number would be a product mistake: the range and the confidence level are
part of the contract with the user. The range widens automatically when the local market is thin,
which is the inference-time counterpart of the heteroscedasticity measured during evaluation.

Nothing here knows how the model was trained: swapping scikit-learn for PyTorch only changes
``src/models/model.py``.
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

#: Colonne cible (absente des payloads d'inférence, présente en évaluation).
TARGET_NAME = "price_eur"

#: Identifiant métier, conservé dans les prédictions pour la traçabilité.
ID_COLUMN = "property_id"

#: Fourchette relative publiée par défaut (± %), écrasable via ``predict.tolerance_pct``.
DEFAULT_TOLERANCE_PCT = 10.0

#: Élargissement de la fourchette selon le niveau de confiance (marché fin = incertitude forte).
CONFIDENCE_BAND_FACTORS: tuple[tuple[str, float], ...] = (
    ("élevée", 1.0),
    ("moyenne", 1.25),
    ("faible", 1.6),
)

#: Seuils de profondeur de marché (nombre de comparables) déterminant le niveau de confiance.
CONFIDENCE_BOUNDS: tuple[tuple[float, str], ...] = (
    (30.0, "élevée"),
    (10.0, "moyenne"),
    (0.0, "faible"),
)

#: Colonnes candidates pour la profondeur de marché (détection si non configurée).
DEPTH_COLUMN_CANDIDATES: tuple[str, ...] = (
    "recent_sales_1km",
    "comparable_sales",
    "market_depth",
    "neighbourhood_sales",
)

#: Colonnes candidates pour le prix unitaire (surface, volume, …).
UNIT_COLUMN_CANDIDATES: tuple[str, ...] = ("surface_m2", "surface", "area_m2", "volume_m3")

#: Colonnes de contexte conservées dans le fichier de prédictions.
CONTEXT_COLUMNS: tuple[str, ...] = (
    "property_id",
    "surface_m2",
    "rooms",
    "floor_level",
    "has_elevator",
    "has_outdoor_space",
    "building_year",
)


class Predictor:
    """Batch and single-record estimation, with validation, range and light explainability."""

    def __init__(
        self,
        *,
        model: BaseModel,
        preprocessing: PreprocessingPipeline,
        feature_builder: FeatureBuilder | None = None,
        config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
        depth_column: str | None = None,
        unit_column: str | None = None,
    ) -> None:
        """Inject the trained artefacts.

        Args:
            model: Fitted model.
            preprocessing: Fitted preprocessing pipeline.
            feature_builder: Fitted feature builder (``None`` when no derived feature is used).
            config: Root configuration mapping (used for paths, seed and options).
            paths: Project layout.
            tolerance_pct: Relative width of the published range (± percent of the estimate).
            depth_column: Column carrying the market depth (drives the confidence level).
            unit_column: Column used to express a unit price (surface, volume, ...).

        Raises:
            ValueError: When the tolerance is not a strictly positive percentage.
        """
        if tolerance_pct <= 0.0:
            msg = f"tolerance_pct must be > 0, got {tolerance_pct}"
            raise ValueError(msg)
        self.model = model
        self.preprocessing = preprocessing
        self.feature_builder = feature_builder or FeatureBuilder([])
        self.config: dict[str, Any] = dict(config or {})
        self.paths = paths or ProjectPaths.from_config(self.config)
        self.tolerance_pct = float(tolerance_pct)
        self.depth_column = depth_column
        self.unit_column = unit_column
        #: Colonne unitaire effectivement résolue lors du dernier appel à :meth:`predict`.
        self.unit_column_used: str | None = unit_column

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
        # La fourchette publiée et les colonnes de confiance vivent dans le noeud `estimation`
        # de `conf/config.yaml` (injecté par le manifeste) ; le noeud standard `predict` sert de
        # repli. Rien n'est codé en dur dans le code source.
        estimation_node = dict(payload.get("estimation") or {})
        predict_node = {**dict(payload.get("predict") or {}), **estimation_node}

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
        tolerance = float(predict_node.get("tolerance_pct") or DEFAULT_TOLERANCE_PCT)
        depth_column = predict_node.get("depth_column") or None
        unit_column = predict_node.get("unit_column") or None

        logger.info(
            "Predictor ready | model={} features={} tolerance=±{:.1f}%",
            model.summary(),
            len(preprocessing.feature_names_out),
            tolerance,
        )
        return cls(
            model=model,
            preprocessing=preprocessing,
            feature_builder=feature_builder,
            config=payload,
            paths=layout,
            tolerance_pct=tolerance,
            depth_column=str(depth_column) if depth_column else None,
            unit_column=str(unit_column) if unit_column else None,
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
            A frame without the target column.
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
        """Estimate a batch of records.

        Args:
            frame: Raw records.
            validate: Enforce ``InferenceDataSchema`` before predicting.

        Returns:
            A frame with the context columns, ``estimate``, ``estimate_low``, ``estimate_high``,
            ``band_pct``, ``confidence`` and ``estimation_reason`` (plus ``unit_price`` when a
            unit column is available).

        Raises:
            RuntimeError: When the model is not fitted.
        """
        self.model.check_is_fitted()
        payload = InferenceDataSchema.validate(frame, lazy=False) if validate else frame.copy()

        matrix = self.prepare(payload)
        estimates = np.asarray(self.model.predict(matrix), dtype="float64").ravel()

        output = self._context_frame(payload).reset_index(drop=True)
        output["estimate"] = estimates
        confidence = self._confidence(payload)
        factors = np.array(
            [dict(CONFIDENCE_BAND_FACTORS).get(level, 1.0) for level in confidence], dtype="float64"
        )
        output["confidence"] = confidence
        output["band_pct"] = self.tolerance_pct * factors
        output["estimate_low"] = output["estimate"] * (1.0 - output["band_pct"] / 100.0)
        output["estimate_high"] = output["estimate"] * (1.0 + output["band_pct"] / 100.0)

        unit_column = self._resolve_unit_column(payload)
        # Mémorisé pour libeller la raison d'estimation (« 3 877 EUR par surface_m2 »).
        self.unit_column_used = unit_column
        if unit_column is not None:
            denominator = pd.to_numeric(payload[unit_column], errors="coerce").to_numpy(
                dtype="float64"
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                output["unit_price"] = np.where(
                    denominator > 0, output["estimate"] / denominator, np.nan
                )
        output["estimation_reason"] = output.apply(self._estimation_reason, axis=1)

        logger.info(
            "Estimations | rows={} median={:,.0f} couverture_confiance_faible={:.1%}",
            len(output),
            float(output["estimate"].median()) if len(output) else float("nan"),
            float((output["confidence"] == "faible").mean()) if len(output) else float("nan"),
        )
        return output

    def predict_one(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Estimate a single record and return a JSON-serialisable payload.

        Args:
            record: Mapping of column name to value.

        Returns:
            The estimation payload (estimate, range, unit price, confidence, drivers).
        """
        frame = pd.DataFrame([dict(record)])
        predictions = self.predict(frame)
        row = predictions.iloc[0]
        drivers = self.top_drivers(frame, index=0, top_n=3)
        return {
            "input": {key: _jsonable(value) for key, value in record.items()},
            "estimate": _jsonable(row.get("estimate")),
            "estimate_low": _jsonable(row.get("estimate_low")),
            "estimate_high": _jsonable(row.get("estimate_high")),
            "band_pct": _jsonable(row.get("band_pct")),
            "unit_price": _jsonable(row.get("unit_price")),
            "confidence": str(row.get("confidence", "n/a")),
            "estimation_reason": str(row.get("estimation_reason", "")),
            "top_drivers": [
                {"feature": name, "contribution": round(float(value), 4)} for name, value in drivers
            ],
            "tolerance_pct": self.tolerance_pct,
        }

    def top_drivers(
        self, frame: pd.DataFrame, *, index: int = 0, top_n: int = 5
    ) -> list[tuple[str, float]]:
        """Return the features that push the estimate the most for one record.

        The contribution is a *local* approximation: feature importance (global) multiplied by the
        standardised value of the record. It is meant for operational explainability, not for
        exact attribution — use SHAP when a formal explanation is required.

        Args:
            frame: Raw records (already validated).
            index: Row index to explain.
            top_n: Number of drivers returned.

        Returns:
            List of ``(feature, contribution)`` sorted by descending absolute contribution.
        """
        matrix = self.prepare(frame)
        if index >= len(matrix):
            msg = f"Row index {index} is out of range ({len(matrix)} rows)"
            raise IndexError(msg)

        importances = self._feature_importances(matrix)
        if importances is None:
            return []
        values = matrix.iloc[index].to_numpy(dtype="float64")
        contributions = np.asarray(importances, dtype="float64") * values
        ranking = np.argsort(-np.abs(contributions))[:top_n]
        return [
            (str(matrix.columns[position]), float(contributions[position])) for position in ranking
        ]

    def save(self, predictions: pd.DataFrame, path: str | Path | None = None) -> Path:
        """Persist predictions.

        Args:
            predictions: Frame produced by :meth:`predict`.
            path: Destination file (defaults to ``artifacts/reports/predictions.csv``).

        Returns:
            The written path.
        """
        destination = Path(path) if path else self.paths.reports_dir / "predictions.csv"
        return write_table(predictions, destination)

    # ------------------------------------------------------------------ internals -------
    def _confidence(self, payload: pd.DataFrame) -> list[str]:
        """Assign a confidence level per record, from the market depth when available.

        Un marché fin (peu de ventes comparables) ne rend pas l'estimation fausse : il la rend
        **incertaine**. La fourchette publiée s'élargit donc, plutôt que de prétendre à une
        précision que les données ne supportent pas.

        Args:
            payload: Validated raw records.

        Returns:
            One confidence label per row (``élevée`` / ``moyenne`` / ``faible``).
        """
        column = self._resolve_depth_column(payload)
        if column is None:
            return ["n/a"] * len(payload)
        depth = pd.to_numeric(payload[column], errors="coerce").to_numpy(dtype="float64")
        labels: list[str] = []
        for value in depth:
            if not np.isfinite(value):
                labels.append("n/a")
                continue
            for bound, label in CONFIDENCE_BOUNDS:
                if value >= bound:
                    labels.append(label)
                    break
            else:  # pragma: no cover - CONFIDENCE_BOUNDS se termine par 0.0
                labels.append("faible")
        return labels

    def _resolve_depth_column(self, payload: pd.DataFrame) -> str | None:
        """Return the market-depth column of the payload (configured or auto-detected)."""
        if self.depth_column and self.depth_column in payload.columns:
            return self.depth_column
        for candidate in DEPTH_COLUMN_CANDIDATES:
            if candidate in payload.columns:
                return candidate
        return None

    def _resolve_unit_column(self, payload: pd.DataFrame) -> str | None:
        """Return the column used for a unit price (configured or auto-detected)."""
        if self.unit_column and self.unit_column in payload.columns:
            return self.unit_column
        for candidate in UNIT_COLUMN_CANDIDATES:
            if candidate in payload.columns:
                return candidate
        return None

    def _feature_importances(self, matrix: pd.DataFrame) -> np.ndarray | None:
        """Extract per-feature importances aligned with the matrix columns."""
        estimator = getattr(self.model, "estimator_", None) or getattr(self.model, "model_", None)
        values = getattr(estimator, "feature_importances_", None)
        if values is None:
            coefficients = getattr(estimator, "coef_", None)
            if coefficients is None:
                return None
            array = np.asarray(coefficients, dtype="float64")
            values = np.abs(array).mean(axis=0) if array.ndim > 1 else np.abs(array).ravel()
        values = np.asarray(values, dtype="float64").ravel()
        return values if len(values) == matrix.shape[1] else None

    def _context_frame(self, payload: pd.DataFrame) -> pd.DataFrame:
        """Keep the useful context columns next to the estimates."""
        columns = [column for column in CONTEXT_COLUMNS if column in payload.columns]
        return payload.loc[:, columns] if columns else payload.iloc[:, :0]

    def _estimation_reason(self, row: pd.Series) -> str:
        """Build a one-line, human readable justification of the published range."""
        estimate = row.get("estimate")
        if estimate is None or (isinstance(estimate, float) and not np.isfinite(estimate)):
            return "estimation indisponible (le modèle n'a pas produit de valeur)"
        band = float(row.get("band_pct", self.tolerance_pct))
        confidence = str(row.get("confidence", "n/a"))
        unit_price = row.get("unit_price")
        unit_label = self.unit_column_used or "unité"
        unit_text = (
            f" • {float(unit_price):,.0f} par {unit_label}" if _is_number(unit_price) else ""
        )
        return (
            f"estimation {float(estimate):,.0f} ± {band:.1f} % (confiance {confidence}){unit_text}"
        )


def _is_number(value: Any) -> bool:
    """Return ``True`` when ``value`` is a finite number."""
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _jsonable(value: Any) -> Any:
    """Coerce numpy/pandas scalars into JSON-serialisable Python values."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


__all__ = ["CONTEXT_COLUMNS", "DEFAULT_TOLERANCE_PCT", "ID_COLUMN", "Predictor", "TARGET_NAME"]
