"""Preprocessing pipeline: composition, fitting, persistence.

The pipeline is the single place where raw columns become a model-ready numeric matrix. It is
built from the Hydra configuration (``conf/preprocessing/default.yaml``), which means the
whole cleaning strategy is declarative, reviewable and reproducible.

Design points:

* ``fit`` is called **once**, on the training split only. Validation/test/inference data are
  only ``transform``ed: that is what prevents target and statistics leakage.
* The fitted pipeline is persisted (``artifacts/models/preprocessing.joblib``) so that
  inference uses exactly the same transformations as training.
* Output is a ``DataFrame`` with explicit feature names: feature importances, reports and
  debugging stay readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.preprocessing.transformers import (
    DataFrameScaler,
    Log1pTransformer,
    OutlierClipper,
    RareCategoryGrouper,
    TypeCaster,
)
from src.utils.io import load_pickle, save_pickle
from src.utils.logging import get_logger

logger = get_logger(__name__)

try:  # TargetEncoder is available from scikit-learn 1.3
    from sklearn.preprocessing import TargetEncoder

    _HAS_TARGET_ENCODER = True
except ImportError:  # pragma: no cover - older scikit-learn
    TargetEncoder = None
    _HAS_TARGET_ENCODER = False


@dataclass(slots=True)
class PreprocessingReport:
    """Traceability of what the pipeline did.

    Attributes:
        n_rows_in: Rows given to ``fit``.
        n_columns_in: Columns given to ``fit``.
        numeric_features: Numeric columns handled.
        categorical_features: Categorical columns handled.
        output_features: Feature names produced by ``transform``.
        missing_before: Number of missing cells before imputation.
        missing_after: Number of missing cells after transformation.
        encoder: Categorical encoder in use.
        scaler: Numeric scaler in use.
        clipped_cells: Cells clipped by :class:`OutlierClipper` on the last transform.
    """

    n_rows_in: int = 0
    n_columns_in: int = 0
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    output_features: list[str] = field(default_factory=list)
    missing_before: int = 0
    missing_after: int = 0
    encoder: str = "onehot"
    scaler: str = "standard"
    clipped_cells: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report as a JSON-friendly mapping."""
        return {
            "n_rows_in": self.n_rows_in,
            "n_columns_in": self.n_columns_in,
            "n_output_features": len(self.output_features),
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "missing_before": self.missing_before,
            "missing_after": self.missing_after,
            "encoder": self.encoder,
            "scaler": self.scaler,
            "clipped_cells": self.clipped_cells,
        }


class PreprocessingPipeline:
    """Configurable, persistable preprocessing pipeline.

    Example:
        >>> pipeline = PreprocessingPipeline(
        ...     numeric_features=["age", "amount"],
        ...     categorical_features=["plan"],
        ...     config={"numeric": {"scaler": "standard"}, "categorical": {"encoder": "onehot"}},
        ... )
        >>> train = pd.DataFrame({"age": [1, 2], "amount": [3.0, 4.0], "plan": ["a", "b"]})
        >>> _ = pipeline.fit(train)
        >>> pipeline.transform(train.head(1)).shape[0]
        1
    """

    #: Branch prefixes used by the underlying ``ColumnTransformer``.
    NUMERIC_BRANCH = "num"
    CATEGORICAL_BRANCH = "cat"

    def __init__(
        self,
        *,
        numeric_features: Sequence[str],
        categorical_features: Sequence[str],
        config: Mapping[str, Any] | None = None,
        target: str | None = None,
    ) -> None:
        """Build the pipeline description (nothing is fitted yet).

        Args:
            numeric_features: Numeric column names.
            categorical_features: Categorical column names.
            config: ``preprocessing`` configuration node (Hydra or plain mapping).
            target: Target column name, required by target encoding.

        Raises:
            ValueError: When the configuration requests target encoding without a target.
        """
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.target = target
        self.config: dict[str, Any] = _normalise_config(config)

        numeric_cfg = self.config["numeric"]
        categorical_cfg = self.config["categorical"]
        if categorical_cfg["encoder"] == "target":
            if target is None:
                msg = "Target encoding requires 'target' to be provided"
                raise ValueError(msg)
            if not _HAS_TARGET_ENCODER:
                logger.warning(
                    "TargetEncoder unavailable (scikit-learn < 1.3); falling back to one-hot"
                )
                categorical_cfg["encoder"] = "onehot"

        self.sklearn_pipeline: Pipeline = self._build()
        self._is_fitted = False
        self._report = PreprocessingReport(
            numeric_features=list(self.numeric_features),
            categorical_features=list(self.categorical_features),
            encoder=str(categorical_cfg["encoder"]),
            scaler=str(numeric_cfg["scaler"]),
        )

    # ------------------------------------------------------------------ construction ----
    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, target: str | None = None
    ) -> PreprocessingPipeline:
        """Build a pipeline from a full application configuration.

        Args:
            config: Root configuration mapping (``data`` and ``preprocessing`` nodes are used).
            target: Optional explicit target name (defaults to ``data.target``).

        Returns:
            The configured (not fitted) pipeline.
        """
        data_node = dict(config.get("data", {}) or {})
        preprocessing_node = dict(config.get("preprocessing", {}) or {})
        numeric, categorical = infer_column_kinds(preprocessing_node, data_node)
        effective_target = target if target is not None else data_node.get("target")
        return cls(
            numeric_features=numeric,
            categorical_features=categorical,
            config=preprocessing_node,
            target=effective_target,
        )

    def _numeric_steps(self) -> list[tuple[str, Any]]:
        """Build the numeric branch steps from the configuration."""
        cfg = self.config["numeric"]
        steps: list[tuple[str, Any]] = []
        if cfg.get("log1p_columns"):
            steps.append(("log1p", Log1pTransformer(columns=list(cfg["log1p_columns"]))))
        if cfg.get("clip_outliers", True):
            steps.append(
                (
                    "clip",
                    OutlierClipper(
                        quantiles=tuple(cfg.get("clip_quantiles", (0.01, 0.99))),
                        columns=self.numeric_features or None,
                    ),
                )
            )
        imputer_strategy = str(cfg.get("imputer", "median"))
        if imputer_strategy != "none":
            kwargs: dict[str, Any] = {"strategy": imputer_strategy}
            if imputer_strategy == "constant":
                kwargs["fill_value"] = float(cfg.get("imputer_fill_value", 0.0))
            steps.append(("impute", SimpleImputer(**kwargs)))
        steps.append(("scale", DataFrameScaler(strategy=str(cfg.get("scaler", "standard")))))
        return steps

    def _categorical_steps(self) -> list[tuple[str, Any]]:
        """Build the categorical branch steps from the configuration."""
        cfg = self.config["categorical"]
        steps: list[tuple[str, Any]] = []
        imputer_strategy = str(cfg.get("imputer", "most_frequent"))
        if imputer_strategy != "none":
            kwargs: dict[str, Any] = {"strategy": imputer_strategy}
            if imputer_strategy == "constant":
                kwargs["fill_value"] = str(cfg.get("imputer_fill_value", "unknown"))
            steps.append(("impute", SimpleImputer(**kwargs)))
        steps.append(
            (
                "rare",
                RareCategoryGrouper(
                    min_frequency=float(cfg.get("min_frequency", 0.01)),
                    max_categories=cfg.get("max_categories", 25),
                ),
            )
        )
        encoder_name = str(cfg.get("encoder", "onehot"))
        handle_unknown = str(cfg.get("handle_unknown", "ignore"))
        if encoder_name == "ordinal":
            steps.append(
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            )
        elif encoder_name == "target" and _HAS_TARGET_ENCODER and TargetEncoder is not None:
            steps.append(
                ("encode", TargetEncoder(random_state=self.config.get("random_state", 42)))
            )
        else:
            steps.append(
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown=handle_unknown
                        if handle_unknown in {"ignore", "infrequent_if_exist"}
                        else "ignore",
                        sparse_output=False,
                        drop=None,
                        min_frequency=float(cfg.get("min_frequency", 0.01)) or None,
                    ),
                )
            )
        return steps

    def _build(self) -> Pipeline:
        """Compose the full scikit-learn pipeline."""
        transformers: list[tuple[str, Any, list[str]]] = []
        if self.numeric_features:
            transformers.append(
                (self.NUMERIC_BRANCH, Pipeline(steps=self._numeric_steps()), self.numeric_features)
            )
        if self.categorical_features:
            transformers.append(
                (
                    self.CATEGORICAL_BRANCH,
                    Pipeline(steps=self._categorical_steps()),
                    self.categorical_features,
                )
            )
        if not transformers:
            msg = (
                "PreprocessingPipeline needs at least one numeric or categorical feature; "
                f"got numeric={self.numeric_features}, categorical={self.categorical_features}"
            )
            raise ValueError(msg)

        column_transformer = ColumnTransformer(
            transformers=transformers, remainder="drop", verbose_feature_names_out=False
        )
        pipeline = Pipeline(
            steps=[
                (
                    "cast",
                    TypeCaster(
                        numeric_columns=self.numeric_features,
                        categorical_columns=self.categorical_features,
                    ),
                ),
                ("transform", column_transformer),
            ]
        )
        try:
            pipeline.set_output(transform="pandas")
        except (ValueError, AttributeError):  # pragma: no cover - very old scikit-learn
            logger.warning("DataFrame output not supported by this scikit-learn version")
        return pipeline

    # ------------------------------------------------------------------ execution -------
    def fit(self, frame: pd.DataFrame, y: pd.Series | None = None) -> PreprocessingPipeline:
        """Learn every statistic on the training split only.

        Args:
            frame: Training data (raw columns).
            y: Target series, required by target encoding.

        Returns:
            ``self``.
        """
        missing_columns = [
            column
            for column in [*self.numeric_features, *self.categorical_features]
            if column not in frame.columns
        ]
        if missing_columns:
            msg = f"PreprocessingPipeline: missing column(s) {missing_columns} in the input frame"
            raise KeyError(msg)

        logger.info(
            "Fitting preprocessing | rows={} numeric={} categorical={} encoder={} scaler={}",
            len(frame),
            len(self.numeric_features),
            len(self.categorical_features),
            self.config["categorical"]["encoder"],
            self.config["numeric"]["scaler"],
        )
        self._report.missing_before = int(frame.isna().to_numpy().sum())
        self._report.n_rows_in = int(len(frame))
        self._report.n_columns_in = int(frame.shape[1])
        self.sklearn_pipeline.fit(frame, y)
        self._is_fitted = True
        self._report.output_features = list(self.feature_names_out)
        logger.info("Preprocessing fitted | output_features={}", len(self._report.output_features))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned transformations.

        Args:
            frame: Data to transform.

        Returns:
            A numeric ``DataFrame`` whose columns are the model features.

        Raises:
            RuntimeError: When the pipeline is used before being fitted.
        """
        if not self._is_fitted:
            msg = "PreprocessingPipeline is not fitted: call fit() before transform()"
            raise RuntimeError(msg)
        transformed = self.sklearn_pipeline.transform(frame)
        output = _as_numeric_dataframe(transformed, self.feature_names_out)
        self._report.missing_after = int(output.isna().to_numpy().sum())
        self._report.clipped_cells = self._collect_clipped_cells()
        return output

    def fit_transform(self, frame: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit on ``frame`` then transform it.

        Args:
            frame: Training data.
            y: Optional target.

        Returns:
            The transformed training data.
        """
        return self.fit(frame, y).transform(frame)

    def _collect_clipped_cells(self) -> dict[str, int]:
        """Retrieve the clipping report from the numeric branch, when available."""
        try:
            numeric_pipeline = self.sklearn_pipeline.named_steps["transform"].transformers_[0][1]
            clipper = numeric_pipeline.named_steps.get("clip")
        except (AttributeError, IndexError, KeyError):
            return {}
        return dict(clipper.report) if clipper is not None else {}

    # ------------------------------------------------------------------ introspection ---
    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` succeeded."""
        return self._is_fitted

    @property
    def feature_names_out(self) -> list[str]:
        """Names of the produced features."""
        try:
            names = list(self.sklearn_pipeline.get_feature_names_out())
        except (AttributeError, ValueError):  # pragma: no cover - not fitted yet
            names = [*self.numeric_features, *self.categorical_features]
        return [_clean_feature_name(name) for name in names]

    @property
    def report(self) -> PreprocessingReport:
        """Traceability report of the last fit/transform (columns, missing values, choices)."""
        return self._report

    # ------------------------------------------------------------------ persistence -----
    def save(self, path: str | Path) -> Path:
        """Persist the fitted pipeline.

        Args:
            path: Destination file.

        Returns:
            The written path.
        """
        if not self._is_fitted:
            msg = "Refusing to save an unfitted PreprocessingPipeline"
            raise RuntimeError(msg)
        return save_pickle(self, path)

    @classmethod
    def load(cls, path: str | Path) -> PreprocessingPipeline:
        """Restore a persisted pipeline.

        Args:
            path: Artefact file.

        Returns:
            The fitted pipeline.
        """
        loaded = load_pickle(path)
        if not isinstance(loaded, cls):
            msg = f"Expected a PreprocessingPipeline in {path}, got {type(loaded).__name__}"
            raise TypeError(msg)
        return loaded


def infer_column_kinds(
    preprocessing_config: Mapping[str, Any], data_config: Mapping[str, Any]
) -> tuple[list[str], list[str]]:
    """Determine numeric and categorical columns from the configuration.

    ``conf/preprocessing/default.yaml`` may declare them explicitly (``columns.numeric`` /
    ``columns.categorical``). When it does not, they are derived from the dataset schema
    exposed by ``src/data/schemas.py`` so that a new column is automatically taken into
    account.

    Args:
        preprocessing_config: The ``preprocessing`` node.
        data_config: The ``data`` node.

    Returns:
        The ``(numeric_columns, categorical_columns)`` pair.
    """
    columns_node = dict(preprocessing_config.get("columns", {}) or {})
    numeric = [str(name) for name in columns_node.get("numeric", []) or []]
    categorical = [str(name) for name in columns_node.get("categorical", []) or []]
    if numeric or categorical:
        return numeric, categorical

    from src.data.schemas import RawDataSchema  # local import: avoids a circular dependency

    drop = {str(name) for name in data_config.get("drop_columns", []) or []}
    target = data_config.get("target")
    if target:
        drop.add(str(target))

    numeric, categorical = [], []
    for name, column in RawDataSchema.to_schema().columns.items():
        if name in drop:
            continue
        # ``str(dtype)`` est la seule introspection fiable entre pandera/pandas/numpy :
        # selon les versions, ``column.dtype`` est un objet pandera, un dtype numpy ou une
        # chaîne ("int64", "string[pyarrow]", "category", ...).
        dtype_name = str(getattr(column, "dtype", "") or "").lower()
        if any(token in dtype_name for token in ("int", "float", "bool", "number")):
            numeric.append(name)
        else:
            categorical.append(name)

    # Les features dérivées par ``FeatureBuilder`` n'existent pas dans le schéma brut : sans
    # cette étape, un pipeline reconstruit depuis la configuration (inférence, notebooks)
    # ignorerait silencieusement ces colonnes.
    for name, kind in _derived_feature_kinds(preprocessing_config, data_config):
        if name in drop or name in numeric or name in categorical:
            continue
        (numeric if kind == "numeric" else categorical).append(name)

    logger.debug("Inferred columns | numeric={} categorical={}", numeric, categorical)
    return numeric, categorical


def _derived_feature_kinds(
    preprocessing_config: Mapping[str, Any], data_config: Mapping[str, Any]
) -> list[tuple[str, str]]:
    """Classify the columns produced by the declarative feature recipes.

    ``FeatureBuilder`` reste la seule source de vérité sur les noms produits : on lui demande
    ses ``output_names`` puis on classe chaque recette (un ``bin`` avec labels produit une
    colonne catégorielle, tout le reste est numérique).

    Args:
        preprocessing_config: The ``preprocessing`` node (its ``features`` recipes are read).
        data_config: The ``data`` node (used to reject recipes reading the target).

    Returns:
        A list of ``(column_name, "numeric" | "categorical")`` pairs.
    """
    from src.features.build_features import FeatureBuilder  # import local : évite un cycle

    recipes = list(preprocessing_config.get("features", []) or [])
    if not recipes:
        return []
    try:
        builder = FeatureBuilder(recipes, target=data_config.get("target"))
    except ValueError as exc:  # pragma: no cover - recette invalide : déjà testée ailleurs
        logger.warning("Feature recipes could not be parsed while inferring columns: {}", exc)
        return []

    labelled_bins = {
        recipe.name
        for recipe in builder.recipes
        if recipe.type == "bin" and recipe.params.get("labels")
    }
    return [
        (name, "categorical" if name in labelled_bins else "numeric")
        for name in builder.output_names
    ]


def _clean_feature_name(name: str) -> str:
    """Strip the ``ColumnTransformer`` branch prefix from a feature name.

    Args:
        name: Raw feature name (e.g. ``num__age`` or ``cat__plan_basic``).

    Returns:
        The cleaned name.
    """
    for prefix in (
        f"{PreprocessingPipeline.NUMERIC_BRANCH}__",
        f"{PreprocessingPipeline.CATEGORICAL_BRANCH}__",
    ):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _as_numeric_dataframe(transformed: Any, feature_names: Sequence[str]) -> pd.DataFrame:
    """Normalise a transformer output into a float ``DataFrame``.

    Args:
        transformed: Output of the scikit-learn pipeline.
        feature_names: Expected column names.

    Returns:
        A numeric ``DataFrame``.
    """
    if isinstance(transformed, pd.DataFrame):
        frame = transformed.copy()
        if list(frame.columns) != list(feature_names) and len(frame.columns) == len(feature_names):
            frame.columns = list(feature_names)
    else:
        array = np.asarray(transformed)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        frame = pd.DataFrame(array, columns=list(feature_names)[: array.shape[1]])
    return frame.astype("float64")


def _normalise_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a config mapping with guaranteed numeric/categorical sub-mappings.

    Args:
        config: Raw configuration node (may be ``None``).

    Returns:
        The normalised configuration.
    """
    raw: dict[str, Any] = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in dict(config or {}).items()
    }
    raw.setdefault("numeric", {})
    raw.setdefault("categorical", {})
    raw["numeric"] = {
        "imputer": "median",
        "imputer_fill_value": 0.0,
        "clip_outliers": True,
        "clip_quantiles": (0.01, 0.99),
        "scaler": "standard",
        "log1p_columns": [],
        **raw["numeric"],
    }
    raw["categorical"] = {
        "imputer": "most_frequent",
        "imputer_fill_value": "unknown",
        "encoder": "onehot",
        "handle_unknown": "ignore",
        "min_frequency": 0.01,
        "max_categories": 25,
        **raw["categorical"],
    }
    quantiles = raw["numeric"].get("clip_quantiles", (0.01, 0.99))
    if isinstance(quantiles, (list, tuple)) and len(quantiles) == 2:
        raw["numeric"]["clip_quantiles"] = (float(quantiles[0]), float(quantiles[1]))
    return raw
