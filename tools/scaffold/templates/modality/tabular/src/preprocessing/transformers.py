"""Custom scikit-learn compatible transformers.

Every transformer follows the scikit-learn contract (``fit`` / ``transform`` /
``get_feature_names_out``), which means they can be composed inside a ``ColumnTransformer``
and persisted with the rest of the pipeline. Rules respected here:

* **no leakage**: everything learned (quantiles, category frequencies, scaler statistics) is
  estimated in ``fit`` on the training split only;
* **DataFrame in, DataFrame out**: column names are preserved, which keeps downstream logs,
  reports and feature importances readable;
* **idempotence**: transforming twice gives the same result;
* **robustness**: unseen categories and unexpected dtypes are handled explicitly.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.utils.logging import get_logger

logger = get_logger(__name__)

SCALERS: dict[str, type[Any]] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def _names_out(
    input_features: Sequence[str] | None,
    fitted_names: Sequence[str] | None,
    fallback: Sequence[str] = (),
) -> np.ndarray:
    """Resolve the output feature names of a width-preserving transformer.

    scikit-learn calls ``get_feature_names_out()`` **without arguments** when it wraps the
    output of a step into a DataFrame, so a transformer must be able to answer on its own:
    the names seen in ``fit`` are the source of truth.

    Args:
        input_features: Names explicitly provided by the caller.
        fitted_names: Names recorded during ``fit`` (``feature_names_in_``).
        fallback: Last resort when the transformer was never fitted.

    Returns:
        The array of output feature names.
    """
    if input_features is not None:
        names = list(input_features)
    elif fitted_names is not None:
        names = list(fitted_names)
    else:
        names = list(fallback)
    return np.asarray(names, dtype=object)


def _as_dataframe(X: Any, columns: Sequence[str]) -> pd.DataFrame:
    """Normalise an input into a ``DataFrame`` with known columns.

    Args:
        X: DataFrame, Series or array-like.
        columns: Column names to use when ``X`` is not a DataFrame.

    Returns:
        A ``DataFrame``.
    """
    if isinstance(X, pd.DataFrame):
        return X
    if isinstance(X, pd.Series):
        return X.to_frame()
    return pd.DataFrame(np.asarray(X), columns=list(columns))


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select a fixed list of columns, optionally tolerating missing ones.

    Attributes:
        columns: Names of the columns to keep.
        allow_missing: Keep only the intersection instead of raising.
    """

    def __init__(self, columns: Sequence[str], *, allow_missing: bool = False) -> None:
        """Store the selection.

        Args:
            columns: Columns to keep, in the desired order.
            allow_missing: When ``True``, missing columns are silently ignored.
        """
        # Contrat scikit-learn : les paramètres du constructeur sont stockés tels quels
        # (aucune conversion), sinon `clone()` échoue.
        self.columns = columns
        self.allow_missing = allow_missing
        self.selected_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> ColumnSelector:  # noqa: ARG002
        """Resolve the effective column list.

        Args:
            X: Input frame.
            y: Ignored (API compatibility).

        Returns:
            ``self``.

        Raises:
            KeyError: When a required column is missing and ``allow_missing`` is ``False``.
        """
        requested = list(self.columns)
        self.feature_names_in_ = list(X.columns)
        missing = [column for column in requested if column not in X.columns]
        if missing and not self.allow_missing:
            msg = f"ColumnSelector: missing column(s) {missing}. Available: {list(X.columns)}"
            raise KeyError(msg)
        self.selected_ = [column for column in requested if column in X.columns]
        if missing:
            logger.warning("ColumnSelector: ignoring missing column(s) {}", missing)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the selected columns.

        Args:
            X: Input frame.

        Returns:
            The projected frame.
        """
        frame = _as_dataframe(X, list(self.columns))
        available = [column for column in self.selected_ if column in frame.columns]
        return frame.loc[:, available]

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:  # noqa: ARG002
        """Return the output feature names.

        Args:
            input_features: Ignored (API compatibility).

        Returns:
            Array of selected column names.
        """
        return np.asarray(self.selected_, dtype=object)


class TypeCaster(BaseEstimator, TransformerMixin):
    """Select the declared columns and cast them to explicit dtypes.

    Real-world loaders return mixed dtypes (strings for numbers, ``object`` for categories)
    and inference payloads often carry extra columns. This transformer solves both problems at
    the entrance of the pipeline:

    * it **projects** the frame onto the declared contract (numeric + categorical columns), so
      the pipeline is insensitive to additional columns,
    * it **casts** them (``float64`` for numerics, ``category`` for categoricals), so every
      downstream step receives predictable types.

    Attributes:
        numeric_columns: Columns cast to ``float64``.
        categorical_columns: Columns cast to ``category``.
    """

    def __init__(
        self, numeric_columns: Sequence[str] = (), categorical_columns: Sequence[str] = ()
    ) -> None:
        """Store the casting contract.

        Args:
            numeric_columns: Columns to coerce to numeric.
            categorical_columns: Columns to coerce to categorical.
        """
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> TypeCaster:  # noqa: ARG002
        """Nothing to learn; the contract is declarative.

        Args:
            X: Input frame.
            y: Ignored.

        Returns:
            ``self``.
        """
        self.feature_names_in_ = list(X.columns)
        return self

    @property
    def declared_columns(self) -> list[str]:
        """Columns of the contract, in output order (numeric first, then categorical)."""
        return [*list(self.numeric_columns), *list(self.categorical_columns)]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Project onto the declared columns and cast them.

        Args:
            X: Input frame (extra columns are ignored).

        Returns:
            The typed frame, whose columns are exactly :attr:`declared_columns`.

        Raises:
            KeyError: When a declared column is missing from the input.
        """
        frame = _as_dataframe(X, self.declared_columns)
        missing = [column for column in self.declared_columns if column not in frame.columns]
        if missing:
            msg = (
                f"TypeCaster: colonne(s) manquante(s) {missing}. Attendues : "
                f"{self.declared_columns}. Reçues : {list(frame.columns)}"
            )
            raise KeyError(msg)

        output = frame.loc[:, self.declared_columns].copy()
        for column in self.numeric_columns:
            output[column] = pd.to_numeric(output[column], errors="coerce").astype("float64")
        for column in self.categorical_columns:
            output[column] = output[column].astype("object").astype("category")
        return output

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:  # noqa: ARG002
        """Return the output feature names: always the declared contract.

        The output width does not depend on the input frame (extra columns are dropped), so
        the declared columns are the only correct answer — whatever scikit-learn passes.

        Args:
            input_features: Ignored.

        Returns:
            Array of declared column names.
        """
        return np.asarray(self.declared_columns, dtype=object)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Winsorise numeric columns between quantiles learned on the training split.

    Clipping (instead of dropping) preserves the row count and the labels, which matters for
    business datasets where an extreme value is often legitimate (a big customer, a peak day).

    Attributes:
        quantiles: ``(low, high)`` quantiles used to compute the bounds.
    """

    def __init__(self, quantiles: tuple[float, float] = (0.01, 0.99), *, columns: Sequence[str] | None = None) -> None:
        """Store the clipping policy.

        Args:
            quantiles: Lower and upper quantiles.
            columns: Columns to clip (``None`` = every numeric column seen in ``fit``).
        """
        self.quantiles = quantiles
        self.columns = columns
        self.bounds_: dict[str, tuple[float, float]] = {}
        self.clipped_cells_: dict[str, int] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> OutlierClipper:  # noqa: ARG002
        """Learn the bounds on the training data.

        Args:
            X: Training frame.
            y: Ignored.

        Returns:
            ``self``.
        """
        low_quantile, high_quantile = self._validated_quantiles()
        frame = _as_dataframe(X, list(self.columns or []))
        self.feature_names_in_ = list(frame.columns)
        targets = list(self.columns) if self.columns is not None else [
            column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])
        ]
        self.bounds_ = {}
        for column in targets:
            if column not in frame.columns:
                continue
            series = pd.to_numeric(frame[column], errors="coerce")
            low = float(series.quantile(low_quantile))
            high = float(series.quantile(high_quantile))
            if np.isfinite(low) and np.isfinite(high):
                self.bounds_[column] = (low, high)
        logger.debug("OutlierClipper bounds: {}", self.bounds_)
        return self

    def _validated_quantiles(self) -> tuple[float, float]:
        """Validate and normalise the quantile window (done in ``fit``, not in ``__init__``)."""
        low, high = float(self.quantiles[0]), float(self.quantiles[1])
        if not 0.0 <= low < high <= 1.0:
            msg = f"quantiles must satisfy 0 <= low < high <= 1, got ({low}, {high})"
            raise ValueError(msg)
        return low, high

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip values outside the learned bounds.

        Args:
            X: Frame to clip.

        Returns:
            The clipped frame (copy).
        """
        frame = _as_dataframe(X, list(self.bounds_))
        self.clipped_cells_ = {}
        # pandas >= 3 (copy-on-write) refuse d'écrire des flottants dans une colonne entière
        # (LossySetitemError) : on reconstruit donc un DataFrame plutôt que d'utiliser setitem.
        data: dict[str, Any] = {column: frame[column] for column in frame.columns}
        for column, (low, high) in self.bounds_.items():
            if column not in frame.columns:
                continue
            series = pd.to_numeric(frame[column], errors="coerce")
            clipped = series.clip(lower=low, upper=high)
            self.clipped_cells_[column] = int((clipped != series).sum())
            data[column] = clipped.astype("float64")
        return pd.DataFrame(data, index=frame.index)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return the output feature names (the frame width is preserved).

        Args:
            input_features: Input names.

        Returns:
            Array of column names.
        """
        return _names_out(input_features, getattr(self, "feature_names_in_", None), fallback=list(self.bounds_))

    @property
    def report(self) -> dict[str, int]:
        """Number of clipped cells per column during the last ``transform``."""
        return dict(self.clipped_cells_)


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Apply ``log(1 + x)`` to strictly positive, heavy-tailed columns.

    Attributes:
        columns: Columns to transform.
    """

    def __init__(self, columns: Sequence[str]) -> None:
        """Store the transformation contract.

        Args:
            columns: Numeric columns to log-transform.
        """
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> Log1pTransformer:  # noqa: ARG002
        """Check that the declared columns are compatible with a log transform.

        Args:
            X: Training frame.
            y: Ignored.

        Returns:
            ``self``.
        """
        frame = _as_dataframe(X, list(self.columns))
        self.feature_names_in_ = list(frame.columns)
        for column in list(self.columns):
            if column not in frame.columns:
                logger.warning("Log1pTransformer: column '{}' missing, skipped", column)
                continue
            minimum = pd.to_numeric(frame[column], errors="coerce").min()
            if minimum is not None and float(minimum) < 0:
                msg = (
                    f"Log1pTransformer: column '{column}' has negative values (min={minimum}); "
                    "log1p requires x >= 0"
                )
                raise ValueError(msg)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply ``log1p``.

        Args:
            X: Frame to transform.

        Returns:
            The transformed frame (copy).
        """
        frame = _as_dataframe(X, list(self.columns)).copy()
        for column in list(self.columns):
            if column in frame.columns:
                frame[column] = np.log1p(pd.to_numeric(frame[column], errors="coerce").clip(lower=0))
        return frame

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return the output feature names.

        Args:
            input_features: Input names.

        Returns:
            Array of column names.
        """
        return _names_out(input_features, getattr(self, "feature_names_in_", None), fallback=list(self.columns))


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Group infrequent categories into a single ``rare`` bucket.

    High-cardinality categoricals blow up one-hot encoding and create unstable coefficients.
    Grouping rare levels keeps the encoding compact and robust to unseen values.

    Attributes:
        min_frequency: Minimum relative frequency to keep a category.
        max_categories: Maximum number of categories kept per column.
        rare_label: Name of the aggregated bucket.
    """

    def __init__(
        self,
        *,
        min_frequency: float = 0.01,
        max_categories: int | None = 25,
        rare_label: str = "rare",
    ) -> None:
        """Store the grouping policy.

        Args:
            min_frequency: Relative frequency threshold.
            max_categories: Hard cap on the number of kept categories.
            rare_label: Label assigned to grouped categories.
        """
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.rare_label = rare_label
        self.kept_: dict[str, list[str]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> RareCategoryGrouper:  # noqa: ARG002
        """Learn the frequent categories per column.

        Args:
            X: Training frame.
            y: Ignored.

        Returns:
            ``self``.
        """
        min_frequency = float(self.min_frequency)
        if not 0.0 <= min_frequency <= 1.0:
            msg = f"min_frequency must be in [0, 1], got {min_frequency}"
            raise ValueError(msg)
        frame = _as_dataframe(X, [])
        self.feature_names_in_ = list(frame.columns)
        self.kept_ = {}
        for column in frame.columns:
            if pd.api.types.is_numeric_dtype(frame[column]) and not isinstance(
                frame[column].dtype, pd.CategoricalDtype
            ):
                continue
            counts = frame[column].astype("object").value_counts(normalize=True)
            kept = [str(value) for value, frequency in counts.items() if frequency >= min_frequency]
            if self.max_categories is not None:
                kept = kept[: int(self.max_categories)]
            self.kept_[column] = kept
            logger.debug("RareCategoryGrouper '{}' keeps {} categories", column, len(kept))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace rare categories by ``rare_label``.

        Args:
            X: Frame to transform.

        Returns:
            The transformed frame (copy).
        """
        frame = _as_dataframe(X, list(self.kept_))
        data: dict[str, Any] = {column: frame[column] for column in frame.columns}
        for column, kept in self.kept_.items():
            if column not in frame.columns:
                continue
            series = frame[column].astype("object").astype(str)
            # Une colonne ``category`` ne peut pas recevoir de chaînes arbitraires sous pandas 3 :
            # la sortie est donc matérialisée en ``object`` (l'encodage one-hot suit).
            data[column] = series.where(series.isin(kept), other=self.rare_label).astype("object")
        return pd.DataFrame(data, index=frame.index)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return the output feature names.

        Args:
            input_features: Input names.

        Returns:
            Array of column names.
        """
        return _names_out(input_features, getattr(self, "feature_names_in_", None), fallback=list(self.kept_))


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """Wrap a scikit-learn scaler while preserving DataFrame semantics.

    Attributes:
        strategy: ``standard``, ``minmax``, ``robust`` or ``none``.
    """

    def __init__(self, strategy: str = "standard") -> None:
        """Store the scaling strategy.

        Args:
            strategy: One of :data:`SCALERS` keys, or ``none`` to disable scaling.

        Raises:
            ValueError: When the strategy is unknown.
        """
        if strategy != "none" and strategy not in SCALERS:
            msg = f"Unknown scaler '{strategy}'. Allowed: {sorted([*SCALERS, 'none'])}"
            raise ValueError(msg)
        self.strategy = strategy
        self.scaler_: Any = None
        self.columns_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> DataFrameScaler:  # noqa: ARG002
        """Fit the underlying scaler on numeric columns.

        Args:
            X: Training frame.
            y: Ignored.

        Returns:
            ``self``.
        """
        frame = _as_dataframe(X, [])
        self.feature_names_in_ = list(frame.columns)
        self.columns_ = [
            column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])
        ]
        if self.strategy == "none" or not self.columns_:
            self.scaler_ = None
            return self
        self.scaler_ = SCALERS[self.strategy]()
        self.scaler_.fit(frame.loc[:, self.columns_].to_numpy(dtype="float64"))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale the numeric columns.

        Args:
            X: Frame to scale.

        Returns:
            The scaled frame (copy).
        """
        frame = _as_dataframe(X, self.columns_)
        if self.scaler_ is None:
            return frame.copy()
        columns = [column for column in self.columns_ if column in frame.columns]
        if not columns:
            return frame.copy()
        scaled = self.scaler_.transform(frame.loc[:, columns].to_numpy(dtype="float64"))
        # Le scaling produit des flottants : sous pandas 3, réécrire une colonne ``int64`` avec
        # des flottants lève LossySetitemError. On reconstruit donc le DataFrame.
        data: dict[str, Any] = {column: frame[column] for column in frame.columns}
        for position, column in enumerate(columns):
            data[column] = np.asarray(scaled[:, position], dtype="float64")
        return pd.DataFrame(data, index=frame.index)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return the output feature names.

        Args:
            input_features: Input names.

        Returns:
            Array of column names.
        """
        return _names_out(input_features, getattr(self, "feature_names_in_", None), fallback=list(self.columns_))
