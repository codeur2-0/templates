"""Declarative feature engineering.

Adding a feature should not require touching the code: it is a new entry in
``conf/preprocessing/default.yaml``. :class:`FeatureBuilder` interprets those *recipes* and
produces the derived columns.

Supported recipe types:

``ratio``            ``numerator / (denominator + epsilon)``
``difference``       ``left - right``
``product``          ``a * b`` (interaction between two numeric columns)
``log1p``            ``log(1 + max(x, 0))``
``bin``              discretisation into quantile or uniform bins (edges learned in ``fit``)
``boolean_flag``     binary flag from a comparison (``gt``, ``lt``, ``ge``, ``le``, ``eq``,
                     ``ne``) against a constant ``value`` **or** another column ``value_column``
``datetime_parts``   calendar components (year, month, day, dayofweek, quarter, is_weekend, hour)
``group_stat``       aggregate of a numeric column per category (learned in ``fit``)

Leakage rules:
    * every statistic (bin edges, group aggregates) is learned on the training split only,
    * no recipe may use the target column; ``FeatureBuilder`` rejects it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_RECIPES: tuple[str, ...] = (
    "ratio",
    "difference",
    "product",
    "log1p",
    "bin",
    "boolean_flag",
    "datetime_parts",
    "group_stat",
)

DATETIME_PARTS: tuple[str, ...] = (
    "year",
    "month",
    "day",
    "dayofweek",
    "quarter",
    "hour",
    "is_weekend",
    "is_month_start",
    "is_month_end",
)

COMPARISONS: dict[str, Any] = {
    "gt": lambda series, value: series > value,
    "lt": lambda series, value: series < value,
    "ge": lambda series, value: series >= value,
    "le": lambda series, value: series <= value,
    "eq": lambda series, value: series == value,
    "ne": lambda series, value: series != value,
}


@dataclass(slots=True)
class FeatureRecipe:
    """One declarative feature definition.

    Attributes:
        type: Recipe type (see :data:`SUPPORTED_RECIPES`).
        name: Output column name (derived from the inputs when omitted).
        params: Recipe parameters (columns, epsilon, bins, ...).
    """

    type: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> FeatureRecipe:
        """Build a recipe from a configuration mapping.

        Args:
            payload: Mapping with a ``type`` key and the recipe parameters.

        Returns:
            The validated recipe.

        Raises:
            ValueError: When the type is unknown or the name cannot be derived.
        """
        data = dict(payload)
        recipe_type = str(data.pop("type", "")).strip().lower()
        if recipe_type not in SUPPORTED_RECIPES:
            msg = f"Unknown feature recipe '{recipe_type}'. Allowed: {SUPPORTED_RECIPES}"
            raise ValueError(msg)
        name = str(data.pop("name", "") or "")
        if not name:
            name = _default_recipe_name(recipe_type, data)
        return cls(type=recipe_type, name=name, params=data)

    def required_columns(self) -> list[str]:
        """Columns this recipe reads from the input frame."""
        columns: list[str] = []
        for key in (
            "column",
            "numerator",
            "denominator",
            "left",
            "right",
            "a",
            "b",
            "value_column",
            "group_by",
        ):
            value = self.params.get(key)
            if isinstance(value, str):
                columns.append(value)
        for key in ("columns",):
            value = self.params.get(key)
            if isinstance(value, (list, tuple)):
                columns.extend(str(item) for item in value)
        return list(dict.fromkeys(columns))


def _default_recipe_name(recipe_type: str, params: Mapping[str, Any]) -> str:
    """Derive an explicit output name from the recipe parameters."""
    parts = [recipe_type]
    for key in (
        "column",
        "numerator",
        "denominator",
        "left",
        "right",
        "a",
        "b",
        "value_column",
        "group_by",
    ):
        if isinstance(params.get(key), str):
            parts.append(str(params[key]))
    return "_".join(parts)


def _flag_operand(series: pd.Series) -> pd.Series:
    """Return a boolean-flag operand: numeric when the column is numeric, textual otherwise.

    Comparing two country codes (``shopper_country != billing_country``) is a legitimate flag
    recipe. Coercing them to numbers would silently yield NaN everywhere and an all-zero flag —
    the worst kind of bug, because it looks like a feature that simply does not help.

    Args:
        series: Column to compare.

    Returns:
        A series comparable with a constant or with another column of the same kind.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int8")
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    if series.notna().any() and float(numeric.notna().mean()) >= 0.99:
        # Colonne objet mais numérique de bout en bout (lecture CSV) : on garde le numérique.
        return numeric
    return series.astype("string").fillna("")


class FeatureBuilder:
    """Build derived features from declarative recipes.

    Example:
        >>> recipe = {"type": "ratio", "name": "r", "numerator": "a", "denominator": "b"}
        >>> builder = FeatureBuilder([recipe])
        >>> frame = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0]})
        >>> _ = builder.fit(frame)
        >>> builder.transform(frame)["r"].round(2).tolist()
        [0.5, 0.5]
    """

    def __init__(
        self,
        recipes: Iterable[Mapping[str, Any] | FeatureRecipe] = (),
        *,
        target: str | None = None,
    ) -> None:
        """Parse and validate the recipes.

        Args:
            recipes: Recipe mappings (from Hydra) or :class:`FeatureRecipe` instances.
            target: Target column name; recipes reading it are rejected.

        Raises:
            ValueError: When a recipe uses the target column or declares unknown parameters.
        """
        self.target = target
        self.recipes: list[FeatureRecipe] = [
            recipe if isinstance(recipe, FeatureRecipe) else FeatureRecipe.from_mapping(recipe)
            for recipe in recipes
        ]
        self._learned: dict[str, Any] = {}
        self._is_fitted = False

        duplicates = [name for name in set(self.output_names) if self.output_names.count(name) > 1]
        if duplicates:
            msg = f"Duplicated feature names in recipes: {duplicates}"
            raise ValueError(msg)
        if target is not None:
            offenders = [
                recipe.name for recipe in self.recipes if target in recipe.required_columns()
            ]
            if offenders:
                msg = f"Feature recipes must not read the target '{target}': {offenders}"
                raise ValueError(msg)

    # ------------------------------------------------------------------ metadata --------
    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, target: str | None = None) -> FeatureBuilder:
        """Build the feature builder from an application configuration.

        Args:
            config: Root configuration mapping (``preprocessing.features`` and ``data.target``).
            target: Optional explicit target override.

        Returns:
            The configured builder.
        """
        preprocessing_node = dict(config.get("preprocessing", {}) or {})
        data_node = dict(config.get("data", {}) or {})
        recipes = list(preprocessing_node.get("features", []) or [])
        effective_target = target if target is not None else data_node.get("target")
        return cls(recipes, target=effective_target)

    @property
    def output_names(self) -> list[str]:
        """Names of the columns produced by :meth:`transform`."""
        names: list[str] = []
        for recipe in self.recipes:
            if recipe.type == "datetime_parts":
                prefix = recipe.params.get("prefix", recipe.name)
                parts = recipe.params.get("parts", ["year", "month", "day", "dayofweek"])
                names.extend(f"{prefix}_{part}" for part in parts)
            else:
                names.append(recipe.name)
        return names

    def describe(self) -> list[dict[str, Any]]:
        """Return a machine readable description of every recipe."""
        return [
            {
                "type": recipe.type,
                "name": recipe.name,
                "inputs": recipe.required_columns(),
                "params": recipe.params,
            }
            for recipe in self.recipes
        ]

    # ------------------------------------------------------------------ execution -------
    def fit(self, frame: pd.DataFrame, y: pd.Series | None = None) -> FeatureBuilder:  # noqa: ARG002
        """Learn the statistics required by ``bin`` and ``group_stat`` recipes.

        Args:
            frame: Training data.
            y: Ignored (features must not depend on the target).

        Returns:
            ``self``.
        """
        self._learned = {}
        for recipe in self.recipes:
            if recipe.type == "bin":
                self._learned[recipe.name] = self._fit_bin(frame, recipe)
            elif recipe.type == "group_stat":
                self._learned[recipe.name] = self._fit_group_stat(frame, recipe)
        self._is_fitted = True
        if self.recipes:
            logger.info(
                "FeatureBuilder fitted | recipes={} learned={}",
                len(self.recipes),
                sorted(self._learned),
            )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add the derived columns to a copy of ``frame``.

        Args:
            frame: Data to enrich.

        Returns:
            A new frame containing the original columns plus the derived features.
        """
        if self.recipes and not self._is_fitted:
            msg = "FeatureBuilder is not fitted: call fit() before transform()"
            raise RuntimeError(msg)

        output = frame.copy()
        for recipe in self.recipes:
            try:
                if recipe.type == "datetime_parts":
                    # Ce type de recette ajoute plusieurs colonnes d'un coup.
                    output = self._apply_datetime_parts(output, recipe)
                elif recipe.type == "bin":
                    output[recipe.name] = self._apply_bin(output, recipe)
                elif recipe.type == "group_stat":
                    output[recipe.name] = self._apply_group_stat(output, recipe)
                else:
                    output[recipe.name] = self._apply_arithmetic(output, recipe)
            except Exception as exc:  # noqa: BLE001 - explicit, actionable message
                msg = f"Feature recipe '{recipe.name}' ({recipe.type}) failed: {exc}"
                raise ValueError(msg) from exc
        return output

    def fit_transform(self, frame: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit on ``frame`` then transform it.

        Args:
            frame: Training data.
            y: Ignored.

        Returns:
            The enriched training frame.
        """
        return self.fit(frame, y).transform(frame)

    # ------------------------------------------------------------------ recipes ---------
    def _apply_arithmetic(self, frame: pd.DataFrame, recipe: FeatureRecipe) -> pd.Series:
        """Evaluate ratio / difference / product / log1p / boolean_flag recipes."""
        params = recipe.params
        if recipe.type == "ratio":
            numerator = pd.to_numeric(frame[_required(frame, params, "numerator")], errors="coerce")
            denominator = pd.to_numeric(
                frame[_required(frame, params, "denominator")], errors="coerce"
            )
            epsilon = float(params.get("epsilon", 1e-6))
            # Un dénominateur nul (ou manquant) rend le ratio *indéfini* : on renvoie 0.0 plutôt
            # que NaN (qui ferait échouer ProcessedDataSchema) ou une valeur explosive
            # (numerator / epsilon), toutes deux nuisibles à l'apprentissage.
            undefined = denominator.isna() | denominator.eq(0)
            safe_denominator = denominator.abs().mask(undefined, 1.0) + epsilon
            return (numerator / safe_denominator).mask(undefined, 0.0).fillna(0.0)
        if recipe.type == "difference":
            left = pd.to_numeric(frame[_required(frame, params, "left")], errors="coerce")
            right = pd.to_numeric(frame[_required(frame, params, "right")], errors="coerce")
            return left - right
        if recipe.type == "product":
            columns = [str(name) for name in params.get("columns", [])] or [
                _required(frame, params, "a"),
                _required(frame, params, "b"),
            ]
            product = pd.Series(1.0, index=frame.index)
            for column in columns:
                product = product * pd.to_numeric(frame[column], errors="coerce")
            return product
        if recipe.type == "log1p":
            series = pd.to_numeric(frame[_required(frame, params, "column")], errors="coerce")
            return np.log1p(series.clip(lower=0))
        if recipe.type == "boolean_flag":
            operator = str(params.get("operator", "gt"))
            if operator not in COMPARISONS:
                msg = f"Unknown operator '{operator}'. Allowed: {sorted(COMPARISONS)}"
                raise ValueError(msg)
            other_column = params.get("value_column")
            threshold = params.get("value")
            if other_column is None and threshold is None:
                msg = "boolean_flag requires a 'value' threshold or a 'value_column'"
                raise ValueError(msg)
            series = _flag_operand(frame[_required(frame, params, "column")])
            if other_column is not None:
                # Comparaison colonne à colonne (p. ex. pays de session != pays de facturation) :
                # les deux opérandes passent par la même coercion pour rester comparables.
                reference: Any = _flag_operand(frame[_required(frame, params, "value_column")])
            elif pd.api.types.is_number(threshold):
                reference = float(threshold)  # type: ignore[arg-type]
            else:
                reference = str(threshold)
            return COMPARISONS[operator](series, reference).astype("int8")
        msg = f"Recipe type '{recipe.type}' is not handled by _apply_arithmetic"
        raise ValueError(msg)

    def _fit_bin(self, frame: pd.DataFrame, recipe: FeatureRecipe) -> dict[str, Any]:
        """Learn bin edges on the training split."""
        column = _required(frame, recipe.params, "column")
        series = pd.to_numeric(frame[column], errors="coerce")
        bins = int(recipe.params.get("bins", 4))
        method = str(recipe.params.get("method", "quantile"))
        if method == "quantile":
            quantiles = np.linspace(0.0, 1.0, bins + 1)
            edges = np.unique(series.quantile(quantiles).to_numpy(dtype="float64"))
        elif method == "uniform":
            edges = np.unique(np.linspace(float(series.min()), float(series.max()), bins + 1))
        else:
            msg = f"Unknown binning method '{method}'. Allowed: ['quantile', 'uniform']"
            raise ValueError(msg)
        if len(edges) < 2:
            logger.warning(
                "Feature '{}' cannot be binned (constant column '{}')", recipe.name, column
            )
            edges = np.array([float(series.min()), float(series.max())], dtype="float64")
        edges[0] = -np.inf
        edges[-1] = np.inf
        return {"column": column, "edges": edges, "labels": recipe.params.get("labels")}

    def _apply_bin(self, frame: pd.DataFrame, recipe: FeatureRecipe) -> pd.Series:
        """Apply the learned bin edges."""
        learned = self._learned.get(recipe.name)
        if learned is None:
            msg = f"Bin recipe '{recipe.name}' was not fitted"
            raise RuntimeError(msg)
        series = pd.to_numeric(frame[learned["column"]], errors="coerce")
        labels = learned.get("labels")
        coded = pd.cut(series, bins=learned["edges"], labels=labels, include_lowest=True)
        if labels is None:
            return coded.cat.codes.astype("int16").where(coded.notna(), other=-1)
        return coded.astype("object").fillna("missing").astype(str)

    def _fit_group_stat(self, frame: pd.DataFrame, recipe: FeatureRecipe) -> dict[str, Any]:
        """Learn a per-category aggregate on the training split."""
        group_by = _required(frame, recipe.params, "group_by")
        value_column = _required(frame, recipe.params, "value_column")
        aggregation = str(recipe.params.get("agg", "mean"))
        if aggregation not in {"mean", "median", "std", "count", "min", "max"}:
            msg = f"Unknown aggregation '{aggregation}'"
            raise ValueError(msg)
        values = pd.to_numeric(frame[value_column], errors="coerce")
        grouped = values.groupby(frame[group_by].astype("object"))
        statistic = getattr(grouped, aggregation)()
        fallback = float(values.mean()) if aggregation != "count" else 0.0
        if not np.isfinite(fallback):
            fallback = 0.0
        return {
            "group_by": group_by,
            "mapping": {str(key): float(value) for key, value in statistic.items()},
            "fallback": fallback,
        }

    def _apply_group_stat(self, frame: pd.DataFrame, recipe: FeatureRecipe) -> pd.Series:
        """Apply the learned per-category aggregate (unseen categories fall back)."""
        learned = self._learned.get(recipe.name)
        if learned is None:
            msg = f"Group-stat recipe '{recipe.name}' was not fitted"
            raise RuntimeError(msg)
        keys = frame[learned["group_by"]].astype("object").astype(str)
        return keys.map(learned["mapping"]).astype("float64").fillna(learned["fallback"])

    def _apply_datetime_parts(self, frame: pd.DataFrame, recipe: FeatureRecipe) -> pd.DataFrame:
        """Expand a timestamp column into calendar features."""
        column = _required(frame, recipe.params, "column")
        parts = [
            str(part) for part in recipe.params.get("parts", ["year", "month", "day", "dayofweek"])
        ]
        unknown = [part for part in parts if part not in DATETIME_PARTS]
        if unknown:
            msg = f"Unknown datetime part(s) {unknown}. Allowed: {DATETIME_PARTS}"
            raise ValueError(msg)
        prefix = str(recipe.params.get("prefix", recipe.name))
        timestamps = pd.to_datetime(frame[column], errors="coerce")
        output = frame.copy()
        for part in parts:
            if part == "is_weekend":
                output[f"{prefix}_{part}"] = (timestamps.dt.dayofweek >= 5).astype("int8")
            elif part == "is_month_start":
                output[f"{prefix}_{part}"] = timestamps.dt.is_month_start.astype("int8")
            elif part == "is_month_end":
                output[f"{prefix}_{part}"] = timestamps.dt.is_month_end.astype("int8")
            else:
                output[f"{prefix}_{part}"] = getattr(timestamps.dt, part).astype("int16")
        return output


def _required(frame: pd.DataFrame, params: Mapping[str, Any], key: str) -> str:
    """Return a mandatory recipe parameter and check that the column exists.

    Args:
        frame: Input frame.
        params: Recipe parameters.
        key: Parameter name holding a column name.

    Returns:
        The column name.

    Raises:
        ValueError: When the parameter is missing.
        KeyError: When the column is absent from the frame.
    """
    value = params.get(key)
    if not isinstance(value, str) or not value:
        msg = f"Recipe parameter '{key}' is required and must be a column name"
        raise ValueError(msg)
    if value not in frame.columns:
        msg = (
            f"Column '{value}' required by the recipe is missing. Available: {list(frame.columns)}"
        )
        raise KeyError(msg)
    return value


def split_by_dtype(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split column names into numeric and categorical according to their dtype.

    C'est la *seule* fonction du projet qui décide du type logique d'une colonne : le
    preprocessing, les tests et les notebooks l'utilisent tous, ce qui évite les divergences
    (par exemple une colonne ``category`` traitée comme numérique).

    Args:
        frame: Frame holding the columns.
        columns: Column names to classify.

    Returns:
        The ``(numeric_columns, categorical_columns)`` pair, in input order.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) and not isinstance(
            series.dtype, pd.CategoricalDtype
        ):
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def select_feature_columns(
    frame: pd.DataFrame, drop_columns: Sequence[str] = (), target: str | None = None
) -> list[str]:
    """Return the modelling columns of a frame.

    Args:
        frame: Input frame.
        drop_columns: Columns to exclude (identifiers, metadata, timestamps).
        target: Target column, always excluded.

    Returns:
        The ordered list of feature columns.
    """
    excluded: set[str] = {str(name) for name in drop_columns}
    if target:
        excluded.add(target)
    return [column for column in frame.columns if column not in excluded]
