"""Data loading, validation and splitting.

Responsibilities of this module (and only these):

* locate the dataset files (Parquet first, CSV fallback),
* load them into pandas,
* enforce the **Pandera contracts** defined in ``src/data/schemas.py``,
* split rows into train / validation / test according to the configuration.

Nothing here knows about models or preprocessing: the loaders are the boundary between the
outside world (files, warehouses) and the typed world of the application.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.schemas import InferenceDataSchema, ProcessedDataSchema, RawDataSchema
from src.utils.io import read_table, resolve_table_path, write_table
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """Train / validation / test splits, ready to be consumed by a model.

    Attributes:
        X_train: Training features (preprocessed).
        y_train: Training target (``None`` for unsupervised tasks).
        X_val: Validation features (may be ``None`` when ``val_size`` is 0).
        y_val: Validation target.
        X_test: Held-out test features.
        y_test: Held-out test target.
        feature_names: Ordered feature names (the model contract).
        task: Learning task identifier (``binary``, ``regression``, ``clustering``, ...).
    """

    X_train: pd.DataFrame
    y_train: pd.Series | None
    X_val: pd.DataFrame | None
    y_val: pd.Series | None
    X_test: pd.DataFrame
    y_test: pd.Series | None
    feature_names: list[str] = field(default_factory=list)
    task: str = "binary"

    @property
    def has_target(self) -> bool:
        """Whether the task is supervised."""
        return self.y_train is not None

    @property
    def n_train(self) -> int:
        """Number of training rows."""
        return int(len(self.X_train))

    @property
    def n_val(self) -> int:
        """Number of validation rows (0 when there is no validation split)."""
        return 0 if self.X_val is None else int(len(self.X_val))

    @property
    def n_test(self) -> int:
        """Number of test rows."""
        return int(len(self.X_test))

    def iter_splits(self) -> Iterator[tuple[str, pd.DataFrame, pd.Series | None]]:
        """Yield ``(name, X, y)`` for every non-empty split."""
        for name, features, target in (
            ("train", self.X_train, self.y_train),
            ("val", self.X_val, self.y_val),
            ("test", self.X_test, self.y_test),
        ):
            if features is not None:
                yield name, features, target

    def describe(self) -> dict[str, Any]:
        """Return a machine readable summary of the bundle."""
        return {
            "task": self.task,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "n_features": len(self.feature_names),
            "has_target": self.has_target,
        }


@dataclass(frozen=True, slots=True)
class SplitFrames:
    """Raw (not yet preprocessed) row splits, target included."""

    train: pd.DataFrame
    val: pd.DataFrame | None
    test: pd.DataFrame

    @property
    def sizes(self) -> dict[str, int]:
        """Number of rows per split."""
        return {
            "train": len(self.train),
            "val": 0 if self.val is None else len(self.val),
            "test": len(self.test),
        }


class BaseDataLoader(ABC):
    """Common behaviour of every loader: paths + optional Pandera validation."""

    #: Name used in logs.
    name: str = "base"

    def __init__(
        self,
        paths: ProjectPaths,
        *,
        dataset_name: str,
        formats: Sequence[str] = ("parquet", "csv"),
        validate: bool = True,
        lazy_validation: bool = False,
        strict: bool = True,
    ) -> None:
        """Store the loading contract.

        Args:
            paths: Project filesystem layout.
            dataset_name: Base name of the dataset (without extension).
            formats: Formats to try, in priority order.
            validate: Whether the Pandera contract must be enforced.
            lazy_validation: Collect every validation error instead of failing fast.
            strict: Reject columns that are not declared in the schema.
        """
        self.paths = paths
        self.dataset_name = dataset_name
        self.formats = tuple(formats)
        self.validate = validate
        self.lazy_validation = lazy_validation
        self.strict = strict

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load and validate the dataset.

        Returns:
            The loaded ``DataFrame``.
        """

    @property
    @abstractmethod
    def schema(self) -> type[Any] | None:
        """Pandera ``DataFrameModel`` enforced by this loader (``None`` to skip)."""

    def _apply_validation(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate a frame against :attr:`schema` when validation is enabled.

        Args:
            frame: Frame to validate.

        Returns:
            The validated (and possibly coerced) frame.
        """
        schema = self.schema
        if not self.validate or schema is None:
            logger.debug("{}: validation skipped", self.name)
            return frame
        logger.debug("{}: validating {} rows against {}", self.name, len(frame), schema.__name__)
        return schema.validate(frame, lazy=self.lazy_validation)

    @staticmethod
    def _log_profile(frame: pd.DataFrame, stage: str) -> None:
        """Log a compact profile of the loaded frame."""
        missing = int(frame.isna().to_numpy().sum())
        logger.info(
            "Loaded {} data | rows={} cols={} missing_cells={} memory={:.1f} KB",
            stage,
            len(frame),
            frame.shape[1],
            missing,
            frame.memory_usage(deep=True).sum() / 1024,
        )


class RawDataLoader(BaseDataLoader):
    """Load the raw dataset from ``data/raw`` and validate it with ``RawDataSchema``."""

    name = "raw-loader"

    def __init__(
        self,
        paths: ProjectPaths,
        *,
        dataset_name: str,
        formats: Sequence[str] = ("parquet", "csv"),
        validate: bool = True,
        lazy_validation: bool = False,
        strict: bool = True,
        limit: int | None = None,
    ) -> None:
        """Store the loading contract.

        Args:
            paths: Project filesystem layout.
            dataset_name: Dataset base name.
            formats: Formats to try.
            validate: Enforce the Pandera contract.
            lazy_validation: Collect all errors before raising.
            strict: Reject undeclared columns.
            limit: Optional maximum number of rows kept (fast iterations).
        """
        super().__init__(
            paths,
            dataset_name=dataset_name,
            formats=formats,
            validate=validate,
            lazy_validation=lazy_validation,
            strict=strict,
        )
        self.limit = limit

    @property
    def schema(self) -> type[Any] | None:
        """Return the raw data contract."""
        return RawDataSchema

    def resolve_path(self) -> Path:
        """Locate the raw dataset file.

        Returns:
            The path of the first existing candidate.

        Raises:
            FileNotFoundError: When the dataset was never generated.
        """
        try:
            return resolve_table_path(self.paths.raw_dir, self.dataset_name, self.formats)
        except FileNotFoundError as exc:
            hint = (
                f"{exc}\nGenerate it first: `python scripts/generate_data.py` "
                "(or `make data`)."
            )
            raise FileNotFoundError(hint) from exc

    def load(self) -> pd.DataFrame:
        """Read, validate and profile the raw dataset.

        Returns:
            The validated raw ``DataFrame``.
        """
        path = self.resolve_path()
        frame = read_table(path)
        if not self.strict and self.schema is not None:
            # ``strict=False`` : on tolère les colonnes non déclarées en les écartant avant
            # validation (comportement attendu pour une source externe qui évolue).
            declared = set(self.schema.to_schema().columns)
            frame = frame.loc[:, [column for column in frame.columns if column in declared]]
        frame = self._apply_validation(frame)
        if self.limit is not None:
            frame = frame.head(int(self.limit)).copy()
        self._log_profile(frame, "raw")
        return frame


class ProcessedDataLoader(BaseDataLoader):
    """Load an already processed dataset from ``data/processed`` (Parquet preferred)."""

    name = "processed-loader"

    def __init__(
        self,
        paths: ProjectPaths,
        *,
        dataset_name: str,
        formats: Sequence[str] = ("parquet",),
        validate: bool = True,
        lazy_validation: bool = False,
        strict: bool = False,
    ) -> None:
        """Store the loading contract (processed data is not necessarily strict on columns).

        Args:
            paths: Project filesystem layout.
            dataset_name: Dataset base name.
            formats: Formats to try.
            validate: Enforce the Pandera contract.
            lazy_validation: Collect all errors before raising.
            strict: Reject undeclared columns.
        """
        super().__init__(
            paths,
            dataset_name=dataset_name,
            formats=formats,
            validate=validate,
            lazy_validation=lazy_validation,
            strict=strict,
        )

    @property
    def schema(self) -> type[Any] | None:
        """Return the processed data contract."""
        return ProcessedDataSchema

    def load(self) -> pd.DataFrame:
        """Read and validate the processed dataset.

        Returns:
            The validated processed ``DataFrame``.
        """
        path = resolve_table_path(self.paths.processed_dir, self.dataset_name, self.formats)
        frame = read_table(path)
        frame = self._apply_validation(frame)
        self._log_profile(frame, "processed")
        return frame


class InferenceDataLoader(BaseDataLoader):
    """Load and validate data submitted for prediction (file or in-memory frame)."""

    name = "inference-loader"

    @property
    def schema(self) -> type[Any] | None:
        """Return the inference data contract."""
        return InferenceDataSchema

    def load(self) -> pd.DataFrame:
        """Read the configured inference input.

        Returns:
            The validated inference ``DataFrame``.

        Raises:
            FileNotFoundError: When no input file was configured or found.
        """
        raise FileNotFoundError(
            "InferenceDataLoader.load() requires an explicit source; use load_from(path) "
            "or load_from_frame(frame)."
        )

    def load_from(self, path: str | Path) -> pd.DataFrame:
        """Load and validate an inference file.

        Args:
            path: Parquet/CSV/JSON file containing the records to score.

        Returns:
            The validated frame.
        """
        file_path = Path(path)
        if not file_path.exists():
            msg = f"Inference input not found: {file_path}"
            raise FileNotFoundError(msg)
        frame = (
            pd.read_json(file_path, orient="records")
            if file_path.suffix.lower() == ".json"
            else read_table(file_path)
        )
        frame = self._apply_validation(frame)
        self._log_profile(frame, "inference")
        return frame

    def load_from_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate an in-memory frame before prediction.

        Args:
            frame: Records to score.

        Returns:
            The validated frame.
        """
        validated = self._apply_validation(frame)
        self._log_profile(validated, "inference")
        return validated


class DatasetSplitter:
    """Split rows into train / validation / test, with or without a temporal constraint.

    The splitter is *row level* only: no feature is computed here, which guarantees that the
    preprocessing statistics are always learned after the split (no leakage).
    """

    def __init__(
        self,
        *,
        test_size: float = 0.2,
        val_size: float = 0.15,
        stratify: bool = True,
        shuffle: bool = True,
        random_state: int = 42,
        time_column: str | None = None,
        time_based: bool = False,
    ) -> None:
        """Store the split policy.

        Args:
            test_size: Fraction of rows kept for the final evaluation.
            val_size: Fraction of rows kept for validation (0 to disable).
            stratify: Preserve the target distribution (supervised tasks only).
            shuffle: Shuffle rows before splitting (ignored for temporal splits).
            random_state: Seed of the split.
            time_column: Column holding the timestamp, required for temporal splits.
            time_based: Force a chronological split.

        Raises:
            ValueError: When the sizes are inconsistent.
        """
        if not 0.0 < test_size < 0.5:
            msg = f"test_size must be in (0, 0.5), got {test_size}"
            raise ValueError(msg)
        if not 0.0 <= val_size < 0.5:
            msg = f"val_size must be in [0, 0.5), got {val_size}"
            raise ValueError(msg)
        if test_size + val_size >= 0.9:
            msg = f"test_size + val_size must stay below 0.9, got {test_size + val_size:.2f}"
            raise ValueError(msg)
        if time_based and not time_column:
            msg = "A chronological split requires 'time_column' to be set"
            raise ValueError(msg)

        self.test_size = float(test_size)
        self.val_size = float(val_size)
        self.stratify = bool(stratify)
        self.shuffle = bool(shuffle) and not time_based
        self.random_state = int(random_state)
        self.time_column = time_column
        self.time_based = bool(time_based)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, seed: int | None = None) -> DatasetSplitter:
        """Build a splitter from the Hydra configuration.

        Args:
            config: Full application config (or its ``train`` / ``data`` nodes).
            seed: Explicit seed. When provided it **overrides** ``config['seed']`` (used by the
                tests to check that the split really depends on the seed); when ``None`` the
                configured seed is used, with ``42`` as last resort.

        Returns:
            The configured splitter.
        """
        train_node = dict(config.get("train", {}) or {})
        data_node = dict(config.get("data", {}) or {})
        split_node = dict(train_node.get("split", {}) or {})
        return cls(
            test_size=float(split_node.get("test_size", 0.2)),
            val_size=float(split_node.get("val_size", 0.15)),
            stratify=bool(split_node.get("stratify", True)),
            shuffle=bool(split_node.get("shuffle", True)),
            random_state=int(seed if seed is not None else config.get("seed", 42)),
            time_column=data_node.get("time_column"),
            time_based=bool(split_node.get("time_based", False)) or bool(data_node.get("time_column")),
        )

    def split(self, frame: pd.DataFrame, target: str | None = None) -> SplitFrames:
        """Split a frame into train / validation / test.

        Args:
            frame: Dataset to split.
            target: Target column used for stratification (optional).

        Returns:
            The :class:`SplitFrames` holding the three splits.
        """
        if self.time_based and self.time_column:
            return self._split_temporal(frame)

        stratify_labels = self._stratify_labels(frame, target)
        remaining_val_fraction = (
            self.val_size / (1.0 - self.test_size) if self.val_size > 0 else 0.0
        )

        train_frame, test_frame = train_test_split(
            frame,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify_labels,
        )
        val_frame: pd.DataFrame | None = None
        if remaining_val_fraction > 0:
            val_stratify = self._stratify_labels(train_frame, target)
            train_frame, val_frame = train_test_split(
                train_frame,
                test_size=remaining_val_fraction,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=val_stratify,
            )

        splits = SplitFrames(
            train=train_frame.reset_index(drop=True),
            val=None if val_frame is None else val_frame.reset_index(drop=True),
            test=test_frame.reset_index(drop=True),
        )
        logger.info(
            "Split (random) | train={} val={} test={} | stratify={}",
            splits.sizes["train"],
            splits.sizes["val"],
            splits.sizes["test"],
            self.stratify and target is not None,
        )
        return splits

    def _split_temporal(self, frame: pd.DataFrame) -> SplitFrames:
        """Chronological split: the past trains, the future is evaluated."""
        assert self.time_column is not None  # noqa: S101 - guarded in __init__
        ordered = frame.sort_values(self.time_column).reset_index(drop=True)
        n_rows = len(ordered)
        n_test = max(int(round(n_rows * self.test_size)), 1)
        n_val = max(int(round(n_rows * self.val_size)), 0) if self.val_size > 0 else 0
        n_train = n_rows - n_test - n_val
        if n_train <= 0:
            msg = (
                f"Chronological split leaves no training row (rows={n_rows}, "
                f"test={n_test}, val={n_val}); reduce test_size/val_size."
            )
            raise ValueError(msg)

        train_frame = ordered.iloc[:n_train]
        val_frame = ordered.iloc[n_train : n_train + n_val] if n_val else None
        test_frame = ordered.iloc[n_train + n_val :]
        splits = SplitFrames(train=train_frame.copy(), val=val_frame.copy() if val_frame is not None else None, test=test_frame.copy())
        logger.info(
            "Split (chronological on '{}') | train={} val={} test={} | {} -> {}",
            self.time_column,
            splits.sizes["train"],
            splits.sizes["val"],
            splits.sizes["test"],
            ordered[self.time_column].min(),
            ordered[self.time_column].max(),
        )
        return splits

    def _stratify_labels(self, frame: pd.DataFrame, target: str | None) -> pd.Series | None:
        """Return stratification labels, or ``None`` when stratification is impossible."""
        if not self.stratify or target is None or target not in frame.columns:
            return None
        labels = frame[target]
        counts = labels.value_counts()
        if counts.min() < 2:
            logger.warning(
                "Stratification disabled: class(es) {} have fewer than 2 rows",
                list(counts[counts < 2].index),
            )
            return None
        return labels


def save_split(split: pd.DataFrame, paths: ProjectPaths, name: str, *, fmt: str = "parquet") -> Path:
    """Persist one split in ``data/processed``.

    Args:
        split: Split to persist.
        paths: Project filesystem layout.
        name: File base name (e.g. ``train``).
        fmt: Output format.

    Returns:
        The written path.
    """
    return write_table(split, paths.processed_dir / f"{name}.{fmt}")


def feature_target_split(
    frame: pd.DataFrame, target: str | None, drop_columns: Sequence[str] = ()
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Separate features and target, dropping non-modelling columns.

    Args:
        frame: Split to separate.
        target: Target column name (``None`` for unsupervised tasks).
        drop_columns: Columns to exclude from the features.

    Returns:
        The ``(X, y)`` pair; ``y`` is ``None`` when there is no target.
    """
    excluded = set(drop_columns)
    if target is not None:
        excluded.add(target)
    feature_columns = [column for column in frame.columns if column not in excluded]
    features = frame.loc[:, feature_columns]
    if target is None or target not in frame.columns:
        return features, None
    labels = frame[target]
    if isinstance(labels, pd.DataFrame):  # pragma: no cover - defensive
        labels = labels.iloc[:, 0]
    return features, labels.astype("object" if labels.dtype == object else labels.dtype)


def target_distribution(labels: pd.Series | None) -> dict[str, float]:
    """Return the relative frequency of each class (useful for logs and reports).

    Args:
        labels: Target series (may be ``None``).

    Returns:
        Mapping of class to proportion, empty when there is no target.
    """
    if labels is None:
        return {}
    counts = labels.value_counts(normalize=True)
    return {str(index): float(value) for index, value in counts.items()}


def assert_no_overlap(*frames: pd.DataFrame, key: str | None = None) -> None:
    """Assert that splits share no row identifier (leakage guard).

    Args:
        *frames: Splits to compare.
        key: Identifier column. When ``None``, the pandas index is used.

    Raises:
        AssertionError: When an identifier appears in more than one split.
    """
    valid = [frame for frame in frames if frame is not None and len(frame) > 0]
    if len(valid) < 2:
        return
    keys: list[set[Any]] = []
    for frame in valid:
        if key is not None and key in frame.columns:
            keys.append(set(frame[key].tolist()))
        else:
            keys.append(set(map(tuple, np.asarray(frame.index).tolist() if hasattr(frame.index, "tolist") else frame.index)))
    for first in range(len(keys)):
        for second in range(first + 1, len(keys)):
            overlap = keys[first] & keys[second]
            if key is not None and overlap:
                msg = f"Data leakage: {len(overlap)} '{key}' value(s) appear in several splits"
                raise AssertionError(msg)
