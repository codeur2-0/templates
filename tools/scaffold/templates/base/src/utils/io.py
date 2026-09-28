"""Input / output helpers.

Reading and writing is centralised here so that:

* the format is chosen from the file extension (``.parquet``, ``.csv``, ``.json``, ``.yaml``),
* every artefact is written atomically (no half-written file when a job is interrupted),
* Parquet is the default for intermediate data (typed, compressed, columnar, pushdown friendly).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

SUPPORTED_TABLE_FORMATS: tuple[str, ...] = ("parquet", "csv")


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory of ``path`` and return it as :class:`Path`.

    Args:
        path: File path whose parent must exist.

    Returns:
        The resolved :class:`Path`.
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Write bytes atomically (temp file + ``os.replace``).

    Args:
        path: Destination file.
        payload: Bytes to write.

    Returns:
        The destination path.
    """
    destination = ensure_parent(path)
    handle, tmp_name = tempfile.mkstemp(dir=str(destination.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as tmp_file:
            tmp_file.write(payload)
        os.replace(tmp_name, destination)
    finally:
        if Path(tmp_name).exists():  # pragma: no cover - defensive cleanup
            Path(tmp_name).unlink()
    return destination


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    """Write text atomically.

    Args:
        path: Destination file.
        text: Text content.
        encoding: Text encoding.

    Returns:
        The destination path.
    """
    return atomic_write_bytes(path, text.encode(encoding))


# -----------------------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------------------
def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a table from Parquet or CSV, dispatching on the file extension.

    Args:
        path: File to read.
        **kwargs: Extra keyword arguments forwarded to the pandas reader.

    Returns:
        The loaded ``DataFrame``.

    Raises:
        FileNotFoundError: When the file does not exist.
        ValueError: When the extension is not supported.
    """
    file_path = Path(path)
    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)

    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(file_path, **kwargs)
    if suffix == ".csv":
        return pd.read_csv(file_path, **kwargs)
    msg = f"Unsupported table format '{suffix}' for {file_path}. Allowed: {SUPPORTED_TABLE_FORMATS}"
    raise ValueError(msg)


def write_table(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    **kwargs: Any,
) -> Path:
    """Write a ``DataFrame`` to Parquet or CSV, dispatching on the file extension.

    Args:
        frame: Data to persist.
        path: Destination file.
        index: Whether to persist the index.
        **kwargs: Extra keyword arguments forwarded to the pandas writer.

    Returns:
        The destination path.
    """
    file_path = ensure_parent(path)
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(file_path, index=index, **kwargs)
    elif suffix == ".csv":
        frame.to_csv(file_path, index=index, **kwargs)
    else:
        allowed = sorted(SUPPORTED_TABLE_FORMATS)
        msg = f"Unsupported table format '{suffix}' for {file_path}. Allowed: {allowed}"
        raise ValueError(msg)
    return file_path


def write_table_multiple(
    frame: pd.DataFrame, base_path: str | Path, formats: Iterable[str] = ("parquet", "csv")
) -> dict[str, Path]:
    """Write the same table in several formats (Parquet for machines, CSV for humans).

    Args:
        frame: Data to persist.
        base_path: Destination path; the extension is replaced per format.
        formats: Formats to write.

    Returns:
        Mapping of format to written path.
    """
    base = Path(base_path)
    written: dict[str, Path] = {}
    for fmt in formats:
        written[fmt] = write_table(frame, base.with_suffix(f".{fmt}"))
    return written


def resolve_table_path(
    directory: str | Path, name: str, formats: Iterable[str] = ("parquet", "csv")
) -> Path:
    """Find an existing table file, trying formats in priority order.

    Args:
        directory: Directory to inspect.
        name: Dataset name without extension.
        formats: Formats to try, in order.

    Returns:
        The first existing path.

    Raises:
        FileNotFoundError: When no candidate exists.
    """
    directory_path = Path(directory)
    for fmt in formats:
        candidate = directory_path / f"{name}.{fmt}"
        if candidate.exists():
            return candidate
    msg = f"No dataset '{name}' found in {directory_path} (tried: {list(formats)})"
    raise FileNotFoundError(msg)


# -----------------------------------------------------------------------------------------
# Structured documents
# -----------------------------------------------------------------------------------------
def read_json(path: str | Path) -> Any:
    """Read a JSON document.

    Args:
        path: File to read.

    Returns:
        The decoded object.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any, *, indent: int = 2, sort_keys: bool = False) -> Path:
    """Write a JSON document atomically.

    Args:
        path: Destination file.
        payload: JSON serialisable object.
        indent: Indentation width.
        sort_keys: Whether keys must be sorted.

    Returns:
        The destination path.
    """
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys, default=str, ensure_ascii=False)
    return atomic_write_text(path, text + "\n")


def read_yaml(path: str | Path) -> Any:
    """Read a YAML document.

    Args:
        path: File to read.

    Returns:
        The decoded object.
    """
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a YAML document atomically.

    Args:
        path: Destination file.
        payload: Mapping to serialise.

    Returns:
        The destination path.
    """
    text = yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True)
    return atomic_write_text(path, text)


def save_pickle(obj: Any, path: str | Path) -> Path:
    """Persist any picklable object with joblib (compressed).

    Args:
        obj: Object to persist.
        path: Destination file.

    Returns:
        The destination path.
    """
    import joblib  # local import: keeps this module usable without joblib installed

    file_path = ensure_parent(path)
    joblib.dump(obj, file_path, compress=3)
    return file_path


def load_pickle(path: str | Path) -> Any:
    """Load an object persisted with :func:`save_pickle`.

    Args:
        path: File to read.

    Returns:
        The restored object.
    """
    import joblib

    if not Path(path).exists():
        msg = f"Artefact not found: {path}"
        raise FileNotFoundError(msg)
    return joblib.load(path)


def write_text(path: str | Path, text: str) -> Path:
    """Write a text file atomically.

    Args:
        path: Destination file.
        text: Content.

    Returns:
        The destination path.
    """
    return atomic_write_text(path, text)
