"""Cross-cutting utilities: logging, paths, IO and small helpers."""

from src.utils.io import (
    atomic_write_text,
    ensure_parent,
    load_pickle,
    read_json,
    read_table,
    read_yaml,
    resolve_table_path,
    save_pickle,
    write_json,
    write_table,
    write_table_multiple,
    write_yaml,
)
from src.utils.logging import get_logger, setup_logging
from src.utils.paths import PROJECT_ROOT, ProjectPaths
from src.utils.utils import (
    chunked,
    flatten_dict,
    frame_fingerprint,
    human_number,
    safe_division,
    set_seed,
    timer,
)

__all__ = [
    "PROJECT_ROOT",
    "ProjectPaths",
    "atomic_write_text",
    "chunked",
    "ensure_parent",
    "flatten_dict",
    "frame_fingerprint",
    "get_logger",
    "human_number",
    "load_pickle",
    "read_json",
    "read_table",
    "read_yaml",
    "resolve_table_path",
    "safe_division",
    "save_pickle",
    "set_seed",
    "setup_logging",
    "timer",
    "write_json",
    "write_table",
    "write_table_multiple",
    "write_yaml",
]
