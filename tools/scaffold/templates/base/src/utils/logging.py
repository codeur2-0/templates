"""Logging configuration.

The project uses `loguru <https://loguru.readthedocs.io>`_ for application logs: a single
importable ``logger`` object, structured formatting, rotation and no configuration burden
in every module. The standard library ``logging`` records emitted by third-party libraries
(numpy, matplotlib, hydra, ...) are intercepted and routed to loguru so that everything
ends up in the same place.

Usage:
    >>> from src.utils.logging import setup_logging, get_logger
    >>> setup_logging(level="INFO")
    >>> log = get_logger(__name__)
    >>> log.info("pipeline started")
"""

from __future__ import annotations

import logging
from types import FrameType
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

#: Console format: readable, colourised, with module and line number.
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

#: File format: the same, without colours.
FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"

_CONFIGURED = False


class InterceptHandler(logging.Handler):
    """Route ``logging`` records to ``loguru``.

    Libraries configured with the standard ``logging`` module keep working, while the
    application only has one logging pipeline to reason about.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Forward a stdlib record to loguru.

        Args:
            record: The standard logging record to forward.
        """
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    *,
    json_logs: bool = False,
    intercept_stdlib: bool = True,
    quiet_third_party: bool = True,
) -> None:
    """Configure loguru once for the whole process.

    Args:
        level: Minimum log level (``DEBUG``, ``INFO``, ``WARNING``, ...).
        log_file: Optional path of a rotating log file.
        json_logs: Emit JSON lines instead of human readable text.
        intercept_stdlib: Redirect ``logging`` records to loguru.
        quiet_third_party: Raise the level of noisy third-party loggers to ``WARNING``.
    """
    global _CONFIGURED  # noqa: PLW0603 - module level singleton, intentional

    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format="{message}" if json_logs else CONSOLE_FORMAT,
        serialize=json_logs,
        colorize=not json_logs,
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            path,
            level=level.upper(),
            format=FILE_FORMAT,
            rotation="10 MB",
            retention=5,
            encoding="utf-8",
            backtrace=False,
            diagnose=False,
        )

    if intercept_stdlib:
        logging.basicConfig(handlers=[InterceptHandler()], level=level.upper(), force=True)

    if quiet_third_party:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True
    logger.debug("Logging configured (level={}, json={}, file={})", level, json_logs, log_file)


_NOISY_LOGGERS: tuple[str, ...] = (
    "matplotlib",
    "PIL",
    "urllib3",
    "filelock",
    "fsspec",
    "asyncio",
    "watchfiles",
    "tensorflow",
    "absl",
    "mlflow",
    "prefect",
    "httpx",
    "httpcore",
    "numexpr",
    "hpack",
)


def get_logger(name: str | None = None) -> Any:
    """Return a bound loguru logger.

    Args:
        name: Optional context name (usually ``__name__``).

    Returns:
        A loguru logger instance.
    """
    if not _CONFIGURED:
        setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
    return logger.bind(context=name) if name else logger


__all__ = ["CONSOLE_FORMAT", "FILE_FORMAT", "InterceptHandler", "get_logger", "setup_logging"]
