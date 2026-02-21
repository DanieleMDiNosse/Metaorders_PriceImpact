"""
Logging helpers shared across the repository's scripts.

Several analysis scripts historically monkey-patched `builtins.print` so that
diagnostics are both visible in the terminal and persisted to a log file. This
module provides a safer context-manager version of that behavior: printing is
only patched while inside the context, and it is always restored afterwards.
"""

from __future__ import annotations

import builtins
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


def setup_file_logger(
    name: str,
    log_path: Path,
    *,
    mode: str = "a",
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(message)s",
    encoding: str = "utf-8",
    errors: Optional[str] = None,
    also_stdout: bool = False,
    reset_handlers: bool = False,
) -> logging.Logger:
    """
    Summary
    -------
    Create (or reuse) a logger writing to a file (and optionally stdout).

    Parameters
    ----------
    name : str
        Logger name (e.g., script stem).
    log_path : Path
        Log file location.
    mode : str, default="a"
        File mode for the log file.
    level : int, default=logging.INFO
        Logging level for the logger and handlers.
    fmt : str, default="%(asctime)s - %(message)s"
        Log formatter string.
    encoding : str, default="utf-8"
        Encoding used by the FileHandler.
    errors : Optional[str], default=None
        Error handling strategy for the FileHandler (e.g., "backslashreplace").
    also_stdout : bool, default=False
        If True, add a StreamHandler to stdout.
    reset_handlers : bool, default=False
        If True, remove and close existing handlers before adding new ones.

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.

    Notes
    -----
    - `logger.propagate` is set to False to avoid duplicate logs from the root logger.
    - When `reset_handlers=False` and handlers already exist, the logger is returned
      unchanged to avoid handler duplication across repeated imports.

    Examples
    --------
    >>> import logging
    >>> from pathlib import Path
    >>> logger = setup_file_logger("demo", Path("demo.log"), reset_handlers=True)
    >>> isinstance(logger, logging.Logger)
    True
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers and not reset_handlers:
        return logger

    if reset_handlers and logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    formatter = logging.Formatter(fmt)

    file_handler = logging.FileHandler(
        log_path,
        mode=mode,
        encoding=encoding,
        errors=errors,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if also_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


@dataclass
class PrintTee:
    """
    Summary
    -------
    Context manager that temporarily tees `print()` calls to a logger.

    Parameters
    ----------
    logger : logging.Logger
        Logger used to persist printed messages (via `logger.info`).
    swallow_logger_errors : bool, default=True
        If True, exceptions from the logger are ignored so logging never breaks
        research runs.

    Returns
    -------
    PrintTee
        The context manager instance.

    Notes
    -----
    - Only active inside the context; always restores `builtins.print` on exit.
    - Logging uses the same `sep` join as `print`, and does not include the `end`
      terminator.
    - If printing raises `UnicodeEncodeError` (common with legacy terminals),
      the output is encoded with a tolerant error handler ("backslashreplace")
      before being written.

    Examples
    --------
    >>> from pathlib import Path
    >>> import logging
    >>> logger = setup_file_logger("demo", Path("demo.log"), reset_handlers=True)
    >>> with PrintTee(logger):
    ...     print("hello", 1, 2, sep="|")
    hello|1|2
    """

    logger: logging.Logger
    swallow_logger_errors: bool = True

    _original_print: Optional[Callable[..., None]] = None

    def __enter__(self) -> "PrintTee":
        self._original_print = builtins.print

        def _patched_print(*args, **kwargs):
            sep = kwargs.get("sep", " ")
            message = sep.join(str(a) for a in args)

            try:
                self.logger.info(message)
            except Exception:
                if not self.swallow_logger_errors:
                    raise

            try:
                self._original_print(*args, **kwargs)
            except UnicodeEncodeError:
                stream = kwargs.get("file", sys.stdout)
                encoding = getattr(stream, "encoding", None) or "utf-8"
                safe = message.encode(encoding, errors="backslashreplace").decode(
                    encoding, errors="ignore"
                )
                end = kwargs.get("end", "\n")
                flush = kwargs.get("flush", False)
                self._original_print(safe, file=stream, end=end, flush=flush)

        builtins.print = _patched_print
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._original_print is not None:
            builtins.print = self._original_print
        self._original_print = None
        return False

