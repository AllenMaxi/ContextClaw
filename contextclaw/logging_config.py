"""Structured logging configuration for ContextClaw.

Call :func:`setup_logging` once at application startup (CLI or server).
All modules use ``logging.getLogger(__name__)`` — this configures the
root ``contextclaw`` logger with a structured JSON formatter for
production and a human-readable formatter for development.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


# Standard LogRecord attribute names — computed once at import time.
_STANDARD_RECORD_ATTRS: frozenset[str] = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
)


class StructuredFormatter(logging.Formatter):
    """JSON-lines formatter for structured log output.

    Each log record becomes a single JSON object with consistent fields:
    ``timestamp``, ``level``, ``logger``, ``message``, plus any extra fields
    attached via the ``extra`` dict.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            )
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        # Merge extra fields (skip standard LogRecord attributes)
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in entry:
                entry[key] = value
        return json.dumps(entry, default=str)


class HumanFormatter(logging.Formatter):
    """Concise human-readable formatter for development."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )


def setup_logging(
    *,
    level: str = "INFO",
    structured: bool = False,
) -> None:
    """Configure the ``contextclaw`` logger hierarchy.

    Parameters
    ----------
    level:
        Log level name (DEBUG, INFO, WARNING, ERROR).
    structured:
        If True, use JSON-lines output. Otherwise use human-readable format.
    """
    root = logging.getLogger("contextclaw")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates on re-init
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(StructuredFormatter() if structured else HumanFormatter())
    root.addHandler(handler)

    # Don't propagate to the root logger
    root.propagate = False
