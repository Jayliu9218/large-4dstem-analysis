"""Lightweight logging support for long-running 4D-STEM workflows.

Every module can obtain a logger via `get_logger(__name__)`.  Log level is
controlled by the ``FOURDSTEM_LOG_LEVEL`` environment variable (default
``INFO``) and can be overridden per invocation through
``configure_pipeline_logging(level=...)``.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any


_LOG_FORMAT = "%(asctime)s  %(levelname)-7s  [%(name)s]  %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of ``"fourdstem_pipeline"``."""
    return logging.getLogger("fourdstem_pipeline").getChild(name.rsplit(".", 1)[-1] if "." in name else name)


def configure_pipeline_logging(
    level: int | str = logging.INFO,
    *,
    stream: Any = sys.stderr,
    format_str: str = _LOG_FORMAT,
    datefmt: str = _DATE_FORMAT,
) -> None:
    """Set up the root pipeline logger once per process.

    Called automatically on first ``get_logger`` if not already configured.
    Safe to call multiple times — subsequent calls update the handler level
    without adding duplicate handlers.
    """
    root = logging.getLogger("fourdstem_pipeline")
    root.setLevel(_resolve_level(level))

    if not root.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter(format_str, datefmt=datefmt))
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setLevel(_resolve_level(level))

    # Keep the root logger quiet so downstream libraries don't spam.
    root.propagate = False


def _resolve_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, level.upper(), logging.INFO)


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def log_stage_start(log: logging.Logger, stage: str, **fields: Any) -> None:
    extra = _fmt_extra(fields)
    log.info("▶  %s started%s", stage, extra)


def log_stage_end(log: logging.Logger, stage: str, elapsed: float, **fields: Any) -> None:
    extra = _fmt_extra(fields)
    log.info("✓  %s completed  (%.1f s)%s", stage, elapsed, extra)


def log_block_progress(log: logging.Logger, *, block: int, total_blocks: int, stage: str, every: int = 20) -> None:
    """Log progress every *every* blocks (or on the first and last block)."""
    if block == 1 or block == total_blocks or block % every == 0:
        pct = block / max(total_blocks, 1) * 100
        log.debug("   block %d/%d  (%.0f%%)  [%s]", block, total_blocks, pct, stage)


def _fmt_extra(fields: dict[str, Any]) -> str:
    if not fields:
        return ""
    parts = ", ".join(f"{k}={v!r}" for k, v in fields.items())
    return f"  ({parts})"


# Auto-configure from env var on import.
_env_level = os.environ.get("FOURDSTEM_LOG_LEVEL", "")
if _env_level:
    configure_pipeline_logging(level=_env_level)
