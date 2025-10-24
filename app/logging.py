"""Logging utilities."""
from __future__ import annotations

import logging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def configure_logging() -> None:
    """Configure the root logger with a sensible default format."""

    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


__all__ = ["configure_logging"]
