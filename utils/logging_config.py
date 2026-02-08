"""Logging configuration for Jarvis."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for console."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
