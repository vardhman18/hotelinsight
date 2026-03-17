"""
HotelInsight - Logging Configuration
======================================
Provides a centralised logger that writes to both the console and a rotating
log file.  Import ``get_logger`` in any module and call it with the module's
``__name__`` to get a pre-configured logger instance.

Usage::

    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started")
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger for the given module name.

    The logger writes:
    - INFO+ messages to the console with a concise format.
    - DEBUG+ messages to a rotating file (10 MB cap, 5 backups) with a
      detailed format including timestamps, level, and module name.

    The log directory is created automatically if it does not exist.

    Args:
        name: Typically ``__name__`` of the calling module.
        level: Minimum severity level as a string (e.g. ``"DEBUG"``,
            ``"INFO"``, ``"WARNING"``).

    Returns:
        A ``logging.Logger`` instance ready for use.
    """
    # Resolve log directory relative to the project root
    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "hotelinsight.log"

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger is already configured
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(logging.DEBUG)  # Capture all; handlers will filter

    # ------------------------------------------------------------------
    # Console handler – INFO and above, concise single-line format
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # ------------------------------------------------------------------
    # File handler – DEBUG and above, detailed multi-field format
    # Rotates when the file exceeds 10 MB, keeping 5 history files.
    # ------------------------------------------------------------------
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger
