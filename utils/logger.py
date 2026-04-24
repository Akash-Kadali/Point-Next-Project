"""Tiny logging helper. Nothing fancy — just a named stream logger."""
import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "pointnext", log_file: Optional[str] = None) -> logging.Logger:
    """Return a configured logger. Safe to call many times.

    Args:
        name: logger name (usually the module name).
        log_file: optional path to also write to a file.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
