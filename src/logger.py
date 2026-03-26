# src/logger.py
import logging
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to console and file."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch = logging.StreamHandler()          # console
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler('logs/pipeline.log')  # file
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
