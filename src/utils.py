from __future__ import annotations

from pathlib import Path

import joblib


def load_model(path: str | Path):
    """Load a joblib-serialised model pipeline from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return joblib.load(p)
