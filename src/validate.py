from __future__ import annotations

import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)


def _check_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error("[validate] Missing required columns: %s", missing)
        raise ValueError(f"Missing columns: {missing}")


def _check_missing_values(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns and df[col].isna().any():
            logger.error("[validate] Null values found in column '%s'", col)
            raise ValueError(f"Null values found in column '{col}'")


def _check_target_values(df: pd.DataFrame, target_col: str, allowed: list) -> None:
    unique_vals = set(df[target_col].dropna().unique())
    unexpected = unique_vals - set(allowed)
    if unexpected:
        logger.error(
            "[validate] Unexpected target values in '%s': %s", target_col, unexpected
        )
        raise ValueError(f"Unexpected target values in '{target_col}': {unexpected}")
    if len(unique_vals) < 2:
        logger.error(
            "[validate] Target '%s' has only 1 class present: %s", target_col, unique