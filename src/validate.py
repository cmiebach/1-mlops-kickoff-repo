from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _check_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _check_missing_values(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns and df[col].isna().any():
            raise ValueError(f"Null values found in column '{col}'")


def _check_target_values(df: pd.DataFrame, target_col: str, allowed: list) -> None:
    unique_vals = set(df[target_col].dropna().unique())
    unexpected = unique_vals - set(allowed)
    if unexpected:
        raise ValueError(f"Unexpected target values in '{target_col}': {unexpected}")
    if len(unique_vals) < 2:
        raise ValueError(
            f"Target '{target_col}' has only 1 class present: {unique_vals}"
        )


def _check_non_negative(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        if (df[col].dropna() < 0).any():
            logger.warning("Column '%s' contains negative values", col)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    check_missing_values: bool = True,
    target_column: str | None = None,
    target_allowed_values: list | None = None,
    numeric_non_negative_cols: list[str] | None = None,
    min_rows: int | None = None,
) -> None:
    if min_rows is not None and len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, need {min_rows}")

    if required_columns:
        _check_required_columns(df, required_columns)

    if check_missing_values and required_columns:
        _check_missing_values(df, required_columns)

    if target_column and target_allowed_values is not None:
        _check_target_values(df, target_column, target_allowed_values)

    if numeric_non_negative_cols:
        _check_non_negative(df, numeric_non_negative_cols)


def validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    _check_required_columns(df, required)


def validate_target_binary(df: pd.DataFrame, target_col: str) -> None:
    vals = sorted(df[target_col].dropna().unique().tolist())
    if vals not in ([0, 1], [0], [1]):
        raise ValueError(f"Target '{target_col}' must be 0/1. Found: {vals}")
