from __future__ import annotations
import pandas as pd


def validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def validate_target_binary(df: pd.DataFrame, target_col: str) -> None:
    vals = sorted(df[target_col].dropna().unique().tolist())
    if vals not in ([0, 1], [0], [1]):
        raise ValueError(f"Target '{target_col}' must be 0/1. Found: {vals}")