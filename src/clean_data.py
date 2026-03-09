"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""

from __future__ import annotations
import datetime
import logging
import pandas as pd

logger = logging.getLogger(__name__)

_FOG_CODES   = {45, 48}
_STORM_CODES = {95, 96, 99, 65, 67, 75, 77}
_NIGHT_HOURS = set(range(0, 6)) | {22, 23}


def _rename_columns(df):
    rename_map = {
        "MONTH": "month", "DAY": "day",
        "CRS_DEP_TIME": "crs_dep_time", "ARR_DELAY": "arr_delay",
        "AIRLINE": "airline", "ORIGIN_AIRPORT": "origin_airport",
        "DESTINATION_AIRPORT": "destination_airport",
        "AIR_TIME": "air_time", "DISTANCE": "distance",
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=existing)


def _engineer_binary_flags(df):
    df = df.copy()
    if "weathercode" in df.columns:
        df["is_foggy"]  = df["weathercode"].isin(_FOG_CODES).astype(int)
        df["is_stormy"] = df["weathercode"].isin(_STORM_CODES).astype(int)
    else:
        df["is_foggy"] = df["is_stormy"] = 0

    if "crs_dep_time" in df.columns:
        dep_hour = (df["crs_dep_time"] // 100).clip(0, 23)
        df["is_night_departure"] = dep_hour.isin(_NIGHT_HOURS).astype(int)
    else:
        df["is_night_departure"] = 0

    if "month" in df.columns and "day" in df.columns:
        def _is_weekend(row):
            try:
                return int(datetime.date(2023, int(row["month"]), int(row["day"]))
                           .weekday() in [5, 6])
            except Exception:
                return 0
        df["is_weekend"] = df.apply(_is_weekend, axis=1)
    else:
        df["is_weekend"] = 0

    return df


def _drop_unused_columns(df, target_column):
    keep = [
        target_column, "temperature_2m", "precipitation", "windspeed_10m",
        "cloudcover", "flight_duration_s", "air_time", "distance",
        "is_foggy", "is_stormy", "is_night_departure", "is_weekend",
    ]
    available = [c for c in keep if c in df.columns]
    return df[available]


def _handle_missing_values(df):
    df = df.copy()
    numeric_cols = [
        "temperature_2m", "precipitation", "windspeed_10m",
        "cloudcover", "flight_duration_s", "air_time", "distance",
    ]
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in ["is_foggy", "is_stormy", "is_night_departure", "is_weekend"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    return df


def clean_dataframe(
    df: pd.DataFrame,
    target_column: str = "delayed",
) -> pd.DataFrame:
    """Run the full cleaning pipeline on the raw DataFrame."""
    logger.info("[clean_data] Starting. Input shape: %s", df.shape)
    df = _rename_columns(df)
    print("[DEBUG] cols after rename:", [c for c in df.columns if c in ("weathercode", "crs_dep_time", "month", "day")])
    df = _engineer_binary_flags(df)
    print("[DEBUG] cols after flags:", [c for c in df.columns if "is_" in c])
    df = _drop_unused_columns(df, target_column)
    print("[DEBUG] final cols:", list(df.columns))
    df = _handle_missing_values(df)
    if target_column not in df.columns:
        raise ValueError(f"[clean_data] Target '{target_column}' not found after cleaning.")
    logger.info("[clean_data] Done. Output shape: %s", df.shape)
    return df