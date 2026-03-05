"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""
"""
clean_data.py — Flight delay pipeline
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