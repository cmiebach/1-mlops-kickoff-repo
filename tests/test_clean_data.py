"""
Tests for src/clean_data.py
"""
import pytest
import pandas as pd
import numpy as np

from src.clean_data import (
    _rename_columns,
    _engineer_binary_flags,
    _drop_unused_columns,
    _handle_missing_values,
    clean_dataframe,
)


@pytest.fixture
def bts_df():
    """Raw BTS-style DataFrame before renaming."""
    return pd.DataFrame({
        "MONTH":               [6, 7, 8],
        "DAY":                 [3, 15, 26],   # Saturday=3 Jun, Saturday=15 Jul, Saturday=26 Aug
        "CRS_DEP_TIME":        [800, 2300, 100],
        "ARR_DELAY":           [10.0, 30.0, -5.0],
        "AIRLINE":             ["BA", "AA", "UA"],
        "ORIGIN_AIRPORT":      ["EGLL", "EGLL", "EGLL"],
        "DESTINATION_AIRPORT": ["CDG", "JFK", "FRA"],
        "AIR_TIME":            [90, 420, 120],
        "DISTANCE":            [340, 5570, 620],
        "temperature_2m":      [18.0, 22.0, 15.0],
        "precipitation":       [0.0, 1.5, 0.0],
        "windspeed_10m":       [10.0, 25.0, 5.0],
        "visibility":          [10000.0, 8000.0, 50000.0],
        "cloudcover":          [50.0, 80.0, 10.0],
        "weathercode":         [0, 45, 95],      # clear, fog, storm
        "flight_duration_s":   [5400, 25200, 7200],
        "delayed":             [0, 1, 0],
    })