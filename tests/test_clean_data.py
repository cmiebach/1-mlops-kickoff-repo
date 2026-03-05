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


# ---------------------------------------------------------------------------
# _rename_columns
# ---------------------------------------------------------------------------

class TestRenameColumns:
    def test_renames_bts_columns(self, bts_df):
        result = _rename_columns(bts_df)
        assert "month" in result.columns
        assert "MONTH" not in result.columns
        assert "crs_dep_time" in result.columns
        assert "air_time" in result.columns

    def test_ignores_already_lowercase_columns(self):
        df = pd.DataFrame({"temperature_2m": [1.0], "delayed": [0]})
        result = _rename_columns(df)
        assert "temperature_2m" in result.columns

    def test_leaves_unknown_columns_unchanged(self, bts_df):
        bts_df["my_custom_col"] = 99
        result = _rename_columns(bts_df)
        assert "my_custom_col" in result.columns


# ---------------------------------------------------------------------------
# _engineer_binary_flags
# ---------------------------------------------------------------------------

class TestEngineerBinaryFlags:
    def test_fog_code_sets_is_foggy(self):
        df = pd.DataFrame({"weathercode": [45, 48, 0, 1]})
        result = _engineer_binary_flags(df)
        assert list(result["is_foggy"]) == [1, 1, 0, 0]

    def test_storm_code_sets_is_stormy(self):
        df = pd.DataFrame({"weathercode": [95, 65, 0, 1]})
        result = _engineer_binary_flags(df)
        assert list(result["is_stormy"]) == [1, 1, 0, 0]

    def test_night_hours_set_is_night_departure(self):
        # 0000 = midnight (night), 0800 = morning (day), 2200 = night
        df = pd.DataFrame({"weathercode": [0, 0, 0], "crs_dep_time": [0, 800, 2200]})
        result = _engineer_binary_flags(df)
        assert list(result["is_night_departure"]) == [1, 0, 1]

    def test_weekend_detection(self):
        # 2023-06-03 = Saturday, 2023-06-05 = Monday
        df = pd.DataFrame({
            "weathercode": [0, 0],
            "month": [6, 6],
            "day": [3, 5],
        })
        result = _engineer_binary_flags(df)
        assert result["is_weekend"].iloc[0] == 1   # Saturday
        assert result["is_weekend"].iloc[1] == 0   # Monday

    def test_defaults_to_zero_if_weathercode_missing(self):
        df = pd.DataFrame({"some_col": [1, 2, 3]})
        result = _engineer_binary_flags(df)
        assert all(result["is_foggy"] == 0)
        assert all(result["is_stormy"] == 0)

    def test_defaults_to_zero_if_dep_time_missing(self):
        df = pd.DataFrame({"weathercode": [0]})
        result = _engineer_binary_flags(df)
        assert result["is_night_departure"].iloc[0] == 0


# ---------------------------------------------------------------------------
# _drop_unused_columns
# ---------------------------------------------------------------------------

class TestDropUnusedColumns:
    def test_keeps_only_expected_columns(self, bts_df):
        df = _rename_columns(bts_df)
        df = _engineer_binary_flags(df)
        result = _drop_unused_columns(df, "delayed")
        expected_keep = {
            "delayed", "temperature_2m", "precipitation", "windspeed_10m",
            "visibility", "cloudcover", "flight_duration_s", "air_time", "distance",
            "is_foggy", "is_stormy", "is_night_departure", "is_weekend",
        }
        assert set(result.columns).issubset(expected_keep)
        assert "AIRLINE" not in result.columns
        assert "airline" not in result.columns

    def test_drops_columns_not_in_keep_list(self, bts_df):
        df = _rename_columns(bts_df)
        df["extra_col"] = 999
        df = _engineer_binary_flags(df)
        result = _drop_unused_columns(df, "delayed")
        assert "extra_col" not in result.columns


# ---------------------------------------------------------------------------
# _handle_missing_values
# ---------------------------------------------------------------------------

class TestHandleMissingValues:
    def test_fills_numeric_nulls_with_median(self):
        df = pd.DataFrame({
            "temperature_2m": [10.0, float("nan"), 20.0],
            "precipitation":  [0.0, 1.0, float("nan")],
        })
        result = _handle_missing_values(df)
        assert result["temperature_2m"].isna().sum() == 0
        assert result["precipitation"].isna().sum() == 0

    def test_fills_binary_flag_nulls_with_zero(self):
        df = pd.DataFrame({
            "is_foggy":  [float("nan"), 1.0, 0.0],
            "is_stormy": [1.0, float("nan"), 0.0],
        })
        result = _handle_missing_values(df)
        assert result["is_foggy"].iloc[0] == 0
        assert result["is_stormy"].iloc[1] == 0

    def test_does_not_modify_complete_columns(self):
        df = pd.DataFrame({"temperature_2m": [10.0, 20.0, 30.0]})
        result = _handle_missing_values(df)
        assert list(result["temperature_2m"]) == [10.0, 20.0, 30.0]


# ---------------------------------------------------------------------------
# clean_dataframe (full pipeline)
# ---------------------------------------------------------------------------

class TestCleanDataframe:
    def test_returns_dataframe(self, bts_df):
        result = clean_dataframe(bts_df)
        assert isinstance(result, pd.DataFrame)

    def test_target_column_present(self, bts_df):
        result = clean_dataframe(bts_df, target_column="delayed")
        assert "delayed" in result.columns

    def test_raises_if_target_missing_after_clean(self, bts_df):
        bts_df = bts_df.drop(columns=["delayed"])
        with pytest.raises(ValueError, match="Target"):
            clean_dataframe(bts_df, target_column="delayed")

    def test_no_nulls_in_output(self, bts_df):
        result = clean_dataframe(bts_df)
        assert result.isna().sum().sum() == 0

    def test_binary_flags_are_integers(self, bts_df):
        result = clean_dataframe(bts_df)
        for col in ["is_foggy", "is_stormy", "is_night_departure", "is_weekend"]:
            if col in result.columns:
                assert result[col].dtype in [int, "int64", "int32"]