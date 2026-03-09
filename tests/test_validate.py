"""
Tests for src/validate.py
"""
import pytest
import pandas as pd
import numpy as np

from src.validate import (
    _check_required_columns,
    _check_missing_values,
    _check_target_values,
    _check_non_negative,
    validate_dataframe,
)


@pytest.fixture
def good_df():
    return pd.DataFrame({
        "delayed":        [0, 1, 0, 1, 0] * 12,  # 60 rows, both classes
        "windspeed_10m":  [5.0] * 60,
        "temperature_2m": [20.0] * 60,
    })


# ---------------------------------------------------------------------------
# _check_required_columns
# ---------------------------------------------------------------------------

class TestCheckRequiredColumns:
    def test_passes_when_all_present(self, good_df):
        _check_required_columns(good_df, ["delayed", "windspeed_10m"])  # no error

    def test_raises_on_missing_column(self, good_df):
        with pytest.raises(ValueError, match="Missing columns"):
            _check_required_columns(good_df, ["delayed", "nonexistent"])


# ---------------------------------------------------------------------------
# _check_missing_values
# ---------------------------------------------------------------------------

class TestCheckMissingValues:
    def test_passes_when_no_nulls(self, good_df):
        _check_missing_values(good_df, ["delayed", "windspeed_10m"])

    def test_raises_on_null(self, good_df):
        good_df.loc[0, "windspeed_10m"] = float("nan")
        with pytest.raises(ValueError, match="Null values"):
            _check_missing_values(good_df, ["windspeed_10m"])


# ---------------------------------------------------------------------------
# _check_target_values
# ---------------------------------------------------------------------------

class TestCheckTargetValues:
    def test_passes_with_binary_target(self, good_df):
        _check_target_values(good_df, "delayed", [0, 1])

    def test_raises_on_unexpected_values(self):
        df = pd.DataFrame({"delayed": [0, 1, 2]})
        with pytest.raises(ValueError, match="Unexpected target values"):
            _check_target_values(df, "delayed", [0, 1])

    def test_raises_if_only_one_class(self):
        df = pd.DataFrame({"delayed": [0] * 10})
        with pytest.raises(ValueError, match="only 1 class"):
            _check_target_values(df, "delayed", [0, 1])


# ---------------------------------------------------------------------------
# _check_non_negative
# ---------------------------------------------------------------------------

class TestCheckNonNegative:
    def test_no_warning_when_all_positive(self, good_df, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            _check_non_negative(good_df, ["windspeed_10m"])
        assert "windspeed_10m" not in caplog.text

    def test_warns_on_negative_values(self, caplog):
        import logging
        df = pd.DataFrame({"windspeed_10m": [-1.0, 5.0, 3.0]})
        with caplog.at_level(logging.WARNING):
            _check_non_negative(df, ["windspeed_10m"])
        assert "windspeed_10m" in caplog.text

    def test_skips_missing_columns_silently(self, good_df):
        _check_non_negative(good_df, ["nonexistent_col"])  # no error


# ---------------------------------------------------------------------------
# validate_dataframe
# ---------------------------------------------------------------------------

class TestValidateDataframe:
    def test_passes_on_valid_data(self, good_df):
        validate_dataframe(
            good_df,
            required_columns=["delayed", "windspeed_10m"],
            check_missing_values=True,
            target_column="delayed",
            target_allowed_values=[0, 1],
            numeric_non_negative_cols=["windspeed_10m"],
            min_rows=50,
        )

    def test_raises_if_too_few_rows(self, good_df):
        small = good_df.head(10)
        with pytest.raises(ValueError, match="rows, need"):
            validate_dataframe(small, required_columns=["delayed"], min_rows=50)

    def test_raises_on_missing_column(self, good_df):
        with pytest.raises(ValueError, match="Missing columns"):
            validate_dataframe(good_df, required_columns=["nonexistent"])

    def test_skips_missing_value_check_when_disabled(self):
        df = pd.DataFrame({"delayed": [0, 1] * 30, "windspeed_10m": [float("nan")] * 60})
        # Should not raise because check_missing_values=False
        validate_dataframe(df, required_columns=["delayed"],
                           check_missing_values=False, min_rows=50)

    def test_raises_on_single_class_target(self, good_df):
        good_df["delayed"] = 0
        with pytest.raises(ValueError, match="only 1 class"):
            validate_dataframe(
                good_df,
                required_columns=["delayed"],
                target_column="delayed",
                target_allowed_values=[0, 1],
            )
