import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.features import _BinarySum, _make_binary_sum, get_feature_preprocessor


# ---------------------------------------------------------------------------
# _BinarySum
# ---------------------------------------------------------------------------

class TestBinarySum:
    def test_sums_columns_from_dataframe(self):
        df = pd.DataFrame({"a": [1, 0, 1], "b": [0, 1, 1]})
        transformer = _BinarySum(["a", "b"])
        result = transformer(df)
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result.flatten(), [1, 1, 2])

    def test_sums_columns_from_numpy_array(self):
        arr = np.array([[1, 0], [0, 1], [1, 1]])
        transformer = _BinarySum(["a", "b"])
        result = transformer(arr)
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result.flatten(), [1, 1, 2])

    def test_all_zeros(self):
        df = pd.DataFrame({"a": [0, 0], "b": [0, 0]})
        result = _BinarySum(["a", "b"])(df)
        np.testing.assert_array_equal(result.flatten(), [0, 0])

    def test_is_picklable(self):
        transformer = _BinarySum(["a", "b"])
        pickled = pickle.dumps(transformer)
        loaded = pickle.loads(pickled)
        df = pd.DataFrame({"a": [1], "b": [1]})
        np.testing.assert_array_equal(loaded(df).flatten(), [2])


# ---------------------------------------------------------------------------
# _make_binary_sum
# ---------------------------------------------------------------------------

class TestMakeBinarySum:
    def test_returns_function_transformer(self):
        from sklearn.preprocessing import FunctionTransformer
        ft = _make_binary_sum(["a", "b"])
        assert isinstance(ft, FunctionTransformer)

    def test_transformer_produces_correct_output(self):
        ft = _make_binary_sum(["a", "b"])
        df = pd.DataFrame({"a": [1, 0], "b": [1, 1]})
        result = ft.transform(df)
        np.testing.assert_array_equal(result.flatten(), [2, 1])


# ---------------------------------------------------------------------------
# get_feature_preprocessor
# ---------------------------------------------------------------------------

class TestGetFeaturePreprocessor:
    def test_returns_column_transformer(self):
        ct = get_feature_preprocessor(
            quantile_bin_cols=["windspeed_10m"],
            categorical_onehot_cols=[],
            numeric_passthrough_cols=["temperature_2m"],
            binary_sum_cols=["is_foggy"],
        )
        assert isinstance(ct, ColumnTransformer)

    def test_transformer_names_present(self):
        ct = get_feature_preprocessor(
            quantile_bin_cols=["windspeed_10m"],
            categorical_onehot_cols=["airline"],
            numeric_passthrough_cols=["temperature_2m"],
            binary_sum_cols=["is_foggy"],
        )
        names = [t[0] for t in ct.transformers]
        assert "quantile_bin" in names
        assert "onehot" in names
        assert "numeric" in names
        assert "binary_sum" in names

    def test_empty_lists_skip_transformers(self):
        ct = get_feature_preprocessor(
            quantile_bin_cols=[],
            categorical_onehot_cols=[],
            numeric_passthrough_cols=["temperature_2m"],
            binary_sum_cols=[],
        )
        names = [t[0] for t in ct.transformers]
        assert "quantile_bin" not in names
        assert "onehot" not in names
        assert "binary_sum" not in names
        assert "numeric" in names

    def test_fits_and_transforms_data(self):
        n = 20
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "windspeed_10m":  rng.uniform(0, 30, n),
            "temperature_2m": rng.uniform(10, 30, n),
            "is_foggy":       rng.integers(0, 2, n),
        })
        ct = get_feature_preprocessor(
            quantile_bin_cols=["windspeed_10m"],
            categorical_onehot_cols=[],
            numeric_passthrough_cols=["temperature_2m"],
            binary_sum_cols=["is_foggy"],
            n_bins=4,
        )
        result = ct.fit_transform(df)
        assert result.shape[0] == n
        # quantile_bin (1) + numeric (1) + binary_sum (1) = 3 columns
        assert result.shape[1] == 3

    def test_n_bins_respected(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"windspeed_10m": rng.uniform(0, 30, 50)})
        ct = get_feature_preprocessor(
            quantile_bin_cols=["windspeed_10m"],
            categorical_onehot_cols=[],
            numeric_passthrough_cols=[],
            binary_sum_cols=[],
            n_bins=6,
        )
        result = ct.fit_transform(df)
        assert set(result.flatten()).issubset(set(range(6)))