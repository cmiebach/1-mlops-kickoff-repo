"""
Tests for src/train.py
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features import get_feature_preprocessor
from src.train import TrainArtifacts, train_model


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def xy():
    """Small feature/label pair for training tests."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "windspeed_10m":       rng.uniform(0, 30, n),
        "temperature_2m":      rng.uniform(10, 30, n),
        "precipitation":       rng.choice([0.0, 0.5, 1.0], size=n),
        "visibility":          rng.uniform(5000, 50000, n),
        "cloudcover":          rng.uniform(0, 100, n),
        "flight_duration_s":   rng.uniform(3600, 28800, n),
        "is_foggy":            rng.integers(0, 2, n),
        "is_stormy":           rng.integers(0, 2, n),
        "is_night_departure":  rng.integers(0, 2, n),
        "is_weekend":          rng.integers(0, 2, n),
    })
    y = pd.Series(rng.integers(0, 2, n), name="delayed")
    return df, y


@pytest.fixture
def preprocessor():
    return get_feature_preprocessor(
        quantile_bin_cols=["windspeed_10m"],
        categorical_onehot_cols=[],
        numeric_passthrough_cols=["temperature_2m", "precipitation", "visibility",
                                  "cloudcover", "flight_duration_s"],
        binary_sum_cols=["is_foggy", "is_stormy", "is_night_departure", "is_weekend"],
        n_bins=4,
    )


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

class TestTrainModel:
    def test_returns_fitted_pipeline(self, xy, preprocessor):
        X, y = xy
        model = train_model(X, y, preprocessor, problem_type="classification")
        assert isinstance(model, Pipeline)

    def test_fitted_pipeline_can_predict(self, xy, preprocessor):
        X, y = xy
        model = train_model(X, y, preprocessor)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

    def test_classification_uses_random_forest(self, xy, preprocessor):
        from sklearn.ensemble import RandomForestClassifier
        X, y = xy
        model = train_model(X, y, preprocessor, problem_type="classification")
        assert isinstance(model.named_steps["model"], RandomForestClassifier)

    def test_regression_uses_random_forest_regressor(self, xy, preprocessor):
        from sklearn.ensemble import RandomForestRegressor
        X, y = xy
        y_reg = y.astype(float)
        model = train_model(X, y_reg, preprocessor, problem_type="regression")
        assert isinstance(model.named_steps["model"], RandomForestRegressor)
