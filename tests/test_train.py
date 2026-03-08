"""
Tests for src/train.py
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features import get_feature_preprocessor
from src.train import TrainArtifacts, build_pipeline, train_model, train


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
# build_pipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def test_returns_sklearn_pipeline(self):
        pipeline = build_pipeline(
            numeric_cols=["temperature_2m"],
            categorical_cols=[],
            numeric_strategy="mean",
            categorical_strategy="most_frequent",
            model_params={"max_iter": 100},
        )
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_preprocess_and_model_steps(self):
        pipeline = build_pipeline(
            numeric_cols=["temperature_2m"],
            categorical_cols=[],
            numeric_strategy="mean",
            categorical_strategy="most_frequent",
            model_params={"max_iter": 100},
        )
        step_names = [name for name, _ in pipeline.steps]
        assert "preprocess" in step_names
        assert "model" in step_names


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


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

class TestTrain:
    def test_returns_train_artifacts(self, xy):
        X, y = xy
        pipeline = build_pipeline(
            numeric_cols=list(X.columns),
            categorical_cols=[],
            numeric_strategy="mean",
            categorical_strategy="most_frequent",
            model_params={"max_iter": 100},
        )
        artifacts = train(X, y, test_size=0.2, random_state=42,
                          stratify=True, pipeline=pipeline)
        assert isinstance(artifacts, TrainArtifacts)

    def test_validation_split_size(self, xy):
        X, y = xy
        pipeline = build_pipeline(
            numeric_cols=list(X.columns),
            categorical_cols=[],
            numeric_strategy="mean",
            categorical_strategy="most_frequent",
            model_params={"max_iter": 100},
        )
        artifacts = train(X, y, test_size=0.2, random_state=42,
                          stratify=False, pipeline=pipeline)
        assert len(artifacts.X_valid) == pytest.approx(20, abs=2)
        assert len(artifacts.y_valid) == len(artifacts.X_valid)

    def test_model_is_fitted_pipeline(self, xy):
        X, y = xy
        pipeline = build_pipeline(
            numeric_cols=list(X.columns),
            categorical_cols=[],
            numeric_strategy="mean",
            categorical_strategy="most_frequent",
            model_params={"max_iter": 100},
        )
        artifacts = train(X, y, test_size=0.2, random_state=42,
                          stratify=False, pipeline=pipeline)
        assert isinstance(artifacts.model, Pipeline)
        preds = artifacts.model.predict(artifacts.X_valid)
        assert len(preds) == len(artifacts.y_valid)
