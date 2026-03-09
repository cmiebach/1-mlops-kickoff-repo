"""
Tests for src/infer.py
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.infer import run_inference


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def model_and_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.uniform(0, 1, 30), "b": rng.uniform(0, 1, 30)})
    y = pd.Series(rng.integers(0, 2, 30))
    clf = DummyClassifier(strategy="stratified", random_state=0).fit(X, y)
    return clf, X


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------

class TestRunInference:
    def test_returns_dataframe(self, model_and_data):
        model, X = model_and_data
        result = run_inference(model, X)
        assert isinstance(result, pd.DataFrame)

    def test_prediction_column_present(self, model_and_data):
        model, X = model_and_data
        result = run_inference(model, X)
        assert "prediction" in result.columns

    def test_prediction_values_are_binary(self, model_and_data):
        model, X = model_and_data
        result = run_inference(model, X)
        assert set(result["prediction"].unique()).issubset({0, 1})

    def test_probability_column_included_by_default(self, model_and_data):
        model, X = model_and_data
        result = run_inference(model, X)
        assert "probability" in result.columns
        assert result["probability"].between(0.0, 1.0).all()

    def test_probability_excluded_when_disabled(self, model_and_data):
        model, X = model_and_data
        result = run_inference(model, X, include_proba=False)
        assert "probability" not in result.columns

    def test_preserves_input_index(self, model_and_data):
        model, X = model_and_data
        X_indexed = X.copy()
        X_indexed.index = range(100, 100 + len(X))
        result = run_inference(model, X_indexed)
        assert list(result.index) == list(X_indexed.index)

    def test_row_count_matches_input(self, model_and_data):
        model, X = model_and_data
        result = run_inference(model, X)
        assert len(result) == len(X)

    def test_no_proba_column_if_model_lacks_predict_proba(self):
        from sklearn.svm import SVC
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"a": rng.uniform(0, 1, 30)})
        y = pd.Series(rng.integers(0, 2, 30))
        clf = SVC(probability=False).fit(X, y)
        result = run_inference(clf, X)
        assert "probability" not in result.columns
