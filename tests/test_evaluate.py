import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from src.evaluate import (
    evaluate_model, 
    make_plots, 
    save_metrics, 
    save_plots
)


@pytest.fixture
def mock_model():
    """Returns model, X_eval, y_eval with known predictions for testing."""
    X = pd.DataFrame({"a": [1,2,3,4,5], "b": [5,4,3,2,1]})
    y = pd.Series([0,1,0,1,0])
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    return clf, X, y


def test_returns_dict(mock_model):
    model, X, y = mock_model
    metrics = evaluate_model(model, X, y, problem_type="classification")
    assert isinstance(metrics, dict)


def test_contains_all_new_keys(mock_model):
    """Test all new metrics are present."""
    model, X, y = mock_model
    metrics = evaluate_model(model, X, y, problem_type="classification")
    expected_keys = [
        "precision_0", "recall_0", "precision_1", "recall_1",
        "accuracy", "f1", "roc_auc", "pr_auc", "brier", "fnr"
    ]
    for key in expected_keys:
        assert key in metrics


def test_metrics_in_valid_range(mock_model):
    model, X, y = mock_model
    metrics = evaluate_model(model, X, y, problem_type="classification")
    # All metrics should be between 0 and 1 (or nan for roc_auc if no proba)
    for key, value in metrics.items():
        if not np.isnan(value):  # Allow nan for some edge cases
            assert 0 <= value <= 1, f"{key}={value} out of range"


def test_precision_recall_correct(mock_model):
    """Verify precision/recall calculations are reasonable."""
    model, X, y = mock_model
    metrics = evaluate_model(model, X, y, problem_type="classification")
    # With this toy data, expect reasonable values
    assert metrics["precision_0"] > 0
    assert metrics["recall_0"] > 0


def test_make_plots_returns_figure(mock_model):
    model, X, y = mock_model
    fig = make_plots(model, X, y)
    assert hasattr(fig, 'savefig')  # Is a matplotlib Figure
    assert len(fig.axes) == 2  # Has PR curve + confusion matrix


def test_save_metrics_creates_json(mock_model, tmp_path):
    model, X, y = mock_model
    metrics = evaluate_model(model, X, y)
    path = tmp_path / "test_metrics.json"
    
    save_metrics(metrics, str(path))
    assert path.exists()
    assert path.read_text()  # File not empty


def test_save_plots_creates_png(mock_model, tmp_path):
    model, X, y = mock_model
    fig = make_plots(model, X, y)
    path = tmp_path / "test_plots.png"
    
    save_plots(fig, str(path))
    assert path.exists()
    assert path.stat().st_size > 1000  # PNG should be >1KB


def test_evaluate_model_raises_on_unsupported_type(mock_model):
    model, X, y = mock_model
    with pytest.raises(ValueError):
        evaluate_model(model, X, y, problem_type="unsupported")
