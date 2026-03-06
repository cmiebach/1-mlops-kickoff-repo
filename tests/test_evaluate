import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.evaluate import evaluate_model

@pytest.fixture
def mock_model():
    X = pd.DataFrame({"a": [1,2,3,4,5], "b": [5,4,3,2,1]})
    y = pd.Series([0,1,0,1,0])
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    return clf, X, y

def test_returns_dict(mock_model):
    model, X, y = mock_model
    metrics = evaluate_model(model, X, y, problem_type="classification")
    assert isinstance(metrics, dict)

def test_contains_required_keys(mock_model):
    model, X, y = mock_model
    metrics = evaluate_model(model, X, y, problem_type="classification")
    assert "accuracy" in metrics
    assert "f1" in metrics

def test_accuracy_in_valid_range(mock_model):
    model, X, y = mock_model
    metrics = evaluate_model(model, X, y, problem_type="classification")
    assert 0 <= metrics["accuracy"] <= 1

def test_raises_on_unsupported_problem_type(mock_model):
    model, X, y = mock_model
    with pytest.raises(ValueError):
        evaluate_model(model, X, y, problem_type="unsupported")
