"""
Tests for src/main.py
"""
import pytest
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.main import load_config, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path) -> dict:
    """Write a minimal config.yaml to tmp_path and return the config dict."""
    raw_path = tmp_path / "data" / "raw" / "flights_raw.csv"
    processed_path = tmp_path / "data" / "processed" / "flights_clean.csv"
    model_path = tmp_path / "models" / "model.joblib"
    metrics_path = tmp_path / "reports" / "metrics.json"
    predictions_path = tmp_path / "reports" / "predictions.csv"

    cfg = {
        "airport": {"icao": "EGLL", "latitude": 51.47, "longitude": -0.4543},
        "data": {
            "raw_path": str(raw_path),
            "processed_path": str(processed_path),
        },
        "target_column": "delayed",
        "problem_type": "classification",
        "features": {
            "quantile_bin": ["windspeed_10m"],
            "numeric_passthrough": [
                "temperature_2m", "precipitation", "visibility",
                "cloudcover", "flight_duration_s",
            ],
            "binary_sum_cols": [
                "is_foggy", "is_stormy", "is_night_departure", "is_weekend",
            ],
            "categorical_onehot": [],
            "n_bins": 4,
        },
        "split": {"test_size": 0.1, "val_size": 0.1, "random_state": 42},
        "validation": {
            "check_missing_values": True,
            "numeric_non_negative_cols": [
                "windspeed_10m", "precipitation", "visibility",
                "cloudcover", "flight_duration_s",
            ],
        },
        "model": {
            "active": "random_forest",
            "random_forest": {"n_estimators": 5, "max_depth": 3, "random_state": 42},
        },
        "paths": {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "predictions_path": str(predictions_path),
        },
        "logging": {"level": "INFO"},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return cfg, cfg_path


def _make_clean_df(n: int = 60) -> pd.DataFrame:
    """Return a small synthetic DataFrame matching the expected schema."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "delayed":              rng.integers(0, 2, n),
        "windspeed_10m":        rng.uniform(0, 30, n),
        "temperature_2m":       rng.uniform(10, 30, n),
        "precipitation":        rng.choice([0.0, 0.5, 1.0], size=n),
        "visibility":           rng.uniform(5000, 50000, n),
        "cloudcover":           rng.uniform(0, 100, n),
        "flight_duration_s":    rng.uniform(3600, 28800, n),
        "is_foggy":             rng.integers(0, 2, n),
        "is_stormy":            rng.integers(0, 2, n),
        "is_night_departure":   rng.integers(0, 2, n),
        "is_weekend":           rng.integers(0, 2, n),
    })


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_returns_dict(self, tmp_path):
        cfg, cfg_path = _make_config(tmp_path)
        result = load_config(str(cfg_path))
        assert isinstance(result, dict)

    def test_contains_expected_keys(self, tmp_path):
        cfg, cfg_path = _make_config(tmp_path)
        result = load_config(str(cfg_path))
        for key in ("target_column", "problem_type", "features", "split", "paths"):
            assert key in result, f"Missing key: {key}"

    def test_raises_file_not_found_on_missing_config(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    """Test the main() pipeline with mocked I/O and sub-module calls."""

    def _run_main(self, tmp_path):
        """Patch all external I/O and run main() with a temp config."""
        cfg, cfg_path = _make_config(tmp_path)
        clean_df = _make_clean_df()

        # Build a real (tiny) fitted model to hand back from train_model
        from src.features import get_feature_preprocessor
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=["windspeed_10m"],
            categorical_onehot_cols=[],
            numeric_passthrough_cols=[
                "temperature_2m", "precipitation", "visibility",
                "cloudcover", "flight_duration_s",
            ],
            binary_sum_cols=["is_foggy", "is_stormy", "is_night_departure", "is_weekend"],
            n_bins=4,
        )
        from src.train import train_model
        X = clean_df.drop(columns=["delayed"])
        y = clean_df["delayed"]
        real_model = train_model(X, y, preprocessor, problem_type="classification")

        preds_df = pd.DataFrame({"prediction": [0, 1], "proba_1": [0.3, 0.7]})

        patches = {
            "src.main.load_config": lambda path: cfg,
            "src.main.load_raw_data": lambda **kw: clean_df,
            "src.main.clean_dataframe": lambda df, **kw: clean_df,
            "src.main.validate_dataframe": lambda **kw: None,
            "src.main.train_model": lambda **kw: real_model,
            "src.main.evaluate_model": lambda **kw: {"accuracy": 0.8, "f1": 0.75},
            "src.main.save_metrics": lambda metrics, path: None,
            "src.main.run_inference": lambda **kw: preds_df,
            "src.main.joblib.dump": lambda obj, path: None,
        }

        with patch("src.main.load_config", return_value=cfg), \
             patch("src.main.load_raw_data", return_value=clean_df), \
             patch("src.main.clean_dataframe", return_value=clean_df), \
             patch("src.main.validate_dataframe", return_value=None), \
             patch("src.main.train_model", return_value=real_model), \
             patch("src.main.evaluate_model", return_value={"accuracy": 0.8, "f1": 0.75}), \
             patch("src.main.save_metrics", return_value=None), \
             patch("src.main.run_inference", return_value=preds_df), \
             patch("src.main.joblib.dump", return_value=None):
            main()

    def test_main_runs_without_error(self, tmp_path):
        """main() should complete without raising exceptions."""
        self._run_main(tmp_path)

    def test_main_calls_evaluate(self, tmp_path):
        """main() should call evaluate_model exactly once."""
        cfg, _ = _make_config(tmp_path)
        clean_df = _make_clean_df()

        from src.features import get_feature_preprocessor
        from src.train import train_model as real_train
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=["windspeed_10m"],
            categorical_onehot_cols=[],
            numeric_passthrough_cols=[
                "temperature_2m", "precipitation", "visibility",
                "cloudcover", "flight_duration_s",
            ],
            binary_sum_cols=["is_foggy", "is_stormy", "is_night_departure", "is_weekend"],
            n_bins=4,
        )
        X = clean_df.drop(columns=["delayed"])
        y = clean_df["delayed"]
        real_model = real_train(X, y, preprocessor, problem_type="classification")
        preds_df = pd.DataFrame({"prediction": [0, 1], "proba_1": [0.3, 0.7]})

        mock_evaluate = MagicMock(return_value={"accuracy": 0.8, "f1": 0.75})

        with patch("src.main.load_config", return_value=cfg), \
             patch("src.main.load_raw_data", return_value=clean_df), \
             patch("src.main.clean_dataframe", return_value=clean_df), \
             patch("src.main.validate_dataframe", return_value=None), \
             patch("src.main.train_model", return_value=real_model), \
             patch("src.main.evaluate_model", mock_evaluate), \
             patch("src.main.save_metrics", return_value=None), \
             patch("src.main.run_inference", return_value=preds_df), \
             patch("src.main.joblib.dump", return_value=None):
            main()

        mock_evaluate.assert_called_once()

    def test_main_calls_train(self, tmp_path):
        """main() should call train_model exactly once."""
        cfg, _ = _make_config(tmp_path)
        clean_df = _make_clean_df()

        from src.features import get_feature_preprocessor
        from src.train import train_model as real_train
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=["windspeed_10m"],
            categorical_onehot_cols=[],
            numeric_passthrough_cols=[
                "temperature_2m", "precipitation", "visibility",
                "cloudcover", "flight_duration_s",
            ],
            binary_sum_cols=["is_foggy", "is_stormy", "is_night_departure", "is_weekend"],
            n_bins=4,
        )
        X = clean_df.drop(columns=["delayed"])
        y = clean_df["delayed"]
        real_model = real_train(X, y, preprocessor, problem_type="classification")
        preds_df = pd.DataFrame({"prediction": [0, 1], "proba_1": [0.3, 0.7]})

        mock_train = MagicMock(return_value=real_model)

        with patch("src.main.load_config", return_value=cfg), \
             patch("src.main.load_raw_data", return_value=clean_df), \
             patch("src.main.clean_dataframe", return_value=clean_df), \
             patch("src.main.validate_dataframe", return_value=None), \
             patch("src.main.train_model", mock_train), \
             patch("src.main.evaluate_model", return_value={"accuracy": 0.8, "f1": 0.75}), \
             patch("src.main.save_metrics", return_value=None), \
             patch("src.main.run_inference", return_value=preds_df), \
             patch("src.main.joblib.dump", return_value=None):
            main()

        mock_train.assert_called_once()
