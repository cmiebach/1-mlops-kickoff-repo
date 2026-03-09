"""
Tests for src/main.py
"""
import pytest
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.main import load_config


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_returns_dict(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"key": "value"}))
        result = load_config(str(cfg_path))
        assert isinstance(result, dict)

    def test_returns_correct_values(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump({"target_column": "delayed", "problem_type": "classification"}))
        result = load_config(str(cfg_path))
        assert result["target_column"] == "delayed"
        assert result["problem_type"] == "classification"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))


# ---------------------------------------------------------------------------
# main (integration — all I/O dependencies mocked)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    return {
        "target_column": "delayed",
        "problem_type": "classification",
        "features": {
            "quantile_bin": ["windspeed_10m"],
            "numeric_passthrough": ["temperature_2m"],
            "binary_sum_cols": ["is_foggy"],
            "categorical_onehot": [],
            "n_bins": 4,
        },
        "validation": {
            "check_missing_values": True,
            "numeric_non_negative_cols": ["windspeed_10m"],
        },
        "split": {
            "test_size": 0.2,
            "val_size": 0.2,
            "random_state": 42,
        },
        "data": {
            "processed_path": "data/processed/flights_clean.csv",
        },
        "paths": {
            "model_path": "models/model.joblib",
            "metrics_path": "reports/metrics.json",
            "predictions_path": "reports/predictions.csv",
            "plots_path": "reports/plots/metrics.png",
        },
    }


class TestMain:
    def _make_df(self):
        rng = np.random.default_rng(0)
        n = 50
        return pd.DataFrame({
            "delayed":        rng.integers(0, 2, n),
            "windspeed_10m":  rng.uniform(0, 30, n),
            "temperature_2m": rng.uniform(10, 30, n),
            "is_foggy":       rng.integers(0, 2, n),
        })

    def test_main_runs_without_error(self, mock_config, tmp_path):
        df = self._make_df()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(len(df))
        mock_predictions = pd.DataFrame({"prediction": np.zeros(len(df)), "probability": np.zeros(len(df))})

        with (
            patch("src.main.load_config", return_value=mock_config),
            patch("src.main.load_raw_data", return_value=df),
            patch("src.main.clean_dataframe", return_value=df),
            patch("src.main.validate_dataframe"),
            patch("src.main.get_feature_preprocessor", return_value=MagicMock()),
            patch("src.main.train_model", return_value=mock_model),
            patch("src.main.evaluate_model", return_value={"accuracy": 0.9}),
            patch("src.main.make_plots", return_value=MagicMock()),
            patch("src.main.save_metrics"),
            patch("src.main.save_plots"),
            patch("src.main.run_inference", return_value=mock_predictions),
            patch("joblib.dump"),
            patch("pandas.DataFrame.to_csv"),
        ):
            from src.main import main
            main()  # Should not raise

    def test_main_calls_evaluate_model(self, mock_config):
        df = self._make_df()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(len(df))
        mock_predictions = pd.DataFrame({"prediction": np.zeros(len(df))})

        with (
            patch("src.main.load_config", return_value=mock_config),
            patch("src.main.load_raw_data", return_value=df),
            patch("src.main.clean_dataframe", return_value=df),
            patch("src.main.validate_dataframe"),
            patch("src.main.get_feature_preprocessor", return_value=MagicMock()),
            patch("src.main.train_model", return_value=mock_model),
            patch("src.main.evaluate_model", return_value={"accuracy": 0.9}) as mock_eval,
            patch("src.main.make_plots", return_value=MagicMock()),
            patch("src.main.save_metrics"),
            patch("src.main.save_plots"),
            patch("src.main.run_inference", return_value=mock_predictions),
            patch("joblib.dump"),
            patch("pandas.DataFrame.to_csv"),
        ):
            from src.main import main
            main()
            mock_eval.assert_called_once()

    def test_main_calls_save_metrics(self, mock_config):
        df = self._make_df()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(len(df))
        mock_predictions = pd.DataFrame({"prediction": np.zeros(len(df))})

        with (
            patch("src.main.load_config", return_value=mock_config),
            patch("src.main.load_raw_data", return_value=df),
            patch("src.main.clean_dataframe", return_value=df),
            patch("src.main.validate_dataframe"),
            patch("src.main.get_feature_preprocessor", return_value=MagicMock()),
            patch("src.main.train_model", return_value=mock_model),
            patch("src.main.evaluate_model", return_value={"accuracy": 0.9}),
            patch("src.main.make_plots", return_value=MagicMock()),
            patch("src.main.save_metrics") as mock_save,
            patch("src.main.save_plots"),
            patch("src.main.run_inference", return_value=mock_predictions),
            patch("joblib.dump"),
            patch("pandas.DataFrame.to_csv"),
        ):
            from src.main import main
            main()
            mock_save.assert_called_once_with({"accuracy": 0.9}, mock_config["paths"]["metrics_path"])
