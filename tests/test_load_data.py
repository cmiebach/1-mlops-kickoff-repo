"""
Tests for src/load_data.py
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import numpy as np

from src.load_data import (
    _load_config,
    _fetch_weather,
    _fetch_flights,
    _merge_weather_flights,
    _build_target,
    load_raw_data,
    generate_sample,
)


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_valid_config(self, minimal_config):
        cfg = _load_config(minimal_config)
        assert "data" in cfg
        assert "airport" in cfg

    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_config(tmp_path / "nonexistent.yaml")

    def test_raises_if_required_key_missing(self, tmp_path):
        import yaml
        cfg_path = tmp_path / "bad_config.yaml"
        cfg_path.write_text(yaml.dump({"airport": {"icao": "EGLL"}}))  # missing 'data'
        with pytest.raises(KeyError, match="data"):
            _load_config(cfg_path)


# ---------------------------------------------------------------------------
# _fetch_weather
# ---------------------------------------------------------------------------

def _mock_weather_response():
    """Build a minimal Open-Meteo-style response payload."""
    times = pd.date_range("2023-06-01", periods=24, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {
        "hourly": {
            "time": times,
            "temperature_2m": [15.0] * 24,
            "precipitation": [0.0] * 24,
        }
    }


class TestFetchWeather:
    def test_returns_dataframe_on_success(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_weather_response()
        mock_resp.raise_for_status.return_value = None

        with patch("src.load_data.requests.get", return_value=mock_resp):
            df = _fetch_weather(51.4, -0.4, "2023-06-01", "2023-06-01",
                                ["temperature_2m", "precipitation"])

        assert isinstance(df, pd.DataFrame)
        assert "datetime" in df.columns
        assert "temperature_2m" in df.columns
        assert len(df) == 24

    def test_raises_runtime_error_after_exhausted_retries(self):
        import requests as req

        with patch("src.load_data.requests.get", side_effect=req.RequestException("timeout")):
            with patch("src.load_data.time.sleep"):
                with pytest.raises(RuntimeError, match="retries exhausted"):
                    _fetch_weather(51.4, -0.4, "2023-06-01", "2023-06-01",
                                   ["temperature_2m"], retries=2, backoff=0)

    def test_raises_if_hourly_time_missing(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"hourly": {}}  # no 'time' key
        mock_resp.raise_for_status.return_value = None

        with patch("src.load_data.requests.get", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="hourly.time"):
                _fetch_weather(51.4, -0.4, "2023-06-01", "2023-06-01", ["temperature_2m"])


# ---------------------------------------------------------------------------
# _fetch_flights
# ---------------------------------------------------------------------------

class TestFetchFlights:
    def test_returns_dataframe_on_success(self):
        flights = [{"icao24": "abc", "firstSeen": 1685577600, "lastSeen": 1685584800,
                    "estDepartureAirport": "EGLL", "estArrivalAirport": "CDG"}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = flights
        mock_resp.raise_for_status.return_value = None

        with patch("src.load_data.requests.get", return_value=mock_resp):
            df = _fetch_flights("EGLL", 1685577600, 1685584800)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "icao24" in df.columns

    def test_raises_runtime_error_on_non_list_response(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "bad request"}
        mock_resp.raise_for_status.return_value = None

        with patch("src.load_data.requests.get", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="Unexpected response format"):
                _fetch_flights("EGLL", 0, 1)

    def test_raises_runtime_error_after_exhausted_retries(self):
        import requests as req

        with patch("src.load_data.requests.get", side_effect=req.RequestException("err")):
            with patch("src.load_data.time.sleep"):
                with pytest.raises(RuntimeError, match="retries exhausted"):
                    _fetch_flights("EGLL", 0, 1, retries=2, backoff=0)


# ---------------------------------------------------------------------------
# _merge_weather_flights
# ---------------------------------------------------------------------------

class TestMergeWeatherFlights:
    def test_merges_on_departure_hour(self):
        df_flights = pd.DataFrame({
            "firstSeen": [1685606400],  # 2023-06-01 08:00 UTC
            "estDepartureAirport": ["EGLL"],
            "estArrivalAirport": ["CDG"],
            "lastSeen": [1685584800],
        })
        df_weather = pd.DataFrame({
            "datetime": pd.to_datetime(["2023-06-01 08:00"]),
            "temperature_2m": [18.0],
        })

        merged = _merge_weather_flights(df_flights, df_weather)
        assert "temperature_2m" in merged.columns
        assert merged["temperature_2m"].iloc[0] == 18.0

    def test_raises_if_firstSeen_missing(self):
        df_flights = pd.DataFrame({"foo": [1]})
        df_weather = pd.DataFrame({"datetime": pd.to_datetime(["2023-06-01"])})
        with pytest.raises(KeyError, match="firstSeen"):
            _merge_weather_flights(df_flights, df_weather)


# ---------------------------------------------------------------------------
# _build_target
# ---------------------------------------------------------------------------

class TestBuildTarget:
    def test_creates_binary_delayed_column(self):
        df = pd.DataFrame({
            "firstSeen":             [1000, 1000, 1000],
            "lastSeen":              [4600, 4000, 3000],  # durations: 3600, 3000, 2000
            "estDepartureAirport":   ["EGLL", "EGLL", "EGLL"],
            "estArrivalAirport":     ["CDG",  "CDG",  "CDG"],
        })
        result = _build_target(df, delay_threshold_minutes=15)
        assert "delayed" in result.columns
        assert set(result["delayed"].unique()).issubset({0, 1})

    def test_threshold_affects_positive_rate(self):
        df = pd.DataFrame({
            "firstSeen":           [0] * 10,
            "lastSeen":            list(range(3600, 3600 + 10 * 600, 600)),
            "estDepartureAirport": ["EGLL"] * 10,
            "estArrivalAirport":   ["CDG"] * 10,
        })
        result_tight = _build_target(df, delay_threshold_minutes=1)
        result_loose = _build_target(df, delay_threshold_minutes=1000)
        assert result_tight["delayed"].sum() >= result_loose["delayed"].sum()


# ---------------------------------------------------------------------------
# load_raw_data
# ---------------------------------------------------------------------------

class TestLoadRawData:
    def test_loads_from_disk_if_file_exists(self, minimal_config, raw_df, tmp_path):
        import yaml
        cfg = yaml.safe_load(minimal_config.read_text())
        raw_path = Path(cfg["data"]["raw_path"])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(raw_path, index=False)

        result = load_raw_data(minimal_config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(raw_df)

    def test_triggers_fetch_if_file_missing(self, minimal_config, raw_df):
        with patch("src.load_data.fetch_and_save_raw", return_value=raw_df) as mock_fetch:
            result = load_raw_data(minimal_config)
        mock_fetch.assert_called_once_with(minimal_config)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# generate_sample
# ---------------------------------------------------------------------------

class TestGenerateSample:
    def test_returns_2000_rows(self, minimal_config):
        df = generate_sample(minimal_config)
        assert len(df) == 2000

    def test_has_required_columns(self, minimal_config):
        df = generate_sample(minimal_config)
        required = ["delayed", "temperature_2m", "windspeed_10m", "weathercode"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_delayed_is_binary(self, minimal_config):
        df = generate_sample(minimal_config)
        assert set(df["delayed"].unique()).issubset({0, 1})

    def test_saves_csv_to_raw_path(self, minimal_config):
        import yaml
        cfg = yaml.safe_load(minimal_config.read_text())
        raw_path = Path(cfg["data"]["raw_path"])
        if raw_path.exists():
            raw_path.unlink()

        generate_sample(minimal_config)
        assert raw_path.exists()
