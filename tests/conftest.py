"""Shared fixtures for the test suite."""
import pytest
import yaml
import pandas as pd
import numpy as np


@pytest.fixture
def minimal_config(tmp_path):
    """Create a minimal config.yaml in a temp directory."""
    raw_path = tmp_path / "data" / "raw" / "flights_raw.csv"
    cfg = {
        "airport": {
            "icao": "EGLL",
            "latitude": 51.47,
            "longitude": -0.4543,
        },
        "data": {
            "raw_path": str(raw_path),
            "start_date": "2023-06-01",
            "end_date": "2023-06-07",
            "weather_variables": ["temperature_2m", "windspeed_10m", "weathercode"],
            "delay_threshold_minutes": 15,
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg))
    return cfg_path


@pytest.fixture
def raw_df():
    """A small raw DataFrame that mirrors the expected schema."""
    rng = np.random.default_rng(0)
    n = 20
    return pd.DataFrame({
        "delayed": rng.integers(0, 2, size=n),
        "temperature_2m": rng.uniform(10, 30, size=n),
        "windspeed_10m": rng.uniform(0, 20, size=n),
        "weathercode": rng.integers(0, 10, size=n),
    })
