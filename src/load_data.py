"""
load_data.py
------------
Fetches and assembles the raw flight delay dataset for the MLOps pipeline.

Data sources (both free, no API key required):
    - Open-Meteo Historical Weather API  https://open-meteo.com/
      Provides hourly weather at the departure airport coordinates.
    - OpenSky Network REST API           https://opensky-network.org/
      Provides historical flight state vectors (departure, arrival, duration).

Design principles (from course guidelines):
    - Config controls all loading  : all URLs, coords, dates come from config.yaml
    - Never overwrite raw data     : write once to data/raw/, then read-only
    - Fail fast                    : raise clear errors on bad config or API failures
    - Log every step               : shape, path, and status always logged
    - No hardcoded values          : zero magic strings outside config.yaml
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str | Path) -> dict:
    """Load and validate the YAML config file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Parsed config dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If required keys are missing from the config.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"[load_data._load_config] Config not found: {config_path}"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required_keys = ["data", "airport"]
    for key in required_keys:
        if key not in config:
            raise KeyError(
                f"[load_data._load_config] Missing required config key: '{key}'"
            )

    logger.info("[load_data._load_config] Config loaded from %s", config_path)
    return config


def _fetch_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_vars: list[str],
    retries: int = 3,
    backoff: float = 2.0,
) -> pd.DataFrame:
    """Call the Open-Meteo historical weather API.

    Open-Meteo is free and requires no API key.
    Endpoint: https://archive-api.open-meteo.com/v1/archive

    Args:
        latitude:    Decimal latitude of the airport.
        longitude:   Decimal longitude of the airport.
        start_date:  ISO date string, e.g. "2023-01-01".
        end_date:    ISO date string, e.g. "2023-12-31".
        hourly_vars: List of Open-Meteo variable names to request.
        retries:     Number of retry attempts on transient failures.
        backoff:     Seconds to wait between retries.

    Returns:
        DataFrame with a 'datetime' column and one column per weather variable.

    Raises:
        RuntimeError: If the API returns a non-200 status after all retries.
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "timezone": "UTC",
    }

    logger.info(
        "[load_data._fetch_weather] Requesting weather for lat=%s lon=%s "
        "from %s to %s",
        latitude, longitude, start_date, end_date,
    )

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            logger.warning(
                "[load_data._fetch_weather] Attempt %d/%d failed: %s",
                attempt, retries, exc,
            )
            if attempt == retries:
                raise RuntimeError(
                    f"[load_data._fetch_weather] All {retries} retries exhausted. "
                    f"Last error: {exc}"
                ) from exc
            time.sleep(backoff * attempt)

    payload = response.json()
    hourly = payload.get("hourly", {})

    if "time" not in hourly:
        raise RuntimeError(
            "[load_data._fetch_weather] API response missing 'hourly.time' key. "
            f"Response keys: {list(payload.keys())}"
        )

    df = pd.DataFrame(hourly)
    df.rename(columns={"time": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])

    logger.info(
        "[load_data._fetch_weather] Weather data fetched. Shape: %s", df.shape
    )
    return df


def _get_opensky_token(client_id: str, client_secret: str) -> Optional[str]:
    """Exchange OpenSky OAuth client credentials for an access token."""
    token_url = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
    try:
        resp = requests.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=15,
        )
        resp.raise_for_status()
        token = resp.json().get("access_token")
        logger.info("[load_data._get_opensky_token] OAuth token obtained")
        return token
    except requests.RequestException as exc:
        logger.warning(
            "[load_data._get_opensky_token] Failed to get token: %s. "
            "Falling back to anonymous access.",
            exc,
        )
        return None


def _fetch_flights(
    airport_icao: str,
    start_ts: int,
    end_ts: int,
    retries: int = 3,
    backoff: float = 2.0,
) -> pd.DataFrame:
    """Call the OpenSky Network REST API for departure flights.

    OpenSky anonymous access is free (rate-limited to ~400 flights per call).
    Endpoint: https://opensky-network.org/api/flights/departure

    Args:
        airport_icao: ICAO code of the departure airport, e.g. "EGLL".
        start_ts:     Unix timestamp for the start of the window.
        end_ts:       Unix timestamp for the end of the window (max 7-day window).
        retries:      Number of retry attempts on transient failures.
        backoff:      Seconds to wait between retries.

    Returns:
        DataFrame with one row per departure flight.

    Raises:
        RuntimeError: If the API returns a non-200 status after all retries.
    """
    base_url = "https://opensky-network.org/api/flights/departure"
    params = {
        "airport": airport_icao,
        "begin": start_ts,
        "end": end_ts,
    }

    logger.info(
        "[load_data._fetch_flights] Requesting departures from %s "
        "(unix %d to %d)",
        airport_icao, start_ts, end_ts,
    )

    # Use OpenSky OAuth token if credentials are available (avoids 403)
    headers = {}
    client_id = os.getenv("OPENSKY_CLIENT_ID")
    client_secret = os.getenv("OPENSKY_CLIENT_SECRET")
    if client_id and client_secret:
        token = _get_opensky_token(client_id, client_secret)
        if token:
            headers["Authorization"] = f"Bearer {token}"

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            logger.warning(
                "[load_data._fetch_flights] Attempt %d/%d failed: %s",
                attempt, retries, exc,
            )
            if attempt == retries:
                raise RuntimeError(
                    f"[load_data._fetch_flights] All {retries} retries exhausted. "
                    f"Last error: {exc}"
                ) from exc
            time.sleep(backoff * attempt)

    flights = response.json()

    if not isinstance(flights, list):
        raise RuntimeError(
            f"[load_data._fetch_flights] Unexpected response format: {type(flights)}"
        )

    df = pd.DataFrame(flights)
    logger.info(
        "[load_data._fetch_flights] Flights fetched. Shape: %s", df.shape
    )
    return df


def _merge_weather_flights(
    df_flights: pd.DataFrame,
    df_weather: pd.DataFrame,
) -> pd.DataFrame:
    """Join flight records with hourly weather on departure hour.

    Strategy:
        - Round each flight's departure timestamp down to the hour.
        - Left-join onto the weather DataFrame on that rounded hour.

    Args:
        df_flights: Flight DataFrame with 'firstSeen' Unix timestamp column.
        df_weather: Weather DataFrame with 'datetime' column (UTC, hourly).

    Returns:
        Merged DataFrame with weather features appended to each flight row.

    Raises:
        KeyError: If expected join columns are missing.
    """
    if "firstSeen" not in df_flights.columns:
        raise KeyError(
            "[load_data._merge_weather_flights] 'firstSeen' column not found "
            f"in flights DataFrame. Available: {list(df_flights.columns)}"
        )

    df_flights = df_flights.copy()
    df_flights["departure_dt"] = pd.to_datetime(
        df_flights["firstSeen"], unit="s", utc=True
    ).dt.tz_localize(None).dt.floor("h")

    df_weather = df_weather.copy()
    df_weather["datetime"] = df_weather["datetime"].dt.floor("h")

    merged = df_flights.merge(
        df_weather,
        left_on="departure_dt",
        right_on="datetime",
        how="left",
    )

    logger.info(
        "[load_data._merge_weather_flights] Merged shape: %s  "
        "Weather match rate: %.1f%%",
        merged.shape,
        100 * merged["datetime"].notna().mean(),
    )
    return merged


def _build_target(
    df: pd.DataFrame,
    delay_threshold_minutes: int,
) -> pd.DataFrame:
    """Engineer the binary delay target column.

    A flight is labelled delayed (1) if the difference between its actual
    arrival time and its estimated arrival time exceeds the threshold.

    Args:
        df:                      Merged DataFrame with 'lastSeen' and
                                 'estArrivalAirportHorizDistance' columns.
        delay_threshold_minutes: Minutes above which a flight is 'delayed'.

    Returns:
        DataFrame with a new binary 'delayed' column (1 = delayed, 0 = on time).
    """
    # OpenSky does not provide scheduled times; we approximate delay as
    # flight duration deviation from the median for the same route pair.
    df = df.copy()

    df["flight_duration_s"] = df["lastSeen"] - df["firstSeen"]

    # Median duration per (origin, destination) route as the baseline
    route_median = (
        df.groupby(["estDepartureAirport", "estArrivalAirport"])["flight_duration_s"]
        .transform("median")
    )

    delay_threshold_s = delay_threshold_minutes * 60
    df["delayed"] = (
        (df["flight_duration_s"] - route_median) > delay_threshold_s
    ).astype(int)

    positive_rate = df["delayed"].mean() * 100
    logger.info(
        "[load_data._build_target] Target built. Delayed rate: %.1f%%  "
        "Threshold: %d min",
        positive_rate,
        delay_threshold_minutes,
    )
    return df


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_and_save_raw(config_path: str | Path = "config.yaml") -> pd.DataFrame:
    """Orchestrate the full data fetch and persist the raw CSV.

    This is the ONLY function that writes to disk (data/raw/flights_raw.csv).
    Subsequent pipeline steps read from that file — they never re-call the API.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Raw DataFrame before any cleaning.

    Raises:
        FileNotFoundError: If config is missing.
        RuntimeError:      On API or merge failures.
    """
    config = _load_config(config_path)

    airport_cfg = config["airport"]
    data_cfg = config["data"]

    # --- 1. Fetch weather ---
    df_weather = _fetch_weather(
        latitude=airport_cfg["latitude"],
        longitude=airport_cfg["longitude"],
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"],
        hourly_vars=data_cfg["weather_variables"],
    )

    # --- 2. Fetch flights (OpenSky limits queries to 1 calendar day) ---
    import datetime
    start = datetime.datetime.fromisoformat(data_cfg["start_date"]).replace(
        tzinfo=datetime.timezone.utc
    )
    end = datetime.datetime.fromisoformat(data_cfg["end_date"]).replace(
        tzinfo=datetime.timezone.utc
    )

    all_flights = []
    window_start = start
    while window_start < end:
        window_end = min(window_start + datetime.timedelta(days=1), end)
        try:
            chunk = _fetch_flights(
                airport_icao=airport_cfg["icao"],
                start_ts=int(window_start.timestamp()),
                end_ts=int(window_end.timestamp()),
            )
            all_flights.append(chunk)
        except RuntimeError as exc:
            logger.warning(
                "[load_data.fetch_and_save_raw] Skipping window %s–%s: %s",
                window_start.date(), window_end.date(), exc,
            )
        window_start = window_end
        time.sleep(1)  # Respectful rate-limiting for anonymous OpenSky access

    if not all_flights:
        raise RuntimeError(
            "[load_data.fetch_and_save_raw] No flight data retrieved. "
            "Check your ICAO code and date range in config.yaml."
        )

    df_flights = pd.concat(all_flights, ignore_index=True)
    logger.info(
        "[load_data.fetch_and_save_raw] Total flights assembled: %d rows",
        len(df_flights),
    )

    # --- 3. Merge weather onto flights ---
    df_merged = _merge_weather_flights(df_flights, df_weather)

    # --- 4. Build binary target ---
    df_raw = _build_target(
        df_merged,
        delay_threshold_minutes=data_cfg["delay_threshold_minutes"],
    )

    # --- 5. Persist raw data (write-once) ---
    raw_path = Path(data_cfg["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    if raw_path.exists():
        logger.warning(
            "[load_data.fetch_and_save_raw] Raw file already exists at %s. "
            "Skipping overwrite to protect raw data integrity. "
            "Delete manually to re-fetch.",
            raw_path,
        )
    else:
        df_raw.to_csv(raw_path, index=False)
        logger.info(
            "[load_data.fetch_and_save_raw] Raw data saved to %s  Shape: %s",
            raw_path, df_raw.shape,
        )

    return df_raw


def load_raw_data(config_path: str | Path = "config.yaml") -> pd.DataFrame:
    """Load the raw CSV from disk for downstream pipeline steps.

    If the raw file does not exist, fetch_and_save_raw() is called first
    so the pipeline is fully self-bootstrapping on first run.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Raw DataFrame loaded from data/raw/flights_raw.csv.
    """
    config = _load_config(config_path)
    raw_path = Path(config["data"]["raw_path"])

    if not raw_path.exists():
        logger.info(
            "[load_data.load_raw_data] Raw file not found. "
            "Triggering API fetch via fetch_and_save_raw()."
        )
        return fetch_and_save_raw(config_path)

    logger.info(
        "[load_data.load_raw_data] Loading raw data from %s", raw_path
    )
    df = pd.read_csv(raw_path)
    logger.info(
        "[load_data.load_raw_data] Loaded. Shape: %s", df.shape
    )
    return df

def generate_sample(config_path="config.yaml"):
    """Generate synthetic flight data for local testing — no API needed."""
    import numpy as np
    
    config = _load_config(config_path)
    data_cfg = config["data"]
    rng = np.random.default_rng(42)
    n = 2000

    dep_hours = rng.integers(5, 23, size=n)
    arr_delay = rng.normal(loc=5, scale=25, size=n).round(0)

    df = pd.DataFrame({
        "delayed": (arr_delay > 15).astype(int),
        "temperature_2m": rng.uniform(5, 35, size=n).round(1),
        "windspeed_10m": rng.uniform(0, 30, size=n).round(1),
        "weathercode": rng.integers(0, 10, size=n),
        "MONTH": rng.integers(6, 9, size=n),
        "DAY": rng.integers(1, 29, size=n),
        "CRS_DEP_TIME": dep_hours * 100,
        "ARR_DELAY": arr_delay,
        "AIRLINE": rng.choice(["AA","UA","DL","BA"], size=n),
        "ORIGIN_AIRPORT": "EGLL",
        "DESTINATION_AIRPORT": rng.choice(["CDG","FRA","AMS","MAD"], size=n),
        "AIR_TIME": rng.integers(60, 480, size=n),
        "DISTANCE": rng.integers(200, 5000, size=n),
    })
    
    raw_path = Path(data_cfg["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"[generate_sample] Saved {len(df)} rows to {raw_path}")
    return df