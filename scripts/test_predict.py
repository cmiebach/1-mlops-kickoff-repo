"""Quick smoke test for the Flight Delay Prediction API."""
import requests
import sys

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

payload = {
    "temperature_2m": 12.5,
    "precipitation": 0.2,
    "windspeed_10m": 25.0,
    "winddirection_10m": 270.0,
    "weathercode": 3,
    "cloudcover": 80.0,
    "flight_duration_s": 7200.0,
}

# 1. Health check
r = requests.get(f"{BASE_URL}/health")
print(f"Health: {r.status_code} → {r.json()}")
assert r.status_code == 200

# 2. Valid prediction
r = requests.post(f"{BASE_URL}/predict", json=payload)
print(f"Predict: {r.status_code} → {r.json()}")
assert r.status_code == 200

# 3. Bad input — cloudcover > 100 should return 422
bad = {**payload, "cloudcover": 999}
r = requests.post(f"{BASE_URL}/predict", json=bad)
print(f"Bad input: {r.status_code} (expect 422)")
assert r.status_code == 422

# 4. Extra field — should return 422 (extra="forbid")
extra = {**payload, "unknown_field": 42}
r = requests.post(f"{BASE_URL}/predict", json=extra)
print(f"Extra field: {r.status_code} (expect 422)")
assert r.status_code == 422

print("\nAll tests passed!")
