# Flight Delay Prediction — London Heathrow (EGLL)

**Authors:** MLOps Student Team (Caspar, Jan, Fouad, Lea, Ghezlan)  
**Course:** MLOps: Master in Business Analytics and Data Science  
**Status:** In Development (Modularised Pipeline)

---

## 1. Business Objective

Airlines and ground operations teams at London Heathrow currently rely on reactive systems to manage delays. This project shifts from reactive monitoring to predictive early warning.

- **The Goal:** Predict whether a departure flight from London Heathrow (EGLL) will be delayed by more than 15 minutes, using historical weather conditions and flight data.
- **The User:** Operations managers and airline staff who can use predicted delay probabilities to proactively re-allocate resources, update passengers, and adjust scheduling.
- **In Scope:** A repeatable, auditable MLOps pipeline generating binary delay classifications and probability scores.
- **Out of Scope:** Automated crew scheduling decisions, causal inference, or real-time ATC integration.

---

## 2. Success Metrics

- **Business KPI:** Reduce the proportion of unmanaged delays by surfacing high-risk departures ahead of time.
- **Technical Metric:** **F1-Score** on the validation set. We balance catching true delays (Recall) without overwhelming operators with false alarms (Precision).
- **Secondary Metrics:** ROC-AUC, PR-AUC, Brier score, and False Negative Rate (FNR) — all computed in `evaluate.py`.

---

## 3. The Data

### Sources

| Source | Purpose | Auth |
|---|---|---|
| [Open-Meteo Historical Weather API](https://open-meteo.com/) | Hourly weather at EGLL coordinates | Free, no key required |
| [OpenSky Network REST API](https://opensky-network.org/) | Historical departure flight state vectors | Optional OAuth2 (see `credentials.env`) |

> **Note:** If OpenSky returns a 403 (anonymous access restriction), or during local development/testing, use `generate_sample()` in `load_data.py` to produce 2,000 rows of synthetic flight data without any API calls.

### Unit of Analysis

One row = one departure flight from EGLL, enriched with the hourly weather at its scheduled departure time.

### Dataset Snapshot (synthetic sample)

- Rows: 2,000 (synthetic) | variable (live API)
- Airport: London Heathrow — ICAO `EGLL` (configurable in `config.yaml`)
- Date range (default): `2023-06-01` to `2023-08-31`
- Positive class prevalence (`delayed=1`): ~approximately 30% depending on threshold

### Target Definition

`delayed`: Binary flag — 1 if a flight's actual duration exceeded the median route duration by more than **15 minutes**, 0 otherwise. Threshold is configurable via `delay_threshold_minutes` in `config.yaml`.

### Data Sensitivity

No personal identifiers are present. In a production context, flight manifest data may fall under aviation data-sharing agreements and should not be committed to public version control without review.

### Feature Dictionary

| Feature | Type | Description |
|---|---|---|
| `delayed` | Binary target | 1 = delayed >15 min, 0 = on time |
| `temperature_2m` | Numeric | Air temperature at 2m height (°C) |
| `precipitation` | Numeric | Rainfall at departure time (mm) |
| `windspeed_10m` | Numeric (binned) | Wind speed at 10m height (km/h) |
| `winddirection_10m` | Numeric | Wind direction at 10m (degrees) |
| `weathercode` | Numeric | WMO weather code (used to derive flags) |
| `cloudcover` | Numeric | Total cloud cover (%) |
| `flight_duration_s` | Numeric | Actual flight duration in seconds |
| `is_foggy` | Binary flag | 1 if WMO weathercode indicates fog |
| `is_stormy` | Binary flag | 1 if WMO weathercode indicates storm |
| `is_night_departure` | Binary flag | 1 if scheduled departure hour is 22–05 |
| `is_weekend` | Binary flag | 1 if departure date falls on Saturday or Sunday |

---

## 4. ML Approach & Design Principles

This repository is a teaching scaffold for **Machine Learning Operations (MLOps)**. It demonstrates the transition from a fragile Jupyter notebook into a testable, modular software engineering architecture.

- **Separation of Concerns:** Every step (loading, cleaning, validating, training, evaluating, inferring) lives in its own single-purpose module.
- **Fail-Fast Quality Gates:** `validate.py` blocks missing values and invalid types before expensive compute begins.
- **Leakage Prevention:** Data is split *before* fitting any feature transformations.
- **Deployable Artifacts:** The orchestrator bundles preprocessing and the estimator into a single `.joblib` file, preventing training–serving skew.
- **Config-Driven:** All data paths, model hyperparameters, feature lists, and split ratios are controlled from `config.yaml` — no magic strings in code.
- **Structured Logging:** All modules write to a structured log at `logs/pipeline.log`.

### Model

A **Random Forest Classifier** (default) or **Logistic Regression** — switchable via `model.active` in `config.yaml`.

### Future Roadmap

- Add MLflow for experiment tracking and model registry
- Replace `generate_sample()` with a live authenticated OpenSky pull via CI secrets
- Containerise and serve predictions via a FastAPI application
- Extend categorical features (airline carrier, aircraft type)

---

## 5. Repository Structure

```text
.
├── config.yaml                  # All pipeline settings — single source of truth
├── credentials.env              # OpenSky OAuth2 credentials (never commit secrets!)
├── environment.yml              # Conda environment definition
├── pytest.ini                   # pytest configuration
├── README.md
│
├── data/
│   ├── raw/
│   │   └── flights_raw.csv      # Written once by load_data.py — do not edit
│   └── processed/
│       └── flights_clean.csv    # Output of clean_data.py
│
├── models/
│   └── model.joblib             # Serialised sklearn Pipeline (preprocessor + model)
│
├── notebooks/
│   └── 01_flight_delay_sandbox.ipynb  # Interactive lab bench (push with outputs cleared)
│
├── reports/
│   ├── metrics.json             # Evaluation metrics (accuracy, F1, ROC-AUC, etc.)
│   ├── predictions.csv          # Inference log with predictions and probabilities
│   └── plots/
│       └── metrics.png          # Precision–Recall curve and Confusion Matrix
│
├── logs/
│   └── pipeline.log             # Structured runtime log
│
├── src/
│   ├── __init__.py
│   ├── main.py                  # Pipeline orchestrator — only authorised writer
│   ├── load_data.py             # API fetch (Open-Meteo + OpenSky) or generate_sample()
│   ├── clean_data.py            # Renaming, flag engineering, imputation
│   ├── validate.py              # Schema and value-range quality gates
│   ├── features.py              # ColumnTransformer feature preprocessor
│   ├── train.py                 # Model training (Random Forest / Logistic Regression)
│   ├── evaluate.py              # Metrics, PR curve, confusion matrix
│   └── infer.py                 # Batch inference on held-out data
│
└── tests/
    ├── conftest.py              # Shared pytest fixtures (minimal_config, raw_df)
    ├── test_load_data.py
    ├── test_clean_data.py
    ├── test_validate.py
    ├── test_features.py
    ├── test_train.py
    ├── test_evaluate.py
    ├── test_infer.py
    └── test_main.py
```

---

## 6. How to Run & Test

### Step 1: Environment Setup

```bash
conda env create -f environment.yml
conda activate mlops
```

### Step 2: (Optional) Add OpenSky Credentials

To fetch live flight data, add your OAuth2 credentials to `credentials.env`:

```
OPENSKY_CLIENT_ID=your_client_id
OPENSKY_CLIENT_SECRET=your_client_secret
```

If credentials are absent or the API returns a 403, the pipeline automatically falls back to synthetic data via `generate_sample()`.

### Step 3: Exploratory Sandbox (The Lab Bench)

Use the Jupyter notebook for interactive exploration and debugging. It runs entirely in memory and does **not** write production artifacts to disk.

```bash
jupyter notebook notebooks/01_flight_delay_sandbox.ipynb
```

> Push the notebook with **outputs cleared** to avoid merge conflicts and binary bloat.

### Step 4: Run the Test Suite

```bash
python -m pytest -q
```

Target: ≥90% coverage across all modules.

### Step 5: Execute the Full Pipeline

```bash
python -m src.main
```

This is the **only** entry point authorised to write canonical production artifacts to disk.

---

## 7. Outputs Generated

| File | Description |
|---|---|
| `data/raw/flights_raw.csv` | Raw merged flight + weather data (write-once) |
| `data/processed/flights_clean.csv` | Cleaned, feature-engineered data |
| `models/model.joblib` | Serialised sklearn Pipeline (preprocessor + estimator) |
| `reports/metrics.json` | Evaluation metrics dict (F1, ROC-AUC, PR-AUC, Brier, FNR, etc.) |
| `reports/predictions.csv` | Inference results with predicted labels and probabilities |
| `reports/plots/metrics.png` | Precision–Recall curve and Confusion Matrix |
| `logs/pipeline.log` | Structured runtime log |

---

## 8. Team & Module Ownership

| Module | Owner | Responsibility |
|---|---|---|
| `load_data.py` | Caspar | API fetch, synthetic fallback, raw data persistence |
| `clean_data.py` | Caspar | Column renaming, binary flag engineering, imputation |
| `validate.py` | Jan | Schema validation, missing value gates, type checks |
| `train.py` | Fouad | Pipeline assembly, Random Forest / LR training |
| `evaluate.py` | Lea | Metrics computation, PR curve, confusion matrix |
| `infer.py` | Ghezlan | Batch inference, probability scoring |

### Branching Convention

- All feature branches cut from `dev`: `feature/membername/module-name`
- PRs go: **feature branch → `dev`** (requires 1 approval)
- Final merge: **`dev` → `main`** at project completion
- Both `main` and `dev` are protected (no force pushes)

---

## 9. Learning Outcomes

- Translate a notebook workflow into a `src/` layout with explicit module contracts
- Implement quality gates before training to prevent silent failure modes
- Enforce leakage prevention by design through split-then-fit boundaries
- Produce model and data artifacts for auditability and reproducibility
- Write tests that validate behaviour, not just that code runs
- Practise collaborative Git workflows with protected branches and PR reviews
