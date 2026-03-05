# Flight Delay Prediction — London Heathrow (EGLL)

**Author:** Group 2 - Jan, Lea, Ghezlan, Fouad and Caspar
**Course:** MLOps: Master in Business Analytics and Data Science
**Status:** Session 1 (Initialization)

---

## 1. Business Objective

Airlines and airport operators lose millions annually due to unmanaged flight delays. This project builds a proactive prediction system that flags likely delay events before they occur, enabling earlier operational interventions.

* **The Goal:** Predict whether a departing flight from London Heathrow (EGLL) will be delayed by more than 15 minutes, based on weather conditions and flight schedule features — enabling ground crews and operations teams to take pre-emptive action.

* **The User:** Airport operations managers and airline dispatchers who use the model's binary predictions and probability scores to prioritize staffing, gate changes, and passenger communications before disruption occurs.

* **In Scope:** A reproducible, modular MLOps pipeline that classifies each flight as `delayed (1)` or `on time (0)` and generates probability scores for operational triage.

* **Out of Scope:** Root-cause analysis of delays, real-time streaming inference, or airspace-level network optimization.

---

## 2. Success Metrics

* **Business KPI (The "Why"):** Reduce last-minute operational scrambles by enabling early identification of at-risk flights, targeting a measurable reduction in unplanned gate changes and missed passenger connections.

* **Technical Metric (The "How"):** F1-Score and PR-AUC (Precision-Recall AUC) on the validation set — chosen because the dataset is imbalanced (~15–25% of flights are delayed), making these metrics more informative than raw accuracy.

* **Acceptance Criteria:** The model must outperform a simple "always predict on time" baseline on both F1 and PR-AUC. Weather-engineered features (fog, storm, wind speed bins) must show positive feature importance.

---

## 3. The Data

* **Source:** Two live APIs combined at ingestion time:
  - **Flight data:** OpenSky Network REST API (historical departures from EGLL)
  - **Weather data:** Open-Meteo Historical Weather API (hourly weather at Heathrow coordinates)

* **Unit of Analysis:** One row = one departing flight, enriched with the weather conditions at its scheduled departure hour.

* **Target Variable:** `delayed` — binary flag indicating whether `arr_delay > 15 minutes` (1 = delayed, 0 = on time).

* **Key Engineered Features** (created in `clean_data.py`):
  - `is_foggy` — derived from WMO weather codes {45, 48}
  - `is_stormy` — derived from WMO weather codes {95, 96, 99, 65, 67, 75, 77}
  - `is_night_departure` — departure hour between 22:00–05:59
  - `is_weekend` — Saturday or Sunday departure

* **Sensitive Info:** No PII or personally identifiable data. Flight identifiers are anonymized. Raw data files are excluded from version control via `.gitignore`.

  > ⚠️ **WARNING:** Ensure `data/` and `models/` remain in your `.gitignore` at all times.

---

## 4. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md                        # This file (Project definition)
├── environment.yml                  # Conda dependencies
├── config.yaml                      # Central configuration (paths, params, airport)
├── .env                             # Secrets placeholder (API keys if needed)
│
├── notebooks/                       # Experimental sandbox
│   └── flight_delay_exploration.ipynb  # EDA and feature prototyping
│
├── src/                             # Production code (The "Factory")
│   ├── __init__.py                  # Python package marker
│   ├── load_data.py                 # Fetch flights + weather from APIs
│   ├── clean_data.py                # Rename, engineer flags, drop unused cols
│   ├── validate.py                  # Data quality gates (schema, nulls, ranges)
│   ├── features.py                  # Feature preprocessing blueprint (ColumnTransformer)
│   ├── train.py                     # Model training (RandomForest pipeline)
│   ├── evaluate.py                  # Metrics computation and artifact saving
│   ├── infer.py                     # Inference logic (predict + probability scores)
│   └── main.py                      # Pipeline orchestrator (single entry point)
│
├── data/                            # Local storage (IGNORED by Git)
│   ├── raw/                         # Immutable API output (flights_raw.csv)
│   └── processed/                   # Cleaned data ready for training
│
├── models/                          # Serialized model artifacts (IGNORED by Git)
│
├── reports/                         # Generated metrics (metrics.json) and predictions
│
└── tests/                           # Automated unit tests
```

---

## 5. Execution

Ensure you have the Conda environment activated:

```bash
conda env create -f environment.yml
conda activate mlops
```

Run the full end-to-end pipeline:

```bash
python src/main.py
```

This will:
1. Fetch raw flight and weather data from the configured APIs
2. Clean and engineer features
3. Validate data quality gates
4. Train a Random Forest classifier
5. Evaluate on the validation split
6. Save the model, metrics, and predictions to `models/` and `reports/`

---

## 6. Configuration

All pipeline parameters are centralized in `config.yaml` — no hardcoded values exist in `src/`.

Key settings:

| Parameter | Default | Description |
|---|---|---|
| `airport.icao` | `EGLL` | Departure airport (London Heathrow) |
| `data.start_date` | `2023-06-01` | Start of historical window |
| `data.end_date` | `2023-08-31` | End of historical window |
| `data.delay_threshold_minutes` | `15` | Minutes above which a flight is labelled delayed |
| `split.test_size` | `0.05` | Hold-out test fraction |
| `split.val_size` | `0.15` | Validation fraction |
| `model.active` | `random_forest` | Active model configuration |
