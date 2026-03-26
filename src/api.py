"""FastAPI serving layer — zero ML logic, pure wrapper."""
import yaml
from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from src.logger import get_logger
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.infer import load_model_from_registry

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request/response contracts (strict — extra="forbid")
# ---------------------------------------------------------------------------
class FlightFeatures(BaseModel):
    """Input contract for a single flight prediction."""
    model_config = ConfigDict(extra="forbid")

    temperature_2m: float = Field(
        ..., description="Air temperature at 2m (C)",
    )
    precipitation: float = Field(
        ..., ge=0, description="Rainfall (mm)",
    )
    windspeed_10m: float = Field(
        ..., ge=0, description="Wind speed at 10m (km/h)",
    )
    winddirection_10m: float = Field(
        ..., ge=0, le=360,
        description="Wind direction (deg)",
    )
    weathercode: int = Field(
        ..., description="WMO weather code",
    )
    cloudcover: float = Field(
        ..., ge=0, le=100,
        description="Cloud cover (%)",
    )
    flight_duration_s: float = Field(
        ..., gt=0,
        description="Flight duration (seconds)",
    )


class PredictResponse(BaseModel):
<<<<<<< caspar/docker-render
    """Structured prediction output."""
=======
>>>>>>> dev
    prediction: int
    probability: float
    label: str


class HealthResponse(BaseModel):
<<<<<<< caspar/docker-render
    """Health check response."""
=======
>>>>>>> dev
    status: str


# ---------------------------------------------------------------------------
# Lifespan — load config + model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config and model into memory at startup."""
    project_root = Path(__file__).resolve().parents[1]
    with open(project_root / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    app.state.cfg = cfg
    app.state.model = load_model_from_registry(cfg)
    logger.info("Model loaded at startup — ready to serve")
    yield
    logger.info("Shutting down API")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Flight Delay Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Heartbeat — confirms server is alive."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
def predict(features: FlightFeatures):
    """Predict flight delay from weather + flight features.

    JSON -> DataFrame -> clean -> validate -> predict
    """
    try:
        df = pd.DataFrame([features.model_dump()])
<<<<<<< caspar/docker-render
        df = clean_dataframe(df, inference_mode=True)
        validate_dataframe(
            df,
            numeric_non_negative_cols=app.state.cfg.get(
                "validation", {}
            ).get("numeric_non_negative_cols", []),
        )
=======
        df = clean_dataframe(df)
        validate_dataframe(df)
>>>>>>> dev

        pred = app.state.model.predict(df)[0]
        proba = app.state.model.predict_proba(df)[0]
        prob_delayed = (
            float(proba[1]) if len(proba) > 1
            else float(proba[0])
        )

        result = PredictResponse(
            prediction=int(pred),
            probability=round(prob_delayed, 4),
            label="delayed" if pred == 1 else "on_time",
        )
        logger.info("Prediction: %s", result.model_dump())
        return result

    except Exception as e:
        logger.error(
            "Prediction error: %s", e, exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=str(e),
        )
