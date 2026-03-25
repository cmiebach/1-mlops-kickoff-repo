"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
import wandb

from src.logger import get_logger

logger = get_logger(__name__)


def load_model_from_registry(cfg: dict):
    """Download the promoted model artifact from W&B.

    Falls back to local model.joblib when
    MODEL_SOURCE != 'wandb'.
    """
    model_source = os.getenv("MODEL_SOURCE", "local")

    if model_source == "wandb":
        wandb_cfg = cfg.get("wandb", {})
        project = wandb_cfg.get("project", "")
        entity = os.getenv("WANDB_ENTITY", "")
        artifact_name = wandb_cfg.get(
            "model_artifact_name", "model"
        )
        alias = os.getenv("WANDB_MODEL_ALIAS", "prod")

        full_name = (
            f"{entity}/{project}/{artifact_name}:{alias}"
        )
        logger.info(
            "Downloading model from W&B: %s", full_name
        )
        api = wandb.Api()
        artifact = api.artifact(
            full_name, type="model"
        )
        artifact_dir = artifact.download()
        model_path = Path(artifact_dir) / "model.joblib"
        logger.info("Model loaded from W&B registry")
        return joblib.load(model_path)

    local_path = Path(cfg["paths"]["model_path"])
    logger.info(
        "Loading model from local path: %s", local_path
    )
    return joblib.load(local_path)


def run_inference(
    model,
    X_infer: pd.DataFrame,
    include_proba: bool = True,
) -> pd.DataFrame:
    """Run the model on new data and return predictions.

    Args:
        model: Fitted sklearn model or pipeline.
        X_infer: Feature DataFrame to predict on.
        include_proba: Include probability column.

    Returns:
        DataFrame with 'prediction' and optional
        'probability'.
    """
    logger.info(
        "[infer] Starting inference on %d rows",
        len(X_infer),
    )

    if X_infer.empty:
        logger.error("[infer] Input DataFrame is empty")
        raise ValueError("Input DataFrame is empty.")

    if not hasattr(model, "predict"):
        logger.error(
            "[infer] Model does not implement predict() "
            "— got type: %s",
            type(model).__name__,
        )
        raise TypeError("Model must implement predict().")

    predictions = model.predict(X_infer)

    result = pd.DataFrame(
        {"prediction": predictions},
        index=X_infer.index,
    )

    if include_proba and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_infer)[:, 1]
        result["probability"] = probabilities

    logger.info(
        "[infer] Done | predicted_delayed=%d, "
        "predicted_on_time=%d",
        int((predictions == 1).sum()),
        int((predictions == 0).sum()),
    )

    return result
