"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import wandb

from src.logger import get_logger

logger = get_logger(__name__)


def load_model_from_registry(cfg: dict):
    """Download the 'prod' model artifact from W&B."""
    wandb_cfg = cfg.get('wandb', {})
    project = wandb_cfg.get('project', '')
    artifact_name = wandb_cfg.get(
        'model_artifact_name', 'model'
    )

    api = wandb.Api()
    artifact = api.artifact(
        f'{project}/{artifact_name}:prod', type='model'
    )
    artifact_dir = artifact.download()
    model_path = Path(artifact_dir) / 'model.joblib'
    logger.info(
        'Model loaded from W&B registry: %s:prod',
        artifact_name,
    )
    return joblib.load(model_path)


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
