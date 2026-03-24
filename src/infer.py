"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""
from __future__ import annotations
import pandas as pd
from src.logger import get_logger

logger = get_logger(__name__)


def run_inference(
    model,
    X_infer: pd.DataFrame,
    include_proba: bool = True,
) -> pd.DataFrame:
    """
    Run the model on new data and return predictions.

    Args:
        model: Fitted sklearn model or pipeline.
        X_infer: Feature DataFrame to predict on.
        include_proba: If True and supported, include probability column.

    Returns:
        DataFrame with 'prediction' and optional 'probability'.
    """
    logger.info("[infer] Starting inference on %d rows", len(X_infer))

    if X_infer.empty:
        logger.error("[infer] Input DataFrame is empty")
        raise ValueError("Input DataFrame is empty.")

    if not hasattr(model, "predict"):
        logger.error("[infer] Model does not implement predict() — got type: %s", type(model).__name__)
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
        "[infer] Done | predicted_delayed=%d, predicted_on_time=%d",
        int((predictions == 1).sum()),
        int((predictions == 0).sum()),
    )

    return result