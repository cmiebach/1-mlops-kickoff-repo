"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""
from __future__ import annotations
import pandas as pd


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

    if X_infer.empty:
        raise ValueError("Input DataFrame is empty.")

    if not hasattr(model, "predict"):
        raise TypeError("Model must implement predict().")

    predictions = model.predict(X_infer)

    result = pd.DataFrame(
        {"prediction": predictions},
        index=X_infer.index,
    )

    if include_proba and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_infer)[:, 1]
        result["probability"] = probabilities

    return result