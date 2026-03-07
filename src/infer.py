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

from pathlib import Path


def predict_and_save(
    model,
    X_test: pd.DataFrame,
    passenger_ids: pd.Series | None,
    out_path: str,
) -> pd.DataFrame:
    """
    Generate predictions and save them to CSV.

    Args:
        model: Fitted model.
        X_test: DataFrame to predict on.
        passenger_ids: Optional IDs to include in output.
        out_path: Path to save CSV.

    Returns:
        DataFrame of saved predictions.
    """

    preds = model.predict(X_test).astype(int)

    out = pd.DataFrame({"Survived": preds})

    if passenger_ids is not None:
        out.insert(0, "PassengerId", passenger_ids.values)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p, index=False)

    return out