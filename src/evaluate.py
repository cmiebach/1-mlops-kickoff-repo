"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""

from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


def evaluate_model(
    model,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    problem_type: str = "classification",
) -> dict[str, float]:
    """Evaluate a fitted pipeline and return a metrics dictionary.

    Args:
        model:        Fitted sklearn Pipeline.
        X_eval:       Feature DataFrame (validation or test split).
        y_eval:       True labels.
        problem_type: "classification" or "regression".

    Returns:
        Dictionary of metric name → float value.
    """
    y_pred = model.predict(X_eval)
    metrics: dict[str, float] = {}

    if problem_type == "classification":
        metrics["accuracy"] = float(accuracy_score(y_eval, y_pred))
        metrics["f1"] = float(f1_score(y_eval, y_pred, zero_division=0))
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_eval)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_eval, proba))
            metrics["pr_auc"] = float(average_precision_score(y_eval, proba))
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type!r}")

    return metrics


def compute_metrics(model, X, y, metrics: list[str]) -> dict[str, float]:
    y_pred = model.predict(X)
    out: dict[str, float] = {}

    if "accuracy" in metrics:
        out["accuracy"] = float(accuracy_score(y, y_pred))
    if "f1" in metrics:
        out["f1"] = float(f1_score(y, y_pred))
    if "roc_auc" in metrics:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            out["roc_auc"] = float(roc_auc_score(y, proba))
        else:
            out["roc_auc"] = float("nan")

    return out


def save_metrics(metrics: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2))
