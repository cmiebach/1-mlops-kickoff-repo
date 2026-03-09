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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, classification_report, precision_score, recall_score


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

        metrics["precision_0"] = float(precision_score(y_eval, y_pred, pos_label=0, zero_division=0))
        metrics["recall_0"] = float(recall_score(y_eval, y_pred, pos_label=0, zero_division=0))
        metrics["precision_1"] = float(precision_score(y_eval, y_pred, pos_label=1, zero_division=0))
        metrics["recall_1"] = float(recall_score(y_eval, y_pred, pos_label=1, zero_division=0))

        metrics["accuracy"] = float(accuracy_score(y_eval, y_pred))
        metrics["f1"] = float(f1_score(y_eval, y_pred, zero_division=0))

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_eval)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_eval, proba))
            metrics["pr_auc"] = float(average_precision_score(y_eval, proba))
            metrics["brier"] = float(np.mean((proba - y_eval)**2))
            
        # False negative rate (missed positives)
        tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()
        metrics["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    else:
        raise ValueError(f"Unsupported problem_type: '{problem_type}'. Expected 'classification'.")

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

def make_plots(model, X_eval, y_eval):
    """Return a matplotlib Figure with PR curve and confusion matrix."""
    proba = model.predict_proba(X_eval)[:, 1]

    precision, recall, _ = precision_recall_curve(y_eval, proba)
    cm = confusion_matrix(y_eval, model.predict(X_eval))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Precision–Recall curve
    ax_pr = axes[0]
    ax_pr.plot(recall, precision, marker='.')
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall Curve")
    ax_pr.grid(True, alpha=0.3)

    # Confusion matrix heatmap
    ax_cm = axes[1]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_ylabel("True label")
    ax_cm.set_xlabel("Predicted label")

    fig.tight_layout()
    return fig


def save_metrics(metrics: dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2))


def save_plots(fig, save_path: str):
    """Save the given figure to disk and close it."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)

