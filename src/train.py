from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline


@dataclass
class TrainArtifacts:
    model: Pipeline
    X_valid: pd.DataFrame
    y_valid: pd.Series


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    problem_type: str = "classification",
) -> Pipeline:
    """Fit a full sklearn Pipeline (preprocessor + estimator) on training data.

    Args:
        X_train:      Training features.
        y_train:      Training labels.
        preprocessor: Unfitted ColumnTransformer from src.features.
        problem_type: "classification" or "regression".

    Returns:
        Fitted sklearn Pipeline.
    """
    if problem_type == "classification":
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    return pipeline