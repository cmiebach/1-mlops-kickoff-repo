from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline
from src.logger import get_logger

logger = get_logger(__name__)


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
    """Fit a full sklearn Pipeline on training data.

    Args:
        X_train:      Training features.
        y_train:      Training labels.
        preprocessor: Unfitted ColumnTransformer.
        problem_type: "classification" or "regression".

    Returns:
        Fitted sklearn Pipeline.
    """
    logger.info(
        "[train] Starting | problem_type=%s, "
        "train_rows=%d, features=%d",
        problem_type, len(X_train), X_train.shape[1],
    )

    if problem_type == "classification":
        estimator = RandomForestClassifier(
            n_estimators=100, random_state=42,
        )
    elif problem_type == "regression":
        estimator = RandomForestRegressor(
            n_estimators=100, random_state=42,
        )
    else:
        logger.error(
            "[train] Unsupported problem_type: '%s'.",
            problem_type,
        )
        raise ValueError(
            f"Unsupported problem_type: "
            f"'{problem_type}'."
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    pipeline.fit(X_train, y_train)

    logger.info(
        "[train] Done | estimator=%s, n_estimators=100",
        type(estimator).__name__,
    )

    return pipeline
