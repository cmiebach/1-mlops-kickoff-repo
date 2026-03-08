from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


@dataclass
class TrainArtifacts:
    model: Pipeline
    X_valid: pd.DataFrame
    y_valid: pd.Series


def build_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    numeric_strategy: str,
    categorical_strategy: str,
    model_params: dict[str, Any],
) -> Pipeline:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=numeric_strategy)),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=categorical_strategy)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ]
    )

    clf = LogisticRegression(**model_params)
    return Pipeline(steps=[("preprocess", pre), ("model", clf)])


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


def train(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    stratify: bool,
    pipeline: Pipeline,
) -> TrainArtifacts:
    strat = y if stratify else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    pipeline.fit(X_train, y_train)
    return TrainArtifacts(model=pipeline, X_valid=X_valid, y_valid=y_valid)