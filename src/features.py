from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from src.logger import get_logger

logger = get_logger(__name__)


class _BinarySum:
    """Picklable transformer that row-sums binary columns."""

    def __init__(self, cols):
        """Store the column names to sum."""
        self.cols = cols

    def __call__(self, X):
        """Sum binary columns and return a single column."""
        import pandas as pd
        if not hasattr(X, "columns"):
            df = pd.DataFrame(X, columns=self.cols)
        else:
            df = X[self.cols]
        return df.sum(axis=1).values.reshape(-1, 1)


def _make_binary_sum(cols):
    """Return a FunctionTransformer that sums binary cols."""
    return FunctionTransformer(_BinarySum(cols))


def get_feature_preprocessor(
    quantile_bin_cols: list[str],
    categorical_onehot_cols: list[str],
    numeric_passthrough_cols: list[str],
    binary_sum_cols: list[str],
    n_bins: int = 4,
) -> ColumnTransformer:
    """Build a feature preprocessing blueprint.

    This function creates the recipe only — no fitting
    happens here. Fitting occurs inside train_model().

    Args:
        quantile_bin_cols:       Columns to discretise.
        categorical_onehot_cols: Columns to one-hot encode.
        numeric_passthrough_cols: Numeric pass-through cols.
        binary_sum_cols:         Binary flags to aggregate.
        n_bins:                  Bins for KBinsDiscretizer.

    Returns:
        An unfitted ColumnTransformer.
    """

    logger.info(
        "[features] Building preprocessor | "
        "bin_cols=%d, onehot_cols=%d, "
        "numeric_cols=%d, binary_sum_cols=%d",
        len(quantile_bin_cols),
        len(categorical_onehot_cols),
        len(numeric_passthrough_cols),
        len(binary_sum_cols),
    )
    transformers = []

    if quantile_bin_cols:
        transformers.append((
            "quantile_bin",
            KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy="quantile",
                quantile_method="averaged_inverted_cdf",
            ),
            quantile_bin_cols,
        ))

    if categorical_onehot_cols:
        transformers.append((
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_onehot_cols,
        ))

    if numeric_passthrough_cols:
        transformers.append((
            "numeric",
            "passthrough",
            numeric_passthrough_cols,
        ))

    if binary_sum_cols:
        transformers.append((
            "binary_sum",
            _make_binary_sum(binary_sum_cols),
            binary_sum_cols,
        ))

    logger.info(
        "[features] Preprocessor built with %d transformer(s)",
        len(transformers),
    )

    return ColumnTransformer(
        transformers=transformers, remainder="drop"
    )
