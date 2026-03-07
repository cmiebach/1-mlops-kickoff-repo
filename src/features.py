from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer


class _BinarySum:
    """Picklable transformer that row-sums a fixed set of binary columns."""

    def __init__(self, cols):
        self.cols = cols

    def __call__(self, X):
        import pandas as pd
        df = pd.DataFrame(X, columns=self.cols) if not hasattr(X, "columns") else X[self.cols]
        return df.sum(axis=1).values.reshape(-1, 1)


def _make_binary_sum(cols):
    """Return a FunctionTransformer that sums the given binary columns."""
    return FunctionTransformer(_BinarySum(cols))

def get_feature_preprocessor(
    quantile_bin_cols: list[str],
    categorical_onehot_cols: list[str],
    numeric_passthrough_cols: list[str],
    binary_sum_cols: list[str],
    n_bins: int = 4,
) -> ColumnTransformer:
    """Build a feature preprocessing blueprint (unfitted ColumnTransformer).

    This function creates the *recipe* only — no fitting happens here.
    Fitting occurs inside train_model() on the training split exclusively.

    Args:
        quantile_bin_cols:       Continuous columns to discretise into quantile bins.
        categorical_onehot_cols: Categorical columns to one-hot encode.
        numeric_passthrough_cols: Numeric columns passed through as-is.
        binary_sum_cols:         Binary flag columns to aggregate into a single count.
        n_bins:                  Number of bins for KBinsDiscretizer.

    Returns:
        An unfitted ColumnTransformer.
    """
    transformers = []

    if quantile_bin_cols:
        transformers.append((
            "quantile_bin",
            KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", quantile_method="averaged_inverted_cdf"),
            quantile_bin_cols,
        ))

    if categorical_onehot_cols:
        transformers.append((
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
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

    return ColumnTransformer(transformers=transformers, remainder="drop")