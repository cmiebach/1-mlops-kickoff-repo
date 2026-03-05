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
"""
Module: Feature Engineering
---------------------------
Role: Define the transformation "recipe" (binning, encoding, scaling) to be bundled with the model.
Input: Configuration (lists of column names).
Output: scikit-learn ColumnTransformer object.
"""
