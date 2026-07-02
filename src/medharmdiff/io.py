from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_feature_csv(
    path: str | Path,
    *,
    sample_id_column: str = "sample_id",
    center_column: str = "center_id",
    label_column: str = "label",
    feature_columns: list[str] | str = "auto",
    feature_prefix: str = "feature_",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Load a feature-level MedHarmDiff CSV into arrays.

    Required columns are ``sample_id``, ``center_id``, ``label``, and at least one
    numeric feature column with the configured prefix.
    """

    df = pd.read_csv(path)
    required = [sample_id_column, center_column, label_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    if df[sample_id_column].duplicated().any():
        raise ValueError("duplicate sample_id values are not allowed")

    resolved_feature_columns = _resolve_feature_columns(
        df,
        feature_columns=feature_columns,
        feature_prefix=feature_prefix,
    )
    if not resolved_feature_columns:
        raise ValueError(f"Expected at least one feature column with prefix {feature_prefix!r}")

    non_numeric = [
        column
        for column in resolved_feature_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(f"Feature column(s) must be numeric: {', '.join(non_numeric)}")

    if df[resolved_feature_columns].isna().any().any():
        raise ValueError("Feature columns must not contain NaN values")

    labels = df[label_column]
    if labels.isna().any():
        raise ValueError("Label column must not contain NaN values")
    unique_labels = set(labels.tolist())
    if not unique_labels.issubset({0, 1, 0.0, 1.0, False, True}):
        raise ValueError("MVP feature loader expects binary labels encoded as 0/1")

    centers = df[center_column].astype(str)
    if centers.nunique() < 3:
        raise ValueError("Feature benchmark requires at least three centers")

    x = df[resolved_feature_columns].to_numpy(dtype=float)
    site = centers.to_numpy()
    y = labels.to_numpy(dtype=int)
    sample_ids = df[sample_id_column].astype(str).tolist()
    return x, site, y, sample_ids, resolved_feature_columns


def _resolve_feature_columns(
    df: pd.DataFrame,
    *,
    feature_columns: list[str] | str,
    feature_prefix: str,
) -> list[str]:
    if feature_columns != "auto":
        resolved = list(feature_columns)
        missing = [column for column in resolved if column not in df.columns]
        if missing:
            raise ValueError(f"Missing feature column(s): {', '.join(missing)}")
        return resolved

    return [column for column in df.columns if column.startswith(feature_prefix)]
