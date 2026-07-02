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


def validate_real_feature_contract(
    path: str | Path,
    *,
    sample_id_column: str = "sample_id",
    center_column: str = "center_id",
    label_column: str = "label",
    feature_prefix: str = "feature_",
) -> dict[str, object]:
    """Validate a real-data feature CSV contract without training any model."""

    df = pd.read_csv(path)
    warnings: list[str] = []
    required = [sample_id_column, center_column, label_column]
    missing = [column for column in required if column not in df.columns]
    for column in missing:
        warnings.append(f"missing required column: {column}")

    feature_columns = [
        column for column in df.columns if str(column).startswith(feature_prefix)
    ]
    metadata_columns = [
        column
        for column in df.columns
        if column not in set(required) and column not in set(feature_columns)
    ]
    if not feature_columns:
        warnings.append(f"no feature columns with prefix {feature_prefix!r}")
    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric_features:
        warnings.append(
            "non-numeric feature columns: " + ", ".join(non_numeric_features)
        )
    if feature_columns and df[feature_columns].isna().any().any():
        warnings.append("feature columns contain NaN values")
    if sample_id_column in df.columns and df[sample_id_column].duplicated().any():
        warnings.append("duplicate sample_id values")
    if label_column in df.columns:
        labels = df[label_column]
        if labels.isna().any():
            warnings.append("label column contains NaN values")
        elif not set(labels.tolist()).issubset({0, 1, 0.0, 1.0, False, True}):
            warnings.append("labels are not binary 0/1")
    if "patient_or_group_id" not in df.columns:
        warnings.append("patient_or_group_id is missing; leakage audit is incomplete")
    elif df["patient_or_group_id"].isna().any():
        warnings.append("patient_or_group_id contains missing values")

    center_count = 0
    center_counts: dict[str, int] = {}
    label_prevalence: dict[str, float] = {}
    if center_column in df.columns:
        centers = df[center_column].astype(str)
        center_count = int(centers.nunique())
        center_counts = {
            str(center): int(count)
            for center, count in centers.value_counts().sort_index().items()
        }
        if center_count < 3:
            warnings.append("fewer than three centers/domains detected")
        if label_column in df.columns:
            prevalence = df.groupby(center_column)[label_column].mean().sort_index()
            label_prevalence = {
                str(center): round(float(value), 6)
                for center, value in prevalence.items()
            }

    return {
        "sample_count": int(len(df)),
        "center_count": center_count,
        "center_counts": center_counts,
        "feature_count": len(feature_columns),
        "label_prevalence_by_center": label_prevalence,
        "metadata_columns": [str(column) for column in metadata_columns],
        "feature_columns": [str(column) for column in feature_columns],
        "has_patient_or_group_id": "patient_or_group_id" in df.columns
        and not df["patient_or_group_id"].isna().any(),
        "warnings": warnings,
    }


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
