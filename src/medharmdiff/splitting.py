from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitPlan:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    audit: dict[str, object]


def plan_feature_benchmark_split(df: pd.DataFrame, config: object) -> SplitPlan:
    """Plan feature benchmark train/validation/test splits with leakage audit."""

    center_column = str(getattr(config, "center_column", "center_id"))
    target_center = str(getattr(config, "target_center"))
    split_group_column = _optional_str(getattr(config, "split_group_column", None))
    source_train_values = _string_list(getattr(config, "source_train_split_values", None))
    source_val_values = _string_list(getattr(config, "source_val_split_values", None))

    official_mode = bool(split_group_column and source_train_values and source_val_values)
    if official_mode:
        return _official_split(
            df,
            config,
            center_column=center_column,
            target_center=target_center,
            split_group_column=str(split_group_column),
            source_train_values=source_train_values,
            source_val_values=source_val_values,
        )
    return _group_safe_random_split(df, config, center_column, target_center)


def _official_split(
    df: pd.DataFrame,
    config: object,
    *,
    center_column: str,
    target_center: str,
    split_group_column: str,
    source_train_values: list[str],
    source_val_values: list[str],
) -> SplitPlan:
    if split_group_column not in df.columns:
        raise ValueError(f"split_group_column {split_group_column!r} is not present")

    split_values = df[split_group_column].astype(str)
    source_mask = df[center_column].astype(str) != target_center
    target_mask = df[center_column].astype(str) == target_center
    train_df = df[source_mask & split_values.isin(source_train_values)].copy()
    val_df = df[source_mask & split_values.isin(source_val_values)].copy()

    warnings: list[str] = []
    target_test_values = _string_list(getattr(config, "target_test_split_values", None))
    if target_test_values:
        test_df = df[target_mask & split_values.isin(target_test_values)].copy()
    else:
        test_df = df[target_mask].copy()
        warnings.append(
            "target_test_split_values not configured; using all target-center rows as test"
        )

    _ensure_non_empty(train_df, val_df, test_df)
    audit = _build_audit(train_df, val_df, test_df, config, warnings)
    return SplitPlan(train_df=train_df, val_df=val_df, test_df=test_df, audit=audit)


def _group_safe_random_split(
    df: pd.DataFrame,
    config: object,
    center_column: str,
    target_center: str,
) -> SplitPlan:
    source_df = df[df[center_column].astype(str) != target_center].copy()
    test_df = df[df[center_column].astype(str) == target_center].copy()
    group_column = _optional_str(getattr(config, "group_column", None))
    require_group_safe = bool(getattr(config, "require_group_safe_split", False))

    if not group_column or group_column not in source_df.columns:
        if require_group_safe:
            missing = group_column or "group_column"
            raise ValueError(f"{missing!r} is required for a group-safe source split")
        train_df, val_df = _row_level_source_split(source_df, config)
        warnings = [
            "group column is missing; source train/validation split is not paper-safe",
        ]
        audit = _build_audit(train_df, val_df, test_df, config, warnings)
        return SplitPlan(train_df=train_df, val_df=val_df, test_df=test_df, audit=audit)

    if source_df[group_column].isna().any():
        if require_group_safe:
            raise ValueError(f"group column {group_column!r} contains missing values")
        train_df, val_df = _row_level_source_split(source_df, config)
        warnings = [
            f"group column {group_column!r} contains missing values; split is not paper-safe",
        ]
        audit = _build_audit(train_df, val_df, test_df, config, warnings)
        return SplitPlan(train_df=train_df, val_df=val_df, test_df=test_df, audit=audit)

    groups = sorted(source_df[group_column].astype(str).unique().tolist())
    if len(groups) < 2:
        raise ValueError("group-safe split requires at least two source groups")

    train_groups, val_groups = _split_groups(
        groups,
        validation_fraction=float(getattr(config, "validation_fraction", 0.15)),
        random_seed=int(getattr(config, "random_seed", 13)),
    )
    group_values = source_df[group_column].astype(str)
    train_df = source_df[group_values.isin(train_groups)].copy()
    val_df = source_df[group_values.isin(val_groups)].copy()

    _ensure_non_empty(train_df, val_df, test_df)
    audit = _build_audit(train_df, val_df, test_df, config, warnings=[])
    return SplitPlan(train_df=train_df, val_df=val_df, test_df=test_df, audit=audit)


def _split_groups(
    groups: list[str],
    *,
    validation_fraction: float,
    random_seed: int,
) -> tuple[set[str], set[str]]:
    try:
        train_groups, val_groups = train_test_split(
            groups,
            test_size=validation_fraction,
            random_state=random_seed,
        )
    except ValueError:
        val_count = min(len(groups) - 1, max(1, int(round(len(groups) * validation_fraction))))
        ordered = sorted(groups)
        val_groups = ordered[:val_count]
        train_groups = ordered[val_count:]
    if not train_groups or not val_groups:
        raise ValueError("group-safe split produced an empty train or validation group set")
    return set(train_groups), set(val_groups)


def _row_level_source_split(
    source_df: pd.DataFrame,
    config: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_column = str(getattr(config, "label_column", "label"))
    validation_fraction = float(getattr(config, "validation_fraction", 0.15))
    random_seed = int(getattr(config, "random_seed", 13))
    labels = source_df[label_column]
    stratify = labels if labels.value_counts().min() >= 2 else None
    try:
        train_df, val_df = train_test_split(
            source_df,
            test_size=validation_fraction,
            random_state=random_seed,
            stratify=stratify,
        )
    except ValueError:
        sample_id_column = str(getattr(config, "sample_id_column", "sample_id"))
        val_count = max(1, int(round(len(source_df) * validation_fraction)))
        val_df = source_df.sort_values(sample_id_column).head(val_count)
        train_df = source_df.drop(val_df.index)
    _ensure_non_empty(train_df, val_df, source_df)
    return train_df.copy(), val_df.copy()


def _build_audit(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: object,
    warnings: list[str],
) -> dict[str, object]:
    group_column = _optional_str(getattr(config, "group_column", None))
    has_group_metadata = bool(
        group_column
        and group_column in train_df.columns
        and group_column in val_df.columns
        and group_column in test_df.columns
        and not train_df[group_column].isna().any()
        and not val_df[group_column].isna().any()
        and not test_df[group_column].isna().any()
    )
    train_groups = _groups(train_df, group_column)
    val_groups = _groups(val_df, group_column)
    test_groups = _groups(test_df, group_column)
    overlap_train_val = sorted(set(train_groups) & set(val_groups))
    overlap_train_test = sorted(set(train_groups) & set(test_groups))
    overlap_val_test = sorted(set(val_groups) & set(test_groups))

    audit_warnings = list(warnings)
    if not has_group_metadata:
        audit_warnings.append(
            "group metadata is missing or incomplete; split audit is not paper-safe"
        )

    has_group_overlap = bool(overlap_train_val or overlap_train_test or overlap_val_test)
    if has_group_overlap:
        audit_warnings.append("group identifiers overlap across train/val/test splits")

    paper_safe_split = has_group_metadata and not has_group_overlap and not audit_warnings
    return {
        "train_count": int(len(train_df)),
        "val_count": int(len(val_df)),
        "test_count": int(len(test_df)),
        "train_groups": train_groups,
        "val_groups": val_groups,
        "test_groups": test_groups,
        "group_overlap_train_val": overlap_train_val,
        "group_overlap_train_test": overlap_train_test,
        "group_overlap_val_test": overlap_val_test,
        "warnings": audit_warnings,
        "paper_safe_split": bool(paper_safe_split),
    }


def _groups(df: pd.DataFrame, group_column: str | None) -> list[str]:
    if not group_column or group_column not in df.columns:
        return []
    return sorted(df[group_column].dropna().astype(str).unique().tolist())


def _ensure_non_empty(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("split produced an empty train, validation, or test set")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
    else:
        parts = [str(item).strip() for item in value]
    resolved = [item for item in parts if item]
    return resolved or None
