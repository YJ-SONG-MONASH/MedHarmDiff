import pandas as pd
import pytest

from medharmdiff.benchmark import FeatureBenchmarkConfig
from medharmdiff.splitting import plan_feature_benchmark_split


def _split_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [
                "a_g1_0",
                "a_g1_1",
                "a_g2_0",
                "a_g2_1",
                "b_g3_0",
                "b_g3_1",
                "b_g4_0",
                "b_g4_1",
                "c_g5_0",
                "c_g5_1",
            ],
            "center_id": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"],
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "patient_or_group_id": [
                "g1",
                "g1",
                "g2",
                "g2",
                "g3",
                "g3",
                "g4",
                "g4",
                "g5",
                "g5",
            ],
            "split_group": [
                "train",
                "train",
                "val",
                "val",
                "train",
                "train",
                "val",
                "val",
                "test",
                "test",
            ],
            "feature_0": [0.0, 1.0, 0.1, 1.1, 0.2, 1.2, 0.3, 1.3, 0.4, 1.4],
            "feature_1": [1.0, 0.0, 1.1, 0.1, 1.2, 0.2, 1.3, 0.3, 1.4, 0.4],
        }
    )


def _config(**overrides: object) -> FeatureBenchmarkConfig:
    values = {
        "run_name": "split_test",
        "feature_path": "unused.csv",
        "target_center": "C",
        "group_column": "patient_or_group_id",
        "validation_fraction": 0.5,
        "random_seed": 7,
    }
    values.update(overrides)
    return FeatureBenchmarkConfig(**values)


def test_group_safe_random_split_has_no_train_val_group_overlap() -> None:
    plan = plan_feature_benchmark_split(_split_df(), _config())

    train_groups = set(plan.train_df["patient_or_group_id"])
    val_groups = set(plan.val_df["patient_or_group_id"])

    assert train_groups
    assert val_groups
    assert train_groups.isdisjoint(val_groups)
    assert plan.audit["group_overlap_train_val"] == []
    assert plan.audit["paper_safe_split"] is True
    assert set(plan.test_df["center_id"]) == {"C"}


def test_official_split_group_mode_respects_configured_split_values() -> None:
    plan = plan_feature_benchmark_split(
        _split_df(),
        _config(
            split_group_column="split_group",
            source_train_split_values=["train"],
            source_val_split_values=["val"],
            target_test_split_values=["test"],
        ),
    )

    assert set(plan.train_df["sample_id"]) == {"a_g1_0", "a_g1_1", "b_g3_0", "b_g3_1"}
    assert set(plan.val_df["sample_id"]) == {"a_g2_0", "a_g2_1", "b_g4_0", "b_g4_1"}
    assert set(plan.test_df["sample_id"]) == {"c_g5_0", "c_g5_1"}
    assert plan.audit["paper_safe_split"] is True


def test_missing_group_column_raises_when_group_safe_split_required() -> None:
    df = _split_df().drop(columns=["patient_or_group_id"])

    with pytest.raises(ValueError, match="patient_or_group_id"):
        plan_feature_benchmark_split(df, _config(require_group_safe_split=True))


def test_missing_group_column_warns_when_group_safe_split_not_required() -> None:
    df = _split_df().drop(columns=["patient_or_group_id"])

    plan = plan_feature_benchmark_split(df, _config(require_group_safe_split=False))

    assert plan.audit["paper_safe_split"] is False
    assert any("not paper-safe" in warning for warning in plan.audit["warnings"])
    assert plan.train_df["center_id"].isin(["A", "B"]).all()
    assert set(plan.test_df["center_id"]) == {"C"}
