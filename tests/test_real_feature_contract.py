import pandas as pd

from medharmdiff.io import validate_real_feature_contract


def test_validate_real_feature_contract_reports_counts_and_metadata(tmp_path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "center_id": ["A", "A", "B", "C"],
            "label": [0, 1, 1, 0],
            "patient_or_group_id": ["p1", "p2", "p3", "p4"],
            "scanner": [1, 2, 3, 4],
            "feature_0": [0.1, 0.2, 0.3, 0.4],
            "feature_1": [1.1, 1.2, 1.3, 1.4],
        }
    ).to_csv(path, index=False)

    summary = validate_real_feature_contract(path)

    assert summary["sample_count"] == 4
    assert summary["center_count"] == 3
    assert summary["feature_count"] == 2
    assert summary["feature_columns"] == ["feature_0", "feature_1"]
    assert summary["metadata_columns"] == ["patient_or_group_id", "scanner"]
    assert summary["has_patient_or_group_id"] is True
    assert summary["has_grouping_key"] is True
    assert summary["label_prevalence_by_center"] == {"A": 0.5, "B": 1.0, "C": 0.0}
    assert summary["warnings"] == []


def test_validate_real_feature_contract_warns_without_patient_group(tmp_path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "center_id": ["A", "B", "C"],
            "label": [0, 1, 1],
            "site_id": [1, 2, 3],
            "feature_0": [0.1, 0.2, 0.3],
        }
    ).to_csv(path, index=False)

    summary = validate_real_feature_contract(path)

    assert summary["metadata_columns"] == ["site_id"]
    assert summary["feature_columns"] == ["feature_0"]
    assert summary["has_patient_or_group_id"] is False
    assert summary["has_grouping_key"] is False
    assert any("patient_or_group_id" in warning for warning in summary["warnings"])


def test_validate_real_feature_contract_accepts_slide_id_grouping(tmp_path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "center_id": ["A", "B", "C"],
            "label": [0, 1, 1],
            "slide_id": ["slide_a", "slide_b", "slide_c"],
            "feature_0": [0.1, 0.2, 0.3],
        }
    ).to_csv(path, index=False)

    summary = validate_real_feature_contract(path)

    assert summary["has_patient_or_group_id"] is False
    assert summary["has_grouping_key"] is True
    assert not any("leakage audit is incomplete" in warning for warning in summary["warnings"])
