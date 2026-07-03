import json
import subprocess
import sys
from time import perf_counter

import pandas as pd

from medharmdiff.io import load_feature_csv
from medharmdiff.real_pilots.camelyon17_wilds import (
    Camelyon17MetadataRow,
    build_camelyon17_feature_csv_rows,
    normalize_camelyon17_metadata_row,
    validate_camelyon17_rows,
)


def test_camelyon17_metadata_alias_normalization_works() -> None:
    row = normalize_camelyon17_metadata_row(
        {
            "id": "patch_001",
            "hospital_id": 2,
            "y": 1,
            "slide_id": "slide_a",
            "split": "train",
            "path": "patches/patch_001.png",
            "scanner": "scanner_x",
        }
    )

    assert row.sample_id == "patch_001"
    assert row.center_id == "2"
    assert row.label == 1
    assert row.patient_or_group_id == "slide_a"
    assert row.split_group == "train"
    assert row.image_path == "patches/patch_001.png"
    assert row.extra == {"scanner": "scanner_x"}


def test_camelyon17_metadata_missing_required_fields_are_clear() -> None:
    try:
        normalize_camelyon17_metadata_row({"id": "patch_001", "y": 1})
    except ValueError as exc:
        assert "center_id" in str(exc)
        assert "domain_id" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_camelyon17_validation_reports_grouping_warning_and_prevalence() -> None:
    rows = [
        normalize_camelyon17_metadata_row(
            {"sample_id": "s1", "center_id": "A", "label": 0, "split_group": "train"}
        ),
        normalize_camelyon17_metadata_row(
            {"sample_id": "s2", "center_id": "A", "label": 1, "split_group": "train"}
        ),
        normalize_camelyon17_metadata_row(
            {"sample_id": "s3", "center_id": "B", "label": 1, "split_group": "test"}
        ),
    ]

    summary = validate_camelyon17_rows(rows)

    assert summary["sample_count"] == 3
    assert summary["center_counts"] == {"A": 2, "B": 1}
    assert summary["label_prevalence_by_center"] == {"A": 0.5, "B": 1.0}
    assert summary["split_counts"] == {"test": 1, "train": 2}
    assert summary["has_patient_or_group_id"] is False
    assert "patient_or_group_id is missing" in " ".join(summary["warnings"])


def test_camelyon17_validation_scales_to_real_metadata_size() -> None:
    rows = [
        Camelyon17MetadataRow(
            sample_id=f"s{index}",
            center_id=str(index % 5),
            label=index % 2,
            patient_or_group_id=f"slide_{index % 50}",
            split_group="train",
        )
        for index in range(20_000)
    ]

    start = perf_counter()
    summary = validate_camelyon17_rows(rows)
    elapsed = perf_counter() - start

    assert summary["sample_count"] == 20_000
    assert summary["has_patient_or_group_id"] is True
    assert elapsed < 1.0


def test_camelyon17_feature_rows_use_feature_prefix_only() -> None:
    metadata_rows = [
        normalize_camelyon17_metadata_row(
            {
                "sample_id": "s1",
                "center_id": "A",
                "label": 0,
                "patient_id": "p1",
                "split": "train",
                "scanner": 7,
            }
        ),
        normalize_camelyon17_metadata_row(
            {
                "sample_id": "s2",
                "center_id": "B",
                "label": 1,
                "patient_id": "p2",
                "split": "test",
                "scanner": 8,
            }
        ),
    ]
    feature_df = build_camelyon17_feature_csv_rows(
        metadata_rows,
        {"s1": [0.1, 0.2], "s2": [0.3, 0.4]},
    )

    assert list(feature_df.columns) == [
        "sample_id",
        "center_id",
        "label",
        "patient_or_group_id",
        "split_group",
        "scanner",
        "feature_0",
        "feature_1",
    ]
    assert [column for column in feature_df.columns if column.startswith("feature_")] == [
        "feature_0",
        "feature_1",
    ]


def test_probe_script_writes_summary_and_report_for_fake_metadata(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    output_dir = tmp_path / "probe"
    _fake_metadata().to_csv(metadata_path, index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/probe_camelyon17_wilds_metadata.py",
            "--metadata-csv",
            str(metadata_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=".",
        capture_output=True,
        text=True,
    )

    summary = json.loads(
        (output_dir / "metadata_probe_summary.json").read_text(encoding="utf-8")
    )
    report = (output_dir / "metadata_probe_report.md").read_text(encoding="utf-8")

    assert summary["row_count"] == 3
    assert summary["center_counts"] == {"A": 1, "B": 1, "C": 1}
    assert summary["feature_csv_ready"] is True
    assert "Camelyon17-WILDS Metadata Probe" in report


def test_converter_script_writes_feature_csv_for_fake_embeddings(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    embeddings_path = tmp_path / "embeddings.csv"
    output_dir = tmp_path / "converted"
    output_csv = output_dir / "camelyon17_features.csv"
    _fake_metadata().to_csv(metadata_path, index=False)
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "embedding_0": [0.1, 0.2, 0.3],
            "embedding_1": [1.1, 1.2, 1.3],
        }
    ).to_csv(embeddings_path, index=False)

    subprocess.run(
        [
            sys.executable,
            "scripts/convert_camelyon17_embeddings_to_feature_csv.py",
            "--metadata-csv",
            str(metadata_path),
            "--embeddings-file",
            str(embeddings_path),
            "--output-csv",
            str(output_csv),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=".",
        capture_output=True,
        text=True,
    )

    feature_df = pd.read_csv(output_csv)
    summary = json.loads((output_dir / "conversion_summary.json").read_text("utf-8"))
    assert (output_dir / "conversion_report.md").exists()
    assert list(feature_df.filter(regex=r"^feature_").columns) == [
        "feature_0",
        "feature_1",
    ]
    assert summary["contract_validation"]["sample_count"] == 3
    assert summary["contract_validation"]["feature_count"] == 2
    x, _, _, _, feature_columns = load_feature_csv(output_csv)
    assert x.shape == (3, 2)
    assert feature_columns == ["feature_0", "feature_1"]


def _fake_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patch_id": ["s1", "s2", "s3"],
            "hospital": ["A", "B", "C"],
            "tumor_label": [0, 1, 1],
            "slide_id": ["p1", "p2", "p3"],
            "split": ["train", "val", "test"],
            "path": ["s1.png", "s2.png", "s3.png"],
        }
    )
