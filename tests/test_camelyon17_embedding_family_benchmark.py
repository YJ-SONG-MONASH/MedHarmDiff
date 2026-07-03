import json
import subprocess
import sys

import pandas as pd


def _fake_feature_csv(path, *, offset: float) -> None:
    rows = []
    split_by_center = {
        "0": ["train", "train", "train", "train", "val", "val"],
        "1": ["train", "train", "train", "train", "val", "val"],
        "4": ["test", "test", "test", "test", "test", "test"],
    }
    for center, split_groups in split_by_center.items():
        for index, split_group in enumerate(split_groups):
            label = index % 2
            rows.append(
                {
                    "sample_id": f"{center}_{index}",
                    "center_id": center,
                    "label": label,
                    "patient_or_group_id": f"{center}_slide_{index}",
                    "split_group": split_group,
                    "feature_0": float(label) + offset + float(center) * 0.01,
                    "feature_1": float(index) * 0.1 + offset,
                    "feature_2": float(label) * 0.5 + float(center) * 0.02,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_embedding_family_benchmark_summarizes_fake_feature_csvs(tmp_path) -> None:
    color_csv = tmp_path / "color_features.csv"
    resnet_csv = tmp_path / "resnet_features.csv"
    output_dir = tmp_path / "embedding_family_results"
    _fake_feature_csv(color_csv, offset=0.0)
    _fake_feature_csv(resnet_csv, offset=0.2)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_camelyon17_embedding_family_benchmark.py",
            "--feature-csv",
            f"color_stats_v0={color_csv}",
            "--feature-csv",
            f"resnet18_imagenet_v0={resnet_csv}",
            "--target-center",
            "4",
            "--group-column",
            "patient_or_group_id",
            "--split-group-column",
            "split_group",
            "--source-train-split-values",
            "train",
            "--source-val-split-values",
            "val",
            "--target-test-split-values",
            "test",
            "--setting",
            "zero_shot",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=".",
        capture_output=True,
        text=True,
    )

    assert "embedding_family_summary:" in completed.stdout
    summary_csv = output_dir / "embedding_family_summary.csv"
    summary_json = output_dir / "embedding_family_summary.json"
    report_md = output_dir / "embedding_family_report.md"
    assert summary_csv.exists()
    assert summary_json.exists()
    assert report_md.exists()

    summary = pd.read_csv(summary_csv)
    assert set(summary["embedding_family"]) == {"color_stats_v0", "resnet18_imagenet_v0"}
    assert summary["paper_safe_split"].all()
    assert not summary["uses_target_labels"].any()
    assert {
        "best_statistical_method",
        "best_ridge_method",
        "latent_diffusion_v0_target_auc",
        "claim_gate_status",
        "diffusion_source_drop_large",
        "diffusion_margin_insufficient",
        "representation_needs_upgrade",
    }.issubset(summary.columns)
    color_row = summary.set_index("embedding_family").loc["color_stats_v0"]
    assert bool(color_row["representation_needs_upgrade"]) == (
        color_row["claim_gate_status"] != "diffusion_contribution_established"
    )

    summary_records = json.loads(summary_json.read_text("utf-8"))
    assert len(summary_records) == 2
    assert "source drop" in report_md.read_text("utf-8").lower()
