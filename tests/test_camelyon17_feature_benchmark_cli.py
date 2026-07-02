import json
import subprocess
import sys

import pandas as pd

from medharmdiff.benchmark import FeatureBenchmarkConfig, run_feature_level_benchmark


def _fake_camelyon17_feature_csv(path) -> None:
    rows = []
    for center, split_groups in {
        "0": ["train", "train", "train", "train", "val", "val"],
        "1": ["train", "train", "train", "train", "val", "val"],
        "4": ["test", "test", "test", "test", "test", "test"],
    }.items():
        for index, split_group in enumerate(split_groups):
            label = index % 2
            rows.append(
                {
                    "sample_id": f"{center}_{index}",
                    "center_id": center,
                    "label": label,
                    "patient_or_group_id": f"{center}_slide_{index}",
                    "split_group": split_group,
                    "feature_0": float(label) + float(center) * 0.01,
                    "feature_1": float(index) * 0.1,
                    "feature_2": float(label) * 0.5 + float(center) * 0.02,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_camelyon17_wrapper_writes_outputs_with_fake_feature_csv(tmp_path) -> None:
    feature_csv = tmp_path / "camelyon17_features.csv"
    output_dir = tmp_path / "results"
    _fake_camelyon17_feature_csv(feature_csv)

    command = [
        sys.executable,
        "scripts/run_camelyon17_feature_benchmark.py",
        "--feature-csv",
        str(feature_csv),
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
        "--run-name",
        "fake_camelyon17",
        "--methods",
        "identity,source_standardize,ridge_denoising,diffusion_placeholder",
    ]

    completed = subprocess.run(command, check=True, cwd=".", capture_output=True, text=True)

    run_dir = output_dir / "fake_camelyon17"
    assert "contract:" in completed.stdout
    assert "metrics:" in completed.stdout
    assert (run_dir / "metrics_by_method.csv").exists()
    assert (run_dir / "metrics_by_method.json").exists()
    assert (run_dir / "claim_gate_summary.json").exists()
    assert (run_dir / "final_report.md").exists()
    assert (run_dir / "run_config.json").exists()

    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["split_audit"]["paper_safe_split"] is True
    assert run_config["paper_safe_split"] is True
    assert run_config["split_audit"]["group_overlap_train_val"] == []
    assert run_config["split_group_column"] == "split_group"

    metrics = pd.read_csv(run_dir / "metrics_by_method.csv")
    assert not metrics["uses_target_labels"].any()


def test_benchmark_artifacts_mark_non_paper_safe_without_group_metadata(tmp_path) -> None:
    feature_csv = tmp_path / "features_without_group.csv"
    _fake_camelyon17_feature_csv(feature_csv)
    df = pd.read_csv(feature_csv).drop(columns=["patient_or_group_id", "split_group"])
    df.to_csv(feature_csv, index=False)

    result = run_feature_level_benchmark(
        FeatureBenchmarkConfig(
            run_name="unsafe",
            feature_path=feature_csv,
            output_dir=tmp_path / "results",
            target_center="4",
            setting="zero_shot",
            methods=["identity", "diffusion_placeholder"],
            require_group_safe_split=False,
            validation_fraction=0.25,
        )
    )

    run_config = json.loads(result.run_config_path.read_text(encoding="utf-8"))
    report = result.report_path.read_text(encoding="utf-8")
    metrics = pd.read_csv(result.metrics_path)

    assert "split_audit" in run_config
    assert run_config["paper_safe_split"] is False
    assert any("not paper-safe" in warning for warning in run_config["warnings"])
    assert "This run is not paper-safe due to missing or leaking group/split metadata." in report
    assert not metrics["uses_target_labels"].any()
