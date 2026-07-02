import json
import subprocess
import sys

import pandas as pd

from medharmdiff.benchmark import FeatureBenchmarkConfig, run_feature_level_benchmark
from medharmdiff.synthetic import generate_synthetic_feature_dataset


def test_feature_level_benchmark_writes_required_artifacts(tmp_path) -> None:
    dataset = generate_synthetic_feature_dataset(
        n_centers=3,
        samples_per_center=36,
        n_features=6,
        clinical_signal_strength=1.4,
        site_shift_strength=1.8,
        site_label_confounding_strength=0.1,
        random_seed=23,
    )
    feature_path = tmp_path / "features.csv"
    dataset.to_csv(feature_path, index=False)

    config = FeatureBenchmarkConfig(
        run_name="synthetic_smoke",
        feature_path=feature_path,
        output_dir=tmp_path / "results",
        target_center="C",
        setting="target_unlabeled",
        methods=[
            "no_harmonization",
            "source_standardize",
            "center_mean",
            "coral_target_unlabeled",
            "mmd_mean_target_unlabeled",
            "diffusion_placeholder",
        ],
        random_seed=31,
    )

    result = run_feature_level_benchmark(config)

    run_dir = tmp_path / "results" / "synthetic_smoke"
    metrics_path = run_dir / "metrics_by_method.csv"
    claim_path = run_dir / "claim_gate_summary.json"
    report_path = run_dir / "final_report.md"
    run_config_path = run_dir / "run_config.json"
    assert result.run_dir == run_dir
    assert metrics_path.exists()
    assert (run_dir / "metrics_by_method.json").exists()
    assert claim_path.exists()
    assert report_path.exists()
    assert run_config_path.exists()

    metrics = pd.read_csv(metrics_path)
    assert set(metrics["method"]) == {
        "no_harmonization",
        "source_standardize",
        "center_mean",
        "coral_target_unlabeled",
        "mmd_mean_target_unlabeled",
        "diffusion_placeholder",
    }
    assert {
        "method",
        "status",
        "target_auc",
        "source_val_auc",
        "site_auc_before",
        "site_auc_after",
        "mmd_before",
        "mmd_after",
        "coral_before",
        "coral_after",
        "uses_target_unlabeled",
        "uses_target_labels",
    }.issubset(metrics.columns)
    assert not metrics["uses_target_labels"].any()
    target_unlabeled = metrics.set_index("method").loc[
        ["coral_target_unlabeled", "mmd_mean_target_unlabeled"]
    ]
    assert target_unlabeled["uses_target_unlabeled"].all()

    claim_summary = json.loads(claim_path.read_text(encoding="utf-8"))
    assert claim_summary["status"] == "insufficient_data"
    assert claim_summary["metrics"]["diffusion_is_placeholder"] is True
    raw_row = metrics.set_index("method").loc["no_harmonization"]
    assert claim_summary["metrics"]["site_metric_before"] == raw_row["site_auc_before"]
    report = report_path.read_text(encoding="utf-8")
    assert "# Feature-Level Harmonization Benchmark" in report
    assert "Safe Conclusion" in report
    assert "site_auc_before" in report
    assert "mmd_before" in report
    assert "coral_before" in report
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert run_config["setting"] == "target_unlabeled"
    assert run_config["target_center"] == "C"


def test_zero_shot_mode_marks_target_adaptation_methods_not_applicable(tmp_path) -> None:
    dataset = generate_synthetic_feature_dataset(
        n_centers=3,
        samples_per_center=24,
        random_seed=29,
    )
    feature_path = tmp_path / "features.csv"
    dataset.to_csv(feature_path, index=False)

    config = FeatureBenchmarkConfig(
        run_name="zero_shot_smoke",
        feature_path=feature_path,
        output_dir=tmp_path / "results",
        target_center="B",
        setting="zero_shot",
        methods=["no_harmonization", "coral_target_unlabeled", "mmd_mean_target_unlabeled"],
    )

    run_feature_level_benchmark(config)

    metrics = pd.read_csv(tmp_path / "results" / "zero_shot_smoke" / "metrics_by_method.csv")
    skipped = metrics.set_index("method").loc[
        ["coral_target_unlabeled", "mmd_mean_target_unlabeled"]
    ]
    assert set(skipped["status"]) == {"not_applicable_zero_shot"}
    assert not metrics["uses_target_labels"].any()


def test_feature_level_benchmark_can_write_directly_to_output_dir(tmp_path) -> None:
    dataset = generate_synthetic_feature_dataset(n_centers=3, samples_per_center=24, random_seed=41)
    feature_path = tmp_path / "features.csv"
    dataset.to_csv(feature_path, index=False)

    config = FeatureBenchmarkConfig(
        run_name="",
        feature_path=feature_path,
        output_dir=tmp_path / "direct_results",
        target_center="A",
        setting="zero_shot",
        methods=["no_harmonization", "diffusion_placeholder"],
    )

    result = run_feature_level_benchmark(config)

    assert result.run_dir == tmp_path / "direct_results"
    assert (tmp_path / "direct_results" / "metrics_by_method.csv").exists()
    assert (tmp_path / "direct_results" / "metrics_by_method.json").exists()
    assert (tmp_path / "direct_results" / "run_config.json").exists()


def test_cli_can_generate_synthetic_smoke_outputs(tmp_path) -> None:
    output_dir = tmp_path / "synthetic_smoke"
    command = [
        sys.executable,
        "scripts/run_feature_level_benchmark.py",
        "--synthetic",
        "--output-dir",
        str(output_dir),
        "--target-center",
        "C",
        "--setting",
        "target_unlabeled",
        "--n-centers",
        "3",
        "--samples-per-center",
        "30",
        "--n-features",
        "6",
    ]

    completed = subprocess.run(command, check=True, cwd=".", capture_output=True, text=True)

    assert "metrics:" in completed.stdout
    assert (output_dir / "synthetic_features.csv").exists()
    assert (output_dir / "metrics_by_method.csv").exists()
    assert (output_dir / "metrics_by_method.json").exists()
    assert (output_dir / "claim_gate_summary.json").exists()
    assert (output_dir / "final_report.md").exists()
    assert (output_dir / "run_config.json").exists()


def test_cli_can_run_from_input_csv(tmp_path) -> None:
    dataset = generate_synthetic_feature_dataset(
        n_centers=3,
        samples_per_center=24,
        n_features=5,
        random_seed=43,
    )
    input_csv = tmp_path / "features.csv"
    output_dir = tmp_path / "csv_smoke"
    dataset.to_csv(input_csv, index=False)
    command = [
        sys.executable,
        "scripts/run_feature_level_benchmark.py",
        "--input-csv",
        str(input_csv),
        "--output-dir",
        str(output_dir),
        "--target-center",
        "C",
        "--setting",
        "zero_shot",
    ]

    completed = subprocess.run(command, check=True, cwd=".", capture_output=True, text=True)

    assert "metrics:" in completed.stdout
    assert (output_dir / "metrics_by_method.csv").exists()
    assert (output_dir / "metrics_by_method.json").exists()
    assert (output_dir / "claim_gate_summary.json").exists()
    assert (output_dir / "final_report.md").exists()
    assert (output_dir / "run_config.json").exists()


def test_cli_supports_csv_alias(tmp_path) -> None:
    dataset = generate_synthetic_feature_dataset(
        n_centers=3,
        samples_per_center=24,
        n_features=5,
        random_seed=47,
    )
    input_csv = tmp_path / "features.csv"
    output_dir = tmp_path / "csv_alias_smoke"
    dataset.to_csv(input_csv, index=False)
    command = [
        sys.executable,
        "scripts/run_feature_level_benchmark.py",
        "--csv",
        str(input_csv),
        "--output-dir",
        str(output_dir),
        "--target-center",
        "C",
        "--setting",
        "zero_shot",
    ]

    subprocess.run(command, check=True, cwd=".", capture_output=True, text=True)

    assert (output_dir / "metrics_by_method.csv").exists()
    assert (output_dir / "metrics_by_method.json").exists()


def test_cli_synthetic_supports_center_2_prompt_command(tmp_path) -> None:
    output_dir = tmp_path / "synthetic_smoke_center_2"
    command = [
        sys.executable,
        "scripts/run_feature_level_benchmark.py",
        "--synthetic",
        "--target-center",
        "center_2",
        "--output-dir",
        str(output_dir),
    ]

    subprocess.run(command, check=True, cwd=".", capture_output=True, text=True)

    assert (output_dir / "metrics_by_method.csv").exists()
    assert (output_dir / "metrics_by_method.json").exists()
    assert (output_dir / "claim_gate_summary.json").exists()
    assert (output_dir / "final_report.md").exists()
    assert (output_dir / "run_config.json").exists()
    assert set(pd.read_csv(output_dir / "synthetic_features.csv")["center_id"]) == {
        "center_0",
        "center_1",
        "center_2",
    }
    methods = set(pd.read_csv(output_dir / "metrics_by_method.csv")["method"])
    assert {"identity", "source_standardize", "center_mean"}.issubset(methods)
    assert {"coral", "mmd_mean_alignment", "diffusion_placeholder"}.issubset(methods)

def test_run_config_contains_hardening_metadata(tmp_path) -> None:
    dataset = generate_synthetic_feature_dataset(
        n_centers=3,
        samples_per_center=20,
        n_features=4,
        random_seed=53,
    )
    feature_path = tmp_path / "features.csv"
    dataset.to_csv(feature_path, index=False)

    config = FeatureBenchmarkConfig(
        run_name="metadata_smoke",
        feature_path=feature_path,
        output_dir=tmp_path / "results",
        target_center="C",
        setting="zero_shot",
        methods=["identity", "diffusion_placeholder"],
    )

    result = run_feature_level_benchmark(config)
    run_config = json.loads(result.run_config_path.read_text(encoding="utf-8"))

    assert run_config["source_centers"] == ["A", "B"]
    assert run_config["data_source"] == "csv"
    assert run_config["feature_columns"] == ["feature_0", "feature_1", "feature_2", "feature_3"]
    assert run_config["sample_count_by_center"] == {"A": 20, "B": 20, "C": 20}
    assert set(run_config["label_prevalence_by_center"]) == {"A", "B", "C"}
    assert run_config["confounding_audit"]["confounding_status"] in {
        "pass",
        "warn_label_site_association",
        "invalidates_claim",
    }


def test_cli_synthetic_with_run_name_writes_dataset_inside_run_dir(tmp_path) -> None:
    output_dir = tmp_path / "results"
    command = [
        sys.executable,
        "scripts/run_feature_level_benchmark.py",
        "--synthetic",
        "--target-center",
        "C",
        "--output-dir",
        str(output_dir),
        "--run-name",
        "synthetic_smoke",
        "--noise-strength",
        "0.7",
    ]

    subprocess.run(command, check=True, cwd=".", capture_output=True, text=True)

    run_dir = output_dir / "synthetic_smoke"
    assert (run_dir / "synthetic_features.csv").exists()
    assert (run_dir / "metrics_by_method.csv").exists()
    assert (run_dir / "metrics_by_method.json").exists()
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["data_source"] == "synthetic"
    assert run_config["synthetic_params"]["noise_strength"] == 0.7
