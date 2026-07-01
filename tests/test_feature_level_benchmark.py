import json

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
        setting="unsupervised_target_adaptation",
        methods=[
            "no_harmonization",
            "source_standardization",
            "coral",
            "mmd",
            "diffusion_harmonization",
        ],
        random_seed=31,
    )

    result = run_feature_level_benchmark(config)

    run_dir = tmp_path / "results" / "synthetic_smoke"
    metrics_path = run_dir / "metrics_by_method.csv"
    claim_path = run_dir / "claim_gate_summary.json"
    report_path = run_dir / "final_report.md"
    assert result.run_dir == run_dir
    assert metrics_path.exists()
    assert claim_path.exists()
    assert report_path.exists()

    metrics = pd.read_csv(metrics_path)
    assert set(metrics["method"]) == {
        "no_harmonization",
        "source_standardization",
        "coral",
        "mmd",
        "diffusion_harmonization",
    }
    assert {
        "method",
        "status",
        "target_auc",
        "source_auc",
        "site_auc",
        "mmd_rbf",
        "coral_distance",
        "uses_target_unlabeled",
        "uses_target_labels",
    }.issubset(metrics.columns)
    assert not metrics["uses_target_labels"].any()

    claim_summary = json.loads(claim_path.read_text(encoding="utf-8"))
    assert claim_summary["status"] != "diffusion_contribution_established"
    assert claim_summary["metrics"]["diffusion_is_placeholder"] is True
    assert "Claim Gate" in report_path.read_text(encoding="utf-8")


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
        setting="zero_shot_unseen_center",
        methods=["no_harmonization", "coral", "mmd"],
    )

    run_feature_level_benchmark(config)

    metrics = pd.read_csv(tmp_path / "results" / "zero_shot_smoke" / "metrics_by_method.csv")
    skipped = metrics.set_index("method").loc[["coral", "mmd"]]
    assert set(skipped["status"]) == {"not_applicable_zero_shot"}
    assert not metrics["uses_target_labels"].any()
