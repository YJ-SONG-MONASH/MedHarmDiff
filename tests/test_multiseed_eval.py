import json
import subprocess
import sys

import pandas as pd


def test_multiseed_eval_writes_aggregate_artifacts_and_safe_summary(tmp_path) -> None:
    output_dir = tmp_path / "results"
    command = [
        sys.executable,
        "scripts/run_multiseed_latent_diffusion_eval.py",
        "--output-dir",
        str(output_dir),
        "--run-name",
        "multiseed_test",
        "--seeds",
        "13",
        "--target-centers",
        "C",
        "--settings",
        "strong_shift_low_confounding,strong_confounding",
        "--samples-per-center",
        "20",
        "--n-features",
        "5",
        "--setting",
        "target_unlabeled",
    ]

    subprocess.run(command, check=True, cwd=".", capture_output=True, text=True)

    run_dir = output_dir / "multiseed_test"
    per_run_path = run_dir / "per_run_metrics.csv"
    aggregate_path = run_dir / "aggregate_metrics.csv"
    summary_path = run_dir / "aggregate_summary.json"
    report_path = run_dir / "aggregate_report.md"
    assert per_run_path.exists()
    assert aggregate_path.exists()
    assert summary_path.exists()
    assert report_path.exists()

    per_run = pd.read_csv(per_run_path)
    aggregate = pd.read_csv(aggregate_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")

    assert {"run_id", "seed", "target_center", "stress_setting", "method"}.issubset(
        per_run.columns
    )
    assert "latent_diffusion_v0" in set(per_run["method"])
    assert "diffusion_placeholder" in set(per_run["method"])
    assert not per_run["uses_target_labels"].fillna(False).astype(bool).any()
    placeholder = per_run[per_run["method"] == "diffusion_placeholder"]
    assert not placeholder["is_real_diffusion_v0"].astype(bool).any()

    assert "target_auc_mean" in aggregate.columns
    assert "target_auc_std" in aggregate.columns
    assert "site_auc_after_mean" in aggregate.columns
    assert "diffusion_v0_win_rate_vs_statistical" in summary
    assert "diffusion_v0_win_rate_vs_ridge" in summary
    assert "claim_gate_pass_rate" in summary
    assert "ablation_configs" in summary
    assert "latent_diffusion_v0_no_time_condition" in summary["ablation_configs"]

    confounded = summary["settings"]["strong_confounding"]
    assert confounded["confounding_invalidations"] >= 1
    assert confounded["claimable_diffusion_win_count"] == 0
    assert "placeholder_not_counted_as_real_diffusion" in report
