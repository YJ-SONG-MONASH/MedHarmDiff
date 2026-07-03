import json
import subprocess
import sys

import pandas as pd


def _write_fake_result_dir(result_dir) -> None:
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "method": "identity",
                "method_family": "statistical_baseline",
                "status": "ok",
                "target_auc": 0.62,
                "source_val_auc": 0.71,
                "site_auc_before": 0.9,
                "site_auc_after": 0.9,
                "uses_target_labels": False,
            },
            {
                "method": "center_mean",
                "method_family": "statistical_baseline",
                "status": "ok",
                "target_auc": 0.7,
                "source_val_auc": 0.73,
                "site_auc_before": 0.9,
                "site_auc_after": 0.8,
                "uses_target_labels": False,
            },
            {
                "method": "ridge_denoising",
                "method_family": "learned_denoising",
                "status": "ok",
                "target_auc": 0.68,
                "source_val_auc": 0.72,
                "site_auc_before": 0.9,
                "site_auc_after": 0.78,
                "uses_target_labels": False,
            },
            {
                "method": "latent_diffusion_v0",
                "method_family": "diffusion_v0",
                "status": "ok",
                "target_auc": 0.66,
                "source_val_auc": 0.69,
                "site_auc_before": 0.9,
                "site_auc_after": 0.76,
                "uses_target_labels": False,
            },
        ]
    ).to_csv(result_dir / "metrics_by_method.csv", index=False)
    (result_dir / "claim_gate_summary.json").write_text(
        json.dumps({"status": "insufficient_data", "reason": "fake fixture"}),
        encoding="utf-8",
    )
    (result_dir / "run_config.json").write_text(
        json.dumps(
            {
                "paper_safe_split": True,
                "split_audit": {
                    "group_overlap_train_val": [],
                    "group_overlap_train_test": [],
                    "group_overlap_val_test": [],
                },
                "confounding_audit": {"confounding_status": "pass"},
            }
        ),
        encoding="utf-8",
    )


def test_summarize_feature_benchmark_result_writes_summary_files(tmp_path) -> None:
    result_dir = tmp_path / "camelyon17_result"
    _write_fake_result_dir(result_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_feature_benchmark_result.py",
            "--result-dir",
            str(result_dir),
            "--write-summary",
        ],
        check=True,
        cwd=".",
        capture_output=True,
        text=True,
    )

    assert "best_statistical: center_mean" in completed.stdout
    assert "diffusion_beats_statistical: False" in completed.stdout
    assert "paper_safe_split: True" in completed.stdout

    summary = json.loads((result_dir / "result_triage_summary.json").read_text(encoding="utf-8"))
    assert summary["best_statistical"]["method"] == "center_mean"
    assert summary["best_learned_denoising"]["method"] == "ridge_denoising"
    assert summary["best_latent_diffusion_v0"]["method"] == "latent_diffusion_v0"
    assert summary["diffusion_beats_statistical"] is False
    assert summary["diffusion_beats_learned_denoising"] is False
    assert summary["uses_target_labels_any"] is False
    assert summary["paper_safe_split"] is True
    assert (result_dir / "result_triage_summary.md").exists()
