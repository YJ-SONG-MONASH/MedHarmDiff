import json
import subprocess
import sys

import pandas as pd


def test_ridge_dominance_script_writes_summary_and_report(tmp_path) -> None:
    run_dir = tmp_path / "multiseed"
    run_dir.mkdir()
    pd.DataFrame(
        [
            _row(
                "low_1",
                "nonlinear_shift_low_confounding",
                "identity",
                "statistical_baseline",
                0.70,
                0.80,
                0.78,
            ),
            _row(
                "low_1",
                "nonlinear_shift_low_confounding",
                "center_mean",
                "statistical_baseline",
                0.72,
                0.82,
                0.76,
            ),
            _row(
                "low_1",
                "nonlinear_shift_low_confounding",
                "ridge_denoising",
                "learned_denoising",
                0.75,
                0.83,
                0.80,
            ),
            _row(
                "low_1",
                "nonlinear_shift_low_confounding",
                "latent_diffusion_v0",
                "diffusion_v0",
                0.68,
                0.74,
                0.77,
            ),
            _row(
                "conf_1",
                "mixed_realistic_shift_strong_confounding",
                "center_mean",
                "statistical_baseline",
                0.72,
                0.82,
                0.75,
                confounding_status="invalidates_claim",
            ),
            _row(
                "conf_1",
                "mixed_realistic_shift_strong_confounding",
                "ridge_denoising",
                "learned_denoising",
                0.73,
                0.84,
                0.78,
                confounding_status="invalidates_claim",
            ),
            _row(
                "conf_1",
                "mixed_realistic_shift_strong_confounding",
                "latent_diffusion_v0",
                "diffusion_v0",
                0.76,
                0.86,
                0.79,
                claimable=True,
                confounding_status="invalidates_claim",
            ),
        ]
    ).to_csv(run_dir / "per_run_metrics.csv", index=False)
    (run_dir / "aggregate_summary.json").write_text(
        json.dumps(
            {
                "run_count": 2,
                "nonlinear_settings_create_headroom": True,
                "diffusion_v0_win_rate_vs_ridge": 0.0,
                "diffusion_v0_win_rate_vs_statistical": 0.0,
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "scripts/analyze_ridge_dominance.py", str(run_dir)],
        check=True,
        cwd=".",
        capture_output=True,
        text=True,
    )

    summary_path = run_dir / "ridge_dominance_summary.json"
    report_path = run_dir / "ridge_dominance_report.md"
    assert summary_path.exists()
    assert report_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    low = summary["settings"]["nonlinear_shift_low_confounding"]
    confounded = summary["settings"]["mixed_realistic_shift_strong_confounding"]

    assert low["best_ridge_target_auc"] == 0.75
    assert low["best_diffusion_v0_target_auc"] == 0.68
    assert low["ridge_minus_diffusion"] == 0.07
    assert low["site_auc_after_comparison"]["diffusion_minus_ridge"] == -0.09
    assert summary["ridge_dominates_all_low_confounding_settings"] is True
    assert summary["statistical_dominates_all_low_confounding_settings"] is False
    assert summary["diffusion_improves_site_but_loses_clinical"] is True
    assert summary["diffusion_loses_site_and_clinical"] is False
    assert summary["headroom_without_diffusion_win"] is True
    assert summary["synthetic_probably_too_linear_for_diffusion"] is True
    assert confounded["claimable_diffusion_win_count"] == 0
    assert "Ridge Dominance Diagnosis" in report_path.read_text(encoding="utf-8")


def _row(
    run_id: str,
    setting: str,
    method: str,
    method_family: str,
    target_auc: float,
    site_auc_after: float,
    source_val_auc: float,
    *,
    claimable: bool = False,
    confounding_status: str = "pass",
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "stress_setting": setting,
        "method": method,
        "method_family": method_family,
        "target_auc": target_auc,
        "site_auc_after": site_auc_after,
        "source_val_auc": source_val_auc,
        "claim_gate_status": "not_established",
        "confounding_status": confounding_status,
        "diffusion_v0_claimable_win": claimable,
    }
