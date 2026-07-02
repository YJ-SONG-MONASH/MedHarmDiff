import json
import subprocess
import sys


def test_synthetic_stress_runner_writes_summary_and_setting_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "results"
    command = [
        sys.executable,
        "scripts/run_synthetic_stress.py",
        "--output-dir",
        str(output_dir),
        "--run-name",
        "denoising",
        "--samples-per-center",
        "24",
        "--n-features",
        "5",
    ]

    subprocess.run(command, check=True, cwd=".", capture_output=True, text=True)

    run_dir = output_dir / "synthetic_stress_denoising"
    summary_path = run_dir / "stress_summary.json"
    assert summary_path.exists()
    assert (run_dir / "stress_summary.md").exists()

    settings = {
        "setting_a_strong_shift_low_confounding",
        "setting_b_strong_confounding",
        "setting_c_weak_site_shift",
    }
    for setting in settings:
        setting_dir = run_dir / setting
        assert (setting_dir / "synthetic_features.csv").exists()
        assert (setting_dir / "metrics_by_method.csv").exists()
        assert (setting_dir / "metrics_by_method.json").exists()
        assert (setting_dir / "claim_gate_summary.json").exists()
        assert (setting_dir / "run_config.json").exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(summary["settings"]) == settings
    confounded = summary["settings"]["setting_b_strong_confounding"]
    assert confounded["claim_gate_status"] != "diffusion_contribution_established"
    assert confounded["confounding_status"] in {
        "warn_label_site_association",
        "invalidates_claim",
    }
