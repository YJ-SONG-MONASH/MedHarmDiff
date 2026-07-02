from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for _thread_env_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_var, "1")

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medharmdiff.benchmark import FeatureBenchmarkConfig, run_feature_level_benchmark
from medharmdiff.synthetic import generate_synthetic_feature_dataset


METHODS = [
    "identity",
    "source_standardize",
    "center_mean",
    "coral",
    "mmd_mean_alignment",
    "ridge_denoising",
    "ridge_denoising_clinical_preserving",
    "diffusion_placeholder",
]

PRESETS = {
    "setting_a_strong_shift_low_confounding": {
        "site_shift_strength": 2.5,
        "site_label_confounding_strength": 0.05,
        "random_seed_offset": 0,
    },
    "setting_b_strong_confounding": {
        "site_shift_strength": 2.0,
        "site_label_confounding_strength": 1.5,
        "random_seed_offset": 100,
    },
    "setting_c_weak_site_shift": {
        "site_shift_strength": 0.2,
        "site_label_confounding_strength": 0.0,
        "random_seed_offset": 200,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic MedHarmDiff stress presets.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="synthetic_stress_denoising")
    parser.add_argument("--samples-per-center", type=int, default=80)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--random-seed", type=int, default=13)
    args = parser.parse_args()

    run_dir = args.output_dir / _stress_run_dir_name(args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = run_stress_presets(
        run_dir=run_dir,
        samples_per_center=args.samples_per_center,
        n_features=args.n_features,
        random_seed=args.random_seed,
    )
    (run_dir / "stress_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "stress_summary.md").write_text(_render_summary_md(summary), encoding="utf-8")
    print(f"stress_summary: {run_dir / 'stress_summary.json'}")


def run_stress_presets(
    *,
    run_dir: Path,
    samples_per_center: int,
    n_features: int,
    random_seed: int,
) -> dict[str, object]:
    settings: dict[str, dict[str, object]] = {}
    for setting_name, preset in PRESETS.items():
        setting_dir = run_dir / setting_name
        setting_dir.mkdir(parents=True, exist_ok=True)
        seed = random_seed + int(preset["random_seed_offset"])
        synthetic_params = {
            "n_centers": 3,
            "samples_per_center": samples_per_center,
            "n_features": n_features,
            "clinical_signal_strength": 1.0,
            "site_shift_strength": preset["site_shift_strength"],
            "site_label_confounding_strength": preset["site_label_confounding_strength"],
            "noise_strength": 1.0,
            "random_seed": seed,
        }
        dataset = generate_synthetic_feature_dataset(
            n_centers=3,
            samples_per_center=samples_per_center,
            n_features=n_features,
            clinical_signal_strength=1.0,
            site_shift_strength=float(preset["site_shift_strength"]),
            site_label_confounding_strength=float(
                preset["site_label_confounding_strength"]
            ),
            noise_strength=1.0,
            random_seed=seed,
        )
        feature_path = setting_dir / "synthetic_features.csv"
        dataset.to_csv(feature_path, index=False)
        result = run_feature_level_benchmark(
            FeatureBenchmarkConfig(
                run_name="",
                feature_path=feature_path,
                output_dir=setting_dir,
                target_center="C",
                setting="target_unlabeled",
                methods=METHODS,
                random_seed=seed,
                data_source="synthetic",
                synthetic_params=synthetic_params,
            )
        )
        settings[setting_name] = _summarize_setting(result.run_dir)

    return {
        "settings": settings,
        "learned_denoising_helps_only_in_strong_shift_setting": _helps_only_in_strong_shift(
            settings
        ),
        "confounding_blocks_claims": _confounding_blocks_claims(settings),
    }


def _summarize_setting(setting_dir: Path) -> dict[str, object]:
    metrics = pd.read_csv(setting_dir / "metrics_by_method.csv")
    claim = json.loads((setting_dir / "claim_gate_summary.json").read_text(encoding="utf-8"))
    run_config = json.loads((setting_dir / "run_config.json").read_text(encoding="utf-8"))
    statistical = metrics[
        (metrics["method_family"] == "statistical_baseline") & metrics["target_auc"].notna()
    ]
    learned = metrics[
        (metrics["method_family"] == "learned_denoising") & metrics["target_auc"].notna()
    ]
    best_statistical = _best_row(statistical)
    best_learned = _best_row(learned)
    learned_beats = (
        best_statistical is not None
        and best_learned is not None
        and float(best_learned["target_auc"]) >= float(best_statistical["target_auc"]) + 0.01
    )
    learned_site_improves = (
        best_learned is not None
        and float(best_learned["site_auc_after"]) < float(best_learned["site_auc_before"])
    )
    return {
        "best_statistical_baseline": None
        if best_statistical is None
        else str(best_statistical["method"]),
        "best_statistical_target_auc": None
        if best_statistical is None
        else float(best_statistical["target_auc"]),
        "best_learned_denoising_method": None
        if best_learned is None
        else str(best_learned["method"]),
        "best_learned_denoising_target_auc": None
        if best_learned is None
        else float(best_learned["target_auc"]),
        "learned_denoising_beats_statistical": learned_beats,
        "learned_denoising_site_improves": learned_site_improves,
        "claim_gate_status": claim["status"],
        "confounding_status": run_config["confounding_audit"]["confounding_status"],
    }


def _best_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    return df.sort_values("target_auc", ascending=False).iloc[0]


def _helps_only_in_strong_shift(settings: dict[str, dict[str, object]]) -> bool:
    strong = settings["setting_a_strong_shift_low_confounding"]
    weak = settings["setting_c_weak_site_shift"]
    return bool(
        strong["learned_denoising_beats_statistical"]
        and not weak["learned_denoising_beats_statistical"]
    )


def _confounding_blocks_claims(settings: dict[str, dict[str, object]]) -> bool:
    confounded = settings["setting_b_strong_confounding"]
    return bool(
        confounded["confounding_status"] != "pass"
        and confounded["claim_gate_status"] != "diffusion_contribution_established"
    )


def _render_summary_md(summary: dict[str, object]) -> str:
    rows = [
        "| setting | best statistical | best learned denoising | claim gate | confounding |",
        "| --- | --- | --- | --- | --- |",
    ]
    for setting, data in summary["settings"].items():
        rows.append(
            "| {setting} | {stat} | {learned} | {claim} | {confounding} |".format(
                setting=setting,
                stat=data["best_statistical_baseline"],
                learned=data["best_learned_denoising_method"],
                claim=data["claim_gate_status"],
                confounding=data["confounding_status"],
            )
        )
    return "\n".join(
        [
            "# Synthetic Stress Summary",
            "",
            *rows,
            "",
            "- Learned denoising helps only in strong-shift setting: "
            f"`{summary['learned_denoising_helps_only_in_strong_shift_setting']}`",
            f"- Confounding blocks claims: `{summary['confounding_blocks_claims']}`",
            "",
        ]
    )


def _stress_run_dir_name(run_name: str) -> str:
    return run_name if run_name.startswith("synthetic_stress_") else f"synthetic_stress_{run_name}"


if __name__ == "__main__":
    main()
