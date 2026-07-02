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
from medharmdiff.synthetic import (
    generate_synthetic_feature_dataset,
    generate_synthetic_nonlinear_multicenter_features,
)


METHODS = [
    "identity",
    "source_standardize",
    "center_mean",
    "coral",
    "mmd_mean_alignment",
    "ridge_denoising",
    "ridge_denoising_clinical_preserving",
    "latent_diffusion_v0",
    "latent_diffusion_clinical_preserving_v0",
    "diffusion_placeholder",
]

PRESETS = {
    "strong_shift_low_confounding": {
        "generator": "additive",
        "site_shift_strength": 2.5,
        "site_label_confounding_strength": 0.05,
        "random_seed_offset": 0,
    },
    "strong_confounding": {
        "generator": "additive",
        "site_shift_strength": 2.0,
        "site_label_confounding_strength": 1.5,
        "random_seed_offset": 100,
    },
    "weak_site_shift": {
        "generator": "additive",
        "site_shift_strength": 0.2,
        "site_label_confounding_strength": 0.0,
        "random_seed_offset": 200,
    },
    "nonlinear_shift_low_confounding": {
        "generator": "nonlinear",
        "additive_shift_strength": 0.25,
        "covariance_shift_strength": 0.35,
        "rotation_shift_strength": 0.75,
        "heteroskedastic_noise_strength": 0.35,
        "nonlinear_warp_strength": 1.1,
        "class_conditional_style_strength": 0.2,
        "site_specific_feature_interaction_strength": 0.25,
        "label_site_confounding_strength": 0.05,
        "random_seed_offset": 300,
    },
    "covariance_shift_low_confounding": {
        "generator": "nonlinear",
        "additive_shift_strength": 0.1,
        "covariance_shift_strength": 1.2,
        "rotation_shift_strength": 0.15,
        "heteroskedastic_noise_strength": 0.1,
        "nonlinear_warp_strength": 0.0,
        "class_conditional_style_strength": 0.0,
        "site_specific_feature_interaction_strength": 0.0,
        "label_site_confounding_strength": 0.05,
        "random_seed_offset": 400,
    },
    "heteroskedastic_shift_low_confounding": {
        "generator": "nonlinear",
        "additive_shift_strength": 0.1,
        "covariance_shift_strength": 0.2,
        "rotation_shift_strength": 0.2,
        "heteroskedastic_noise_strength": 1.1,
        "nonlinear_warp_strength": 0.15,
        "class_conditional_style_strength": 0.0,
        "site_specific_feature_interaction_strength": 0.1,
        "label_site_confounding_strength": 0.05,
        "random_seed_offset": 500,
    },
    "mixed_realistic_shift_low_confounding": {
        "generator": "nonlinear",
        "additive_shift_strength": 0.35,
        "covariance_shift_strength": 0.8,
        "rotation_shift_strength": 0.8,
        "heteroskedastic_noise_strength": 0.6,
        "nonlinear_warp_strength": 0.8,
        "class_conditional_style_strength": 0.25,
        "site_specific_feature_interaction_strength": 0.35,
        "label_site_confounding_strength": 0.05,
        "random_seed_offset": 600,
    },
    "mixed_realistic_shift_strong_confounding": {
        "generator": "nonlinear",
        "additive_shift_strength": 0.35,
        "covariance_shift_strength": 0.8,
        "rotation_shift_strength": 0.8,
        "heteroskedastic_noise_strength": 0.6,
        "nonlinear_warp_strength": 0.8,
        "class_conditional_style_strength": 0.25,
        "site_specific_feature_interaction_strength": 0.35,
        "label_site_confounding_strength": 1.5,
        "random_seed_offset": 700,
    },
}

ABLATION_CONFIGS = {
    "latent_diffusion_v0_default": {
        "num_steps": 10,
        "n_augments": 8,
        "noise_strength": 0.2,
        "uses_time_conditioning": True,
    },
    "latent_diffusion_v0_no_time_condition": {
        "num_steps": 10,
        "n_augments": 8,
        "noise_strength": 0.2,
        "uses_time_conditioning": False,
    },
    "latent_diffusion_v0_one_step": {
        "num_steps": 1,
        "n_augments": 8,
        "noise_strength": 0.2,
        "uses_time_conditioning": True,
    },
    "latent_diffusion_v0_low_noise": {
        "num_steps": 10,
        "n_augments": 8,
        "noise_strength": 0.05,
        "uses_time_conditioning": True,
    },
    "latent_diffusion_v0_high_noise": {
        "num_steps": 10,
        "n_augments": 8,
        "noise_strength": 0.5,
        "uses_time_conditioning": True,
    },
    "latent_diffusion_clinical_preserving_v0": {
        "num_steps": 10,
        "n_augments": 8,
        "noise_strength": 0.2,
        "clinical_preservation_strength": 0.5,
        "uses_time_conditioning": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-seed latent diffusion v0 synthetic evaluation."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="multiseed_latent_diffusion_v0")
    parser.add_argument("--seeds", default="13,17,19,23,29")
    parser.add_argument("--target-centers", default="A,B,C")
    parser.add_argument(
        "--settings",
        default="strong_shift_low_confounding,strong_confounding,weak_site_shift",
    )
    parser.add_argument(
        "--setting",
        choices=["zero_shot", "target_unlabeled"],
        default="target_unlabeled",
    )
    parser.add_argument("--samples-per-center", type=int, default=80)
    parser.add_argument("--n-features", type=int, default=16)
    args = parser.parse_args()

    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    result = run_multiseed_eval(
        run_dir=run_dir,
        seeds=_parse_ints(args.seeds),
        target_centers=_parse_strings(args.target_centers),
        settings=_parse_strings(args.settings),
        benchmark_setting=args.setting,
        samples_per_center=args.samples_per_center,
        n_features=args.n_features,
    )
    result["per_run_metrics"].to_csv(run_dir / "per_run_metrics.csv", index=False)
    result["aggregate_metrics"].to_csv(run_dir / "aggregate_metrics.csv", index=False)
    (run_dir / "aggregate_summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "aggregate_report.md").write_text(
        _render_report(result["summary"], result["aggregate_metrics"]),
        encoding="utf-8",
    )
    print(f"aggregate_summary: {run_dir / 'aggregate_summary.json'}")


def run_multiseed_eval(
    *,
    run_dir: Path,
    seeds: list[int],
    target_centers: list[str],
    settings: list[str],
    benchmark_setting: str,
    samples_per_center: int,
    n_features: int,
) -> dict[str, object]:
    rows = []
    run_summaries = []
    for setting_name in settings:
        preset = PRESETS[setting_name]
        for seed in seeds:
            dataset_seed = seed + int(preset["random_seed_offset"])
            dataset = _generate_dataset(
                preset=preset,
                dataset_seed=dataset_seed,
                samples_per_center=samples_per_center,
                n_features=n_features,
            )
            for target_center in target_centers:
                run_id = f"{setting_name}_seed_{seed}_target_{target_center}"
                single_run_dir = run_dir / "runs" / run_id
                single_run_dir.mkdir(parents=True, exist_ok=True)
                feature_path = single_run_dir / "synthetic_features.csv"
                dataset.to_csv(feature_path, index=False)
                benchmark_result = run_feature_level_benchmark(
                    FeatureBenchmarkConfig(
                        run_name="",
                        feature_path=feature_path,
                        output_dir=single_run_dir,
                        target_center=target_center,
                        setting=benchmark_setting,
                        methods=METHODS,
                        random_seed=dataset_seed,
                        data_source="synthetic",
                        synthetic_params={
                            "stress_setting": setting_name,
                            "seed": seed,
                            "target_center": target_center,
                            "samples_per_center": samples_per_center,
                            "n_features": n_features,
                            **preset,
                        },
                    )
                )
                metrics = pd.read_csv(benchmark_result.metrics_path)
                claim = json.loads(
                    benchmark_result.claim_gate_path.read_text(encoding="utf-8")
                )
                run_config = json.loads(
                    benchmark_result.run_config_path.read_text(encoding="utf-8")
                )
                run_summary = _summarize_run(metrics, claim, run_config)
                run_summaries.append(
                    {
                        "run_id": run_id,
                        "seed": seed,
                        "target_center": target_center,
                        "stress_setting": setting_name,
                        **run_summary,
                    }
                )
                for metric_row in metrics.to_dict(orient="records"):
                    rows.append(
                        {
                            "run_id": run_id,
                            "seed": seed,
                            "target_center": target_center,
                            "stress_setting": setting_name,
                            "claim_gate_status": claim["status"],
                            "confounding_status": run_summary["confounding_status"],
                            "best_statistical_method": run_summary["best_statistical_method"],
                            "best_learned_denoising_method": run_summary[
                                "best_learned_denoising_method"
                            ],
                            "best_diffusion_v0_method": run_summary[
                                "best_diffusion_v0_method"
                            ],
                            "diffusion_v0_claimable_win": run_summary[
                                "diffusion_v0_claimable_win"
                            ],
                            "is_real_diffusion_v0": (
                                metric_row["method_family"] == "diffusion_v0"
                                and not bool(metric_row["is_placeholder"])
                            ),
                            **metric_row,
                        }
                    )

    per_run_metrics = pd.DataFrame(rows)
    run_summary_df = pd.DataFrame(run_summaries)
    aggregate_metrics = _aggregate_metrics(per_run_metrics)
    summary = _aggregate_summary(
        run_summary_df,
        per_run_metrics,
        seeds,
        target_centers,
        settings,
        benchmark_setting,
    )
    return {
        "per_run_metrics": per_run_metrics,
        "aggregate_metrics": aggregate_metrics,
        "summary": summary,
    }


def _generate_dataset(
    *,
    preset: dict[str, object],
    dataset_seed: int,
    samples_per_center: int,
    n_features: int,
) -> pd.DataFrame:
    if preset.get("generator") == "nonlinear":
        return generate_synthetic_nonlinear_multicenter_features(
            n_centers=3,
            samples_per_center=samples_per_center,
            n_features=n_features,
            clinical_signal_strength=1.0,
            additive_shift_strength=float(preset["additive_shift_strength"]),
            covariance_shift_strength=float(preset["covariance_shift_strength"]),
            rotation_shift_strength=float(preset["rotation_shift_strength"]),
            heteroskedastic_noise_strength=float(
                preset["heteroskedastic_noise_strength"]
            ),
            nonlinear_warp_strength=float(preset["nonlinear_warp_strength"]),
            class_conditional_style_strength=float(
                preset["class_conditional_style_strength"]
            ),
            site_specific_feature_interaction_strength=float(
                preset["site_specific_feature_interaction_strength"]
            ),
            label_site_confounding_strength=float(
                preset["label_site_confounding_strength"]
            ),
            noise_strength=1.0,
            random_seed=dataset_seed,
        )
    return generate_synthetic_feature_dataset(
        n_centers=3,
        samples_per_center=samples_per_center,
        n_features=n_features,
        clinical_signal_strength=1.0,
        site_shift_strength=float(preset["site_shift_strength"]),
        site_label_confounding_strength=float(preset["site_label_confounding_strength"]),
        noise_strength=1.0,
        random_seed=dataset_seed,
    )


def _summarize_run(
    metrics: pd.DataFrame,
    claim: dict[str, object],
    run_config: dict[str, object],
) -> dict[str, object]:
    statistical = _family(metrics, "statistical_baseline")
    learned = _family(metrics, "learned_denoising")
    diffusion = _family(metrics, "diffusion_v0")
    identity = metrics[metrics["method"] == "identity"]
    best_statistical = _best_row(statistical)
    best_learned = _best_row(learned)
    best_diffusion = _best_row(diffusion)
    identity_source = _metric_or_none(identity, "source_val_auc")
    win = _is_claimable_diffusion_win(
        best_diffusion=best_diffusion,
        best_statistical=best_statistical,
        best_learned=best_learned,
        identity_source=identity_source,
        claim_status=str(claim["status"]),
        confounding_status=str(run_config["confounding_audit"]["confounding_status"]),
    )
    win_vs_statistical = _is_gate_aware_win(
        best_diffusion=best_diffusion,
        baseline=best_statistical,
        best_statistical=best_statistical,
        identity_source=identity_source,
        claim_status=str(claim["status"]),
        confounding_status=str(run_config["confounding_audit"]["confounding_status"]),
    )
    win_vs_ridge = _is_gate_aware_win(
        best_diffusion=best_diffusion,
        baseline=best_learned,
        best_statistical=best_statistical,
        identity_source=identity_source,
        claim_status=str(claim["status"]),
        confounding_status=str(run_config["confounding_audit"]["confounding_status"]),
    )
    return {
        "best_statistical_method": None
        if best_statistical is None
        else str(best_statistical["method"]),
        "best_statistical_target_auc": _row_float(best_statistical, "target_auc"),
        "best_statistical_source_val_auc": _row_float(best_statistical, "source_val_auc"),
        "best_learned_denoising_method": None
        if best_learned is None
        else str(best_learned["method"]),
        "best_learned_denoising_target_auc": _row_float(best_learned, "target_auc"),
        "best_diffusion_v0_method": None
        if best_diffusion is None
        else str(best_diffusion["method"]),
        "best_diffusion_v0_target_auc": _row_float(best_diffusion, "target_auc"),
        "best_diffusion_v0_source_val_auc": _row_float(best_diffusion, "source_val_auc"),
        "best_diffusion_v0_site_auc_before": _row_float(best_diffusion, "site_auc_before"),
        "best_diffusion_v0_site_auc_after": _row_float(best_diffusion, "site_auc_after"),
        "diffusion_v0_beats_statistical_margin": _beats(best_diffusion, best_statistical),
        "diffusion_v0_beats_ridge_margin": _beats(best_diffusion, best_learned),
        "diffusion_v0_win_vs_statistical": win_vs_statistical,
        "diffusion_v0_win_vs_ridge": win_vs_ridge,
        "diffusion_v0_source_preserved": _source_preserved(
            best_diffusion,
            best_statistical,
            identity_source,
        ),
        "diffusion_v0_site_improved": _site_improved(best_diffusion),
        "diffusion_v0_claimable_win": win,
        "claim_gate_status": str(claim["status"]),
        "confounding_status": str(run_config["confounding_audit"]["confounding_status"]),
    }


def _aggregate_metrics(per_run_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "target_auc",
        "source_val_auc",
        "site_auc_before",
        "site_auc_after",
        "mmd_before",
        "mmd_after",
        "coral_before",
        "coral_after",
    ]
    grouped = per_run_metrics.groupby(["method", "method_family"], dropna=False)
    aggregate = grouped[metric_columns].agg(["mean", "std"])
    aggregate.columns = [f"{name}_{stat}" for name, stat in aggregate.columns]
    return aggregate.reset_index()


def _aggregate_summary(
    run_summary_df: pd.DataFrame,
    per_run_metrics: pd.DataFrame,
    seeds: list[int],
    target_centers: list[str],
    settings: list[str],
    benchmark_setting: str,
) -> dict[str, object]:
    total_runs = int(len(run_summary_df))
    settings_summary = {}
    for setting_name, setting_df in run_summary_df.groupby("stress_setting"):
        settings_summary[str(setting_name)] = {
            "run_count": int(len(setting_df)),
            "claimable_diffusion_win_count": int(
                setting_df["diffusion_v0_claimable_win"].sum()
            ),
            "claim_gate_pass_count": int(
                (setting_df["claim_gate_status"] == "diffusion_contribution_established").sum()
            ),
            "confounding_warnings": int(
                (setting_df["confounding_status"] == "warn_label_site_association").sum()
            ),
            "confounding_invalidations": int(
                (setting_df["confounding_status"] == "invalidates_claim").sum()
            ),
        }
    setting_diagnostics = _setting_diagnostics(run_summary_df, per_run_metrics)
    difficulty_flags = _difficulty_flags(setting_diagnostics)
    return {
        "seeds": seeds,
        "target_centers": target_centers,
        "settings_evaluated": settings,
        "benchmark_setting": benchmark_setting,
        "run_count": total_runs,
        "diffusion_v0_win_rate_vs_statistical": _mean_bool(
            run_summary_df["diffusion_v0_win_vs_statistical"]
        ),
        "diffusion_v0_win_rate_vs_ridge": _mean_bool(
            run_summary_df["diffusion_v0_win_vs_ridge"]
        ),
        "diffusion_v0_target_margin_win_rate_vs_statistical": _mean_bool(
            run_summary_df["diffusion_v0_beats_statistical_margin"]
        ),
        "diffusion_v0_target_margin_win_rate_vs_ridge": _mean_bool(
            run_summary_df["diffusion_v0_beats_ridge_margin"]
        ),
        "claim_gate_pass_rate": _mean_bool(
            run_summary_df["claim_gate_status"] == "diffusion_contribution_established"
        ),
        "claimable_diffusion_win_rate": _mean_bool(
            run_summary_df["diffusion_v0_claimable_win"]
        ),
        "confounding_warning_count": int(
            (run_summary_df["confounding_status"] == "warn_label_site_association").sum()
        ),
        "confounding_invalidation_count": int(
            (run_summary_df["confounding_status"] == "invalidates_claim").sum()
        ),
        "target_labels_used_count": int(
            per_run_metrics["uses_target_labels"].fillna(False).astype(bool).sum()
        ),
        "placeholder_diffusion_rows": int(
            (per_run_metrics["method"] == "diffusion_placeholder").sum()
        ),
        "placeholder_real_diffusion_rows": int(
            per_run_metrics[
                (per_run_metrics["method"] == "diffusion_placeholder")
                & (per_run_metrics["is_real_diffusion_v0"].astype(bool))
            ].shape[0]
        ),
        "settings": settings_summary,
        "setting_diagnostics": setting_diagnostics,
        **difficulty_flags,
        "ablation_configs": ABLATION_CONFIGS,
        "decision": _decision_from_summary(run_summary_df),
    }


def _setting_diagnostics(
    run_summary_df: pd.DataFrame,
    per_run_metrics: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    diagnostics = {}
    for setting_name in sorted(per_run_metrics["stress_setting"].dropna().unique()):
        method_metrics = per_run_metrics[per_run_metrics["stress_setting"] == setting_name]
        run_metrics = run_summary_df[run_summary_df["stress_setting"] == setting_name]
        method_target = _mean_by_method(method_metrics, "target_auc")
        method_site_after = _mean_by_method(method_metrics, "site_auc_after")
        method_mmd_before = _mean_by_method(method_metrics, "mmd_before")
        method_mmd_after = _mean_by_method(method_metrics, "mmd_after")
        method_coral_before = _mean_by_method(method_metrics, "coral_before")
        method_coral_after = _mean_by_method(method_metrics, "coral_after")

        best_statistical = _best_family_mean(method_metrics, "statistical_baseline")
        best_ridge = _best_family_mean(method_metrics, "learned_denoising")
        best_diffusion = _best_family_mean(method_metrics, "diffusion_v0")
        center_mean_auc = method_target.get("center_mean")
        statistical_values = [
            value
            for method, value in method_target.items()
            if _method_family(method_metrics, method) == "statistical_baseline"
        ]
        center_mean_dominates = (
            center_mean_auc is not None
            and statistical_values
            and center_mean_auc >= max(statistical_values)
        )
        diagnostics[str(setting_name)] = {
            "raw_identity_target_auc": _optional_round(method_target.get("identity")),
            "best_statistical_target_auc": _optional_round(best_statistical),
            "best_ridge_target_auc": _optional_round(best_ridge),
            "best_diffusion_v0_target_auc": _optional_round(best_diffusion),
            "site_auc_before": _optional_round(_identity_mean(method_metrics, "site_auc_before")),
            "site_auc_after_by_method": _round_mapping(method_site_after),
            "mmd_before_by_method": _round_mapping(method_mmd_before),
            "mmd_after_by_method": _round_mapping(method_mmd_after),
            "coral_before_by_method": _round_mapping(method_coral_before),
            "coral_after_by_method": _round_mapping(method_coral_after),
            "confounding_status_counts": _value_counts(run_metrics, "confounding_status"),
            "center_mean_dominates": bool(center_mean_dominates),
            "nonlinear_shift_makes_statistical_baselines_weaker": False,
            "diffusion_v0_beats_statistical": bool(
                _mean_bool(run_metrics["diffusion_v0_beats_statistical_margin"]) > 0
            ),
            "diffusion_v0_beats_ridge": bool(
                _mean_bool(run_metrics["diffusion_v0_beats_ridge_margin"]) > 0
            ),
            "diffusion_v0_win_rate_vs_statistical": _mean_bool(
                run_metrics["diffusion_v0_win_vs_statistical"]
            ),
            "diffusion_v0_win_rate_vs_ridge": _mean_bool(
                run_metrics["diffusion_v0_win_vs_ridge"]
            ),
            "diffusion_v0_target_margin_win_rate_vs_statistical": _mean_bool(
                run_metrics["diffusion_v0_beats_statistical_margin"]
            ),
            "diffusion_v0_target_margin_win_rate_vs_ridge": _mean_bool(
                run_metrics["diffusion_v0_beats_ridge_margin"]
            ),
            "claimable_diffusion_win_count": int(
                run_metrics["diffusion_v0_claimable_win"].sum()
            ),
        }

    reference = _reference_statistical_auc(diagnostics)
    for setting_name, data in diagnostics.items():
        best_statistical = data["best_statistical_target_auc"]
        data["nonlinear_shift_makes_statistical_baselines_weaker"] = bool(
            _is_nonlinear_setting(setting_name)
            and reference is not None
            and best_statistical is not None
            and float(best_statistical) < reference - 0.01
        )
    return diagnostics


def _difficulty_flags(
    setting_diagnostics: dict[str, dict[str, object]]
) -> dict[str, object]:
    values = list(setting_diagnostics.values())
    statistical_dominates = bool(
        values
        and all(
            _dominates(
                data["best_statistical_target_auc"],
                data["best_ridge_target_auc"],
                data["best_diffusion_v0_target_auc"],
            )
            for data in values
        )
    )
    ridge_dominates = bool(
        values
        and all(
            _dominates(
                data["best_ridge_target_auc"],
                data["best_statistical_target_auc"],
                data["best_diffusion_v0_target_auc"],
            )
            for data in values
        )
    )
    diffusion_low_confounding_win = any(
        _is_low_confounding_setting(setting_name)
        and int(data["claimable_diffusion_win_count"]) > 0
        for setting_name, data in setting_diagnostics.items()
    )
    nonlinear_headroom = any(
        bool(data["nonlinear_shift_makes_statistical_baselines_weaker"])
        for data in values
    )
    return {
        "statistical_baselines_dominate_all_settings": statistical_dominates,
        "ridge_denoising_dominates_all_settings": ridge_dominates,
        "diffusion_v0_has_any_low_confounding_win": bool(diffusion_low_confounding_win),
        "nonlinear_settings_create_headroom": bool(nonlinear_headroom),
    }


def _mean_by_method(metrics: pd.DataFrame, column: str) -> dict[str, float]:
    valid = metrics[metrics[column].notna()]
    means = valid.groupby("method")[column].mean().sort_index()
    return {str(method): float(value) for method, value in means.items()}


def _best_family_mean(metrics: pd.DataFrame, family: str) -> float | None:
    family_metrics = metrics[
        (metrics["method_family"] == family) & metrics["target_auc"].notna()
    ]
    if family_metrics.empty:
        return None
    means = family_metrics.groupby("method")["target_auc"].mean()
    return float(means.max())


def _method_family(metrics: pd.DataFrame, method: str) -> str | None:
    rows = metrics[metrics["method"] == method]
    if rows.empty:
        return None
    return str(rows.iloc[0]["method_family"])


def _optional_round(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def _identity_mean(metrics: pd.DataFrame, column: str) -> float | None:
    rows = metrics[(metrics["method"] == "identity") & metrics[column].notna()]
    if rows.empty:
        return None
    return float(rows[column].mean())


def _round_mapping(values: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in values.items()}


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df:
        return {}
    counts = df[column].value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def _reference_statistical_auc(
    setting_diagnostics: dict[str, dict[str, object]]
) -> float | None:
    values = [
        data["best_statistical_target_auc"]
        for setting_name, data in setting_diagnostics.items()
        if not _is_nonlinear_setting(setting_name)
        and data["best_statistical_target_auc"] is not None
    ]
    if not values:
        return None
    return max(float(value) for value in values)


def _is_nonlinear_setting(setting_name: str) -> bool:
    return any(
        token in setting_name
        for token in ("nonlinear", "covariance", "heteroskedastic", "mixed_realistic")
    )


def _is_low_confounding_setting(setting_name: str) -> bool:
    return "strong_confounding" not in setting_name


def _dominates(candidate: object, *baselines: object) -> bool:
    if candidate is None:
        return False
    candidate_value = float(candidate)
    return all(baseline is None or candidate_value >= float(baseline) for baseline in baselines)


def _decision_from_summary(run_summary_df: pd.DataFrame) -> str:
    win_stat = _mean_bool(run_summary_df["diffusion_v0_beats_statistical_margin"])
    win_ridge = _mean_bool(run_summary_df["diffusion_v0_beats_ridge_margin"])
    pass_rate = _mean_bool(
        run_summary_df["claim_gate_status"] == "diffusion_contribution_established"
    )
    if win_stat >= 0.6 and win_ridge >= 0.6 and pass_rate > 0:
        return "consider_latent_diffusion_v1_optional_torch"
    return "pause_diffusion_and_improve_synthetic_or_data_selection"


def _render_report(summary: dict[str, object], aggregate: pd.DataFrame) -> str:
    table = aggregate.to_csv(index=False, lineterminator="\n")
    return "\n".join(
        [
            "# Multi-Seed Latent Diffusion v0 Evaluation",
            "",
            f"- Seeds: `{summary['seeds']}`",
            f"- Target centers: `{summary['target_centers']}`",
            f"- Settings: `{summary['settings_evaluated']}`",
            f"- Win rate vs statistical: `{summary['diffusion_v0_win_rate_vs_statistical']}`",
            f"- Win rate vs ridge: `{summary['diffusion_v0_win_rate_vs_ridge']}`",
            f"- Claim gate pass rate: `{summary['claim_gate_pass_rate']}`",
            f"- Confounding invalidations: `{summary['confounding_invalidation_count']}`",
            f"- Target labels used count: `{summary['target_labels_used_count']}`",
            "- Statistical baselines dominate all settings: "
            f"`{summary['statistical_baselines_dominate_all_settings']}`",
            "- Ridge denoising dominates all settings: "
            f"`{summary['ridge_denoising_dominates_all_settings']}`",
            "- Diffusion v0 has any low-confounding win: "
            f"`{summary['diffusion_v0_has_any_low_confounding_win']}`",
            "- Nonlinear settings create headroom: "
            f"`{summary['nonlinear_settings_create_headroom']}`",
            "- placeholder_not_counted_as_real_diffusion: "
            f"`{summary['placeholder_real_diffusion_rows'] == 0}`",
            f"- Decision: `{summary['decision']}`",
            "",
            "## Aggregate Metrics",
            "",
            "```csv",
            table.strip(),
            "```",
            "",
        ]
    )


def _family(metrics: pd.DataFrame, family: str) -> pd.DataFrame:
    return metrics[(metrics["method_family"] == family) & metrics["target_auc"].notna()]


def _best_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    return df.sort_values("target_auc", ascending=False).iloc[0]


def _metric_or_none(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or df[column].isna().all():
        return None
    return float(df.iloc[0][column])


def _row_float(row: pd.Series | None, column: str) -> float | None:
    if row is None or pd.isna(row[column]):
        return None
    return float(row[column])


def _beats(candidate: pd.Series | None, baseline: pd.Series | None) -> bool:
    if candidate is None or baseline is None:
        return False
    return float(candidate["target_auc"]) >= float(baseline["target_auc"]) + 0.01


def _site_improved(row: pd.Series | None) -> bool:
    if row is None:
        return False
    return float(row["site_auc_after"]) < float(row["site_auc_before"])


def _source_preserved(
    candidate: pd.Series | None,
    best_statistical: pd.Series | None,
    identity_source: float | None,
) -> bool:
    if candidate is None:
        return False
    references = []
    if best_statistical is not None:
        references.append(float(best_statistical["source_val_auc"]))
    if identity_source is not None:
        references.append(identity_source)
    if not references:
        return False
    return float(candidate["source_val_auc"]) >= max(references) - 0.01


def _is_claimable_diffusion_win(
    *,
    best_diffusion: pd.Series | None,
    best_statistical: pd.Series | None,
    best_learned: pd.Series | None,
    identity_source: float | None,
    claim_status: str,
    confounding_status: str,
) -> bool:
    return bool(
        _beats(best_diffusion, best_statistical)
        and _beats(best_diffusion, best_learned)
        and _site_improved(best_diffusion)
        and _source_preserved(best_diffusion, best_statistical, identity_source)
        and claim_status == "diffusion_contribution_established"
        and confounding_status == "pass"
    )


def _is_gate_aware_win(
    *,
    best_diffusion: pd.Series | None,
    baseline: pd.Series | None,
    best_statistical: pd.Series | None,
    identity_source: float | None,
    claim_status: str,
    confounding_status: str,
) -> bool:
    return bool(
        _beats(best_diffusion, baseline)
        and _site_improved(best_diffusion)
        and _source_preserved(best_diffusion, best_statistical, identity_source)
        and claim_status == "diffusion_contribution_established"
        and confounding_status == "pass"
    )


def _mean_bool(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return round(float(values.astype(bool).mean()), 6)


def _parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_strings(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one value")
    return values


if __name__ == "__main__":
    main()
