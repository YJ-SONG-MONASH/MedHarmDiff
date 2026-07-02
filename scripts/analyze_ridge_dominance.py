from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose ridge/statistical dominance in multiseed outputs."
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.result_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze_ridge_dominance(args.result_dir)
    (output_dir / "ridge_dominance_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "ridge_dominance_report.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    print(f"ridge_dominance_summary: {output_dir / 'ridge_dominance_summary.json'}")


def analyze_ridge_dominance(result_dir: Path) -> dict[str, object]:
    per_run_path = result_dir / "per_run_metrics.csv"
    aggregate_summary_path = result_dir / "aggregate_summary.json"
    if not per_run_path.exists():
        raise FileNotFoundError(f"Missing {per_run_path}")
    if not aggregate_summary_path.exists():
        raise FileNotFoundError(f"Missing {aggregate_summary_path}")

    metrics = pd.read_csv(per_run_path)
    aggregate_summary = json.loads(aggregate_summary_path.read_text(encoding="utf-8"))
    _validate_columns(metrics)

    settings = {
        setting: _setting_summary(setting, setting_df)
        for setting, setting_df in metrics.groupby("stress_setting")
    }
    flags = _diagnostic_flags(settings, aggregate_summary)
    return {
        "input_dir": str(result_dir),
        "run_count": int(aggregate_summary.get("run_count", _run_count(metrics))),
        "source_multiseed_flags": {
            "nonlinear_settings_create_headroom": bool(
                aggregate_summary.get("nonlinear_settings_create_headroom", False)
            ),
            "diffusion_v0_win_rate_vs_ridge": float(
                aggregate_summary.get("diffusion_v0_win_rate_vs_ridge", 0.0)
            ),
            "diffusion_v0_win_rate_vs_statistical": float(
                aggregate_summary.get("diffusion_v0_win_rate_vs_statistical", 0.0)
            ),
        },
        "settings": settings,
        **flags,
        "decision": _decision(flags),
    }


def _setting_summary(setting: str, metrics: pd.DataFrame) -> dict[str, object]:
    run_groups = _run_groups(metrics)
    best_statistical = _best_family(metrics, "statistical_baseline")
    best_ridge = _best_family(metrics, "learned_denoising")
    best_diffusion = _best_family(metrics, "diffusion_v0")
    confounding_counts = _run_value_counts(run_groups, "confounding_status")
    claim_gate_counts = _run_value_counts(run_groups, "claim_gate_status")
    claimable_count = _claimable_diffusion_win_count(run_groups)
    low_confounding = (
        "strong_confounding" not in str(setting)
        and confounding_counts.get("invalidates_claim", 0) == 0
    )
    return {
        "low_confounding": bool(low_confounding),
        "best_statistical_method": best_statistical["method"],
        "best_statistical_target_auc": best_statistical["target_auc"],
        "best_ridge_method": best_ridge["method"],
        "best_ridge_target_auc": best_ridge["target_auc"],
        "best_diffusion_v0_method": best_diffusion["method"],
        "best_diffusion_v0_target_auc": best_diffusion["target_auc"],
        "ridge_minus_diffusion": _subtract(
            best_ridge["target_auc"], best_diffusion["target_auc"]
        ),
        "statistical_minus_diffusion": _subtract(
            best_statistical["target_auc"], best_diffusion["target_auc"]
        ),
        "site_auc_after_comparison": _comparison(
            best_statistical,
            best_ridge,
            best_diffusion,
            "site_auc_after",
        ),
        "source_val_auc_comparison": _comparison(
            best_statistical,
            best_ridge,
            best_diffusion,
            "source_val_auc",
        ),
        "claim_gate_counts": claim_gate_counts,
        "confounding_counts": confounding_counts,
        "claimable_diffusion_win_count": claimable_count,
    }


def _diagnostic_flags(
    settings: dict[str, dict[str, object]],
    aggregate_summary: dict[str, object],
) -> dict[str, bool]:
    low_settings = [data for data in settings.values() if data["low_confounding"]]
    ridge_dominates = bool(
        low_settings
        and all(
            _dominates(
                data["best_ridge_target_auc"],
                data["best_statistical_target_auc"],
                data["best_diffusion_v0_target_auc"],
            )
            for data in low_settings
        )
    )
    statistical_dominates = bool(
        low_settings
        and all(
            _dominates(
                data["best_statistical_target_auc"],
                data["best_ridge_target_auc"],
                data["best_diffusion_v0_target_auc"],
            )
            for data in low_settings
        )
    )
    diffusion_site_gain_clinical_loss = any(
        _diffusion_improves_site_but_loses_clinical(data) for data in low_settings
    )
    diffusion_site_loss_clinical_loss = any(
        _diffusion_loses_site_and_clinical(data) for data in low_settings
    )
    low_confounding_claims = sum(
        int(data["claimable_diffusion_win_count"]) for data in low_settings
    )
    headroom_without_win = bool(
        aggregate_summary.get("nonlinear_settings_create_headroom", False)
        and low_confounding_claims == 0
    )
    synthetic_linear = bool(
        (ridge_dominates or statistical_dominates)
        and low_confounding_claims == 0
    )
    return {
        "ridge_dominates_all_low_confounding_settings": ridge_dominates,
        "statistical_dominates_all_low_confounding_settings": statistical_dominates,
        "diffusion_improves_site_but_loses_clinical": diffusion_site_gain_clinical_loss,
        "diffusion_loses_site_and_clinical": diffusion_site_loss_clinical_loss,
        "headroom_without_diffusion_win": headroom_without_win,
        "synthetic_probably_too_linear_for_diffusion": synthetic_linear,
    }


def _best_family(metrics: pd.DataFrame, family: str) -> dict[str, object]:
    family_rows = metrics[
        (metrics["method_family"] == family) & metrics["target_auc"].notna()
    ]
    if family_rows.empty:
        return {
            "method": None,
            "target_auc": None,
            "site_auc_after": None,
            "source_val_auc": None,
        }
    means = (
        family_rows.groupby("method")[["target_auc", "site_auc_after", "source_val_auc"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    best = means.sort_values("target_auc", ascending=False).iloc[0]
    return {
        "method": str(best["method"]),
        "target_auc": _round_or_none(best["target_auc"]),
        "site_auc_after": _round_or_none(best["site_auc_after"]),
        "source_val_auc": _round_or_none(best["source_val_auc"]),
    }


def _comparison(
    best_statistical: dict[str, object],
    best_ridge: dict[str, object],
    best_diffusion: dict[str, object],
    metric: str,
) -> dict[str, float | None]:
    return {
        "best_statistical": best_statistical[metric],
        "best_ridge": best_ridge[metric],
        "best_diffusion_v0": best_diffusion[metric],
        "diffusion_minus_statistical": _subtract(
            best_diffusion[metric], best_statistical[metric]
        ),
        "diffusion_minus_ridge": _subtract(best_diffusion[metric], best_ridge[metric]),
    }


def _run_groups(metrics: pd.DataFrame) -> list[pd.DataFrame]:
    if "run_id" not in metrics:
        return [metrics]
    return [group for _, group in metrics.groupby("run_id")]


def _run_value_counts(groups: list[pd.DataFrame], column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not groups or column not in groups[0]:
        return counts
    for group in groups:
        values = group[column].dropna().astype(str).unique().tolist()
        value = values[0] if values else "missing"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _claimable_diffusion_win_count(groups: list[pd.DataFrame]) -> int:
    count = 0
    for group in groups:
        claimable = (
            group.get("diffusion_v0_claimable_win", pd.Series(dtype=bool))
            .fillna(False)
            .astype(bool)
            .any()
        )
        confounding_values = (
            group.get("confounding_status", pd.Series(dtype=str))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if claimable and confounding_values == ["pass"]:
            count += 1
    return count


def _diffusion_improves_site_but_loses_clinical(data: dict[str, object]) -> bool:
    diffusion_target = data["best_diffusion_v0_target_auc"]
    if diffusion_target is None:
        return False
    site = data["site_auc_after_comparison"]
    improves_site = (
        site["diffusion_minus_ridge"] is not None
        and site["diffusion_minus_statistical"] is not None
        and float(site["diffusion_minus_ridge"]) < 0
        and float(site["diffusion_minus_statistical"]) < 0
    )
    loses_clinical = (
        _subtract(diffusion_target, data["best_ridge_target_auc"]) is not None
        and _subtract(diffusion_target, data["best_statistical_target_auc"]) is not None
        and float(_subtract(diffusion_target, data["best_ridge_target_auc"])) < 0
        and float(_subtract(diffusion_target, data["best_statistical_target_auc"])) < 0
    )
    return bool(improves_site and loses_clinical)


def _diffusion_loses_site_and_clinical(data: dict[str, object]) -> bool:
    diffusion_target = data["best_diffusion_v0_target_auc"]
    if diffusion_target is None:
        return False
    site = data["site_auc_after_comparison"]
    loses_site = (
        site["diffusion_minus_ridge"] is not None
        and site["diffusion_minus_statistical"] is not None
        and float(site["diffusion_minus_ridge"]) >= 0
        and float(site["diffusion_minus_statistical"]) >= 0
    )
    loses_clinical = (
        _subtract(diffusion_target, data["best_ridge_target_auc"]) is not None
        and _subtract(diffusion_target, data["best_statistical_target_auc"]) is not None
        and float(_subtract(diffusion_target, data["best_ridge_target_auc"])) < 0
        and float(_subtract(diffusion_target, data["best_statistical_target_auc"])) < 0
    )
    return bool(loses_site and loses_clinical)


def _dominates(candidate: object, *baselines: object) -> bool:
    if candidate is None:
        return False
    return all(baseline is None or float(candidate) >= float(baseline) for baseline in baselines)


def _subtract(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def _round_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def _run_count(metrics: pd.DataFrame) -> int:
    if "run_id" not in metrics:
        return int(len(metrics))
    return int(metrics["run_id"].nunique())


def _validate_columns(metrics: pd.DataFrame) -> None:
    required = {
        "stress_setting",
        "method",
        "method_family",
        "target_auc",
        "site_auc_after",
        "source_val_auc",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def _decision(flags: dict[str, bool]) -> str:
    if (
        flags["ridge_dominates_all_low_confounding_settings"]
        and flags["headroom_without_diffusion_win"]
    ):
        return "pause_diffusion_and_prioritize_real_dataset_metadata"
    if flags["diffusion_improves_site_but_loses_clinical"]:
        return "diagnose_over_harmonization_before_model_upgrade"
    return "continue_real_dataset_readiness_before_neural_diffusion"


def _render_report(summary: dict[str, object]) -> str:
    rows = []
    for setting, data in summary["settings"].items():
        rows.append(
            "| "
            + " | ".join(
                [
                    str(setting),
                    str(data["best_statistical_target_auc"]),
                    str(data["best_ridge_target_auc"]),
                    str(data["best_diffusion_v0_target_auc"]),
                    str(data["ridge_minus_diffusion"]),
                    str(data["statistical_minus_diffusion"]),
                    str(data["claimable_diffusion_win_count"]),
                    str(data["confounding_counts"]),
                ]
            )
            + " |"
        )
    return "\n".join(
        [
            "# Ridge Dominance Diagnosis",
            "",
            f"- Input directory: `{summary['input_dir']}`",
            f"- Ridge dominates low-confounding settings: "
            f"`{summary['ridge_dominates_all_low_confounding_settings']}`",
            f"- Statistical baselines dominate low-confounding settings: "
            f"`{summary['statistical_dominates_all_low_confounding_settings']}`",
            f"- Diffusion improves site but loses clinical: "
            f"`{summary['diffusion_improves_site_but_loses_clinical']}`",
            f"- Diffusion loses site and clinical: "
            f"`{summary['diffusion_loses_site_and_clinical']}`",
            f"- Headroom without diffusion win: "
            f"`{summary['headroom_without_diffusion_win']}`",
            f"- Synthetic probably too linear for diffusion: "
            f"`{summary['synthetic_probably_too_linear_for_diffusion']}`",
            f"- Decision: `{summary['decision']}`",
            "",
            "## Per-Setting Summary",
            "",
            "| Setting | Best statistical AUC | Best ridge AUC | Best diffusion AUC | "
            "Ridge - diffusion | Statistical - diffusion | Claimable wins | Confounding |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            *rows,
            "",
            "## Interpretation",
            "",
            "A site-AUC reduction without target clinical AUC improvement is not a "
            "diffusion contribution. Confounded settings are counted as invalidated "
            "rather than claimable wins.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
