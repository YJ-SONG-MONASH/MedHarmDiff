from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _thread_env_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_var, "1")

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medharmdiff.benchmark import FeatureBenchmarkConfig, run_feature_level_benchmark
from medharmdiff.io import validate_real_feature_contract


DEFAULT_METHODS = [
    "identity",
    "source_standardize",
    "center_mean",
    "ridge_denoising",
    "latent_diffusion_v0",
    "diffusion_placeholder",
]
TARGET_MARGIN_REQUIRED = 0.01
SOURCE_DROP_TOLERANCE = 0.01


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Camelyon17 benchmark across prepared embedding family CSVs."
    )
    parser.add_argument(
        "--feature-csv",
        action="append",
        required=True,
        help="Embedding family feature CSV as name=path. Repeat for multiple families.",
    )
    parser.add_argument("--target-center", required=True)
    parser.add_argument("--group-column", default="patient_or_group_id")
    parser.add_argument("--split-group-column", default=None)
    parser.add_argument("--source-train-split-values", default=None)
    parser.add_argument("--source-val-split-values", default=None)
    parser.add_argument("--target-test-split-values", default=None)
    parser.add_argument("--setting", choices=["zero_shot", "target_unlabeled"], default="zero_shot")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for family, feature_csv in _parse_feature_csv_args(args.feature_csv):
        row = run_embedding_family(
            embedding_family=family,
            feature_csv=feature_csv,
            output_dir=args.output_dir,
            target_center=args.target_center,
            group_column=args.group_column,
            split_group_column=args.split_group_column,
            source_train_split_values=_comma_list(args.source_train_split_values),
            source_val_split_values=_comma_list(args.source_val_split_values),
            target_test_split_values=_comma_list(args.target_test_split_values),
            setting=args.setting,
            methods=_comma_list(args.methods) or DEFAULT_METHODS,
            random_seed=args.random_seed,
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_csv = args.output_dir / "embedding_family_summary.csv"
    summary_json = args.output_dir / "embedding_family_summary.json"
    report_path = args.output_dir / "embedding_family_report.md"
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(_json_ready(rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(render_report(rows), encoding="utf-8")
    print(f"embedding_family_summary: {summary_csv}")
    print(f"embedding_family_report: {report_path}")


def run_embedding_family(
    *,
    embedding_family: str,
    feature_csv: Path,
    output_dir: Path,
    target_center: str,
    group_column: str,
    split_group_column: str | None,
    source_train_split_values: list[str] | None,
    source_val_split_values: list[str] | None,
    target_test_split_values: list[str] | None,
    setting: str,
    methods: list[str],
    random_seed: int,
) -> dict[str, object]:
    contract = validate_real_feature_contract(feature_csv)
    result = run_feature_level_benchmark(
        FeatureBenchmarkConfig(
            run_name=_safe_name(embedding_family),
            feature_path=feature_csv,
            output_dir=output_dir,
            target_center=target_center,
            setting=setting,
            methods=methods,
            random_seed=random_seed,
            data_source=f"camelyon17_{embedding_family}",
            group_column=group_column,
            split_group_column=split_group_column,
            source_train_split_values=source_train_split_values,
            source_val_split_values=source_val_split_values,
            target_test_split_values=target_test_split_values,
            require_group_safe_split=True,
        )
    )
    metrics = pd.read_csv(result.metrics_path)
    run_config = json.loads(result.run_config_path.read_text(encoding="utf-8"))
    claim_gate = json.loads(result.claim_gate_path.read_text(encoding="utf-8"))
    return summarize_family_result(
        embedding_family=embedding_family,
        feature_csv=feature_csv,
        run_dir=result.run_dir,
        metrics=metrics,
        run_config=run_config,
        claim_gate=claim_gate,
        feature_count=int(contract["feature_count"]),
        sample_count=int(contract["sample_count"]),
    )


def summarize_family_result(
    *,
    embedding_family: str,
    feature_csv: Path,
    run_dir: Path,
    metrics: pd.DataFrame,
    run_config: dict[str, object],
    claim_gate: dict[str, object],
    feature_count: int,
    sample_count: int,
) -> dict[str, object]:
    successful = metrics[metrics["status"].isin(["ok", "placeholder"])].copy()
    best_statistical = _best_row(
        successful[successful["method_family"] == "statistical_baseline"]
    )
    best_ridge = _best_row(successful[successful["method_family"] == "learned_denoising"])
    latent = _method_row(successful, "latent_diffusion_v0")
    identity = _method_row(successful, "identity")
    non_diffusion = successful[~successful["is_diffusion"].astype(bool)]
    best_non_diffusion_target = _max_metric(non_diffusion, "target_auc")
    latent_target = _metric(latent, "target_auc")
    latent_source = _metric(latent, "source_val_auc")
    identity_source = _metric(identity, "source_val_auc")
    gain_vs_best_non_diffusion = _difference(latent_target, best_non_diffusion_target)
    source_drop = _difference(identity_source, latent_source)
    split_audit = dict(run_config.get("split_audit", {}))
    paper_safe_split = bool(run_config.get("paper_safe_split", False))
    uses_target_labels = _uses_target_labels(metrics)
    claim_gate_status = str(claim_gate.get("status"))

    return {
        "embedding_family": embedding_family,
        "feature_csv": str(feature_csv),
        "run_dir": str(run_dir),
        "sample_count": sample_count,
        "feature_count": feature_count,
        "best_statistical_method": _row_value(best_statistical, "method"),
        "best_statistical_target_auc": _metric(best_statistical, "target_auc"),
        "best_statistical_source_val_auc": _metric(best_statistical, "source_val_auc"),
        "best_ridge_method": _row_value(best_ridge, "method"),
        "best_ridge_target_auc": _metric(best_ridge, "target_auc"),
        "best_ridge_source_val_auc": _metric(best_ridge, "source_val_auc"),
        "latent_diffusion_v0_target_auc": latent_target,
        "latent_diffusion_v0_source_val_auc": latent_source,
        "claim_gate_status": claim_gate_status,
        "paper_safe_split": paper_safe_split,
        "uses_target_labels": uses_target_labels,
        "group_overlap_train_val": split_audit.get("group_overlap_train_val", []),
        "group_overlap_train_test": split_audit.get("group_overlap_train_test", []),
        "group_overlap_val_test": split_audit.get("group_overlap_val_test", []),
        "source_drop_vs_no_harmonization": source_drop,
        "gain_vs_best_non_diffusion": gain_vs_best_non_diffusion,
        "diffusion_source_drop_large": bool(
            source_drop is not None and source_drop > SOURCE_DROP_TOLERANCE
        ),
        "diffusion_margin_insufficient": bool(
            gain_vs_best_non_diffusion is None
            or gain_vs_best_non_diffusion < TARGET_MARGIN_REQUIRED
        ),
        "representation_needs_upgrade": bool(
            embedding_family == "color_stats_v0"
            and claim_gate_status != "diffusion_contribution_established"
        ),
    }


def render_report(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Camelyon17 Embedding Family Benchmark",
        "",
        "This report compares prepared feature CSVs without reading raw images.",
        "",
        "| embedding_family | paper_safe | best_statistical | best_ridge | "
        "latent_target_auc | latent_source_auc | source drop flag | margin flag | claim_gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {embedding_family} | {paper_safe_split} | {best_statistical_method} | "
            "{best_ridge_method} | {latent_diffusion_v0_target_auc} | "
            "{latent_diffusion_v0_source_val_auc} | {diffusion_source_drop_large} | "
            "{diffusion_margin_insufficient} | {claim_gate_status} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"Source drop flag threshold: `{SOURCE_DROP_TOLERANCE}`.",
            f"Target AUC margin threshold: `{TARGET_MARGIN_REQUIRED}`.",
            "",
            "Do not continue diffusion modeling unless a paper-safe stronger embedding "
            "beats the best non-diffusion baseline by the required margin without a "
            "large source validation drop.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_feature_csv_args(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for value in values:
        if "=" not in value:
            raise ValueError("--feature-csv must use name=path")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("feature CSV family name must not be blank")
        parsed.append((name, Path(path)))
    return parsed


def _best_row(rows: pd.DataFrame) -> pd.Series | None:
    if rows.empty or not rows["target_auc"].notna().any():
        return None
    return rows.sort_values("target_auc", ascending=False).iloc[0]


def _method_row(rows: pd.DataFrame, method: str) -> pd.Series | None:
    matched = rows[rows["method"] == method]
    if matched.empty:
        return None
    return matched.iloc[0]


def _max_metric(rows: pd.DataFrame, metric: str) -> float | None:
    if rows.empty or metric not in rows.columns or not rows[metric].notna().any():
        return None
    return round(float(rows[metric].max()), 6)


def _metric(row: pd.Series | None, metric: str) -> float | None:
    if row is None or metric not in row or pd.isna(row[metric]):
        return None
    return round(float(row[metric]), 6)


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 6)


def _row_value(row: pd.Series | None, key: str) -> object | None:
    if row is None or key not in row or pd.isna(row[key]):
        return None
    return row[key]


def _uses_target_labels(metrics: pd.DataFrame) -> bool:
    if "uses_target_labels" not in metrics.columns:
        return True
    return bool(metrics["uses_target_labels"].fillna(False).astype(bool).any())


def _comma_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    resolved = [item for item in items if item]
    return resolved or None


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe or "embedding_family"


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
