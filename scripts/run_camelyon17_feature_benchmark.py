from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a group-safe Camelyon17/WILDS feature-level benchmark."
    )
    parser.add_argument("--feature-csv", type=Path, required=True)
    parser.add_argument("--target-center", required=True)
    parser.add_argument("--group-column", default="patient_or_group_id")
    parser.add_argument("--split-group-column", default=None)
    parser.add_argument("--source-train-split-values", default=None)
    parser.add_argument("--source-val-split-values", default=None)
    parser.add_argument("--target-test-split-values", default=None)
    parser.add_argument(
        "--setting",
        choices=["zero_shot", "target_unlabeled"],
        default="zero_shot",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/camelyon17_smoke"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--center-column", default="center_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument(
        "--allow-unsafe-split",
        action="store_true",
        help="Allow row-level fallback when group metadata is missing.",
    )
    args = parser.parse_args()

    contract = validate_real_feature_contract(
        args.feature_csv,
        sample_id_column=args.sample_id_column,
        center_column=args.center_column,
        label_column=args.label_column,
    )
    methods = _comma_list(args.methods) or DEFAULT_METHODS
    config = FeatureBenchmarkConfig(
        run_name=args.run_name,
        feature_path=args.feature_csv,
        output_dir=args.output_dir,
        target_center=args.target_center,
        setting=args.setting,
        methods=methods,
        sample_id_column=args.sample_id_column,
        center_column=args.center_column,
        label_column=args.label_column,
        random_seed=args.random_seed,
        data_source="camelyon17_feature_csv",
        group_column=args.group_column,
        split_group_column=args.split_group_column,
        source_train_split_values=_comma_list(args.source_train_split_values),
        source_val_split_values=_comma_list(args.source_val_split_values),
        target_test_split_values=_comma_list(args.target_test_split_values),
        require_group_safe_split=not args.allow_unsafe_split,
    )

    result = run_feature_level_benchmark(config)
    print(
        "contract: "
        f"{contract['sample_count']} rows, "
        f"{contract['center_count']} centers, "
        f"{contract['feature_count']} features, "
        f"{len(contract['warnings'])} warning(s)"
    )
    print(f"metrics: {result.metrics_path}")
    print(f"claim_gate: {result.claim_gate_path}")
    print(f"report: {result.report_path}")
    print(f"run_config: {result.run_config_path}")


def _comma_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",")]
    resolved = [item for item in items if item]
    return resolved or None


if __name__ == "__main__":
    main()
