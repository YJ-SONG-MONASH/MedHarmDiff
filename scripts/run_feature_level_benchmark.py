from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medharmdiff.benchmark import (
    FeatureBenchmarkConfig,
    config_from_yaml,
    run_feature_level_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a feature-level MedHarmDiff benchmark.")
    parser.add_argument("--config", type=Path, help="YAML config path.", default=None)
    parser.add_argument("--feature-path", type=Path, help="Input feature CSV path.")
    parser.add_argument("--run-name", default="feature_level_run")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--target-center")
    parser.add_argument(
        "--setting",
        choices=["zero_shot_unseen_center", "unsupervised_target_adaptation"],
        default="zero_shot_unseen_center",
    )
    parser.add_argument(
        "--methods",
        help="Comma-separated methods. Defaults to the scaffold baseline set.",
        default=None,
    )
    args = parser.parse_args()

    if args.config is not None:
        config = config_from_yaml(args.config)
    else:
        if args.feature_path is None:
            parser.error("--feature-path is required when --config is not provided")
        methods = (
            None if args.methods is None else [item.strip() for item in args.methods.split(",")]
        )
        config = FeatureBenchmarkConfig(
            run_name=args.run_name,
            feature_path=args.feature_path,
            output_dir=args.output_dir,
            target_center=args.target_center,
            setting=args.setting,
            methods=methods or FeatureBenchmarkConfig("default", args.feature_path).methods,
        )

    result = run_feature_level_benchmark(config)
    print(f"metrics: {result.metrics_path}")
    print(f"claim_gate: {result.claim_gate_path}")
    print(f"report: {result.report_path}")


if __name__ == "__main__":
    main()
