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
from medharmdiff.synthetic import (
    generate_synthetic_feature_dataset,
    generate_synthetic_multicenter_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a feature-level MedHarmDiff benchmark.")
    parser.add_argument("--config", type=Path, help="YAML config path.", default=None)
    parser.add_argument("--csv", type=Path, help="Input feature CSV path.")
    parser.add_argument("--input-csv", type=Path, help="Input feature CSV path.")
    parser.add_argument("--feature-path", type=Path, help="Input feature CSV path.")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--target-center")
    parser.add_argument("--synthetic", action="store_true", help="Generate a synthetic input CSV.")
    parser.add_argument("--n-centers", type=int, default=3)
    parser.add_argument("--samples-per-center", type=int, default=80)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--clinical-signal-strength", type=float, default=1.0)
    parser.add_argument("--site-shift-strength", type=float, default=1.0)
    parser.add_argument("--site-label-confounding-strength", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--noise-strength", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument(
        "--setting",
        choices=[
            "zero_shot",
            "target_unlabeled",
            "zero_shot_unseen_center",
            "unsupervised_target_adaptation",
        ],
        default="zero_shot",
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
        output_dir = args.output_dir
        feature_path = args.csv or args.input_csv or args.feature_path
        run_name = args.run_name
        data_source = "csv"
        synthetic_params: dict[str, object] = {}
        if args.synthetic:
            run_dir = output_dir if not run_name else output_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            feature_path = run_dir / "synthetic_features.csv"
            noise_strength = (
                args.noise_strength if args.noise_strength is not None else args.noise_std
            )
            synthetic_params = {
                "n_centers": args.n_centers,
                "samples_per_center": args.samples_per_center,
                "n_features": args.n_features,
                "clinical_signal_strength": args.clinical_signal_strength,
                "site_shift_strength": args.site_shift_strength,
                "site_label_confounding_strength": args.site_label_confounding_strength,
                "noise_strength": noise_strength,
                "random_seed": args.random_seed,
            }
            if (args.target_center or "").startswith("center_"):
                synthetic = generate_synthetic_multicenter_features(
                    n_centers=args.n_centers,
                    samples_per_center=args.samples_per_center,
                    n_features=args.n_features,
                    clinical_signal_strength=args.clinical_signal_strength,
                    site_shift_strength=args.site_shift_strength,
                    label_site_confounding=args.site_label_confounding_strength,
                    noise_strength=noise_strength,
                    seed=args.random_seed,
                )
            else:
                synthetic = generate_synthetic_feature_dataset(
                    n_centers=args.n_centers,
                    samples_per_center=args.samples_per_center,
                    n_features=args.n_features,
                    clinical_signal_strength=args.clinical_signal_strength,
                    site_shift_strength=args.site_shift_strength,
                    site_label_confounding_strength=args.site_label_confounding_strength,
                    noise_strength=noise_strength,
                    random_seed=args.random_seed,
                )
            synthetic.to_csv(feature_path, index=False)
            data_source = "synthetic"
        if feature_path is None:
            parser.error("--input-csv or --feature-path is required unless --synthetic is set")
        methods = (
            None if args.methods is None else [item.strip() for item in args.methods.split(",")]
        )
        config = FeatureBenchmarkConfig(
            run_name=run_name,
            feature_path=feature_path,
            output_dir=output_dir,
            target_center=args.target_center,
            setting=args.setting,
            methods=methods or FeatureBenchmarkConfig("default", feature_path).methods,
            random_seed=args.random_seed,
            data_source=data_source,
            synthetic_params=synthetic_params,
        )

    result = run_feature_level_benchmark(config)
    print(f"metrics: {result.metrics_path}")
    print(f"claim_gate: {result.claim_gate_path}")
    print(f"report: {result.report_path}")


if __name__ == "__main__":
    main()
