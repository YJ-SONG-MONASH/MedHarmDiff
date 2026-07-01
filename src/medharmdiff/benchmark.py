from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import (
    CenterMeanHarmonizer,
    CoralHarmonizer,
    DiffusionPlaceholderHarmonizer,
    Harmonizer,
    IdentityHarmonizer,
    MMDMeanAlignmentHarmonizer,
    StandardizeBySourceHarmonizer,
)
from .claim_gate import evaluate_harmonization_claim
from .metrics import binary_classification_metrics, coral_distance, mmd_rbf


@dataclass(frozen=True)
class FeatureBenchmarkConfig:
    run_name: str
    feature_path: Path | str
    output_dir: Path | str = Path("results")
    target_center: str | None = None
    setting: str = "zero_shot_unseen_center"
    methods: list[str] = field(
        default_factory=lambda: [
            "no_harmonization",
            "source_standardization",
            "combat",
            "coral",
            "mmd",
            "adversarial_domain_adaptation",
            "vae_harmonization",
            "diffusion_harmonization",
        ]
    )
    sample_id_column: str = "sample_id"
    center_column: str = "center_id"
    label_column: str = "label"
    feature_columns: list[str] | str = "auto"
    validation_fraction: float = 0.15
    random_seed: int = 13


@dataclass(frozen=True)
class FeatureBenchmarkResult:
    run_dir: Path
    metrics_path: Path
    claim_gate_path: Path
    report_path: Path


@dataclass(frozen=True)
class _MethodSpec:
    factory: Callable[[], Harmonizer] | None
    uses_target_unlabeled: bool = False
    placeholder: bool = False
    runnable: bool = True


_METHODS: dict[str, _MethodSpec] = {
    "no_harmonization": _MethodSpec(lambda: IdentityHarmonizer()),
    "identity": _MethodSpec(lambda: IdentityHarmonizer()),
    "source_standardization": _MethodSpec(lambda: StandardizeBySourceHarmonizer()),
    "combat": _MethodSpec(lambda: CenterMeanHarmonizer(), uses_target_unlabeled=True),
    "coral": _MethodSpec(lambda: CoralHarmonizer(), uses_target_unlabeled=True),
    "mmd": _MethodSpec(lambda: MMDMeanAlignmentHarmonizer(), uses_target_unlabeled=True),
    "adversarial_domain_adaptation": _MethodSpec(None, placeholder=True, runnable=False),
    "vae_harmonization": _MethodSpec(None, placeholder=True, runnable=False),
    "diffusion_harmonization": _MethodSpec(
        lambda: DiffusionPlaceholderHarmonizer(), placeholder=True
    ),
    "diffusion_clinical_preserving": _MethodSpec(
        lambda: DiffusionPlaceholderHarmonizer(), placeholder=True
    ),
}


def run_feature_level_benchmark(config: FeatureBenchmarkConfig) -> FeatureBenchmarkResult:
    df = pd.read_csv(config.feature_path)
    feature_columns = _resolve_feature_columns(df, config)
    target_center = config.target_center or sorted(
        df[config.center_column].astype(str).unique()
    )[-1]
    if target_center not in set(df[config.center_column].astype(str)):
        raise ValueError(f"target_center {target_center!r} is not present in the dataset")

    source_df = df[df[config.center_column].astype(str) != target_center].copy()
    target_df = df[df[config.center_column].astype(str) == target_center].copy()
    train_df, val_df = _split_source(source_df, config)

    rows = [
        _evaluate_method(method, config, feature_columns, train_df, val_df, target_df)
        for method in config.methods
    ]
    metrics = pd.DataFrame(rows)

    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics_by_method.csv"
    claim_gate_path = run_dir / "claim_gate_summary.json"
    report_path = run_dir / "final_report.md"

    metrics.to_csv(metrics_path, index=False)
    claim_summary = _build_claim_summary(metrics, df, config)
    claim_gate_path.write_text(
        json.dumps(_json_ready(claim_summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(_render_report(config, metrics, claim_summary), encoding="utf-8")

    return FeatureBenchmarkResult(
        run_dir=run_dir,
        metrics_path=metrics_path,
        claim_gate_path=claim_gate_path,
        report_path=report_path,
    )


def config_from_yaml(path: Path | str) -> FeatureBenchmarkConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("Reading YAML configs requires PyYAML.") from exc

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    data = raw.get("data", {})
    split = raw.get("split", {})
    return FeatureBenchmarkConfig(
        run_name=raw.get("experiment_name", raw.get("run_name", "feature_level_run")),
        feature_path=data.get("feature_path", raw.get("feature_path")),
        output_dir=raw.get("output_dir", "results"),
        target_center=split.get("target_center", raw.get("target_center")),
        setting=raw.get("setting", "zero_shot_unseen_center"),
        methods=list(raw.get("methods", [])) or FeatureBenchmarkConfig("x", "x").methods,
        sample_id_column=data.get("sample_id_column", "sample_id"),
        center_column=data.get("center_column", "center_id"),
        label_column=data.get("label_column", "label"),
        feature_columns=data.get("feature_columns", "auto"),
        validation_fraction=float(split.get("validation_fraction", 0.15)),
        random_seed=int(raw.get("random_seed", 13)),
    )


def _evaluate_method(
    method: str,
    config: FeatureBenchmarkConfig,
    feature_columns: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> dict[str, object]:
    spec = _METHODS.get(method)
    base_row: dict[str, object] = {
        "method": method,
        "status": "ok",
        "target_auc": None,
        "target_accuracy": None,
        "source_auc": None,
        "source_accuracy": None,
        "site_auc": None,
        "mmd_rbf": None,
        "coral_distance": None,
        "uses_target_unlabeled": False,
        "uses_target_labels": False,
        "is_placeholder": False,
    }
    if spec is None:
        return {**base_row, "status": "unknown_method"}
    if spec.uses_target_unlabeled and config.setting == "zero_shot_unseen_center":
        return {**base_row, "status": "not_applicable_zero_shot"}
    if not spec.runnable or spec.factory is None:
        return {**base_row, "status": "placeholder_not_implemented", "is_placeholder": True}

    train_x, train_site, train_y = _arrays(train_df, config, feature_columns)
    val_x, val_site, val_y = _arrays(val_df, config, feature_columns)
    target_x, target_site, target_y = _arrays(target_df, config, feature_columns)

    harmonizer = spec.factory()
    fit_kwargs = {}
    uses_target_unlabeled = spec.uses_target_unlabeled and (
        config.setting == "unsupervised_target_adaptation"
    )
    if uses_target_unlabeled:
        fit_kwargs = {"target_x": target_x, "target_site": target_site}
    harmonizer.fit(train_x, train_site, train_y, **fit_kwargs)

    train_h = harmonizer.transform(train_x, train_site)
    val_h = harmonizer.transform(val_x, val_site)
    target_h = harmonizer.transform(target_x, target_site)

    source_scores, target_scores = _task_scores(
        train_h, train_y, val_h, target_h, config.random_seed
    )
    source_metrics = binary_classification_metrics(val_y, source_scores)
    target_metrics = binary_classification_metrics(target_y, target_scores)

    return {
        **base_row,
        "status": "placeholder" if spec.placeholder else "ok",
        "target_auc": target_metrics.auc,
        "target_accuracy": target_metrics.accuracy,
        "source_auc": source_metrics.auc,
        "source_accuracy": source_metrics.accuracy,
        "site_auc": _site_auc(val_h, target_h, config.random_seed),
        "mmd_rbf": mmd_rbf(val_h, target_h),
        "coral_distance": coral_distance(val_h, target_h),
        "uses_target_unlabeled": uses_target_unlabeled,
        "uses_target_labels": False,
        "is_placeholder": spec.placeholder,
    }


def _resolve_feature_columns(df: pd.DataFrame, config: FeatureBenchmarkConfig) -> list[str]:
    if config.feature_columns != "auto":
        return list(config.feature_columns)
    reserved = {config.sample_id_column, config.center_column, config.label_column}
    feature_columns = [
        column
        for column in df.columns
        if column not in reserved and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns found")
    return feature_columns


def _split_source(
    source_df: pd.DataFrame, config: FeatureBenchmarkConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = source_df[config.label_column]
    stratify = labels if labels.value_counts().min() >= 2 else None
    try:
        train_df, val_df = train_test_split(
            source_df,
            test_size=config.validation_fraction,
            random_state=config.random_seed,
            stratify=stratify,
        )
    except ValueError:
        val_count = max(1, int(round(len(source_df) * config.validation_fraction)))
        val_df = source_df.sort_values(config.sample_id_column).head(val_count)
        train_df = source_df.drop(val_df.index)
    if train_df.empty or val_df.empty:
        raise ValueError("source split produced an empty train or validation set")
    return train_df.copy(), val_df.copy()


def _arrays(
    df: pd.DataFrame, config: FeatureBenchmarkConfig, feature_columns: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        df[feature_columns].to_numpy(dtype=float),
        df[config.center_column].astype(str).to_numpy(),
        df[config.label_column].to_numpy(dtype=int),
    )


def _task_scores(
    train_x: np.ndarray,
    train_y: np.ndarray,
    source_x: np.ndarray,
    target_x: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(set(train_y.tolist())) < 2:
        constant = float(np.mean(train_y))
        return np.full(source_x.shape[0], constant), np.full(target_x.shape[0], constant)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_seed),
    )
    model.fit(train_x, train_y)
    return model.predict_proba(source_x)[:, 1], model.predict_proba(target_x)[:, 1]


def _site_auc(source_x: np.ndarray, target_x: np.ndarray, random_seed: int) -> float:
    x = np.vstack([source_x, target_x])
    y = np.concatenate(
        [np.zeros(source_x.shape[0], dtype=int), np.ones(target_x.shape[0], dtype=int)]
    )
    if len(set(y.tolist())) < 2:
        return 0.5
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_seed),
    )
    model.fit(x, y)
    score = model.predict_proba(x)[:, 1]
    return round(float(roc_auc_score(y, score)), 4)


def _build_claim_summary(
    metrics: pd.DataFrame, full_df: pd.DataFrame, config: FeatureBenchmarkConfig
) -> dict[str, object]:
    successful = metrics[metrics["status"].isin(["ok", "placeholder"])].copy()
    diffusion_rows = successful[successful["method"].str.startswith("diffusion")]
    baseline_rows = successful[~successful["method"].str.startswith("diffusion")]
    no_harm_rows = successful[successful["method"] == "no_harmonization"]

    diffusion = diffusion_rows.iloc[0] if not diffusion_rows.empty else None
    no_harm = no_harm_rows.iloc[0] if not no_harm_rows.empty else None
    best_baseline = (
        float(baseline_rows["target_auc"].max())
        if not baseline_rows.empty and baseline_rows["target_auc"].notna().any()
        else None
    )
    diffusion_placeholder = bool(diffusion is not None and diffusion.get("is_placeholder", False))
    clinical_preservation_pass = (
        False
        if diffusion is None or no_harm is None
        else float(no_harm["source_auc"]) - float(diffusion["source_auc"]) <= 0.01
    )
    claim_metrics = {
        "diffusion_target_metric": None if diffusion is None else float(diffusion["target_auc"]),
        "best_baseline_target_metric": best_baseline,
        "diffusion_source_metric": None if diffusion is None else float(diffusion["source_auc"]),
        "no_harmonization_source_metric": None if no_harm is None else float(no_harm["source_auc"]),
        "site_metric_before": None if no_harm is None else float(no_harm["site_auc"]),
        "site_metric_after": None if diffusion is None else float(diffusion["site_auc"]),
        "clinical_preservation_pass": clinical_preservation_pass,
        "confounding_status": _label_site_confounding_status(full_df, config),
        "diffusion_is_placeholder": diffusion_placeholder,
    }
    return evaluate_harmonization_claim(claim_metrics).to_dict()


def _label_site_confounding_status(full_df: pd.DataFrame, config: FeatureBenchmarkConfig) -> str:
    prevalence = full_df.groupby(config.center_column)[config.label_column].mean()
    spread = float(prevalence.max() - prevalence.min())
    if spread >= 0.75:
        return "invalidates_claim"
    if spread >= 0.4:
        return "needs_review"
    return "pass"


def _render_report(
    config: FeatureBenchmarkConfig, metrics: pd.DataFrame, claim_summary: dict[str, object]
) -> str:
    metrics_csv = metrics.to_csv(index=False)
    return "\n".join(
        [
            "# Feature-Level Benchmark Report",
            "",
            f"- Run: `{config.run_name}`",
            f"- Setting: `{config.setting}`",
            f"- Target center: `{config.target_center or 'auto'}`",
            "",
            "## Metrics",
            "",
            "```csv",
            metrics_csv.strip(),
            "```",
            "",
            "## Claim Gate",
            "",
            f"- Status: `{claim_summary['status']}`",
            f"- Reason: {claim_summary['reason']}",
            "",
            "Diffusion rows marked as placeholders are not claimable evidence.",
            "",
        ]
    )


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
