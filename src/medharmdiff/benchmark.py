from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
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
from .confounding import audit_label_site_confounding
from .io import load_feature_csv
from .latent_denoising import (
    ClinicalPreservingRidgeDenoisingHarmonizer,
    RidgeDenoisingHarmonizer,
)
from .metrics import binary_classification_metrics, coral_distance, mmd_rbf


@dataclass(frozen=True)
class FeatureBenchmarkConfig:
    run_name: str
    feature_path: Path | str
    output_dir: Path | str = Path("results")
    target_center: str | None = None
    setting: str = "zero_shot"
    methods: list[str] = field(
        default_factory=lambda: [
            "identity",
            "source_standardize",
            "center_mean",
            "coral",
            "mmd_mean_alignment",
            "ridge_denoising",
            "ridge_denoising_clinical_preserving",
            "diffusion_placeholder",
        ]
    )
    sample_id_column: str = "sample_id"
    center_column: str = "center_id"
    label_column: str = "label"
    feature_columns: list[str] | str = "auto"
    validation_fraction: float = 0.15
    random_seed: int = 13
    data_source: str = "csv"
    synthetic_params: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureBenchmarkResult:
    run_dir: Path
    metrics_path: Path
    metrics_json_path: Path
    claim_gate_path: Path
    report_path: Path
    run_config_path: Path


@dataclass(frozen=True)
class _MethodSpec:
    factory: Callable[[], Harmonizer] | None
    uses_target_unlabeled: bool = False
    placeholder: bool = False
    runnable: bool = True
    method_family: str = "statistical_baseline"
    is_learned_denoising: bool = False
    is_diffusion: bool = False


_METHODS: dict[str, _MethodSpec] = {
    "no_harmonization": _MethodSpec(lambda: IdentityHarmonizer()),
    "identity": _MethodSpec(lambda: IdentityHarmonizer()),
    "source_standardization": _MethodSpec(lambda: StandardizeBySourceHarmonizer()),
    "source_standardize": _MethodSpec(lambda: StandardizeBySourceHarmonizer()),
    "combat": _MethodSpec(lambda: CenterMeanHarmonizer()),
    "center_mean": _MethodSpec(lambda: CenterMeanHarmonizer()),
    "coral": _MethodSpec(lambda: CoralHarmonizer(), uses_target_unlabeled=True),
    "coral_target_unlabeled": _MethodSpec(
        lambda: CoralHarmonizer(), uses_target_unlabeled=True
    ),
    "mmd": _MethodSpec(lambda: MMDMeanAlignmentHarmonizer(), uses_target_unlabeled=True),
    "mmd_mean_alignment": _MethodSpec(
        lambda: MMDMeanAlignmentHarmonizer(), uses_target_unlabeled=True
    ),
    "mmd_mean_target_unlabeled": _MethodSpec(
        lambda: MMDMeanAlignmentHarmonizer(), uses_target_unlabeled=True
    ),
    "ridge_denoising": _MethodSpec(
        lambda: RidgeDenoisingHarmonizer(),
        method_family="learned_denoising",
        is_learned_denoising=True,
    ),
    "ridge_denoising_clinical_preserving": _MethodSpec(
        lambda: ClinicalPreservingRidgeDenoisingHarmonizer(),
        method_family="learned_denoising",
        is_learned_denoising=True,
    ),
    "adversarial_domain_adaptation": _MethodSpec(None, placeholder=True, runnable=False),
    "vae_harmonization": _MethodSpec(None, placeholder=True, runnable=False),
    "diffusion_harmonization": _MethodSpec(
        lambda: DiffusionPlaceholderHarmonizer(),
        placeholder=True,
        method_family="diffusion_placeholder",
        is_diffusion=True,
    ),
    "diffusion_placeholder": _MethodSpec(
        lambda: DiffusionPlaceholderHarmonizer(),
        placeholder=True,
        method_family="diffusion_placeholder",
        is_diffusion=True,
    ),
    "diffusion_clinical_preserving": _MethodSpec(
        lambda: DiffusionPlaceholderHarmonizer(),
        placeholder=True,
        method_family="diffusion_placeholder",
        is_diffusion=True,
    ),
}


def run_feature_level_benchmark(config: FeatureBenchmarkConfig) -> FeatureBenchmarkResult:
    config = _normalize_config(config)
    _, site, _, _, feature_columns = load_feature_csv(
        config.feature_path,
        sample_id_column=config.sample_id_column,
        center_column=config.center_column,
        label_column=config.label_column,
        feature_columns=config.feature_columns,
    )
    df = pd.read_csv(config.feature_path)
    target_center = config.target_center or sorted(set(site))[-1]
    if target_center not in set(site):
        raise ValueError(f"target_center {target_center!r} is not present in the dataset")
    config = replace(config, target_center=target_center)

    source_df = df[df[config.center_column].astype(str) != target_center].copy()
    target_df = df[df[config.center_column].astype(str) == target_center].copy()
    train_df, val_df = _split_source(source_df, config)

    rows = [
        _evaluate_method(method, config, feature_columns, train_df, val_df, target_df)
        for method in config.methods
    ]
    metrics = pd.DataFrame(rows)

    run_dir = (
        Path(config.output_dir)
        if not config.run_name
        else Path(config.output_dir) / config.run_name
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics_by_method.csv"
    metrics_json_path = run_dir / "metrics_by_method.json"
    claim_gate_path = run_dir / "claim_gate_summary.json"
    report_path = run_dir / "final_report.md"
    run_config_path = run_dir / "run_config.json"

    metrics.to_csv(metrics_path, index=False)
    metrics_json_path.write_text(
        metrics.to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    claim_summary = _build_claim_summary(metrics, df, config)
    claim_gate_path.write_text(
        json.dumps(_json_ready(claim_summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    run_config_path.write_text(
        json.dumps(
            _json_ready(_run_config_dict(config, df, source_df, feature_columns)),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        _render_report(config, metrics, claim_summary, source_df, df, feature_columns),
        encoding="utf-8",
    )

    return FeatureBenchmarkResult(
        run_dir=run_dir,
        metrics_path=metrics_path,
        metrics_json_path=metrics_json_path,
        claim_gate_path=claim_gate_path,
        report_path=report_path,
        run_config_path=run_config_path,
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
        setting=raw.get("setting", "zero_shot"),
        methods=list(raw.get("methods", [])) or FeatureBenchmarkConfig("x", "x").methods,
        sample_id_column=data.get("sample_id_column", "sample_id"),
        center_column=data.get("center_column", "center_id"),
        label_column=data.get("label_column", "label"),
        feature_columns=data.get("feature_columns", "auto"),
        validation_fraction=float(split.get("validation_fraction", 0.15)),
        random_seed=int(raw.get("random_seed", 13)),
        data_source=str(raw.get("data_source", data.get("data_source", "csv"))),
        synthetic_params=dict(raw.get("synthetic_params", {})),
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
        "method_family": "unknown",
        "status": "ok",
        "target_auc": None,
        "target_accuracy": None,
        "source_val_auc": None,
        "source_val_accuracy": None,
        "source_auc": None,
        "source_accuracy": None,
        "site_auc_before": None,
        "site_auc_after": None,
        "site_auc": None,
        "mmd_before": None,
        "mmd_after": None,
        "mmd_rbf": None,
        "coral_before": None,
        "coral_after": None,
        "coral_distance": None,
        "uses_target_unlabeled": False,
        "uses_target_labels": False,
        "is_learned_denoising": False,
        "is_diffusion": False,
        "is_placeholder": False,
    }
    if spec is None:
        return {**base_row, "status": "unknown_method"}
    train_x, train_site, train_y = _arrays(train_df, config, feature_columns)
    val_x, val_site, val_y = _arrays(val_df, config, feature_columns)
    target_x, target_site, target_y = _arrays(target_df, config, feature_columns)
    before = _site_shift_metrics(val_x, target_x, config.random_seed, phase="before")

    if spec.uses_target_unlabeled and config.setting == "zero_shot":
        return {
            **base_row,
            **_method_flags(spec),
            **before,
            "status": "not_applicable_zero_shot",
        }
    if not spec.runnable or spec.factory is None:
        return {
            **base_row,
            **_method_flags(spec),
            **before,
            "status": "placeholder_not_implemented",
            "is_placeholder": True,
        }

    harmonizer = spec.factory()
    fit_kwargs = {}
    uses_target_unlabeled = spec.uses_target_unlabeled and config.setting == "target_unlabeled"
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
    after = _site_shift_metrics(val_h, target_h, config.random_seed, phase="after")

    return {
        **base_row,
        **_method_flags(spec),
        **before,
        **after,
        "status": "placeholder" if spec.placeholder else "ok",
        "target_auc": target_metrics.auc,
        "target_accuracy": target_metrics.accuracy,
        "source_val_auc": source_metrics.auc,
        "source_val_accuracy": source_metrics.accuracy,
        "source_auc": source_metrics.auc,
        "source_accuracy": source_metrics.accuracy,
        "site_auc": after["site_auc_after"],
        "mmd_rbf": after["mmd_after"],
        "coral_distance": after["coral_after"],
        "uses_target_unlabeled": uses_target_unlabeled,
        "uses_target_labels": False,
        "is_placeholder": spec.placeholder,
    }


def _method_flags(spec: _MethodSpec) -> dict[str, object]:
    return {
        "method_family": spec.method_family,
        "is_learned_denoising": spec.is_learned_denoising,
        "is_diffusion": spec.is_diffusion,
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
    diffusion_rows = successful[successful["is_diffusion"].astype(bool)]
    baseline_rows = successful[~successful["is_diffusion"].astype(bool)]
    no_harm_rows = successful[successful["method"].isin(["no_harmonization", "identity"])]

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
        else float(no_harm["source_val_auc"]) - float(diffusion["source_val_auc"]) <= 0.01
    )
    claim_metrics = {
        "diffusion_target_metric": None if diffusion is None else float(diffusion["target_auc"]),
        "best_baseline_target_metric": best_baseline,
        "diffusion_source_metric": (
            None if diffusion is None else float(diffusion["source_val_auc"])
        ),
        "no_harmonization_source_metric": (
            None if no_harm is None else float(no_harm["source_val_auc"])
        ),
        "site_metric_before": None if no_harm is None else float(no_harm["site_auc_before"]),
        "site_metric_after": None if diffusion is None else float(diffusion["site_auc_after"]),
        "clinical_preservation_pass": clinical_preservation_pass,
        "confounding_status": audit_label_site_confounding(
            full_df,
            center_column=config.center_column,
            label_column=config.label_column,
        ).confounding_status,
        "diffusion_is_placeholder": diffusion_placeholder,
    }
    return evaluate_harmonization_claim(claim_metrics).to_dict()


def _render_report(
    config: FeatureBenchmarkConfig,
    metrics: pd.DataFrame,
    claim_summary: dict[str, object],
    source_df: pd.DataFrame,
    full_df: pd.DataFrame,
    feature_columns: list[str],
) -> str:
    successful = metrics[metrics["status"].isin(["ok", "placeholder"])]
    baselines = successful[~successful["is_diffusion"].astype(bool)]
    statistical_baselines = baselines[baselines["method_family"] == "statistical_baseline"]
    learned_denoising = successful[successful["method_family"] == "learned_denoising"]
    strongest = (
        "none"
        if statistical_baselines.empty or not statistical_baselines["target_auc"].notna().any()
        else str(
            statistical_baselines.sort_values("target_auc", ascending=False).iloc[0]["method"]
        )
    )
    learned_summary = _learned_denoising_summary(learned_denoising, statistical_baselines)
    diffusion_rows = successful[successful["is_diffusion"].astype(bool)]
    diffusion_placeholder = (
        False if diffusion_rows.empty else bool(diffusion_rows.iloc[0]["is_placeholder"])
    )
    train_centers = ", ".join(sorted(source_df[config.center_column].astype(str).unique()))
    confounding_audit = audit_label_site_confounding(
        full_df,
        center_column=config.center_column,
        label_column=config.label_column,
    )
    sample_counts = _sample_count_by_center(full_df, config)
    label_prevalence = _label_prevalence_by_center(full_df, config)
    adaptation_note = (
        "Target-center labels are evaluation-only; target-unlabeled adaptation methods are skipped."
        if config.setting == "zero_shot"
        else (
            "Target-center labels are evaluation-only; target features may be used only by "
            "target-unlabeled methods."
        )
    )
    result_columns = [
        "method",
        "method_family",
        "target_auc",
        "source_val_auc",
        "site_auc_before",
        "site_auc_after",
        "mmd_before",
        "mmd_after",
        "coral_before",
        "coral_after",
        "is_learned_denoising",
        "is_diffusion",
        "is_placeholder",
        "status",
    ]
    metrics_csv = metrics[result_columns].to_csv(index=False, lineterminator="\n")
    return "\n".join(
        [
            "# Feature-Level Harmonization Benchmark",
            "",
            "## Setting",
            "",
            f"- Target center: `{config.target_center or 'auto'}`",
            f"- Source centers: `{train_centers}`",
            f"- Setting: `{config.setting}`",
            f"- Data source: `{config.data_source}`",
            f"- Feature columns: `{len(feature_columns)}`",
            f"- Adaptation note: {adaptation_note}",
            "",
            "## Data Audit",
            "",
            f"- Sample count by center: `{sample_counts}`",
            f"- Label prevalence by center: `{label_prevalence}`",
            f"- Confounding status: `{confounding_audit.confounding_status}`",
            f"- Label/site imbalance score: `{confounding_audit.imbalance_score}`",
            "",
            "## Results",
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
            "## Interpretation",
            "",
            f"- Strongest statistical non-diffusion baseline: `{strongest}`",
            f"- Diffusion placeholder: `{diffusion_placeholder}`",
            "- Site reduction and clinical preservation are reported separately above.",
            "",
            "## Learned Denoising Baselines",
            "",
            f"- Best learned denoising method: `{learned_summary['best_method']}`",
            f"- Beats strongest statistical baseline: `{learned_summary['beats_statistical']}`",
            f"- Site metrics improve: `{learned_summary['site_improves']}`",
            f"- Source clinical performance preserved: `{learned_summary['source_preserved']}`",
            "- Warning: ridge denoising is not yet diffusion and is not a diffusion contribution.",
            "",
            "## Safe Conclusion",
            "",
            "Do not claim diffusion works yet if the row is a placeholder or the gate fails.",
            "",
        ]
    )


def _learned_denoising_summary(
    learned: pd.DataFrame, statistical: pd.DataFrame
) -> dict[str, object]:
    if learned.empty or not learned["target_auc"].notna().any():
        return {
            "best_method": "none",
            "beats_statistical": False,
            "site_improves": False,
            "source_preserved": False,
        }
    best_learned = learned.sort_values("target_auc", ascending=False).iloc[0]
    best_stat_auc = (
        None
        if statistical.empty or not statistical["target_auc"].notna().any()
        else float(statistical["target_auc"].max())
    )
    best_stat_source = (
        None
        if statistical.empty or not statistical["source_val_auc"].notna().any()
        else float(statistical["source_val_auc"].max())
    )
    learned_target = float(best_learned["target_auc"])
    learned_source = float(best_learned["source_val_auc"])
    return {
        "best_method": str(best_learned["method"]),
        "beats_statistical": best_stat_auc is not None and learned_target >= best_stat_auc + 0.01,
        "site_improves": float(best_learned["site_auc_after"])
        < float(best_learned["site_auc_before"]),
        "source_preserved": best_stat_source is None or learned_source >= best_stat_source - 0.01,
    }


def _site_shift_metrics(
    source_x: np.ndarray,
    target_x: np.ndarray,
    random_seed: int,
    *,
    phase: str,
) -> dict[str, float]:
    return {
        f"site_auc_{phase}": _site_auc(source_x, target_x, random_seed),
        f"mmd_{phase}": mmd_rbf(source_x, target_x),
        f"coral_{phase}": coral_distance(source_x, target_x),
    }


def _normalize_config(config: FeatureBenchmarkConfig) -> FeatureBenchmarkConfig:
    setting_aliases = {
        "zero_shot_unseen_center": "zero_shot",
        "zero-shot": "zero_shot",
        "unsupervised_target_adaptation": "target_unlabeled",
        "target-unlabeled": "target_unlabeled",
    }
    setting = setting_aliases.get(config.setting, config.setting)
    if setting not in {"zero_shot", "target_unlabeled"}:
        raise ValueError("setting must be zero_shot or target_unlabeled")
    return FeatureBenchmarkConfig(
        run_name=config.run_name,
        feature_path=config.feature_path,
        output_dir=config.output_dir,
        target_center=config.target_center,
        setting=setting,
        methods=config.methods,
        sample_id_column=config.sample_id_column,
        center_column=config.center_column,
        label_column=config.label_column,
        feature_columns=config.feature_columns,
        validation_fraction=config.validation_fraction,
        random_seed=config.random_seed,
        data_source=config.data_source,
        synthetic_params=dict(config.synthetic_params),
    )


def _run_config_dict(
    config: FeatureBenchmarkConfig,
    full_df: pd.DataFrame,
    source_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, object]:
    confounding_audit = audit_label_site_confounding(
        full_df,
        center_column=config.center_column,
        label_column=config.label_column,
    )
    return {
        "run_name": config.run_name,
        "feature_path": str(config.feature_path),
        "output_dir": str(config.output_dir),
        "target_center": config.target_center,
        "source_centers": sorted(source_df[config.center_column].astype(str).unique().tolist()),
        "train_centers": sorted(source_df[config.center_column].astype(str).unique().tolist()),
        "setting": config.setting,
        "methods": config.methods,
        "sample_id_column": config.sample_id_column,
        "center_column": config.center_column,
        "label_column": config.label_column,
        "feature_columns": feature_columns,
        "validation_fraction": config.validation_fraction,
        "random_seed": config.random_seed,
        "data_source": config.data_source,
        "synthetic_params": config.synthetic_params,
        "sample_count_by_center": _sample_count_by_center(full_df, config),
        "label_prevalence_by_center": _label_prevalence_by_center(full_df, config),
        "confounding_audit": confounding_audit.to_dict(),
    }


def _sample_count_by_center(
    df: pd.DataFrame, config: FeatureBenchmarkConfig
) -> dict[str, int]:
    counts = df.groupby(config.center_column).size().sort_index()
    return {str(center): int(count) for center, count in counts.items()}


def _label_prevalence_by_center(
    df: pd.DataFrame, config: FeatureBenchmarkConfig
) -> dict[str, float]:
    prevalence = df.groupby(config.center_column)[config.label_column].mean().sort_index()
    return {str(center): round(float(value), 6) for center, value in prevalence.items()}


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
