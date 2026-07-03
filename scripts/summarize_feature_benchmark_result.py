from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_MARGIN_REQUIRED = 0.01


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a feature benchmark result directory without raw data."
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help="Write result_triage_summary.json and .md under the result directory.",
    )
    args = parser.parse_args()

    summary = summarize_result_dir(args.result_dir)
    if args.write_summary:
        (args.result_dir / "result_triage_summary.json").write_text(
            json.dumps(_json_ready(summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (args.result_dir / "result_triage_summary.md").write_text(
            render_markdown_summary(summary),
            encoding="utf-8",
        )
    print(render_console_summary(summary))


def summarize_result_dir(result_dir: Path) -> dict[str, object]:
    metrics_path = result_dir / "metrics_by_method.csv"
    claim_path = result_dir / "claim_gate_summary.json"
    run_config_path = result_dir / "run_config.json"
    missing = [
        path.name
        for path in (metrics_path, claim_path, run_config_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Result directory is missing required artifact(s): {', '.join(missing)}"
        )

    metrics = pd.read_csv(metrics_path)
    claim_gate = json.loads(claim_path.read_text(encoding="utf-8"))
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    successful = metrics[metrics["status"].isin(["ok", "placeholder"])].copy()

    best_statistical = _best_row(
        successful[successful["method_family"] == "statistical_baseline"]
    )
    best_learned = _best_row(
        successful[successful["method_family"] == "learned_denoising"]
    )
    best_diffusion = _best_row(successful[successful["method_family"] == "diffusion_v0"])
    split_audit = dict(run_config.get("split_audit", {}))
    confounding = dict(run_config.get("confounding_audit", {}))
    diffusion_target = _row_metric(best_diffusion, "target_auc")
    statistical_target = _row_metric(best_statistical, "target_auc")
    learned_target = _row_metric(best_learned, "target_auc")

    return {
        "result_dir": str(result_dir),
        "best_statistical": best_statistical,
        "best_learned_denoising": best_learned,
        "best_latent_diffusion_v0": best_diffusion,
        "target_margin_required": TARGET_MARGIN_REQUIRED,
        "diffusion_beats_statistical": _beats(
            diffusion_target,
            statistical_target,
            margin=TARGET_MARGIN_REQUIRED,
        ),
        "diffusion_beats_learned_denoising": _beats(
            diffusion_target,
            learned_target,
            margin=TARGET_MARGIN_REQUIRED,
        ),
        "claim_gate_status": claim_gate.get("status"),
        "claim_gate_reason": claim_gate.get("reason"),
        "paper_safe_split": bool(run_config.get("paper_safe_split", False)),
        "group_overlap_train_val": split_audit.get("group_overlap_train_val", []),
        "group_overlap_train_test": split_audit.get("group_overlap_train_test", []),
        "group_overlap_val_test": split_audit.get("group_overlap_val_test", []),
        "uses_target_labels_any": _uses_target_labels(metrics),
        "confounding_status": confounding.get("confounding_status"),
        "recommendation": _recommendation(
            best_statistical=best_statistical,
            best_learned=best_learned,
            best_diffusion=best_diffusion,
            claim_gate_status=str(claim_gate.get("status")),
            paper_safe_split=bool(run_config.get("paper_safe_split", False)),
        ),
    }


def render_console_summary(summary: dict[str, object]) -> str:
    best_statistical = dict(summary.get("best_statistical") or {})
    best_learned = dict(summary.get("best_learned_denoising") or {})
    best_diffusion = dict(summary.get("best_latent_diffusion_v0") or {})
    lines = [
        f"result_dir: {summary['result_dir']}",
        f"paper_safe_split: {summary['paper_safe_split']}",
        f"uses_target_labels_any: {summary['uses_target_labels_any']}",
        f"claim_gate_status: {summary['claim_gate_status']}",
        f"best_statistical: {best_statistical.get('method', 'none')}",
        f"best_learned_denoising: {best_learned.get('method', 'none')}",
        f"best_latent_diffusion_v0: {best_diffusion.get('method', 'none')}",
        f"target_margin_required: {summary['target_margin_required']}",
        f"diffusion_beats_statistical: {summary['diffusion_beats_statistical']}",
        "diffusion_beats_learned_denoising: "
        f"{summary['diffusion_beats_learned_denoising']}",
        f"confounding_status: {summary['confounding_status']}",
        f"recommendation: {summary['recommendation']}",
    ]
    return "\n".join(lines)


def render_markdown_summary(summary: dict[str, object]) -> str:
    best_statistical = dict(summary.get("best_statistical") or {})
    best_learned = dict(summary.get("best_learned_denoising") or {})
    best_diffusion = dict(summary.get("best_latent_diffusion_v0") or {})
    return "\n".join(
        [
            "# Feature Benchmark Result Triage",
            "",
            f"- Result directory: `{summary['result_dir']}`",
            f"- Paper-safe split: `{summary['paper_safe_split']}`",
            f"- Uses target labels: `{summary['uses_target_labels_any']}`",
            f"- Claim gate status: `{summary['claim_gate_status']}`",
            f"- Best statistical baseline: `{best_statistical.get('method', 'none')}`",
            f"- Best learned denoising baseline: `{best_learned.get('method', 'none')}`",
            f"- Best latent diffusion v0 method: `{best_diffusion.get('method', 'none')}`",
            f"- Target margin required: `{summary['target_margin_required']}`",
            f"- Diffusion beats statistical: `{summary['diffusion_beats_statistical']}`",
            "- Diffusion beats learned denoising: "
            f"`{summary['diffusion_beats_learned_denoising']}`",
            f"- Confounding status: `{summary['confounding_status']}`",
            f"- Recommendation: {summary['recommendation']}",
            "",
            "This summary reads benchmark artifacts only, not raw metadata or features.",
            "",
        ]
    )


def _best_row(rows: pd.DataFrame) -> dict[str, object] | None:
    if rows.empty or not rows["target_auc"].notna().any():
        return None
    row = rows.sort_values("target_auc", ascending=False).iloc[0]
    columns = [
        "method",
        "method_family",
        "status",
        "target_auc",
        "source_val_auc",
        "site_auc_before",
        "site_auc_after",
    ]
    return {column: _json_ready(row.get(column)) for column in columns if column in row}


def _row_metric(row: dict[str, object] | None, metric: str) -> float | None:
    if not row or row.get(metric) is None:
        return None
    return float(row[metric])


def _beats(left: float | None, right: float | None, *, margin: float) -> bool:
    return bool(left is not None and right is not None and left >= right + margin)


def _uses_target_labels(metrics: pd.DataFrame) -> bool:
    if "uses_target_labels" not in metrics.columns:
        return True
    return bool(metrics["uses_target_labels"].fillna(False).astype(bool).any())


def _recommendation(
    *,
    best_statistical: dict[str, object] | None,
    best_learned: dict[str, object] | None,
    best_diffusion: dict[str, object] | None,
    claim_gate_status: str,
    paper_safe_split: bool,
) -> str:
    diffusion_target = _row_metric(best_diffusion, "target_auc")
    statistical_target = _row_metric(best_statistical, "target_auc")
    learned_target = _row_metric(best_learned, "target_auc")
    if not paper_safe_split:
        return "diagnostic only: split is not paper-safe"
    if diffusion_target is None:
        return "pause diffusion: no runnable latent diffusion v0 row"
    if (
        statistical_target is not None
        and diffusion_target < statistical_target + TARGET_MARGIN_REQUIRED
        or learned_target is not None
        and diffusion_target < learned_target + TARGET_MARGIN_REQUIRED
    ):
        return "pause diffusion: ridge/statistical baselines still dominate"
    if claim_gate_status != "diffusion_contribution_established":
        return "inspect source drop, site metrics, and confounding before modeling"
    return "repeat across target centers and seeds before any claim"


def _json_ready(value: Any) -> Any:
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
