from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HarmonizationClaimCriteria:
    min_target_gain: float = 0.01
    max_source_drop: float = 0.01
    require_site_reduction: bool = True
    require_clinical_preservation: bool = True


@dataclass(frozen=True)
class HarmonizationClaimDecision:
    status: str
    reason: str
    metrics: dict[str, Any]
    criteria: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "metrics": self.metrics,
            "criteria": self.criteria,
        }


def evaluate_harmonization_claim(
    metrics: dict[str, Any],
    *,
    criteria: HarmonizationClaimCriteria | None = None,
) -> HarmonizationClaimDecision:
    """Decide whether a diffusion harmonization result is claimable.

    Expected metric keys:

    - diffusion_target_metric
    - best_baseline_target_metric
    - diffusion_source_metric
    - no_harmonization_source_metric
    - site_metric_before
    - site_metric_after
    - clinical_preservation_pass
    - confounding_status
    """

    gate = criteria or HarmonizationClaimCriteria()
    if metrics.get("diffusion_is_placeholder"):
        return _decision(
            "insufficient_data",
            "Diffusion result is a placeholder and cannot establish a contribution.",
            metrics,
            gate,
        )
    required_keys = (
        "diffusion_target_metric",
        "best_baseline_target_metric",
        "diffusion_source_metric",
        "no_harmonization_source_metric",
        "site_metric_before",
        "site_metric_after",
    )
    if any(metrics.get(key) is None for key in required_keys):
        return _decision(
            "insufficient_data",
            "Required claim-gate metrics are missing.",
            metrics,
            gate,
        )
    target_gain = _float(metrics.get("diffusion_target_metric")) - _float(
        metrics.get("best_baseline_target_metric")
    )
    source_drop = _float(metrics.get("no_harmonization_source_metric")) - _float(
        metrics.get("diffusion_source_metric")
    )
    site_reduction = _float(metrics.get("site_metric_before")) - _float(
        metrics.get("site_metric_after")
    )
    normalized = {
        **metrics,
        "target_gain_vs_best_baseline": round(target_gain, 4),
        "source_drop_vs_no_harmonization": round(source_drop, 4),
        "site_metric_reduction": round(site_reduction, 4),
    }
    if metrics.get("confounding_status") == "invalidates_claim":
        return _decision(
            "confounded_result_needs_review",
            "Label-site confounding invalidates the claim.",
            normalized,
            gate,
        )
    if round(target_gain, 8) < gate.min_target_gain:
        return _decision(
            "baseline_not_beaten",
            "Diffusion does not beat the strongest non-diffusion baseline by the required margin.",
            normalized,
            gate,
        )
    if round(source_drop, 8) > gate.max_source_drop:
        return _decision(
            "site_removed_but_clinical_signal_lost",
            "Source-center clinical signal drops beyond tolerance.",
            normalized,
            gate,
        )
    if gate.require_site_reduction and site_reduction <= 0:
        return _decision(
            "candidate_signal_only_not_established",
            "Site predictability or distance does not decrease.",
            normalized,
            gate,
        )
    if gate.require_clinical_preservation and not bool(
        metrics.get("clinical_preservation_pass", False)
    ):
        return _decision(
            "site_removed_but_clinical_signal_lost",
            "Clinical preservation check failed.",
            normalized,
            gate,
        )
    return _decision(
        "diffusion_contribution_established",
        "Diffusion beats baselines while reducing site shift and preserving clinical signal.",
        normalized,
        gate,
    )


def _decision(
    status: str,
    reason: str,
    metrics: dict[str, Any],
    gate: HarmonizationClaimCriteria,
) -> HarmonizationClaimDecision:
    return HarmonizationClaimDecision(
        status=status,
        reason=reason,
        metrics=metrics,
        criteria=asdict(gate),
    )


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
