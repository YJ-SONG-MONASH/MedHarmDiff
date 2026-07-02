from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ConfoundingAudit:
    label_distribution_by_center: dict[str, float]
    imbalance_score: float
    confounding_status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "label_distribution_by_center": self.label_distribution_by_center,
            "imbalance_score": self.imbalance_score,
            "confounding_status": self.confounding_status,
        }


def audit_label_site_confounding(
    df: pd.DataFrame,
    *,
    center_column: str = "center_id",
    label_column: str = "label",
    warning_threshold: float = 0.4,
    invalidates_threshold: float = 0.75,
) -> ConfoundingAudit:
    """Audit label prevalence imbalance across centers.

    This first-pass audit uses max-min label prevalence spread. It is intentionally
    conservative and should be replaced by richer stratified analysis before any
    clinical claim.
    """

    for column in (center_column, label_column):
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    prevalence = df.groupby(center_column)[label_column].mean().sort_index()
    distribution = {str(center): round(float(value), 6) for center, value in prevalence.items()}
    if prevalence.empty:
        score = 0.0
    else:
        score = round(float(prevalence.max() - prevalence.min()), 6)

    if score >= invalidates_threshold:
        status = "invalidates_claim"
    elif score >= warning_threshold:
        status = "warn_label_site_association"
    else:
        status = "pass"

    return ConfoundingAudit(
        label_distribution_by_center=distribution,
        imbalance_score=score,
        confounding_status=status,
    )
