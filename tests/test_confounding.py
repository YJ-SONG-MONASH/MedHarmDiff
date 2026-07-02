import pandas as pd

from medharmdiff.confounding import audit_label_site_confounding


def test_confounding_audit_passes_when_center_prevalence_is_similar() -> None:
    df = pd.DataFrame(
        {
            "center_id": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "label": [0, 1, 0, 1] * 3,
        }
    )

    audit = audit_label_site_confounding(df)

    assert audit.confounding_status == "pass"
    assert audit.imbalance_score == 0.0
    assert audit.label_distribution_by_center["A"] == 0.5


def test_confounding_audit_warns_for_moderate_imbalance() -> None:
    df = pd.DataFrame(
        {
            "center_id": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "label": [0] * 5 + [1] * 5 + [0] * 8 + [1] * 2 + [0] * 1 + [1] * 9,
        }
    )

    audit = audit_label_site_confounding(df)

    assert audit.confounding_status == "warn_label_site_association"
    assert 0.4 <= audit.imbalance_score < 0.75


def test_confounding_audit_invalidates_for_extreme_imbalance() -> None:
    df = pd.DataFrame(
        {
            "center_id": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "label": [0] * 10 + [1] * 10 + [1] * 9 + [0],
        }
    )

    audit = audit_label_site_confounding(df)

    assert audit.confounding_status == "invalidates_claim"
    assert audit.imbalance_score >= 0.75
