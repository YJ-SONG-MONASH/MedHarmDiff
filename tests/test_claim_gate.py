from medharmdiff.claim_gate import evaluate_harmonization_claim


def test_claim_gate_accepts_safe_diffusion_gain() -> None:
    decision = evaluate_harmonization_claim(
        {
            "diffusion_target_metric": 0.78,
            "best_baseline_target_metric": 0.75,
            "diffusion_source_metric": 0.82,
            "no_harmonization_source_metric": 0.83,
            "site_metric_before": 0.90,
            "site_metric_after": 0.60,
            "clinical_preservation_pass": True,
            "confounding_status": "pass",
        }
    )

    assert decision.status == "diffusion_contribution_established"


def test_claim_gate_rejects_when_baseline_not_beaten() -> None:
    decision = evaluate_harmonization_claim(
        {
            "diffusion_target_metric": 0.751,
            "best_baseline_target_metric": 0.75,
            "diffusion_source_metric": 0.83,
            "no_harmonization_source_metric": 0.83,
            "site_metric_before": 0.90,
            "site_metric_after": 0.60,
            "clinical_preservation_pass": True,
            "confounding_status": "pass",
        }
    )

    assert decision.status == "baseline_not_beaten"


def test_claim_gate_rejects_clinical_signal_loss() -> None:
    decision = evaluate_harmonization_claim(
        {
            "diffusion_target_metric": 0.80,
            "best_baseline_target_metric": 0.75,
            "diffusion_source_metric": 0.70,
            "no_harmonization_source_metric": 0.83,
            "site_metric_before": 0.90,
            "site_metric_after": 0.60,
            "clinical_preservation_pass": True,
            "confounding_status": "pass",
        }
    )

    assert decision.status == "site_removed_but_clinical_signal_lost"


def test_claim_gate_rejects_confounding() -> None:
    decision = evaluate_harmonization_claim(
        {
            "diffusion_target_metric": 0.80,
            "best_baseline_target_metric": 0.75,
            "diffusion_source_metric": 0.83,
            "no_harmonization_source_metric": 0.83,
            "site_metric_before": 0.90,
            "site_metric_after": 0.60,
            "clinical_preservation_pass": True,
            "confounding_status": "invalidates_claim",
        }
    )

    assert decision.status == "confounded_result_needs_review"


def test_claim_gate_rejects_placeholder_diffusion_as_insufficient_data() -> None:
    decision = evaluate_harmonization_claim(
        {
            "diffusion_target_metric": 0.90,
            "best_baseline_target_metric": 0.75,
            "diffusion_source_metric": 0.85,
            "no_harmonization_source_metric": 0.85,
            "site_metric_before": 0.90,
            "site_metric_after": 0.50,
            "clinical_preservation_pass": True,
            "confounding_status": "pass",
            "diffusion_is_placeholder": True,
        }
    )

    assert decision.status == "insufficient_data"
