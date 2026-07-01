from __future__ import annotations

from medharmdiff.claim_gate import evaluate_harmonization_claim


if __name__ == "__main__":
    example = {
        "diffusion_target_metric": 0.78,
        "best_baseline_target_metric": 0.75,
        "diffusion_source_metric": 0.82,
        "no_harmonization_source_metric": 0.83,
        "site_metric_before": 0.90,
        "site_metric_after": 0.62,
        "clinical_preservation_pass": True,
        "confounding_status": "pass",
    }
    print(evaluate_harmonization_claim(example).to_dict())
