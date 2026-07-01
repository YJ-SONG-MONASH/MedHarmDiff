# Diffusion Harmonization Claim Gate

Use this gate before making any paper claim.

## Inputs

For each experiment run:

```text
source clinical metric
target clinical metric
site classifier metric
harmonization distance metrics
clinical preservation metrics
baseline comparison table
label-site confounding audit
```

## Gate Logic

Diffusion contribution is established if:

```text
diffusion_target_metric > strongest_non_diffusion_target_metric + margin
site_predictability_after < site_predictability_before
source_metric_drop <= tolerance
clinical_preservation_pass == true
label_site_confounding_status != invalidates_claim
```

Default margin:

```text
classification AUC: +0.01 minimum, +0.02 preferred
segmentation Dice: +0.01 minimum, +0.02 preferred
regression: task-specific clinically meaningful delta
```

## Status Labels

```text
diffusion_contribution_established
candidate_signal_only_not_established
site_removed_but_clinical_signal_lost
baseline_not_beaten
confounded_result_needs_review
insufficient_data
```

## Safe Conclusion Templates

### Success

> Diffusion harmonization improves held-out-center clinical performance over the strongest non-diffusion harmonization baseline while reducing site predictability and preserving clinical signal.

### Partial

> Diffusion reduces site shift but does not yet outperform strong non-diffusion baselines on the clinical task.

### Failure

> Diffusion harmonization is not currently justified as a main contribution; results suggest prioritizing baseline calibration, confounding control, and clinical preservation.
