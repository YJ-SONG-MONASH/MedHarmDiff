# Codex Next Step Prompt: Multi-Seed Latent Diffusion v0 Evaluation

You are taking over `MedHarmDiff` on branch `codex/mvp-benchmark` after the first latent diffusion v0 implementation.

Current expected code state:

```text
src/medharmdiff/latent_diffusion.py exists
latent_diffusion_v0 is registered in the benchmark
latent_diffusion_clinical_preserving_v0 is registered in the benchmark
scripts/run_synthetic_stress.py includes diffusion_v0 summaries
tests/test_latent_diffusion.py exists
```

The previous task added a lightweight time-conditioned ridge diffusion-style harmonizer. This is a real v0 method, but it is not yet enough for a paper claim.

## Goal

Do **not** implement a bigger model yet. First determine whether latent diffusion v0 has stable signal across seeds, target centers, and synthetic stress settings.

The main question:

```text
Does latent_diffusion_v0 consistently beat both statistical harmonization baselines and ridge_denoising baselines, while reducing site shift and preserving clinical signal?
```

## Hard rules

1. Do not use target-center labels for training.
2. Do not tune hyperparameters on target test performance.
3. Do not claim diffusion works unless the claim gate passes across a predefined evaluation protocol.
4. Report failures honestly.
5. Keep generated results out of git unless they are tiny summary artifacts intentionally committed.
6. Add tests for new summary/aggregation logic.
7. Run `pytest -q` and report the exact result.

## Step 1: Add multi-seed evaluation script

Create:

```text
scripts/run_multiseed_latent_diffusion_eval.py
```

It should run synthetic benchmarks over:

```text
seeds: e.g. 5 or 10 seeds
target centers: all available centers, default A/B/C
settings:
  strong_shift_low_confounding
  strong_confounding
  weak_site_shift
methods:
  identity
  source_standardize
  center_mean
  coral
  mmd_mean_alignment
  ridge_denoising
  ridge_denoising_clinical_preserving
  latent_diffusion_v0
  latent_diffusion_clinical_preserving_v0
  diffusion_placeholder
```

For zero-shot setting, CORAL/MMD target-unlabeled methods should be skipped or clearly marked not applicable. For target-unlabeled setting, they may use target features without target labels.

Recommended CLI:

```bash
python scripts/run_multiseed_latent_diffusion_eval.py \
  --output-dir results \
  --run-name multiseed_latent_diffusion_v0 \
  --seeds 13,17,19,23,29 \
  --setting target_unlabeled
```

Outputs:

```text
results/multiseed_latent_diffusion_v0/
  per_run_metrics.csv
  aggregate_metrics.csv
  aggregate_summary.json
  aggregate_report.md
```

## Step 2: Aggregate metrics

Aggregate at least:

```text
mean/std target_auc by method
mean/std source_val_auc by method
mean/std site_auc_before / site_auc_after
mean/std mmd_before / mmd_after
mean/std coral_before / coral_after
win rate vs best statistical baseline
win rate vs best ridge_denoising baseline
claim gate pass rate
confounding warning / invalidation count
```

Define winners carefully:

```text
best_statistical = best method where method_family == statistical_baseline
best_learned_denoising = best method where method_family == learned_denoising
best_diffusion_v0 = best method where method_family == diffusion_v0
```

For a run-level diffusion win, require:

```text
best_diffusion_v0 target_auc >= best_statistical target_auc + 0.01
best_diffusion_v0 target_auc >= best_learned_denoising target_auc + 0.01
best_diffusion_v0 site_auc_after < site_auc_before
source_val_auc drop <= 0.01 vs best statistical or identity
claim gate status == diffusion_contribution_established
confounding_status == pass
```

## Step 3: Add ablation options

Add preconfigured ablations:

```text
latent_diffusion_v0_default
latent_diffusion_v0_no_time_condition
latent_diffusion_v0_one_step
latent_diffusion_v0_low_noise
latent_diffusion_v0_high_noise
latent_diffusion_clinical_preserving_v0
```

If adding separate methods is too much for this step, expose config parameters in the script and record them in `aggregate_summary.json`.

The goal is to know whether the time-step diffusion design matters beyond ridge denoising.

## Step 4: Add tests

Add:

```text
tests/test_multiseed_eval.py
```

Required checks:

```text
small multiseed run writes per_run_metrics.csv
aggregate_metrics.csv exists
aggregate_summary.json exists
aggregate_report.md exists
aggregate summary includes win rates
no target labels are reported as used
confounded setting does not count as claimable win
placeholder diffusion does not count as real diffusion
```

Keep test sizes small for runtime.

## Step 5: Required commands

Run:

```bash
python scripts/run_multiseed_latent_diffusion_eval.py \
  --output-dir results \
  --run-name multiseed_latent_diffusion_v0 \
  --seeds 13,17,19,23,29 \
  --setting target_unlabeled
```

Then run:

```bash
pytest -q
```

## Decision criteria

### If latent diffusion v0 wins consistently

If diffusion v0 has:

```text
win rate vs statistical >= 0.6
win rate vs ridge >= 0.6
claim gate pass rate > 0 under low-confounding settings
no claimable wins under strong confounding
source preservation passes
```

then next task can be:

```text
latent_diffusion_v1_optional_torch
```

with a small MLP epsilon-prediction objective.

### If latent diffusion v0 does not beat ridge/center_mean

Do not escalate to neural DDPM yet. Instead:

```text
1. analyze whether synthetic benchmark is too easy for center_mean;
2. add nonlinear site shift / feature interactions;
3. add modality-style covariance shift;
4. add domain-specific noise variance;
5. rerun multiseed before adding neural diffusion.
```

### If latent diffusion improves site removal but hurts clinical AUC

Prioritize clinical preservation and confounding controls. Do not claim diffusion contribution.

## Final report template

Report:

```text
Files added/modified
Commands run
Number of seeds / target centers / settings
Best statistical baseline aggregate
Best ridge denoising aggregate
Best latent diffusion v0 aggregate
Win rate vs statistical
Win rate vs ridge
Claim gate pass rate
Confounding invalidation behavior
Whether diffusion v0 is worth escalating to optional torch v1
pytest result
```

## Do not do yet

Do not add real medical data.

Do not implement full image-level harmonization.

Do not claim paper contribution from a single synthetic run.

Do not tune on target labels.

Do not add a large neural diffusion model until multiseed evidence says it is worth doing.
