# Codex Next Step Prompt: Nonlinear Shift Benchmark + Dataset Scout

You are taking over `MedHarmDiff` on branch `codex/mvp-benchmark` after multi-seed latent diffusion v0 evaluation.

Current reported result:

```text
45 runs = 5 seeds x 3 target centers x 3 settings
pytest -q -> 50 passed
diffusion_v0_win_rate_vs_statistical = 0.0
diffusion_v0_win_rate_vs_ridge = 0.0
claim_gate_pass_rate = 0.0
confounding_invalidation_count = 15
target_labels_used_count = 0
best overall target AUC:
  ridge_denoising = 0.7901
  latent_diffusion_v0 = 0.7602
  latent_diffusion_clinical_preserving_v0 = 0.7637
```

Interpretation:

```text
Do not upgrade to optional torch / neural diffusion yet.
The current synthetic benchmark is probably too linear/additive, and ridge/statistical methods dominate.
The next task is to diagnose whether the benchmark contains a realistic shift regime where diffusion-style modeling is necessary.
```

## Hard rules

1. Do not claim diffusion contribution from the current result.
2. Do not implement optional torch v1 in this task.
3. Do not tune hyperparameters on target test labels.
4. Keep target labels evaluation-only.
5. Keep generated result directories out of git unless they are tiny summary docs.
6. Add tests for new synthetic shift mechanisms and summary logic.
7. Run `pytest -q` and report the exact result.

## Goal

Improve the synthetic stress benchmark so it tests realistic representation shift beyond simple additive center offsets.

The question is:

```text
Is latent diffusion v0 failing because diffusion is unhelpful, or because the current synthetic shift is too easily solved by center_mean / ridge_denoising?
```

## Step 1: Add nonlinear synthetic shift generator

Update or extend:

```text
src/medharmdiff/synthetic.py
```

Add a new function, for example:

```python
generate_synthetic_nonlinear_multicenter_features(...)
```

It should keep the same output schema:

```text
sample_id
center_id
label
feature_0 ... feature_N
```

Add optional hidden/evaluation-only columns only if clearly prefixed, for example:

```text
oracle_clinical_latent_*
oracle_site_style_*
```

Do not let benchmark methods use oracle columns as features by default.

Required shift knobs:

```text
additive_shift_strength
covariance_shift_strength
rotation_shift_strength
heteroskedastic_noise_strength
nonlinear_warp_strength
class_conditional_style_strength
site_specific_feature_interaction_strength
label_site_confounding_strength
clinical_signal_strength
noise_strength
```

At least implement these four in v1:

```text
covariance_shift_strength
rotation_shift_strength
heteroskedastic_noise_strength
nonlinear_warp_strength
```

Design intent:

```text
center_mean should solve additive shift;
CORAL/MMD should help covariance/mean shift;
ridge_denoising should help linear denoising;
diffusion_v0 should only have a plausible advantage when shift is nonlinear/noisy/multi-step and clinical signal must be preserved.
```

## Step 2: Add benchmark presets

Update:

```text
scripts/run_multiseed_latent_diffusion_eval.py
```

Add new stress settings:

```text
nonlinear_shift_low_confounding
covariance_shift_low_confounding
heteroskedastic_shift_low_confounding
mixed_realistic_shift_low_confounding
mixed_realistic_shift_strong_confounding
```

Keep existing settings:

```text
strong_shift_low_confounding
strong_confounding
weak_site_shift
```

The aggregate summary should report settings separately.

## Step 3: Add benchmark difficulty diagnostics

Add a diagnostic script or module:

```text
scripts/analyze_synthetic_shift_difficulty.py
```

or integrate into the multiseed summary.

Report per setting:

```text
raw identity target_auc
best statistical target_auc
best ridge target_auc
best diffusion_v0 target_auc
site_auc_before
site_auc_after by method
MMD/CORAL before/after
label-site confounding status
whether center_mean dominates
whether nonlinear shift makes statistical baselines weaker
whether diffusion_v0 beats ridge/statistical
```

Add summary flags:

```text
statistical_baselines_dominate_all_settings
ridge_denoising_dominates_all_settings
diffusion_v0_has_any_low_confounding_win
nonlinear_settings_create_headroom
```

## Step 4: Add synthetic oracle diagnostics, evaluation-only

If adding oracle latent columns, add a diagnostic only, not an inference feature.

Possible metrics:

```text
clinical_latent_correlation_preserved
site_style_correlation_removed
oracle_content_mse
```

These should be excluded from feature columns by default. Add tests to ensure `load_feature_csv(..., feature_columns="auto")` does not include `oracle_*` columns.

If this is too much for one pass, leave oracle diagnostics as TODO in the final report and focus on nonlinear shift generation.

## Step 5: Dataset scout document

Create:

```text
docs/real_dataset_scout.md
```

Do not add any real data.

The document should list candidate real-world multi-center/domain-shift datasets for future evaluation and classify them by:

```text
data modality
number of centers/domains
label/task
whether center/domain metadata exists
whether target labels are public
whether feature-level extraction is feasible
license/access burden
recommended first-use priority
```

Suggested categories to investigate:

```text
medical imaging domain generalization datasets
histopathology multi-center datasets
dermatology multi-source datasets
chest X-ray multi-hospital datasets
MRI multi-scanner datasets
radiomics datasets with scanner/site metadata
```

If using web research, cite sources in the document. If not using web, clearly mark as preliminary and source-needed.

## Step 6: Tests

Add/update tests:

```text
tests/test_synthetic_nonlinear.py
tests/test_multiseed_eval.py
tests/test_io.py
```

Required checks:

```text
nonlinear generator is deterministic
output schema is compatible with benchmark
center count and labels valid
stronger covariance/rotation/nonlinear shift increases site separability or distribution distance
oracle_* columns are excluded from auto feature selection
multiseed eval accepts new nonlinear settings
aggregate summary includes new diagnostic flags
confounded nonlinear setting does not count as claimable win
```

## Step 7: Required commands

Run:

```bash
python scripts/run_multiseed_latent_diffusion_eval.py \
  --output-dir results \
  --run-name multiseed_nonlinear_shift_eval \
  --seeds 13,17,19,23,29 \
  --setting target_unlabeled \
  --settings strong_shift_low_confounding,strong_confounding,weak_site_shift,nonlinear_shift_low_confounding,covariance_shift_low_confounding,heteroskedastic_shift_low_confounding,mixed_realistic_shift_low_confounding,mixed_realistic_shift_strong_confounding
```

Then run:

```bash
pytest -q
```

## Decision criteria

### If diffusion_v0 still has zero wins

Do not build neural diffusion yet. Instead:

```text
1. prioritize real dataset selection;
2. improve shift realism only if justified;
3. consider whether diffusion is the wrong model for feature-level harmonization.
```

### If nonlinear settings create headroom but diffusion_v0 still loses to ridge

Do not build neural diffusion yet. First inspect why ridge wins:

```text
linear target is enough?
clinical signal easier than site signal?
diffusion reverse loop oversmooths?
noise schedule too destructive?
```

### If diffusion_v0 wins only in low-confounding nonlinear/mixed settings

Then consider:

```text
latent_diffusion_v1_optional_torch
```

with an actual MLP epsilon-prediction objective.

Minimum for escalation:

```text
diffusion_v0_has_any_low_confounding_win == true
win rate vs ridge/statistical > 0 in nonlinear/mixed settings
no claimable wins under strong confounding
source preservation passes
```

## Final report template

Report:

```text
Files added/modified
Commands run
New synthetic shift settings
Whether nonlinear/mixed settings create headroom
Whether center_mean/ridge still dominate
Diffusion v0 win rate vs statistical/ridge overall and by setting
Claim gate pass rate
Confounding invalidation behavior
Whether to proceed to optional torch diffusion v1
pytest result
```

## Do not do yet

Do not add real medical datasets to git.

Do not implement neural DDPM / optional torch v1 unless this task shows evidence that diffusion v0 has at least some low-confounding wins.

Do not claim paper contribution from synthetic-only results.

Do not tune on target labels.
