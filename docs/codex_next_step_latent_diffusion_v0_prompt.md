# Codex Next Step Prompt: Latent Diffusion Harmonizer v0

You are taking over `MedHarmDiff` on branch `codex/mvp-benchmark` after the lightweight latent denoising baseline.

Current status from the previous task:

```text
src/medharmdiff/latent_denoising.py exists
ridge_denoising and ridge_denoising_clinical_preserving are registered
scripts/run_synthetic_stress.py exists
pytest -q -> 36 passed in 22.49s
```

Important: ridge denoising is **not** a diffusion model. It is a learned denoising baseline. Do not claim a diffusion contribution from ridge results.

## Goal

Implement the first real lightweight **latent diffusion / time-step denoising** harmonizer for feature-level multi-center harmonization.

The goal is to test whether a time-conditioned denoising model can outperform:

```text
identity
source_standardize
center_mean
coral / mmd_mean_alignment when target-unlabeled setting is used
ridge_denoising
ridge_denoising_clinical_preserving
```

while preserving clinical signal and keeping the claim gate honest.

## Hard rules

1. Do not use target-center labels for training.
2. In `zero_shot`, do not use target-center inputs for fitting harmonizers.
3. In `target_unlabeled`, target-center inputs may be used without labels only if the method is explicitly target-unlabeled.
4. Do not tune hyperparameters on target test performance.
5. Report both clinical metrics and site/domain removal metrics.
6. Do not claim success if site shift drops but clinical target/source performance collapses.
7. Keep generated data/results out of git.
8. Add tests and run `pytest -q`.

## Implementation preference

Start with a **torch-free or optional-torch** v0 if possible.

Preferred v0:

```text
TimeConditionedRidgeDiffusionHarmonizer
```

This is a lightweight diffusion-style baseline:

```text
1. Build source canonical features by removing source-center offsets.
2. Sample diffusion time t in [0, 1].
3. Add Gaussian noise with schedule sigma(t) to canonical features.
4. Add sampled source-site residual/style offsets to create shifted noisy inputs.
5. Train a time-conditioned Ridge model to predict either:
   a. clean canonical feature h0, or
   b. noise epsilon.
6. At transform time, perform K denoising steps from source/target features toward canonical representation.
```

Acceptable v0 simplification:

```text
Train one Ridge model with input [x_t, t, site embedding / site offset estimate] -> h0.
Run a small deterministic reverse loop with num_steps = 5-20.
```

Do not present this as a full DDPM paper model. Call it:

```text
latent_diffusion_v0
```

or:

```text
time_conditioned_ridge_diffusion
```

## Files to add/update

Add or update:

```text
src/medharmdiff/latent_diffusion.py
src/medharmdiff/baselines.py
src/medharmdiff/benchmark.py
scripts/run_synthetic_stress.py
tests/test_latent_diffusion.py
tests/test_feature_level_benchmark.py
tests/test_synthetic_stress.py
```

## Required class

Implement something like:

```python
class TimeConditionedRidgeDiffusionHarmonizer:
    def __init__(
        self,
        *,
        num_steps: int = 10,
        n_augments: int = 8,
        noise_strength: float = 0.2,
        alpha: float = 1.0,
        clinical_preservation_strength: float = 0.0,
        random_seed: int = 13,
    ): ...

    def fit(self, x, site, y=None, *, target_x=None, target_site=None): ...
    def transform(self, x, site=None): ...
```

The model must expose:

```text
is_fitted
uses_target_unlabeled = False by default
uses_target_labels = False always
```

Do not use `target_y` anywhere.

## Clinical-preserving variant

Implement either via parameter or separate class:

```text
latent_diffusion_clinical_preserving_v0
```

It should preserve source label-predictive direction similarly to ridge denoising:

```text
estimate clinical direction from source labels only
blend denoised output with original projection along that direction
```

## Register benchmark methods

Update benchmark method registry with:

```text
latent_diffusion_v0
latent_diffusion_clinical_preserving_v0
```

Metrics fields:

```text
method_family = diffusion_v0
is_diffusion = True
is_placeholder = False
is_learned_denoising = False
```

Keep `diffusion_placeholder` available and non-claimable.

## Claim gate behavior

For real `latent_diffusion_v0`, the claim gate should evaluate it as a diffusion row.

Diffusion contribution is claimable only if:

```text
diffusion target AUC > strongest non-diffusion baseline target AUC + margin
site_auc_after < site_auc_before
source validation AUC drop <= tolerance
clinical_preservation_pass == true
confounding_status does not invalidate claim
```

Do not relax this gate.

If `latent_diffusion_v0` fails, report it honestly.

## Synthetic stress experiments

Update stress runner to include:

```text
latent_diffusion_v0
latent_diffusion_clinical_preserving_v0
```

Keep existing three settings:

```text
setting_a_strong_shift_low_confounding
setting_b_strong_confounding
setting_c_weak_site_shift
```

The stress summary should explicitly report:

```text
best statistical baseline
best learned denoising baseline
best diffusion_v0 method
diffusion claim gate status
whether diffusion beats ridge denoising
whether diffusion only helps in strong-shift/low-confounding setting
whether confounding blocks claim
```

## Tests

Add tests for:

```text
latent diffusion fit/transform preserves shape
transform before fit raises RuntimeError
deterministic with same random_seed
num_steps affects transform path but stays finite
clinical-preserving variant preserves shape
method appears in metrics_by_method with method_family = diffusion_v0
is_diffusion = True and is_placeholder = False for latent_diffusion_v0
diffusion_placeholder remains is_placeholder = True
claim gate rejects latent diffusion if it does not beat baselines
stress runner writes diffusion_v0 fields
```

## Required commands

Run smoke benchmark:

```bash
python scripts/run_feature_level_benchmark.py \
  --synthetic \
  --target-center C \
  --output-dir results \
  --run-name synthetic_latent_diffusion_smoke \
  --methods identity,source_standardize,center_mean,coral,mmd_mean_alignment,ridge_denoising,ridge_denoising_clinical_preserving,latent_diffusion_v0,latent_diffusion_clinical_preserving_v0,diffusion_placeholder
```

Run stress:

```bash
python scripts/run_synthetic_stress.py --output-dir results --run-name synthetic_stress_latent_diffusion
```

Run tests:

```bash
pytest -q
```

## Success criteria

This task succeeds if:

```text
1. A real time-conditioned latent diffusion-style harmonizer runs end-to-end.
2. It is clearly separated from ridge denoising and diffusion placeholder.
3. It never uses target labels.
4. It is compared against statistical and learned-denoising baselines.
5. Claim gate passes or fails honestly.
6. Stress presets include diffusion_v0 fields.
7. pytest -q passes.
```

## Do not do yet

Do not implement full neural DDPM unless this lightweight v0 demonstrates enough signal.

Do not add real medical datasets.

Do not claim a paper contribution unless claim gate passes.

Do not tune hyperparameters on target test performance.

Do not move to image-level harmonization.

## Final response template

Report:

```text
Files added/modified
Commands run
Smoke benchmark method table summary
Best statistical baseline
Best ridge/learned denoising baseline
Best latent diffusion v0 method
Claim gate status
Whether latent diffusion beats ridge denoising and statistical baselines
Whether confounding blocks claims in stress test
pytest result
Next recommended task
```

## Next task after this

If latent diffusion v0 shows consistent benefit under strong site shift with low confounding, implement:

```text
latent_diffusion_v1_optional_torch
```

with an actual MLP epsilon-prediction objective and stronger ablations.

If latent diffusion v0 does not beat ridge denoising or center_mean, pause diffusion development and improve synthetic stress design / real dataset selection first.
