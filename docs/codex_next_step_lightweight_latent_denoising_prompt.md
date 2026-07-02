# Codex Next Step Prompt: Lightweight Latent Denoising Baseline v0

You are taking over `MedHarmDiff` on branch `codex/mvp-benchmark` after MVP benchmark hardening.

Current verified status from the previous agent:

```text
python scripts/run_feature_level_benchmark.py --synthetic --target-center C --output-dir results --run-name synthetic_smoke
pytest -q -> 36 passed in 22.49s
```

The MVP harness now supports synthetic data, CSV loading/validation, benchmark artifacts, run config, before/after domain metrics, confounding audit, and placeholder diffusion rejection.

Your task is to implement the first **real lightweight learned denoising baseline**, not a full diffusion model yet.

## Why this step

The repository goal is to test whether diffusion-style harmonization can improve held-out-center clinical generalization while reducing site/device/protocol shift and preserving clinical signal.

Before implementing DDPM/score diffusion, we need a small learned denoising baseline that tests whether a learned site-shift denoiser can beat statistical baselines on the synthetic feature benchmark.

This is an intermediate method, not yet the paper diffusion method.

## Hard rules

1. Do not use target-center labels for training.
2. In `zero_shot`, do not use target-center inputs for fitting harmonizers.
3. In `target_unlabeled` / `unsupervised_target_adaptation`, target inputs may be used without labels.
4. Keep `DiffusionHarmonizer` / `diffusion_placeholder` non-claimable unless a real diffusion implementation exists.
5. Do not claim diffusion contribution from this lightweight denoising baseline.
6. Report both clinical metrics and site/domain removal metrics.
7. Do not accept site AUC reduction if target clinical AUC/source preservation collapses.
8. Add tests and run `pytest -q`.

## Method to implement

Create:

```text
src/medharmdiff/latent_denoising.py
```

Implement at least:

```python
class RidgeDenoisingHarmonizer:
    def fit(self, x, site, y=None, *, target_x=None, target_site=None): ...
    def transform(self, x, site=None): ...
```

Recommended v0 design:

```text
1. Estimate a canonical source representation from source training data.
2. Estimate site-specific residual shifts from source sites.
3. Create noisy training pairs by adding Gaussian perturbations to source features.
4. Train a deterministic Ridge regressor to map noisy/site-shifted features back to canonicalized features.
5. Transform features by applying the learned denoising map.
```

Use only dependencies already in the project, preferably scikit-learn:

```text
sklearn.linear_model.Ridge
sklearn.preprocessing.StandardScaler if needed
```

The class must be deterministic with `random_seed`.

## Clinical-preserving variant

Also implement a conservative variant:

```python
class ClinicalPreservingRidgeDenoisingHarmonizer(RidgeDenoisingHarmonizer): ...
```

The goal is to reduce over-harmonization. Keep it simple. Acceptable v0 approaches:

```text
A. Preserve the label-predictive direction estimated from source labels.
B. Blend original and denoised representation along a clinical direction.
C. Use a lower denoising strength for features highly correlated with y.
```

Do not use target labels.

Expose parameters:

```text
noise_strength
n_augments
alpha
clinical_preservation_strength
random_seed
```

## Register methods

Update:

```text
src/medharmdiff/baselines.py
src/medharmdiff/benchmark.py
```

Add method names:

```text
ridge_denoising
ridge_denoising_clinical_preserving
```

Set method family in metrics:

```text
method_family = learned_denoising
```

Keep diffusion placeholder separate:

```text
diffusion_placeholder -> method_family = diffusion_placeholder
```

Do not rename ridge denoising as diffusion.

## Benchmark/report updates

Update output metrics to include:

```text
method_family
is_learned_denoising
is_diffusion
is_placeholder
```

Update `final_report.md` to include a section:

```text
## Learned Denoising Baselines

- best learned denoising method
- whether it beats strongest statistical non-diffusion baseline
- whether site metrics improve
- whether source clinical performance is preserved
- warning that this is not yet diffusion
```

Claim gate should still focus on real diffusion rows. If only placeholder diffusion exists, it must return:

```text
insufficient_data
```

## Synthetic stress runner

Add either a script or benchmark helper for three synthetic presets.

Suggested script:

```text
scripts/run_synthetic_stress.py
```

Run settings:

### setting_a_strong_shift_low_confounding

```text
site_shift_strength = 2.0 or higher
site_label_confounding_strength = 0.0 or low
```

Expected: harmonization may help.

### setting_b_strong_confounding

```text
site_label_confounding_strength = high
```

Expected: confounding warning or invalidation; do not claim.

### setting_c_weak_site_shift

```text
site_shift_strength = low
```

Expected: harmonization should not create fake gains.

Output:

```text
results/synthetic_stress_<run_name>/
  setting_a_strong_shift_low_confounding/
  setting_b_strong_confounding/
  setting_c_weak_site_shift/
  stress_summary.json
  stress_summary.md
```

The stress summary should report:

```text
best statistical baseline
best learned denoising method
claim gate status
whether learned denoising helps only in strong-shift setting
whether confounding blocks claims
```

## Tests to add

Add:

```text
tests/test_latent_denoising.py
tests/test_synthetic_stress.py
```

Required tests:

```text
RidgeDenoisingHarmonizer fit/transform preserves shape
transform before fit raises RuntimeError
deterministic with same random_seed
no target labels are used
methods appear in metrics_by_method.csv/json
method_family is reported
placeholder diffusion still returns insufficient_data
strong confounding setting does not pass claim gate
stress runner writes stress_summary.json/md
```

Also update existing benchmark tests to include:

```text
ridge_denoising
ridge_denoising_clinical_preserving
```

## Required commands

Run a smoke benchmark:

```bash
python scripts/run_feature_level_benchmark.py \
  --synthetic \
  --target-center C \
  --output-dir results \
  --run-name synthetic_denoising_smoke \
  --methods identity,source_standardize,center_mean,coral,mmd_mean_alignment,ridge_denoising,ridge_denoising_clinical_preserving,diffusion_placeholder
```

Run stress test if implemented as script:

```bash
python scripts/run_synthetic_stress.py --output-dir results --run-name synthetic_stress_denoising
```

Then run:

```bash
pytest -q
```

## Success criteria

This task succeeds if:

```text
1. A real learned denoising baseline runs end-to-end.
2. It is clearly separated from diffusion placeholder.
3. It never uses target labels.
4. It is compared against center_mean/CORAL/MMD when applicable.
5. Claim gate still rejects placeholder diffusion.
6. Stress presets run and produce summary artifacts.
7. pytest -q passes.
```

## What not to do

Do not implement full DDPM / score model yet.

Do not make any paper claim.

Do not add real medical datasets.

Do not use target labels.

Do not tune hyperparameters on target test performance.

Do not move to image-level harmonization.

## Final report template

After completing, report:

```text
Files added/modified
Commands run
Synthetic smoke metrics
Best statistical baseline
Best learned denoising baseline
Whether learned denoising beats any baseline
Whether claim gate still rejects diffusion placeholder
Whether confounding warning appears
pytest result
Next recommended task
```

## Next task after this

If learned denoising shows promise, implement a real:

```text
latent_diffusion_harmonizer_v0
```

with time-step conditioned noise prediction / denoising objective. Only then should diffusion start being tested as a possible paper contribution.
