# Codex Next Step Prompt: MVP Feature-Level Benchmark Runner

You are taking over the `MedHarmDiff` repository.

Repository goal:

```text
Clinical-Signal-Preserving Diffusion Harmonization for Multi-Center Medical Data
```

The target paper question is:

```text
Can site-/device-conditioned diffusion harmonization improve held-out-center clinical generalization while reducing site/device/protocol shift and preserving clinically meaningful signal?
```

Current repository status:

```text
README / AGENTS / research docs: present
claim_gate.py: present
basic metrics.py: present
leave-center-out protocol.py: present
baseline skeletons: present
diffusion_harmonizer.py: placeholder only
benchmark runner: missing
synthetic multi-center dataset: missing
```

Your immediate task is **not** to train a real diffusion model. Your task is to make the project a runnable, fair benchmark package that can later test diffusion honestly.

## Hard rules

1. Do not claim diffusion works while `DiffusionHarmonizer` is still a placeholder.
2. Do not use target-center labels during training in zero-shot or target-unlabeled adaptation.
3. Always report both:
   - clinical task performance;
   - site/domain removal metrics.
4. Do not count a method as successful if site predictability drops but clinical performance also drops.
5. Keep data and generated results out of git except tiny test fixtures.
6. Every new module must have tests.
7. Run `pytest -q` and report the exact result.

## Phase 1 target

Implement a reproducible feature-level benchmark MVP:

```text
synthetic multi-center feature generator
+ CSV loader
+ leave-one-center-out runner
+ baselines
+ claim gate integration
+ markdown/json/csv reports
```

The MVP should support:

```text
Train centers: A+B
Validation: source-center split from A+B
Test center: C
```

and later rotate target center for leave-one-center-out.

## Required files to add or update

Add:

```text
src/medharmdiff/synthetic.py
src/medharmdiff/io.py
src/medharmdiff/benchmark.py
scripts/run_feature_level_benchmark.py
tests/test_synthetic.py
tests/test_io.py
tests/test_benchmark.py
```

You may also update:

```text
src/medharmdiff/baselines.py
src/medharmdiff/metrics.py
src/medharmdiff/protocol.py
README.md
docs/experiment_protocol.md
```

## Synthetic generator requirements

Create `src/medharmdiff/synthetic.py`.

Implement a function such as:

```python
generate_synthetic_multicenter_features(
    *,
    n_centers: int = 3,
    samples_per_center: int = 200,
    n_features: int = 32,
    clinical_signal_strength: float = 1.0,
    site_shift_strength: float = 1.0,
    label_site_confounding: float = 0.0,
    noise_strength: float = 0.5,
    seed: int = 13,
) -> pandas.DataFrame
```

The output must contain:

```text
sample_id
center_id
label
feature_000 ... feature_N
```

The generator should simulate:

```text
clinical signal component
center/site shift component
noise component
optional label-site confounding
```

Add tests that verify:

```text
number of rows is correct
number of centers is correct
feature columns exist
labels are binary
same seed gives same data
stronger site_shift_strength increases center separability or center mean distance
stronger clinical_signal_strength increases label-feature association
```

## IO requirements

Create `src/medharmdiff/io.py`.

Implement:

```python
load_feature_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]
```

Expected columns:

```text
sample_id
center_id
label
feature_* columns
```

Return:

```text
X: feature matrix
site: center labels
label: clinical labels
sample_ids
```

Validate:

```text
required columns exist
at least one feature column exists
labels are numeric / binary for MVP
no NaN in features
```

## Benchmark runner requirements

Create `src/medharmdiff/benchmark.py`.

Implement a function such as:

```python
run_leave_center_out_benchmark(
    df: pandas.DataFrame,
    *,
    target_center: str,
    methods: list[str],
    setting: str = "zero_shot",
    seed: int = 13,
) -> dict
```

Required methods for MVP:

```text
identity
source_standardize
center_mean
coral
dmmd_mean_alignment
```

For now, diffusion should be either:

```text
not included
```

or explicitly reported as:

```text
diffusion_placeholder / insufficient_data
```

Do not create fake diffusion results.

For each method:

1. Fit harmonizer on source train only.
2. For adaptation methods requiring target unlabeled data, use target X without target labels only.
3. Train a simple clinical classifier on harmonized source train features.
4. Evaluate clinical metrics on source validation and target test.
5. Train or evaluate a site classifier / site separability metric before and after harmonization.
6. Compute MMD and CORAL distances between source and target before and after harmonization.
7. Produce per-method metrics.

Suggested simple clinical model:

```text
LogisticRegression from sklearn
```

Suggested site classifier:

```text
LogisticRegression or RandomForest, but start with LogisticRegression
```

Be careful with binary vs multicenter site AUC. For MVP, site accuracy is enough; site AUC can be optional / None for multiclass.

## Output requirements

Create CLI:

```text
scripts/run_feature_level_benchmark.py
```

It should support:

```bash
python scripts/run_feature_level_benchmark.py --synthetic --target-center center_2 --output-dir results/smoke
```

and optionally:

```bash
python scripts/run_feature_level_benchmark.py --csv data/my_features.csv --target-center center_2 --output-dir results/my_run
```

Outputs:

```text
metrics_by_method.csv
metrics_by_method.json
claim_gate_summary.json
final_report.md
run_config.json
```

`final_report.md` must include:

```text
run setting
source centers
target center
method table
strongest non-diffusion baseline
diffusion claim status
site metric before/after
clinical metric source/target
warnings about confounding or placeholder diffusion
```

## Claim gate integration

Use `src/medharmdiff/claim_gate.py`.

For MVP, if no real diffusion method is run, write:

```text
diffusion_is_placeholder: true
```

and claim gate should return:

```text
insufficient_data
```

If a placeholder diffusion method is included, it must not be allowed to pass the gate.

## Baseline notes

Current `baselines.py` includes:

```text
IdentityHarmonizer
StandardizeBySourceHarmonizer
CenterMeanHarmonizer
CoralHarmonizer
MMDMeanAlignmentHarmonizer
DiffusionPlaceholderHarmonizer
```

Keep the names clear in reports:

```text
identity
source_standardize
center_mean
coral
mmd_mean_alignment
```

Do not call `CenterMeanHarmonizer` full ComBat. It is only a ComBat-style fallback.

## Metrics

Use existing `metrics.py` where possible.

Add if needed:

```text
balanced accuracy
site classifier accuracy
source-target MMD before/after
source-target CORAL before/after
clinical preservation flag
```

Clinical preservation flag can start as:

```text
source_metric_drop <= 0.01
```

## Confounding audit MVP

Add a simple audit:

```text
label distribution by center
max label prevalence difference across centers
confounding_status
```

Status can be:

```text
pass
warn_label_site_association
invalidates_claim
```

For MVP, use a conservative warning if prevalence difference is large, e.g. > 0.25.

Do not over-engineer this yet.

## Tests

Add tests for:

```text
synthetic generator determinism
CSV loader validation
leave-one-center-out benchmark produces all required methods
claim gate returns insufficient_data for placeholder diffusion
coral / mmd metrics are finite
no target labels are used for fitting harmonizers in zero_shot setting
```

At the end run:

```bash
pytest -q
```

Expected final answer to user should include exact command and result.

## Success criteria for this task

This task is successful if:

```text
python scripts/run_feature_level_benchmark.py --synthetic --target-center center_2 --output-dir results/synthetic_smoke
```

creates all expected output files and:

```text
pytest -q
```

passes.

It is okay if diffusion is not implemented yet. The point is to build the honest benchmark harness first.

## Do not do yet

Do not implement a large PyTorch diffusion model in this task.

Do not add real medical datasets to git.

Do not claim a paper contribution.

Do not tune on target labels.

Do not merge image-level harmonization into this MVP.

## Final report requirements

After completing the task, provide a concise summary:

```text
1. What files were added/modified?
2. What command was run?
3. What metrics did the synthetic smoke run produce?
4. Did claim gate correctly reject placeholder diffusion?
5. What is the next task after MVP?
```

Recommended next task after this MVP:

```text
Implement a real lightweight latent diffusion / denoising autoencoder baseline and compare it against CORAL/MMD/center_mean on the synthetic benchmark before using real medical data.
```
