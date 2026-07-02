# Codex Next Step Prompt: MVP Benchmark Hardening

You are taking over the `MedHarmDiff` repository.

Current repository state:

```text
src/medharmdiff/synthetic.py exists
src/medharmdiff/benchmark.py exists
scripts/run_feature_level_benchmark.py exists
tests/test_synthetic.py exists
tests/test_feature_level_benchmark.py exists
```

The MVP is partially implemented. The synthetic generator and CSV-driven feature-level benchmark exist, but the repository is not yet a fully reproducible benchmark package.

Your task is to harden the MVP so it can be used as a fair baseline harness before implementing any real diffusion model.

## Project goal

MedHarmDiff tests whether diffusion harmonization can improve held-out-center clinical generalization while reducing site/device/protocol shift and preserving clinically meaningful signal.

The current claim boundary remains:

```text
No real diffusion contribution can be claimed while DiffusionHarmonizer is still a placeholder.
```

## Hard rules

1. Do not claim diffusion works while `DiffusionHarmonizer` is a placeholder.
2. Do not use target-center labels during training in zero-shot or target-unlabeled adaptation.
3. Always report both clinical task metrics and site/domain removal metrics.
4. Do not count a method as successful if site predictability drops but clinical performance also drops.
5. Keep data and generated results out of git except tiny test fixtures.
6. Add tests for every new module or important behavior.
7. Run `pytest -q` and report the exact result.

## Current known gaps

The current MVP still needs:

```text
1. CLI --synthetic mode
2. src/medharmdiff/io.py CSV loader
3. metrics_by_method.json output
4. run_config.json output
5. richer final_report.md
6. explicit before/after site metrics
7. CLI smoke test
8. target-label leakage tests
```

## Step 1: Add CLI `--synthetic` mode

Update:

```text
scripts/run_feature_level_benchmark.py
```

The CLI currently requires `--feature-path` unless a YAML config is provided. Add:

```text
--synthetic
--n-centers
--samples-per-center
--n-features
--clinical-signal-strength
--site-shift-strength
--site-label-confounding-strength
--noise-strength
--random-seed
```

Expected smoke command:

```bash
python scripts/run_feature_level_benchmark.py \
  --synthetic \
  --target-center C \
  --output-dir results \
  --run-name synthetic_smoke
```

When `--synthetic` is used:

```text
1. Generate a synthetic feature dataset with generate_synthetic_feature_dataset.
2. Save it to <run_dir>/synthetic_features.csv.
3. Run the benchmark using that CSV.
4. Write all normal benchmark artifacts.
```

Important: current synthetic centers are named `A`, `B`, `C`, ... so use `--target-center C` in docs/tests unless you change the generator naming consistently.

## Step 2: Add `src/medharmdiff/io.py`

Create:

```text
src/medharmdiff/io.py
```

Implement:

```python
load_feature_csv(
    path,
    *,
    sample_id_column="sample_id",
    center_column="center_id",
    label_column="label",
    feature_columns="auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]
```

Return:

```text
X
site
y
sample_ids
feature_columns
```

Validation requirements:

```text
required columns exist
at least one feature column exists
features contain no NaN
labels are binary / numeric for MVP
center count >= 3
sample_id is unique
```

Update `benchmark.py` to reuse this loader rather than duplicating CSV parsing logic.

## Step 3: Complete benchmark artifacts

Current benchmark writes:

```text
metrics_by_method.csv
claim_gate_summary.json
final_report.md
```

Add:

```text
metrics_by_method.json
run_config.json
synthetic_features.csv  # synthetic mode only
```

`run_config.json` should include at least:

```text
run_name
setting
target_center
source_centers
methods
random_seed
data_source: synthetic/csv
synthetic_params
feature_columns
sample_count_by_center
label_prevalence_by_center
```

## Step 4: Add explicit before/after harmonization metrics

Currently each method has after-harmonization metrics such as:

```text
site_auc
mmd_rbf
coral_distance
```

Add explicit columns:

```text
site_auc_before
site_auc_after
mmd_before
mmd_after
coral_before
coral_after
```

For `no_harmonization`:

```text
before == after
```

For other methods:

```text
before = raw source-validation vs raw target
after = harmonized source-validation vs harmonized target
```

Use these fields in the claim gate:

```text
site_metric_before = site_auc_before for no_harmonization/raw baseline
site_metric_after = diffusion site_auc_after
```

If diffusion remains placeholder, the claim gate must still return:

```text
insufficient_data
```

## Step 5: Improve `final_report.md`

The report should include:

```text
run setting
source centers
target center
method table
strongest non-diffusion baseline
claim gate status
placeholder diffusion warning
label-site confounding summary
source clinical metric
target clinical metric
site metric before/after
MMD/CORAL before/after
zero-shot vs target-unlabeled adaptation note
```

Keep the report factual. Do not imply diffusion works.

## Step 6: Improve confounding audit

Add a simple label-site confounding section to `run_config.json` and `final_report.md`:

```text
label prevalence by center
max label prevalence difference across centers
confounding_status
```

Use conservative statuses:

```text
pass
warn_label_site_association
invalidates_claim
```

Suggested thresholds:

```text
max prevalence spread >= 0.75 -> invalidates_claim
max prevalence spread >= 0.40 -> warn_label_site_association
otherwise -> pass
```

If you use a different threshold, document it.

## Step 7: Add tests

Add or update:

```text
tests/test_io.py
tests/test_feature_level_benchmark.py
tests/test_cli.py
```

Required test coverage:

```text
valid CSV loads correctly
missing sample_id / center_id / label raises ValueError
missing feature columns raises ValueError
NaN feature raises ValueError
duplicate sample_id raises ValueError
less than 3 centers raises ValueError
CLI --synthetic creates expected files
claim gate returns insufficient_data for placeholder diffusion
metrics_by_method.json and run_config.json are written
before/after site metrics exist
zero-shot mode marks CORAL/MMD target-adaptation methods as not_applicable_zero_shot
no method uses target labels in zero-shot or unsupervised target adaptation
```

## Step 8: Required commands

Run:

```bash
python scripts/run_feature_level_benchmark.py \
  --synthetic \
  --target-center C \
  --output-dir results \
  --run-name synthetic_smoke
```

Then run:

```bash
pytest -q
```

Report exact commands and results.

## Success criteria

This task is successful if:

```text
1. One CLI command can generate and evaluate synthetic multi-center data.
2. All required artifacts are produced.
3. Claim gate rejects placeholder diffusion as insufficient_data.
4. Site/domain removal and clinical performance metrics are both reported.
5. Target-label leakage is explicitly prevented and tested.
6. pytest -q passes.
```

## Do not do yet

Do not implement a real PyTorch diffusion model in this task.

Do not add real medical datasets to git.

Do not make a paper contribution claim.

Do not tune on target labels.

Do not start image-level harmonization.

## Final response template

After completing the task, respond with:

```text
Files added/modified:
- ...

Commands run:
- python scripts/run_feature_level_benchmark.py --synthetic --target-center C --output-dir results --run-name synthetic_smoke
- pytest -q

Key smoke result:
- claim status: insufficient_data
- diffusion placeholder rejected: yes/no
- strongest non-diffusion baseline: ...
- target AUC table summary: ...

Next recommended task:
- implement a lightweight latent denoising / diffusion baseline only after this benchmark hardening is complete.
```
