# Codex Next Step Prompt: Group-Safe Camelyon17 Feature Benchmark Dry Run

You are taking over `MedHarmDiff` on branch `codex/mvp-benchmark` after the Camelyon17 metadata probe and feature CSV converter scaffold.

Current status:

```text
src/medharmdiff/real_pilots/camelyon17_wilds.py exists
scripts/probe_camelyon17_wilds_metadata.py exists
scripts/convert_camelyon17_embeddings_to_feature_csv.py exists
validate_real_feature_contract exists
pytest -q -> 65 passed
```

The real-data entrypoint is scaffolded, but the benchmark runner still needs group-safe split support before any real Camelyon17 result can be trusted.

## Goal

Make the feature-level benchmark safe for real Camelyon17/WILDS-style data by supporting:

```text
official split_group columns
group-safe source train/validation split using patient_or_group_id / slide_id
leakage audit across train/val/test
real feature CSV dry-run commands
```

Do not train on real data in this task unless the user has local ignored features and explicitly runs the command. Do not add real data or embeddings to git.

## Hard rules

1. Do not add real data, patch images, embeddings, labels, or result directories to git.
2. Do not use target labels for training.
3. Do not split patches from the same patient/slide/group across train and validation.
4. Prefer official `split_group` if present and explicitly configured.
5. If no group column is available for real data, emit a warning and mark benchmark as not paper-safe.
6. Do not implement new diffusion models in this task.
7. Add tests with tiny fake feature CSVs only.
8. Run `pytest -q` and report exact result.

## Step 1: Add split/group config support

Update:

```text
src/medharmdiff/benchmark.py
```

Extend `FeatureBenchmarkConfig` with optional fields:

```python
group_column: str | None = None
split_group_column: str | None = None
source_train_split_values: list[str] | None = None
source_val_split_values: list[str] | None = None
target_test_split_values: list[str] | None = None
require_group_safe_split: bool = False
```

Keep default synthetic behavior unchanged.

## Step 2: Implement split planner

Add helper module or benchmark helpers, for example:

```text
src/medharmdiff/splitting.py
```

Implement:

```python
plan_feature_benchmark_split(df, config) -> SplitPlan
```

It should support two modes:

### Mode A: official split_group mode

If `split_group_column` and split value lists are provided:

```text
train_df = rows with split_group in source_train_split_values and center != target_center
val_df = rows with split_group in source_val_split_values and center != target_center
test_df = rows with split_group in target_test_split_values and center == target_center
```

If target split values are not provided, keep existing target-center filtering but record warning.

### Mode B: group-safe random source split

If official split is not configured, split source rows into train/validation by `group_column` if available.

Rules:

```text
same group cannot appear in both train and val
target center remains test only
if group_column missing and require_group_safe_split=True, raise ValueError
if group_column missing and require_group_safe_split=False, warn that result is not paper-safe
```

Output an audit object:

```text
train_count
val_count
test_count
train_groups
val_groups
test_groups
group_overlap_train_val
group_overlap_train_test
group_overlap_val_test
warnings
paper_safe_split: bool
```

## Step 3: Add leakage audit output

Update benchmark artifacts:

```text
run_config.json
final_report.md
```

Add:

```text
split_audit
paper_safe_split
warnings
```

If `paper_safe_split` is false, report should say:

```text
This run is not paper-safe due to missing or leaking group/split metadata.
```

## Step 4: Add Camelyon17 benchmark CLI wrapper

Add:

```text
scripts/run_camelyon17_feature_benchmark.py
```

This script should call existing `run_feature_level_benchmark` with real-data-safe defaults.

Suggested CLI:

```bash
python scripts/run_camelyon17_feature_benchmark.py \
  --feature-csv data/camelyon17/features/camelyon17_features.csv \
  --target-center 4 \
  --group-column patient_or_group_id \
  --split-group-column split_group \
  --source-train-split-values train \
  --source-val-split-values val,id_val \
  --target-test-split-values test,ood_test \
  --setting zero_shot \
  --output-dir results/camelyon17_smoke
```

It should:

```text
validate feature contract
require group-safe split by default
run identity/source_standardize/center_mean/ridge/latent_diffusion_v0/diffusion_placeholder
write outputs under results/
```

Do not assume exact WILDS split names; make them configurable.

## Step 5: Update converter/probe docs

Update:

```text
docs/real_dataset_pilot_plan.md
docs/real_feature_csv_contract.md
```

Add:

```text
real benchmark requires group-safe split audit
patient_or_group_id or slide_id is mandatory for paper-safe result
split_group values must be documented
```

## Step 6: Tests

Add/update:

```text
tests/test_splitting.py
tests/test_camelyon17_feature_benchmark_cli.py
```

Use tiny fake CSV fixtures. Required tests:

```text
group-safe split has no train/val group overlap
official split_group mode respects configured split values
missing group column raises when require_group_safe_split=True
missing group column warns when require_group_safe_split=False
run_config.json contains split_audit
final_report.md marks non-paper-safe splits
Camelyon17 wrapper writes outputs with fake feature CSV
no target labels used for training
```

## Step 7: Required commands

Run:

```bash
python -m py_compile scripts/run_camelyon17_feature_benchmark.py
pytest -q
```

If local ignored Camelyon17 fake fixture exists, optionally run a fake smoke command. Do not require real WILDS data in tests.

## Success criteria

This task succeeds if:

```text
1. Benchmark runner supports group-safe real-data splits.
2. Official split_group mode is configurable.
3. Leakage audit is written to run_config and final_report.
4. Camelyon17 feature benchmark wrapper exists.
5. Fake fixtures prove no group leakage.
6. pytest -q passes.
```

## Do not do yet

Do not download Camelyon17.
Do not extract real embeddings.
Do not run paper-facing benchmark without local metadata audit.
Do not implement neural diffusion v1.
Do not claim diffusion works.

## Final response template

Report:

```text
Files added/modified
Commands run
Split planner behavior
Leakage audit behavior
Camelyon17 wrapper behavior
Whether real benchmark is now paper-safe when group metadata is present
Remaining blockers for actual local Camelyon17 run
pytest result
Next recommended task
```

## Next task after this

If group-safe benchmark support is ready, run a local Camelyon17 feature dry run using ignored data:

```text
1. probe metadata CSV
2. convert local embeddings to feature CSV
3. validate feature contract
4. run group-safe Camelyon17 feature benchmark
5. inspect whether ridge/statistical baselines still dominate
```
