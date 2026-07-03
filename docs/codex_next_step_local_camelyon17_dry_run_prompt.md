# Codex Next Step Prompt: Local Camelyon17 Feature Dry Run

You are taking over `MedHarmDiff` on branch `codex/mvp-benchmark` after group-safe Camelyon17 benchmark support.

Current status:

```text
scripts/probe_camelyon17_wilds_metadata.py exists
scripts/convert_camelyon17_embeddings_to_feature_csv.py exists
scripts/run_camelyon17_feature_benchmark.py exists
src/medharmdiff/splitting.py exists
benchmark writes split_audit and paper_safe_split to run_config/final_report
pytest -q -> 72 passed
```

The next task is a local dry run with ignored data if available. Do not add any real data or generated result directories to git.

## Goal

Run the full local Camelyon17 feature-level pipeline:

```text
metadata probe -> embedding conversion -> feature contract validation -> group-safe benchmark -> result triage
```

The goal is not to claim diffusion works. The goal is to determine whether the real-data pilot is runnable and whether ridge/statistical baselines still dominate.

## Hard rules

1. Do not commit real metadata, image paths, embeddings, feature CSVs, or results.
2. Keep all real data under ignored `data/` and results under ignored `results/`.
3. Do not use target labels for training.
4. Do not run a paper-facing benchmark unless `paper_safe_split=true` and all group overlap lists are empty.
5. Do not implement new diffusion models in this task.
6. If local data is unavailable, do not fabricate real results; report blockers and run only fake-fixture tests.
7. Run `pytest -q` and report exact result.

## Step 1: Probe local metadata

If local metadata exists, run for example:

```bash
python scripts/probe_camelyon17_wilds_metadata.py \
  --metadata-csv data/camelyon17/metadata.csv \
  --output-dir results/camelyon17_metadata_probe
```

If using WILDS package, only run if local data exists:

```bash
python scripts/probe_camelyon17_wilds_metadata.py \
  --use-wilds \
  --wilds-root data/wilds \
  --output-dir results/camelyon17_wilds_probe
```

Do not use `--allow-download` unless the user explicitly approves.

Inspect:

```text
center_count
center_counts
label_prevalence_by_center
split_counts
has_patient_or_group_id
candidate_leakage_risks
feature_csv_ready
warnings
```

If `has_patient_or_group_id=false`, stop before paper-safe benchmark and report blocker.

## Step 2: Convert local embeddings

If local embeddings exist in CSV or NPZ, run:

```bash
python scripts/convert_camelyon17_embeddings_to_feature_csv.py \
  --metadata-csv data/camelyon17/metadata.csv \
  --embeddings-file data/camelyon17/embeddings/camelyon17_embeddings.npz \
  --output-csv data/camelyon17/features/camelyon17_features.csv \
  --output-dir results/camelyon17_conversion
```

Check:

```text
sample_count
center_count
feature_count
contract warnings
metadata warnings
```

The output feature CSV must follow:

```text
sample_id,center_id,label,patient_or_group_id,split_group,feature_0,...,feature_N
```

## Step 3: Run group-safe benchmark

Only run if feature contract validation passes and grouping metadata is available.

Example official split command:

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
  --output-dir results/camelyon17_group_safe_smoke
```

If exact split values differ, use the local metadata probe output. Do not guess.

The benchmark must report:

```text
paper_safe_split=true
group_overlap_train_val=[]
group_overlap_train_test=[]
group_overlap_val_test=[]
uses_target_labels=false for all methods
```

If `paper_safe_split=false`, mark the result as diagnostic only.

## Step 4: Triage results

Create a local ignored report or console summary. Do not commit result files.

Report:

```text
best statistical baseline
best ridge / learned-denoising baseline
best latent diffusion v0 method
claim gate status
site_auc_before/after
source_val_auc
target_auc
whether diffusion beats ridge/statistical
whether paper_safe_split is true
whether label-site confounding needs review
```

Decision rules:

```text
If ridge/statistical still dominate, do not continue diffusion modeling.
If latent_diffusion_v0 wins but claim gate fails, inspect source drop/site metrics/confounding.
If latent_diffusion_v0 wins and claim gate passes on paper-safe split, repeat with additional target centers/seeds before any claim.
```

## Step 5: Add a lightweight result-triage helper if needed

Optional but recommended: add a script that summarizes an existing benchmark result directory without reading raw data:

```text
scripts/summarize_feature_benchmark_result.py
```

Input:

```text
results/camelyon17_group_safe_smoke
```

Output:

```text
summary printed to console or summary JSON/MD under the same ignored result dir
```

If you add this script, add tests with tiny fake result artifacts.

## Tests

Run:

```bash
python -m py_compile scripts/probe_camelyon17_wilds_metadata.py scripts/convert_camelyon17_embeddings_to_feature_csv.py scripts/run_camelyon17_feature_benchmark.py
pytest -q
```

If you add a summarizer, include it in py_compile and tests.

## Final response template

Report:

```text
Files added/modified
Commands run
Whether local metadata was available
Metadata probe summary
Whether local embeddings were available
Conversion summary
Whether group-safe benchmark ran
paper_safe_split status
Best statistical/ridge/diffusion rows
Claim gate status
Whether diffusion development should continue or pause
Remaining blockers
pytest result
```

## Do not do

Do not commit data/results.
Do not download WILDS data without explicit user approval.
Do not implement neural diffusion.
Do not claim paper contribution from one local dry run.
