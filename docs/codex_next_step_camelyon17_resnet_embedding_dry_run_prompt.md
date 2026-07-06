# Codex Next Step Prompt: Camelyon17 ResNet Embedding Dry Run

You are taking over `MedHarmDiff` on branch `codex/mvp-benchmark` after the embedding-family benchmark infrastructure.

Current status:

```text
scripts/extract_camelyon17_patch_embeddings.py supports:
  color_stats_v0
  resnet18_imagenet_v0
  resnet50_imagenet_v0
  pathology_encoder_placeholder
scripts/run_camelyon17_embedding_family_benchmark.py exists
pytest -q -> 81 passed
Current torch environment reported by agent:
  torch 2.12.1+cpu
  cuda_available False
No ResNet real extraction has been run yet.
```

Previous real Camelyon17 color-stat dry run:

```text
paper_safe_split = true
uses_target_labels = false
confounding_status = pass
best statistical target_auc = 0.8628
best ridge target_auc = 0.8729
latent_diffusion_v0 target_auc = 0.8769
claim gate = baseline_not_beaten
latent_diffusion_v0 margin over ridge = +0.004, below +0.01
source_val_auc drop was large
```

## Goal

Run a controlled stronger-embedding dry run for Camelyon17 using ResNet ImageNet features, without committing any data/results.

This task is primarily an execution/environment task, not a modeling task.

## Hard rules

1. Do not commit real data, embeddings, feature CSVs, or results.
2. Keep generated artifacts under ignored `data/` and `results/` only.
3. Do not use target labels for training or tuning.
4. Do not implement neural diffusion v1 in this task.
5. Do not claim diffusion contribution from one run.
6. Do not download weights unless explicitly allowed by the user or already cached locally.
7. If CUDA PyTorch must be restored, document the exact command and resulting torch/torchvision versions.
8. Run `pytest -q` and report exact result.

## Step 1: Environment audit

Run and report:

```bash
python - <<'PY'
import torch
import torchvision
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_version', getattr(torch.version, 'cuda', None))
PY
```

If CUDA is required and unavailable, do not silently reinstall. Report the environment and ask whether to restore CUDA PyTorch, unless the user has already explicitly approved environment changes.

If running CPU-only, only do a small limited smoke extraction first.

## Step 2: Check local data paths

Confirm these exist locally:

```text
data/camelyon17/metadata.csv
H:/MedHarmDiff_data/wilds/camelyon17_v1.0
```

or the equivalent local paths from the previous dry run.

Do not commit path-specific outputs to git.

## Step 3: ResNet extraction smoke test

Run a limited extraction first.

If local ImageNet weights are available:

```bash
python scripts/extract_camelyon17_patch_embeddings.py \
  --metadata-csv data/camelyon17/metadata.csv \
  --image-root H:/MedHarmDiff_data/wilds/camelyon17_v1.0 \
  --output-npz data/camelyon17/embeddings/camelyon17_resnet18_imagenet_v0_limit1000.npz \
  --output-dir results/camelyon17_resnet18_limit1000 \
  --extractor resnet18_imagenet_v0 \
  --weights-path <LOCAL_WEIGHTS_PATH> \
  --limit 1000 \
  --batch-size 32
```

If weights download is explicitly approved:

```bash
python scripts/extract_camelyon17_patch_embeddings.py \
  --metadata-csv data/camelyon17/metadata.csv \
  --image-root H:/MedHarmDiff_data/wilds/camelyon17_v1.0 \
  --output-npz data/camelyon17/embeddings/camelyon17_resnet18_imagenet_v0_limit1000.npz \
  --output-dir results/camelyon17_resnet18_limit1000 \
  --extractor resnet18_imagenet_v0 \
  --allow-download-weights \
  --limit 1000 \
  --batch-size 32
```

Check:

```text
embedding_manifest.json exists
feature_dim is expected for ResNet-18
sample_count == 1000
weights_source recorded
weights_downloaded recorded
```

If this fails, stop and report the reason.

## Step 4: Convert smoke embeddings to feature CSV

Run:

```bash
python scripts/convert_camelyon17_embeddings_to_feature_csv.py \
  --metadata-csv data/camelyon17/metadata.csv \
  --embeddings-file data/camelyon17/embeddings/camelyon17_resnet18_imagenet_v0_limit1000.npz \
  --output-csv data/camelyon17/features/camelyon17_resnet18_imagenet_v0_limit1000_features.csv \
  --output-dir results/camelyon17_resnet18_limit1000_conversion
```

If conversion fails because metadata has more rows than embeddings, implement a safe option or helper to allow subset conversion by available sample_ids. Add tests for this behavior.

Do not fake missing embeddings.

## Step 5: Limited group-safe benchmark smoke

Run the group-safe benchmark only if the limited feature CSV still has enough rows/groups/centers for a meaningful split. If 1000 sequential samples are not enough, use a stratified/subsampled extraction plan instead.

Example:

```bash
python scripts/run_camelyon17_feature_benchmark.py \
  --feature-csv data/camelyon17/features/camelyon17_resnet18_imagenet_v0_limit1000_features.csv \
  --target-center 2 \
  --group-column patient_or_group_id \
  --setting zero_shot \
  --output-dir results/camelyon17_resnet18_limit1000_benchmark
```

Then summarize:

```bash
python scripts/summarize_feature_benchmark_result.py \
  results/camelyon17_resnet18_limit1000_benchmark
```

If `paper_safe_split=false`, label result diagnostic only.

## Step 6: Full extraction decision

Only run full ResNet extraction if smoke extraction and conversion succeed.

Before full extraction, estimate:

```text
expected runtime
expected disk usage
CPU/GPU mode
feature dim
sample count
```

If GPU is unavailable, consider whether full extraction is practical. Do not start a multi-hour full extraction unless the user has approved.

If approved, run full extraction and then:

```text
convert to feature CSV
run paper-safe Camelyon17 benchmarkun embedding-family benchmark comparing color_stats_v0 and resnet18/resnet50 if available
summarize results
```

## Step 7: Embedding family benchmark

When at least two feature CSVs are available, run:

```bash
python scripts/run_camelyon17_embedding_family_benchmark.py \
  --feature-csv color_stats_v0=data/camelyon17/features/camelyon17_color_stats_v0_features.csv \
  --feature-csv resnet18=data/camelyon17/features/camelyon17_resnet18_imagenet_v0_features.csv \
  --target-center 2 \
  --group-column patient_or_group_id \
  --setting zero_shot \
  --output-dir results/camelyon17_embedding_family_benchmark
```

Report:

```text
best statistical by family
best ridge by family
latent_diffusion_v0 by family
claim gate by family
paper_safe_split by family
source_drop and margin flags
```

## Tests

Run:

```bash
python -m py_compile scripts/extract_camelyon17_patch_embeddings.py scripts/run_camelyon17_embedding_family_benchmark.py scripts/summarize_feature_benchmark_result.py
pytest -q
```

If you add subset conversion or subsampling helpers, add tests.

## Decision rule

Continue diffusion development only if stronger embeddings produce:

```text
paper_safe_split = true
uses_target_labels = false
latent_diffusion_v0 target_auc >= best_non_diffusion + 0.01
source validation drop <= claim-gate tolerance
claim gate passes
```

Otherwise keep diffusion paused and report whether the bottleneck is:

```text
representation quality
ridge/statistical dominance
source clinical signal loss
lack of practical GPU/weights environment
```

## Final response template

Report:

```text
Files added/modified
Commands run
Torch/CUDA status
Weights source / whether downloaded
Whether ResNet smoke extraction ran
Smoke extraction manifest summary
Whether conversion ran
Whether benchmark ran
paper_safe_split status
Best statistical/ridge/diffusion rows if benchmark ran
Claim gate status
Whether full extraction is recommended
Whether diffusion development remains paused
pytest result
```

## Do not do

Do not commit generated embeddings/results.
Do not download weights without approval.
Do not run full extraction without reporting feasibility.
Do not implement neural diffusion v1.
Do not claim paper contribution.
