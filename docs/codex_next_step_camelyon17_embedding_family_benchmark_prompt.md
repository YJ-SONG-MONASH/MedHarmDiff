# Codex Next Step Prompt: Camelyon17 Embedding Family Benchmark

You are taking over `MedHarmDiff` on branch `codex/mvp-benchmark` after the first real-scale Camelyon17 dry run.

Current reported real dry-run result:

```text
Camelyon17 feature CSV: 455,954 samples, 37 color_stats_v0 features
paper_safe_split = true
group overlaps train/val/test = []
uses_target_labels = false for all methods
confounding_status = pass

best statistical: source_standardize
  target_auc = 0.8628
  source_val_auc = 0.9528

best ridge: ridge_denoising
  target_auc = 0.8729
  source_val_auc = 0.9247

latent_diffusion_v0:
  target_auc = 0.8769
  source_val_auc = 0.8994

claim gate:
  status = baseline_not_beaten
  target_gain_vs_best_baseline = 0.004
  source_drop_vs_no_harmonization = 0.0533
```

Decision:

```text
Do not continue diffusion modeling yet.
The current color_stats_v0 features are too weak/simple for a diffusion claim.
Next step is to benchmark stronger frozen embedding families and see whether any real representation creates diffusion-relevant headroom.
```

## Hard rules

1. Do not commit real data, images, embeddings, feature CSVs, or result directories.
2. Keep all generated data under ignored `data/` and `results/`.
3. Do not use target labels for training or tuning.
4. Do not implement neural diffusion v1 in this task.
5. Do not claim diffusion contribution from color_stats_v0.
6. If CUDA is needed, report the local PyTorch/CUDA status before running deep extractors.
7. Add tests for new extractor registry / metadata logic using tiny fake images only.
8. Run `pytest -q` and report exact result.

## Goal

Create a safe, extensible embedding-family benchmark for Camelyon17:

```text
color_stats_v0
image_resnet18_imagenet_v0 or image_resnet50_imagenet_v0
optional pathology encoder placeholder/scaffold if dependencies/weights are unavailable
```

The goal is to decide whether representation quality, not diffusion model capacity, is the current bottleneck.

## Step 1: Add embedding family registry

Refactor or extend:

```text
scripts/extract_camelyon17_patch_embeddings.py
```

Add a small registry:

```text
color_stats_v0
resnet18_imagenet_v0
resnet50_imagenet_v0
pathology_encoder_placeholder
```

For `color_stats_v0`, keep the current deterministic 37-feature extractor.

For ImageNet encoders:

```text
Use torchvision if available.
Do not download weights unless the user explicitly passes --allow-download-weights.
If weights are missing, fail clearly or allow --weights-path for local weights.
If running CPU-only, support --limit and --batch-size small values.
```

Do not require real model weights in tests.

## Step 2: Add feature versioning

Every embedding output should include a manifest:

```text
embedding_manifest.json
```

Fields:

```text
extractor_name
model_family
weights_source
weights_downloaded: bool
feature_dim
sample_count
image_root
metadata_csv
limit
batch_size
device
created_by_script
```

The feature CSV converter should preserve extractor metadata in its conversion summary when available.

## Step 3: Add multi-embedding benchmark script

Add:

```text
scripts/run_camelyon17_embedding_family_benchmark.py
```

It should accept one or more prepared feature CSVs:

```bash
python scripts/run_camelyon17_embedding_family_benchmark.py \
  --feature-csv color_stats_v0=data/camelyon17/features/camelyon17_color_stats_v0_features.csv \
  --feature-csv resnet18=data/camelyon17/features/camelyon17_resnet18_features.csv \
  --target-center 2 \
  --group-column patient_or_group_id \
  --setting zero_shot \
  --output-dir results/camelyon17_embedding_family_benchmark
```

For each feature CSV, run:

```text
identity
source_standardize
center_mean
ridge_denoising
latent_diffusion_v0
diffusion_placeholder
```

Write:

```text
embedding_family_summary.csv
embedding_family_summary.json
embedding_family_report.md
```

Report per embedding family:

```text
best statistical method / target_auc / source_val_auc
best ridge method / target_auc / source_val_auc
latent_diffusion_v0 target_auc / source_val_auc
claim gate status
paper_safe_split
uses_target_labels
```

## Step 4: Add source-drop guard to interpretation

The dry run showed latent_diffusion_v0 had higher target AUC than ridge by only 0.004 but source validation AUC dropped substantially.

Add interpretation logic:

```text
flag diffusion_source_drop_large if source_drop_vs_no_harmonization > 0.01
flag diffusion_margin_insufficient if gain_vs_best_non_diffusion < 0.01
flag representation_needs_upgrade if color_stats_v0 and claim gate fails
```

Do not alter the claim gate margin.

## Step 5: Add tests

Add or update tests:

```text
tests/test_camelyon17_embedding_extractors.py
tests/test_camelyon17_embedding_family_benchmark.py
```

Use tiny fake images and fake feature CSVs only. Required checks:

```text
color_stats_v0 still extracts 37 features
embedding manifest is written
missing torchvision/weights gives clear nonzero/diagnostic error or skipped status
embedding family benchmark summarizes multiple fake feature CSVs
paper_safe_split is propagated
uses_target_labels is false
source-drop and margin-insufficient flags are present
```

## Step 6: Required commands

Run tests:

```bash
python -m py_compile scripts/extract_camelyon17_patch_embeddings.py scripts/run_camelyon17_embedding_family_benchmark.py
pytest -q
```

If local CUDA and weights are available, optionally run a limited smoke extraction:

```bash
python scripts/extract_camelyon17_patch_embeddings.py \
  --metadata-csv data/camelyon17/metadata.csv \
  --image-root H:/MedHarmDiff_data/wilds/camelyon17_v1.0 \
  --output-npz data/camelyon17/embeddings/camelyon17_resnet18_imagenet_v0_limit1000.npz \
  --output-dir results/camelyon17_resnet18_limit1000 \
  --extractor resnet18_imagenet_v0 \
  --limit 1000
```

Do not require this optional run for tests.

## Decision rule after this task

### Continue toward diffusion only if:

```text
A stronger embedding family produces a paper-safe benchmark where latent_diffusion_v0 has at least +0.01 target AUC over best non-diffusion baseline and source drop is within tolerance.
```

### Pause diffusion if:

```text
ridge/statistical baselines still dominate across stronger embeddings;
or latent_diffusion_v0 only gains target AUC while source validation collapses;
or no stronger embeddings can be generated locally.
```

## Final response template

Report:

```text
Files added/modified
Commands run
Embedding families supported
Whether CUDA/PyTorch was available
Whether any optional real extraction was run
Color_stats baseline recap
Stronger embedding benchmark result if run
Claim gate status by embedding family
Whether diffusion development remains paused
pytest result
Next recommended task
```

## Do not do

Do not commit embeddings or real data.
Do not use target labels for tuning.
Do not implement neural diffusion v1.
Do not claim paper contribution from a single embedding run.
