# Codex Next Step Prompt: Camelyon17 Decision Report and Diffusion Pause

You are taking over `MedHarmDiff` on branch `codex/mvp-benchmark` after the full Camelyon17 ResNet18 benchmark.

Current reported real-data results:

```text
paper_safe_split = true
uses_target_labels = false for all methods
group_overlap_train_val = []
group_overlap_train_test = []
group_overlap_val_test = []
confounding_status = pass
pytest -q -> 82 passed
```

Full feature-family results:

```text
color_stats_v0:
  best statistical target_auc = 0.8853
  ridge target_auc            = 0.8991
  latent diffusion target_auc = 0.8975
  claim_gate                  = baseline_not_beaten

resnet18_imagenet_v0:
  best statistical target_auc = 0.9685
  ridge target_auc            = 0.9685
  latent diffusion target_auc = 0.9665
  claim_gate                  = baseline_not_beaten
```

Full-scale engineering fix:

```text
latent_diffusion_v0 now caps training pairs with max_training_pairs=100_000
uses float32 preallocation
prevents 18.5GB training-pair allocation on full benchmark
```

Decision:

```text
Do not implement neural/optional-torch diffusion v1 now.
Diffusion route is paused because strong non-diffusion baselines dominate on paper-safe real Camelyon17 features.
```

## Goal

Create a clean, versioned project decision report that records:

```text
1. what was built,
2. what was tested,
3. what the real Camelyon17 results show,
4. why diffusion is paused,
5. what future work would be needed before resuming diffusion.
```

This is a documentation/consolidation task, not a modeling task.

## Hard rules

1. Do not commit real data, embeddings, feature CSVs, or result directories.
2. Do not modify model behavior unless a bug is found.
3. Do not implement new diffusion models.
4. Do not claim diffusion contribution.
5. If using local result artifacts, summarize only safe aggregate metrics in docs.
6. Run `pytest -q` after any code change. If only docs change, still run a lightweight check if practical.

## Required docs

Add:

```text
docs/camelyon17_pilot_decision_report.md
```

Update if appropriate:

```text
README.md
docs/real_dataset_pilot_plan.md
```

## Report contents

The report should include these sections:

### 1. Executive summary

State clearly:

```text
MedHarmDiff built a paper-safe multi-center harmonization benchmark and ran a real Camelyon17-WILDS feature-level pilot. Diffusion v0 did not outperform strong non-diffusion baselines under the claim gate, so diffusion modeling is paused.
```

### 2. Pipeline built

List:

```text
synthetic benchmark
nonlinear stress benchmark
claim gate
ridge_denoising baseline
latent_diffusion_v0
data scout
Camelyon17 metadata probe
embedding converter
feature CSV contract
group-safe split audit
Camelyon17 benchmark wrapper
embedding-family benchmark
result summarizer
```

### 3. Synthetic findings

Summarize:

```text
multi-seed synthetic: diffusion_v0 win rate 0.0 vs statistical/ridge
nonlinear stress created headroom but diffusion still had no gate-aware wins
ridge/statistical baselines dominated
```

### 4. Real Camelyon17 dry run

Summarize color_stats_v0 and ResNet18 results exactly as above.

Include safety conditions:

```text
paper_safe_split=true
no target label training
group overlaps empty
confounding pass
```

### 5. Claim gate interpretation

Explain:

```text
latent_diffusion_v0 did not exceed best non-diffusion baseline by required +0.01 target AUC margin.
In earlier color_stats dry run, latent diffusion had small target gain but source validation drop was too large.
In full ResNet18 run, non-diffusion baselines dominated outright.
```

### 6. Why diffusion is paused

State:

```text
The current failure is not a pipeline failure.
It is an empirical result: current feature-level Camelyon17 setting is well handled by source_standardize/center_mean/ridge.
Increasing diffusion model capacity is not justified without new evidence of diffusion-relevant headroom.
```

### 7. What would justify resuming diffusion

Require:

```text
paper-safe real benchmark
latent diffusion target_auc >= best_non_diffusion + 0.01
source validation drop within claim-gate tolerance
site/domain metric improves
confounding_status pass
result replicated across target centers / feature families
```

### 8. Recommended next directions

Suggest non-diffusion next steps:

```text
1. Try stronger pathology-specific frozen encoders only as representation study.
2. Try a different multi-center dataset where acquisition/style shift is stronger.
3. Keep diffusion paused until benchmark evidence supports it.
4. Consider writing this as a negative/benchmarking result or using it to justify pivot.
```

## Optional result table

Include a markdown table:

| Feature family | Best statistical AUC | Ridge AUC | Latent diffusion AUC | Claim gate | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| color_stats_v0 | 0.8853 | 0.8991 | 0.8975 | baseline_not_beaten | pause diffusion |
| resnet18_imagenet_v0 | 0.9685 | 0.9685 | 0.9665 | baseline_not_beaten | pause diffusion |

## Tests / validation

Run:

```bash
pytest -q
```

If only docs changed and runtime is a concern, at least run:

```bash
python -m py_compile scripts/summarize_feature_benchmark_result.py scripts/run_camelyon17_embedding_family_benchmark.py
```

## Final response template

Report:

```text
Files added/modified
Commands run
Decision report location
Final diffusion decision
Whether any code changed
pytest result
Recommended next research direction
```

## Do not do

Do not add real result CSVs or embeddings.
Do not implement diffusion v1.
Do not tune latent diffusion on Camelyon17 target labels.
Do not claim paper contribution for diffusion.
