# Codex Next Step Prompt: Real-Data Pilot and Ridge-Dominance Diagnosis

Current result from the nonlinear multiseed run:

```text
120 runs = 8 settings x 5 seeds x 3 target centers
nonlinear_settings_create_headroom = true
diffusion_v0_win_rate_vs_statistical = 0.0
diffusion_v0_win_rate_vs_ridge = 0.0
claim_gate_pass_rate = 0.0
target_labels_used_count = 0
best overall target AUC: ridge_denoising 0.7148, latent_diffusion_v0 0.6899
pytest -q -> 55 passed
```

Conclusion: do not implement neural/optional-torch diffusion yet. The next task is analysis and real-data readiness, not another model.

## Goals

1. Explain why ridge/statistical baselines still dominate.
2. Select the first realistic public-domain benchmark candidate.
3. Define a safe feature-CSV contract for real data.
4. Decide whether diffusion development should continue or pause.

## Tasks

### 1. Ridge dominance analysis

Add:

```text
scripts/analyze_ridge_dominance.py
tests/test_ridge_dominance_analysis.py
```

Input should be an existing multiseed result directory containing:

```text
per_run_metrics.csv
aggregate_summary.json
```

Output:

```text
ridge_dominance_summary.json
ridge_dominance_report.md
```

Compute, per stress setting:

```text
best statistical target_auc
best ridge target_auc
best diffusion_v0 target_auc
ridge_minus_diffusion
statistical_minus_diffusion
site_auc_after comparison
source_val_auc comparison
claim gate counts
confounding counts
```

Add flags:

```text
ridge_dominates_all_low_confounding_settings
statistical_dominates_all_low_confounding_settings
diffusion_improves_site_but_loses_clinical
diffusion_loses_site_and_clinical
headroom_without_diffusion_win
synthetic_probably_too_linear_for_diffusion
```

### 2. Real-data pilot plan

Add:

```text
docs/real_dataset_pilot_plan.md
```

Use `docs/real_dataset_scout.md` as starting point. Do not download data.

Recommended structure:

```text
first recommended pilot
backup pilot
why suitable
why risky
minimum metadata needed
minimum feature file needed
expected split strategy
expected baseline table
what would count as diffusion-relevant headroom
source_verification_needed where appropriate
```

Current likely priority:

```text
1. Camelyon17-WILDS style domain-generalization benchmark if accessible.
2. IXI-style scanner/hospital harmonization mechanics if clinical labels are not required.
3. CXR cross-dataset setup only if embedding extraction is easy and metadata is clear.
```

### 3. Real feature CSV contract

Add:

```text
docs/real_feature_csv_contract.md
```

Required fields:

```text
sample_id
domain_id or center_id
label
feature_0 ... feature_N
```

Optional metadata fields:

```text
patient_or_group_id
split_group
scanner
protocol
device
collection_id
site_id
view_position
```

Clarify that automatic feature selection should use `feature_*` columns by default when they exist, not numeric metadata. If loader behavior needs a safety update, add tests.

### 4. Tests

Run:

```bash
pytest -q
```

Required coverage:

```text
ridge dominance script writes json and md
flags exist in json
confounded settings are not counted as diffusion wins
feature CSV contract / loader safety is tested if loader changes
```

## Decision rule

Do not continue to neural diffusion unless at least one is true:

```text
a real dataset pilot is ready with clear domain metadata and a feasible feature file plan;
or ridge analysis reveals a specific fixable failure mode for diffusion v0.
```

If ridge dominates all low-confounding settings and no real pilot is ready, pause diffusion modeling and complete data access/metadata work first.

## Final response

Report:

```text
files added/modified
commands run
ridge dominance conclusion
first real-data pilot recommendation
feature CSV contract summary
whether diffusion development should continue or pause
pytest result
next recommended task
```
