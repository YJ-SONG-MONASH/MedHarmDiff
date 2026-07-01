# Codex Initial Research Prompt

You are taking over a new repository: `MedHarmDiff`.

Your task is to turn this scaffold into a reproducible research package for diffusion-based harmonization of multi-center medical data.

## Goal

Test whether diffusion harmonization can improve cross-center clinical generalization while preserving clinical signal.

Core setting:

```text
Train centers A+B
Test center C
```

Compare:

```text
no harmonization
ComBat
CORAL
MMD
adversarial domain adaptation
VAE harmonization
diffusion harmonization
```

## First Tasks

1. Read:

```text
README.md
AGENTS.md
docs/research_plan.md
docs/experiment_protocol.md
docs/method_design.md
docs/claim_gate.md
```

2. Implement a feature-level benchmark runner:

```text
scripts/run_feature_level_benchmark.py
```

Required inputs:

```text
CSV with sample_id, center_id, label, feature columns
config YAML or CLI args
```

Required outputs:

```text
results/<run_name>/metrics_by_method.csv
results/<run_name>/claim_gate_summary.json
results/<run_name>/final_report.md
```

3. Implement or wrap baselines:

```text
identity / no harmonization
source standardization
ComBat if dependency available, otherwise documented placeholder
CORAL
MMD alignment
adversarial domain adaptation placeholder or simple implementation
VAE placeholder or simple implementation
```

4. Implement claim gate integration using:

```text
src/medharmdiff/claim_gate.py
```

5. Add tests.

## Important Rules

- Do not claim diffusion works unless the claim gate passes.
- Do not use target-center labels for training in zero-shot or unsupervised adaptation settings.
- Always separate site-removal metrics from clinical task metrics.
- Always audit source performance drop.
- Always compare to strongest non-diffusion baseline.

## Suggested First Milestone

A synthetic multi-center feature dataset generator for testing pipeline mechanics.

Create:

```text
src/medharmdiff/synthetic.py
tests/test_synthetic.py
```

The synthetic generator should allow:

```text
clinical signal strength
site shift strength
site-label confounding strength
number of centers
```

Then run:

```text
no harmonization vs CORAL/MMD/simple diffusion placeholder
```

The first goal is not to win. The first goal is to make the protocol hard to fool.
