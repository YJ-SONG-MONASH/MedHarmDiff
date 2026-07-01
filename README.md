# MedHarmDiff

**Clinical-Signal-Preserving Diffusion Harmonization for Multi-Center Medical Data**

MedHarmDiff is a research scaffold for testing whether diffusion models can reduce multi-center representation shift while preserving clinically predictive signal.

The core question is:

> Can a site-/device-conditioned diffusion harmonizer improve cross-center generalization, e.g. train on centers A+B and test on unseen center C, better than no harmonization, ComBat, MMD, CORAL, adversarial domain adaptation, and VAE-based harmonization?

## Current Research Hypothesis

Multi-center medical data often contains two entangled factors:

```text
observed representation = clinical content + center / scanner / department / device style + noise
```

A useful harmonizer should remove center/device/protocol style while preserving clinical content. Diffusion is potentially suitable because harmonization can be framed as conditional denoising:

```text
source-center latent representation -> canonical / site-invariant latent representation
```

## Main Experimental Protocol

Start with leave-one-center-out evaluation:

```text
Train centers: A + B
Validation: held-out split from A + B
Test center: C
```

Rotate target center when more than three centers are available.

Compare:

1. No harmonization
2. ComBat
3. CORAL / Deep CORAL
4. MMD / domain adaptation
5. Adversarial domain adaptation
6. VAE harmonization
7. Diffusion harmonization
8. Diffusion + clinical preservation + domain confusion

## Claim Gate

Diffusion can be claimed as a paper contribution only if it satisfies all of:

```text
held-out center performance improves over strongest non-diffusion baseline
site predictability decreases
clinical task signal is preserved or improves
source-center performance does not collapse
label/site confounding audit passes or is explicitly controlled
```

## Repository Structure

```text
configs/                 Experiment configs
docs/                    Research plans, method details, agent prompts
src/medharmdiff/          Python package skeleton
  data_schema.py          Data schema definitions
  metrics.py              Clinical and harmonization metrics
  claim_gate.py           Gate for deciding whether diffusion contribution is claimable
  baselines.py            Baseline interface placeholders
  diffusion_harmonizer.py Diffusion harmonizer skeleton
  protocol.py             Leave-center-out split helpers
scripts/                 CLI entry points / future runners
tests/                   Unit tests
experiments/             Experiment notes and run manifests
data/                    Ignored data placeholder
results/                 Ignored results placeholder
```

## Quick Start

```bash
python -m pip install -e .[dev]
pytest -q
```

## Safe Paper Position for Now

This repository is a scaffold. It does **not** yet prove that diffusion harmonization works. The first goal is to build a fair, reproducible cross-center benchmark and decide whether diffusion can beat strong harmonization baselines without erasing clinical signal.
