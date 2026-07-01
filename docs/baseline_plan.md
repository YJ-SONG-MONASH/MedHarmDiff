# Baseline Plan

## Required Baselines

### 1. No Harmonization

Train task model on raw source-center features. Test directly on target center.

### 2. ComBat

Classical harmonization baseline for feature/radiomics/tabular representations.

Important variants:

```text
ComBat without labels
ComBat with clinical covariates, if allowed
ComBat source-only vs source+unlabeled-target fitting
```

### 3. CORAL

Align covariance statistics between source and target feature distributions.

Variants:

```text
shallow CORAL on fixed features
Deep CORAL in encoder representation
```

### 4. MMD

Use maximum mean discrepancy to align source and target representation distributions.

### 5. Adversarial Domain Adaptation

Domain classifier tries to predict site; encoder/harmonizer tries to fool it.

### 6. VAE Harmonization

Generative baseline that encodes data into a latent variable and reconstructs a site-reduced representation.

### 7. Diffusion Harmonization

Main method.

### 8. Diffusion + Clinical Preservation + Domain Confusion

Expected strongest variant.

## Baseline Fairness Rules

```text
same train/dev/test split
same encoder backbone where possible
same clinical task head capacity where possible
same access to target-unlabeled data within a setting
same hyperparameter budget or explicitly reported differences
```

## Result Interpretation

A method that only lowers site classification but reduces clinical AUC is not successful.

A method that only improves over no harmonization but not over ComBat/CORAL/MMD/DANN is not enough for a strong diffusion paper.

A method that improves only when target labels are used must be framed as supervised adaptation, not unsupervised harmonization.
