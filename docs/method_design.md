# Method Design

## High-Level Architecture

```text
raw input x
  -> encoder E
  -> latent representation h
  -> site-conditioned diffusion harmonizer H
  -> harmonized latent h*
  -> task head T
  -> clinical prediction y
```

## Why Latent-Level First

Latent-level harmonization is the recommended starting point because:

```text
it is cheaper than 2D/3D image diffusion
it is easier to compare with ComBat/CORAL/MMD
it reduces risk of directly hallucinating or erasing pathology
it can be tightly tied to downstream clinical task performance
```

## Diffusion Formulation

Let:

```text
h = encoder representation
s = site / scanner / protocol / department label
y = clinical label
```

Forward process:

```text
q(h_t | h_0) = add Gaussian noise to latent h_0
```

Reverse process:

```text
epsilon_theta(h_t, t, s_source, s_target or canonical)
```

The harmonizer learns to denoise site-shifted latent features into a canonical/site-invariant representation.

## Loss Function

Recommended objective:

```text
L = L_diffusion
  + lambda_task * L_clinical
  + lambda_domain * L_domain_confusion
  + lambda_content * L_content_preservation
  + lambda_recon * L_reconstruction_optional
```

### Diffusion Loss

```text
L_diffusion = || epsilon - epsilon_theta(h_t, t, site_condition) ||^2
```

### Clinical Preservation Loss

Classification:

```text
L_clinical = CrossEntropy(T(h*), y)
```

Segmentation / regression should use task-appropriate losses.

### Domain Confusion Loss

A site classifier should fail to identify the center from harmonized features.

```text
L_domain = adversarial site classifier loss
```

### Content Preservation Loss

Prevent the harmonizer from removing disease/anatomy signal.

Examples:

```text
|| content_encoder(h*) - content_encoder(h) ||
contrastive clinical consistency
radiomics consistency
segmentation consistency
```

## Model Variants

### Variant A: Latent DDPM Harmonizer

```text
E frozen or jointly trained
DDPM denoises h
T predicts y from h*
```

### Variant B: Conditional Score Matching Harmonizer

```text
score model estimates gradient toward canonical representation
```

### Variant C: Diffusion Residual Harmonizer

```text
h* = h + delta_theta(h, site)
```

where `delta_theta` is produced by a diffusion denoiser.

### Variant D: Image-Level Harmonizer, Later Stage

Only after latent-level evidence is strong.

Required extra safety metrics:

```text
SSIM / PSNR
segmentation consistency
lesion preservation
radiomics preservation
```

## Key Ablations

```text
without diffusion
without task preservation loss
without domain confusion
without content preservation
without site condition
latent vs image level
source-only vs target-unlabeled setting
```

## Failure Modes

```text
over-harmonization removes clinical signal
site-label confounding creates false gains
site classifier drops because representation collapses
harmonizer improves source but not target
model needs target unlabeled data but claim says zero-shot
```
