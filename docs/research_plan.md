# Research Plan

## Working Title

**Clinical-Signal-Preserving Diffusion Harmonization for Multi-Center Medical Generalization**

## Motivation

Medical models often fail when moved across hospitals, departments, scanners, protocols, or devices. A model trained on centers A+B may underperform on center C because representation differences encode acquisition/site artifacts rather than clinical content.

Diffusion models may be useful if we treat center/device/protocol shift as a structured noise or style perturbation and learn a conditional denoising map into a canonical, site-invariant representation.

## Main Hypothesis

A site-conditioned diffusion harmonizer can improve held-out-center clinical generalization while reducing center predictability and preserving disease/anatomical signal.

## Primary Research Question

```text
Train centers A+B, test center C:
Does diffusion harmonization outperform no harmonization, ComBat, CORAL, MMD, adversarial domain adaptation, and VAE harmonization?
```

## Subquestions

1. Does diffusion reduce site/device/protocol information in the representation?
2. Does diffusion preserve clinical signal?
3. Does diffusion improve held-out-center performance?
4. Does diffusion still help after controlling for label-site confounding?
5. Is latent-level harmonization enough, or is image-level harmonization required?
6. Is the gain due to diffusion specifically, or due to auxiliary losses such as domain confusion and task preservation?

## Recommended First Scope

Start with **feature-level or latent-level harmonization**.

Possible data types:

```text
radiomics features
pretrained imaging embeddings
tabular clinical + device features
intermediate CNN/ViT features
```

Avoid starting with full 3D image diffusion unless a clean dataset and compute budget are available.

## Minimum Viable Experiment

Dataset requirements:

```text
>= 3 centers or domains
center/site metadata
clinical label or downstream target
train/dev/test split by patient and center
```

Protocol:

```text
Train: centers A+B
Validation: source-center dev from A+B
Test: held-out center C
```

Baselines:

```text
no_harmonization
combat
coral
mmd
domain_adversarial
vae_harmonization
diffusion_harmonization
diffusion_plus_clinical_preservation
diffusion_plus_clinical_preservation_plus_domain_confusion
```

## Success Criteria

Diffusion contribution is claimable only if:

```text
held-out-center clinical metric > strongest non-diffusion baseline
site classifier performance decreases after harmonization
source validation performance does not collapse
clinical signal preservation metrics remain stable or improve
label-site confounding audit does not invalidate the interpretation
```

## Main Risks

### 1. Label-site confounding

If center C has a different disease distribution from A+B, a harmonizer may mistakenly remove real disease signal as if it were site noise.

Mitigations:

```text
stratified analysis
site-label association audit
negative control labels
clinical preservation loss
subgroup metrics
```

### 2. Over-harmonization

The harmonizer may remove pathology or clinically relevant variation.

Mitigations:

```text
clinical task loss
content consistency loss
segmentation/radiomics consistency
lesion/anatomy preservation metrics
```

### 3. Weak baselines

A diffusion method only matters if it beats strong baselines.

Mitigations:

```text
include ComBat, CORAL, MMD, adversarial, VAE
perform ablations
report source and target metrics
```

## Long-Term Paper Claim

A strong final claim would look like:

> We propose a clinical-signal-preserving site-conditioned diffusion harmonizer for multi-center medical representation shift. Across leave-one-center-out experiments, our method improves held-out-center clinical performance over ComBat, CORAL, MMD, adversarial adaptation, and VAE baselines while reducing center predictability and preserving clinical signal.
