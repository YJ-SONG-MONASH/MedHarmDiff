# Experiment Protocol

## Core Setting

The primary setting is leave-one-center-out evaluation.

For centers `{A, B, C}`:

```text
Train: A + B
Validation: held-out split from A + B
Test: C
```

Repeat for:

```text
Train: A + C -> Test: B
Train: B + C -> Test: A
```

When more centers are available, perform leave-one-site-out cross-validation.

## Experimental Settings

### Setting 1: Zero-shot unseen-center generalization

```text
Train labels: source centers only
Target center data: not visible during training or harmonization fitting
Test labels: target center only for evaluation
```

This is the strongest deployment setting.

### Setting 2: Unsupervised target-domain adaptation

```text
Train labels: source centers only
Target center inputs: visible without labels
Target center labels: used only for evaluation
```

This is easier and should be reported separately.

### Setting 3: Supervised target fine-tuning

```text
Small labeled sample from target center is available
```

This is not the main setting unless explicitly framed.

## Baseline Table

| Method | Uses target unlabeled data? | Uses target labels? | Notes |
| --- | --- | --- | --- |
| No harmonization | no | no | Raw baseline |
| ComBat | optional | no | Classical site-effect adjustment |
| CORAL | yes for adaptation | no | Covariance alignment |
| MMD | yes for adaptation | no | Distribution matching |
| Adversarial domain adaptation | yes or source-only variant | no | Domain confusion baseline |
| VAE harmonization | optional | no | Generative latent baseline |
| Diffusion harmonization | optional | no | Main method |

## Required Metrics

### Clinical task metrics

Classification:

```text
AUC
F1
sensitivity
specificity
balanced accuracy
ECE / calibration
```

Segmentation:

```text
Dice
Hausdorff distance
lesion volume correlation
```

Regression:

```text
MAE
RMSE
Pearson/Spearman correlation
```

### Harmonization / site-removal metrics

```text
site classifier AUC / accuracy
MMD distance
CORAL distance
feature distribution distance
center clustering silhouette
UMAP/PCA site mixing visualization
```

### Clinical preservation metrics

```text
source-center task metric preservation
label association before/after harmonization
anatomy/lesion/radiomics consistency if available
```

## Claim Gate

A diffusion method passes only if:

```text
target clinical metric improves over strongest non-diffusion baseline
site classifier performance decreases
source validation performance drop <= allowed tolerance
clinical signal preservation passes
no evidence of label-site confounding driving the result
```

Default tolerance:

```text
source performance drop <= 0.01 AUC or task-specific equivalent
```

## Reporting Template

| Method | Target clinical ↑ | Source clinical ↑ | Site AUC ↓ | MMD ↓ | Calibration ↓ | Claim status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| No harmonization | | | | | | baseline |
| ComBat | | | | | | baseline |
| CORAL | | | | | | baseline |
| MMD | | | | | | baseline |
| DANN | | | | | | baseline |
| VAE | | | | | | baseline |
| Diffusion | | | | | | candidate |
| Diffusion + preservation | | | | | | candidate |
