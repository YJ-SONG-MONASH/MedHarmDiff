# Real Dataset Pilot Plan

This plan uses `docs/real_dataset_scout.md` as the starting point. It does not
download data and does not claim diffusion works.

## First Recommended Pilot

**Camelyon17-WILDS / CAMELYON17 patch-level domain generalization.**

Sources:

- WILDS dataset page: https://wilds.stanford.edu/datasets/
- CAMELYON17 data page: https://camelyon17.grand-challenge.org/Data/

### Why Suitable

- The domain is hospital/site, which directly matches the MedHarmDiff
  leave-center-out question.
- WILDS frames Camelyon17 as domain generalization to data from an unseen
  hospital, with binary tumor-region labels and hospital domain IDs.
- The benchmark is already patch-level, so a feature-level pilot can be built
  from pretrained pathology image embeddings without full-slide diffusion.
- WILDS reports the Camelyon17 benchmark as public-domain/CC0 in its dataset
  metadata, making it the cleanest first reproducibility target.

### Why Risky

- Patch-level pathology embeddings may already make the task mostly linearly
  separable, which could reproduce the current ridge-dominance outcome.
- Slide/patient grouping must be respected; patch-level random splitting would
  create leakage and invalidate the held-out-domain claim.
- If stain variation is already normalized by preprocessing or a strong
  foundation encoder, harmonization headroom may be small.

### Minimum Metadata Needed

- `sample_id`: stable patch identifier.
- `patient_or_group_id` or slide ID: grouping key to prevent leakage.
- `center_id` or `domain_id`: hospital/domain ID.
- `label`: binary tumor-region label.
- `split_group`: source train/source validation/held-out target assignment if
  using an existing benchmark split.
- Optional: slide ID, tissue type, scanner/staining metadata if available.

### Minimum Feature File Needed

Use the contract in `docs/real_feature_csv_contract.md`:

```text
sample_id,center_id,label,patient_or_group_id,split_group,feature_0,...,feature_N
```

Feature extraction should start with frozen embeddings from a documented
pathology encoder. Raw image files and extracted embeddings stay under ignored
`data/`.

### Expected Split Strategy

- Primary: follow the WILDS official train/validation/test domain split if the
  split metadata is available.
- Secondary: leave-one-hospital-out rotation using hospital/domain ID, with
  patient/slide grouping enforced inside source train/validation splits.
- Every real benchmark run must write a split audit with train/validation/test
  row counts, group sets, cross-split group overlaps, warnings, and a
  `paper_safe_split` flag.
- A Camelyon17 result is paper-safe only when a patient, slide, or equivalent
  grouping key is present and no group overlaps across train/validation/test.
- If official `split_group` metadata is used, document the exact configured
  train, validation, and target-test split values in the run command and
  `run_config.json`.
- Target labels are evaluation-only. Target unlabeled inputs may be used only
  under an explicitly named `target_unlabeled` adaptation setting.

### Expected Baseline Table

| Method | Role | Uses target labels? | Uses target unlabeled? |
| --- | --- | --- | --- |
| identity | no harmonization | no | no |
| source_standardize | simple source scaling | no | no |
| center_mean / ComBat-like | statistical baseline | no | optional |
| coral | covariance alignment | no | yes in target-unlabeled setting |
| mmd_mean_alignment | distribution matching | no | yes in target-unlabeled setting |
| ridge_denoising | learned non-diffusion denoising | no | no |
| latent_diffusion_v0 | current diffusion-style baseline | no | no |
| diffusion_placeholder | placeholder control only | no | no |

### Diffusion-Relevant Headroom

Diffusion development should resume only if the pilot shows at least one
low-confounding target domain where:

- best non-diffusion performance leaves measurable headroom;
- site/domain AUC can be reduced without lowering target clinical AUC;
- source validation AUC drop stays within the claim-gate tolerance;
- target labels remain evaluation-only;
- ridge/statistical baselines do not explain the gain.

### Source Verification Needed

- Confirm the exact WILDS/Camelyon17 file fields exposed by the installed WILDS
  version before writing a converter.
- Confirm whether patient/slide IDs are available in the feature export path.
- Confirm any local storage and license requirements before downloading data.

## Backup Pilot

**IXI MRI scanner/hospital harmonization mechanics.**

Source: https://brain-development.org/ixi-dataset/

### Why Suitable

- IXI provides MRI data from three London hospitals/scanner configurations,
  which is useful for domain-removal and scanner-shift mechanics.
- The data page lists multiple MRI modalities and scanner setups, enabling a
  clean feature-level harmonization test from radiomics or image embeddings.
- It is lighter-weight than a large chest X-ray corpus for an initial metadata
  audit.

### Why Risky

- IXI is healthy-subject MRI and does not provide a disease-label task by
  default, so it cannot support the main clinical generalization claim alone.
- A proxy task such as age/sex prediction or anatomy-derived metrics must be
  framed as a mechanics pilot, not as evidence of clinical benefit.

### Minimum Metadata Needed

- `sample_id`
- `domain_id` or `center_id` from hospital/scanner.
- `label` only if a defensible proxy task is chosen.
- `patient_or_group_id`
- scanner field and MRI modality/protocol.

### Diffusion-Relevant Headroom

IXI can justify better harmonization diagnostics, but not a paper-level
diffusion claim unless paired with a clinically meaningful downstream target.

## CXR Route, Later

MIMIC-CXR-JPG and CheXpert are useful for chest X-ray embedding experiments, but
they are higher-friction for this immediate pilot:

- MIMIC-CXR-JPG has structured labels and metadata, but PhysioNet access is
  credentialed and governed by a DUA.
- CheXpert is a large Stanford Health Care dataset with report-derived labels,
  but a MIMIC-vs-CheXpert setup is cross-dataset transfer rather than a clean
  within-dataset leave-center-out study.

Sources:

- https://www.physionet.org/content/mimic-cxr-jpg/
- https://aimi.stanford.edu/datasets/chexpert-chest-x-rays

## Current Decision

Do not continue to neural diffusion yet. The next concrete task is a
Camelyon17-WILDS metadata probe and converter plan that emits only a feature
CSV contract sample, not raw data.

## Camelyon17-WILDS Pilot Implementation Status

The pilot is now scaffolded for local, ignored data only:

- Metadata probe script:
  `scripts/probe_camelyon17_wilds_metadata.py`
- Embedding-to-feature CSV converter:
  `scripts/convert_camelyon17_embeddings_to_feature_csv.py`
- Feature CSV contract:
  `docs/real_feature_csv_contract.md`
- Pure metadata/conversion helpers:
  `src/medharmdiff/real_pilots/camelyon17_wilds.py`

The metadata probe can read a local metadata CSV and report row counts,
hospital/domain counts, label prevalence by center, split counts, grouping-key
availability, and candidate leakage risks. The optional WILDS package path is
explicitly gated behind `--use-wilds` and does not download by default.

The converter can join local metadata with local precomputed embeddings in CSV
or NPZ format and write a MedHarmDiff feature CSV with:

```text
sample_id,center_id,label,patient_or_group_id,split_group,feature_0,...,feature_N
```

Remaining blockers before a real benchmark run:

- Confirm the exact WILDS/Camelyon17 metadata fields exposed by the local WILDS
  version or exported CSV.
- Confirm patient/slide grouping IDs are available and can prevent leakage.
- Confirm official `split_group` values or choose a group-safe source split
  strategy before running the benchmark.
- Generate frozen patch embeddings under ignored `data/`.
- Run the feature CSV contract validator before any benchmark.
- Run `scripts/run_camelyon17_feature_benchmark.py` only after the metadata
  audit passes; it requires group-safe split metadata by default.
