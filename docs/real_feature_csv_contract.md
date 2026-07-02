# Real Feature CSV Contract

This contract defines the feature-level input format for real-data pilots. It
is intentionally conservative: metadata can be present for audits and grouping,
but automatic model features must come only from `feature_*` columns by default.

## Required Columns

| Column | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Unique row identifier. Must be stable across reruns. |
| `center_id` or `domain_id` | string | Hospital, scanner, site, source collection, or benchmark domain. |
| `label` | integer/binary for MVP | Evaluation label for the clinical or proxy task. |
| `feature_0 ... feature_N` | numeric | Extracted model/radiomics/tabular features used by harmonizers. |

The current loader expects `center_id`. If a real source uses `domain_id`, the
converter should either rename it to `center_id` or call the loader with
`center_column="domain_id"`.

## Optional Metadata Columns

These fields are allowed and useful for audits, but must not be automatically
treated as model features:

```text
patient_or_group_id
split_group
scanner
protocol
device
collection_id
site_id
view_position
slide_id
study_id
subject_id
modality
acquisition_date_or_bucket
```

Numeric metadata such as `site_id`, scanner code, patient ID, or acquisition
bucket must stay out of automatic feature selection. If a metadata field is
scientifically justified as an input feature, the converter should copy it into
an explicit `feature_*` column and document that decision.

## Automatic Feature Selection Rule

When `feature_columns="auto"`, MedHarmDiff should use only columns with the
configured feature prefix, defaulting to:

```text
feature_
```

Examples included by default:

```text
feature_0
feature_1
feature_127
```

Examples excluded by default:

```text
patient_or_group_id
scanner
site_id
oracle_clinical_latent_0
oracle_site_style_0
```

This rule protects against accidental leakage from numeric metadata.

## Split and Leakage Requirements

- `sample_id` must be unique.
- If multiple rows come from the same patient, slide, study, or subject, include
  `patient_or_group_id`.
- Source train/source validation splits must not split one patient/slide/study
  across both sides.
- Target-center labels are evaluation-only in `zero_shot` and
  `target_unlabeled` settings.
- `split_group` may encode source train/source validation/target test if using
  an official benchmark split.

## Minimum Camelyon17-WILDS Feature CSV

```text
sample_id,center_id,label,patient_or_group_id,split_group,feature_0,...,feature_N
```

Recommended mappings:

- `center_id`: WILDS/Camelyon17 hospital/domain ID.
- `label`: binary tumor-region label.
- `patient_or_group_id`: slide or patient grouping key if exposed by the data
  loader or converter.
- `split_group`: official WILDS split when available.

## Minimum IXI Mechanics Feature CSV

```text
sample_id,center_id,label,patient_or_group_id,scanner,protocol,feature_0,...,feature_N
```

For IXI, `label` must be a clearly named proxy target such as age bucket or sex
only if that target is available and ethically appropriate. IXI should be
reported as scanner/hospital harmonization mechanics unless a real clinical
label is introduced.

## Validation Checklist

- CSV has at least three domains/centers.
- Labels are binary for the current MVP runner.
- Feature columns are numeric and non-null.
- Metadata columns are not included in auto feature selection.
- Patient/slide/study grouping is available before any result is described as
  patient-safe.
- License and data access terms are recorded outside the CSV.
