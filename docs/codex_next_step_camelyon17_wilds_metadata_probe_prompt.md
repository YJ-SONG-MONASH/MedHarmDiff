# Codex Next Step Prompt: Camelyon17-WILDS Metadata Probe and Feature CSV Converter Scaffold

You are taking over `MedHarmDiff` on branch `codex/mvp-benchmark` after ridge-dominance diagnosis and real-data pilot planning.

Current project decision:

```text
Synthetic nonlinear benchmark created headroom, but diffusion_v0 still had zero wins.
Ridge/statistical baselines remain stronger.
Do not implement neural diffusion v1 yet.
First real-data pilot recommendation: Camelyon17-WILDS / CAMELYON17.
```

The next task is to make the real-data pilot executable **without downloading or committing data**.

## Goal

Create a Camelyon17-WILDS metadata probe and feature CSV converter scaffold that can run when a user has local WILDS/Camelyon17 data or precomputed embeddings under ignored `data/`.

Do not add real data. Do not add embeddings. Do not add raw images.

## Hard rules

1. Do not download datasets into git.
2. Do not commit any real data, embeddings, images, patches, labels beyond tiny synthetic fixtures.
3. Do not use target labels for training.
4. Do not claim diffusion works.
5. Do not implement new diffusion models in this task.
6. Keep all generated artifacts under ignored `data/` or `results/` unless they are tiny schema examples.
7. Add tests for metadata parsing/converter logic using tiny fake fixtures.
8. Run `pytest -q` and report exact result.

## Context

WILDS Camelyon17 is the preferred first pilot because it is a hospital-domain generalization task with binary tumor-region labels and patch-level images. The feature-level pilot should convert local metadata + local/precomputed embeddings into the MedHarmDiff real feature CSV contract.

Required contract is documented in:

```text
docs/real_feature_csv_contract.md
```

Minimum output columns:

```text
sample_id
center_id
label
patient_or_group_id
split_group
feature_0 ... feature_N
```

Metadata columns are allowed, but auto feature selection must only include `feature_*` columns.

## Step 1: Add Camelyon17 pilot module

Add:

```text
src/medharmdiff/real_pilots/camelyon17_wilds.py
src/medharmdiff/real_pilots/__init__.py
```

Implement pure-Python helpers that do not require downloading data:

```python
@dataclass
class Camelyon17MetadataRow:
    sample_id: str
    center_id: str
    label: int
    patient_or_group_id: str | None = None
    split_group: str | None = None
    image_path: str | None = None
    extra: dict[str, object] | None = None
```

Functions:

```python
normalize_camelyon17_metadata_row(raw: dict) -> Camelyon17MetadataRow
validate_camelyon17_rows(rows: list[Camelyon17MetadataRow]) -> dict
build_camelyon17_feature_csv_rows(metadata_rows, embeddings_by_sample_id) -> pandas.DataFrame
write_camelyon17_feature_csv(...)
```

The normalizer should be flexible because WILDS/Camelyon17 field names may vary by loader/export. Accept aliases such as:

```text
sample_id: sample_id, id, patch_id, image_id
center_id: center_id, domain_id, hospital, hospital_id, metadata_hospital
label: label, y, tumor_label
patient_or_group_id: patient_id, slide_id, patient_or_group_id, group_id
split_group: split, split_group
```

## Step 2: Add local metadata probe script

Add:

```text
scripts/probe_camelyon17_wilds_metadata.py
```

This script should support two modes:

### Mode A: Probe a local metadata CSV

```bash
python scripts/probe_camelyon17_wilds_metadata.py \
  --metadata-csv data/camelyon17/metadata.csv \
  --output-dir results/camelyon17_metadata_probe
```

### Mode B: Optional WILDS package probe

If `wilds` is installed, optionally try to instantiate the Camelyon17 dataset only when the user explicitly passes:

```bash
--use-wilds --wilds-root data/wilds
```

If `wilds` is not installed or data is unavailable, fail gracefully with a clear message. Do not auto-download unless the user explicitly passes an option such as `--allow-download`, and even then write only under ignored `data/`.

Probe output:

```text
metadata_probe_summary.json
metadata_probe_report.md
```

Report:

```text
row count
center/domain counts
label prevalence by center
split counts
patient_or_group_id availability
candidate leakage risks
whether metadata is ready for feature CSV conversion
```

## Step 3: Add embedding-to-feature CSV converter script

Add:

```text
scripts/convert_camelyon17_embeddings_to_feature_csv.py
```

Input:

```text
metadata CSV
embeddings file under ignored data/ path
```

Supported embedding formats for v0:

```text
CSV with sample_id + embedding columns
NPZ with arrays: sample_id and embeddings
```

Output:

```text
feature CSV following docs/real_feature_csv_contract.md
conversion_summary.json
conversion_report.md
```

Do not require real data in tests. Use tiny fake fixtures.

## Step 4: Add feature CSV contract validation helper

Add or extend:

```text
src/medharmdiff/io.py
```

Add a helper:

```python
validate_real_feature_contract(path, *, center_column="center_id") -> dict
```

It should report:

```text
sample_count
center_count
feature_count
label_prevalence_by_center
metadata_columns
feature_columns
has_patient_or_group_id
warnings
```

It should not train a model.

## Step 5: Add tests

Add:

```text
tests/test_camelyon17_wilds_pilot.py
tests/test_real_feature_contract.py
```

Use tiny fake metadata and fake embeddings. Required checks:

```text
metadata alias normalization works
missing center_id/label/sample_id is reported clearly
feature CSV rows include only feature_* as features
numeric metadata is excluded from auto features
probe script writes summary/report on fake metadata
converter writes feature CSV on fake metadata + fake embeddings
patient_or_group_id absence produces warning
label prevalence by center is computed
```

## Step 6: Documentation update

Update:

```text
docs/real_dataset_pilot_plan.md
```

Add a short section:

```text
Camelyon17-WILDS pilot implementation status
```

Include:

```text
metadata probe script path
embedding converter script path
feature CSV contract path
what remains blocked before running real benchmark
```

## Required commands

Run:

```bash
python -m py_compile scripts/probe_camelyon17_wilds_metadata.py scripts/convert_camelyon17_embeddings_to_feature_csv.py
pytest -q
```

If you include CLI tests, they must use tiny fake fixtures only.

## Success criteria

This task succeeds if:

```text
1. Camelyon17 metadata can be probed from a local fake/real metadata CSV.
2. Fake embeddings can be converted into a valid MedHarmDiff feature CSV.
3. The converter enforces feature_* only for model features.
4. Patient/slide grouping risks are reported.
5. No real data is committed.
6. pytest -q passes.
```

## Do not do yet

Do not train on Camelyon17.
Do not download WILDS data by default.
Do not extract real embeddings unless data is local and ignored.
Do not implement neural diffusion.
Do not claim diffusion contribution.

## Final response template

Report:

```text
Files added/modified
Commands run
Metadata probe behavior
Feature CSV converter behavior
Contract validation result on fake fixtures
Whether real Camelyon17 pilot is now executable with local data
Remaining blockers
pytest result
Next recommended task
```

## Recommended next task after this

If the metadata probe/converter is ready:

```text
Run a local Camelyon17-WILDS feature extraction dry run using frozen embeddings under ignored data/, then run MedHarmDiff benchmark with identity/center_mean/ridge/latent_diffusion_v0.
```
