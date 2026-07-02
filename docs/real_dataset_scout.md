# Real Dataset Scout

This scout lists candidate public or access-controlled multi-center/domain-shift
datasets for future MedHarmDiff evaluation. It does not add or download data.

## Selection Criteria

- At least plausible center/domain/scanner/source metadata.
- A clinical label, segmentation target, or report-derived task that can be
  evaluated on held-out domains.
- Feasible feature-level extraction path, such as radiomics, pretrained imaging
  embeddings, or tabular metadata.
- Clear enough access terms to support reproducible research.

## Candidate Datasets

| Dataset | Modality | Centers / domains | Label / task | Domain metadata | Target labels public? | Feature-level extraction | Access burden | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Camelyon17-WILDS](https://wilds.stanford.edu/datasets/) / [CAMELYON17](https://camelyon17.grand-challenge.org/Data/) | Histopathology whole-slide lymph node images | 5 hospitals in WILDS/CAMELYON17 framing | Tumor classification / metastasis detection | Hospital domain is explicit in WILDS split | Yes for benchmark labels | Patch embeddings from pretrained pathology encoders are feasible | WILDS loader is easiest; Grand Challenge terms may apply | High |
| [MIMIC-CXR-JPG](https://www.physionet.org/content/mimic-cxr-jpg/) | Chest X-ray | Single hospital system, many devices/views/time periods | Report-derived findings | DICOM/JPG metadata can support acquisition/view/device proxies, but not true multi-hospital | Yes after PhysioNet credentialing | CXR embeddings or radiomics feasible | PhysioNet credentialing and data-use agreement | Medium |
| [CheXpert](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays) | Chest X-ray | Stanford Health Care; inpatient/outpatient centers | Multi-label chest findings | Institution is mostly single-system; view/time/device metadata can define domains | Public labels after access | CXR embeddings feasible | Stanford terms and registration | Medium |
| [PadChest](https://pubmed.ncbi.nlm.nih.gov/32877839/) | Chest X-ray | San Juan Hospital, Spain | Multi-label radiology findings/reports | Acquisition and demographic metadata are reported, but center count is not multi-hospital | Public/research access varies by mirror | CXR embeddings feasible | Moderate; verify official download route and license | Medium |
| [ISIC Archive](https://www.isic-archive.com/) and [ISIC challenge data](https://challenge.isic-archive.com/data/) | Dermoscopy | Multiple collections/sources | Lesion diagnosis / segmentation | Collection/source metadata can define domains; metadata may change in the live archive | Labels available for public challenge sets | Dermoscopy embeddings feasible | Low to moderate; use fixed challenge versions for reproducibility | Medium |
| [HAM10000](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091241/) | Dermoscopy | Multi-source lesion images | Seven-class lesion diagnosis | Source/site metadata exists in the descriptor, but domain granularity must be checked in files | Yes for academic use | Dermoscopy embeddings feasible | Low; confirm academic-use terms | Medium |
| [IXI](https://brain-development.org/ixi-dataset/) | Brain MRI | 3 London hospitals/scanner setups | No disease label; anatomy/segmentation or reconstruction proxy tasks | Hospital/scanner domain is explicit | No clinical disease labels; images public | Radiomics/brain embeddings feasible; segmentation labels require external processing | Low; license and preprocessing need confirmation | Medium |
| [ADNI](https://adni.loni.usc.edu/) | Brain MRI plus clinical biomarkers | Longitudinal, multi-center observational study | AD/MCI/cognition/biomarker tasks | Site/scanner/protocol metadata likely available after access | Labels available after data access | MRI radiomics and embeddings feasible | High; registration and data-use approval | Medium |
| [NSCLC-Radiomics](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/) | Lung CT | Collection-level CT cohort; site/scanner metadata needs audit | Survival / tumor annotations / radiomics | DICOM metadata exists; true center metadata must be verified | Clinical outcomes available in collection | Radiomics path is direct | Moderate; TCIA download and metadata audit | Medium |
| [RIDER-Lung-CT](https://www.cancerimagingarchive.net/collection/rider-lung-ct/) | Lung CT test-retest | Same-day repeated scans and reconstruction settings | Measurement variability / lesion analysis | Reconstruction setting is a domain-like factor; not a broad hospital split | Labels/lesion metadata limited | Radiomics harmonization is feasible | Moderate; small sample size | Low |
| [BraTS](https://braintumorsegmentation.org/) | Multi-parametric brain MRI | Multi-institutional clinical MRI | Brain tumor segmentation | Multi-institutional/protocol variation exists; released data are heavily preprocessed | Segmentation labels public for training subsets | Feature extraction feasible; raw scanner harmonization partly reduced by preprocessing | Moderate; challenge account/terms | Medium |
| [fastMRI](https://fastmri.med.nyu.edu/) | Knee/brain/prostate/breast MRI raw k-space and DICOM | Magnet strength and acquisition protocol domains; mainly NYU-derived public release | Reconstruction; fastMRI+ adds pathology annotations | Scanner strength/protocol metadata useful, but multi-center labels may be limited | Reconstruction targets public; pathology labels require fastMRI+ route | Embeddings and reconstruction-derived features feasible | Moderate; large storage and MRI preprocessing | Low |

## Recommended First-Use Order

1. Camelyon17-WILDS: best first real benchmark because the domain split is
   explicit, benchmarked, and directly aligned with domain generalization.
2. IXI: useful for scanner/hospital harmonization mechanics, but less suitable
   for a clinical label claim unless paired with a defensible downstream task.
3. MIMIC-CXR-JPG plus CheXpert or PadChest: useful for cross-dataset CXR
   generalization, but this becomes cross-institution dataset transfer rather
   than within-dataset leave-center-out harmonization.
4. NSCLC-Radiomics or RIDER-Lung-CT: useful for radiomics harmonization, but
   metadata and sample-size constraints should be audited before committing.
5. ISIC/HAM10000: useful dermatology source shift candidates; use fixed
   challenge snapshots or paper-released metadata rather than mutable archive
   state.

## Immediate Feasibility Notes

- The first real-data MVP should avoid downloading raw data into this repo.
- Start with extracted embeddings stored under ignored `data/` only.
- Before any paper-facing claim, create a metadata audit table with center/site,
  label prevalence, device/protocol fields, and patient-level split keys.
- Do not compare synthetic nonlinear results directly to real-data results; use
  synthetic only to test failure modes and benchmark mechanics.
