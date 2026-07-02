from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medharmdiff.real_pilots.camelyon17_wilds import (
    normalize_camelyon17_metadata_frame,
    validate_camelyon17_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe local Camelyon17-WILDS metadata without downloading data."
    )
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-wilds", action="store_true")
    parser.add_argument("--wilds-root", type=Path, default=Path("data/wilds"))
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.metadata_csv is not None:
        summary = probe_metadata_csv(args.metadata_csv)
    elif args.use_wilds:
        summary = probe_wilds_package(args.wilds_root, allow_download=args.allow_download)
    else:
        raise SystemExit("Provide --metadata-csv or pass --use-wilds explicitly.")

    (args.output_dir / "metadata_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "metadata_probe_report.md").write_text(
        render_probe_report(summary),
        encoding="utf-8",
    )
    print(f"metadata_probe_summary: {args.output_dir / 'metadata_probe_summary.json'}")


def probe_metadata_csv(metadata_csv: Path) -> dict[str, object]:
    df = pd.read_csv(metadata_csv)
    rows = normalize_camelyon17_metadata_frame(df)
    validation = validate_camelyon17_rows(rows)
    return {
        "mode": "metadata_csv",
        "metadata_csv": str(metadata_csv),
        "source_columns": [str(column) for column in df.columns],
        "row_count": validation["sample_count"],
        "center_counts": validation["center_counts"],
        "center_count": validation["center_count"],
        "label_prevalence_by_center": validation["label_prevalence_by_center"],
        "split_counts": validation["split_counts"],
        "has_patient_or_group_id": validation["has_patient_or_group_id"],
        "missing_patient_or_group_id_count": validation[
            "missing_patient_or_group_id_count"
        ],
        "candidate_leakage_risks": _candidate_leakage_risks(validation),
        "feature_csv_ready": validation["feature_csv_ready"],
        "warnings": validation["warnings"],
    }


def probe_wilds_package(wilds_root: Path, *, allow_download: bool) -> dict[str, object]:
    try:
        from wilds import get_dataset
    except ImportError:
        return {
            "mode": "wilds",
            "status": "wilds_not_installed",
            "message": "Install the optional wilds package to probe local WILDS data.",
            "wilds_root": str(wilds_root),
            "allow_download": allow_download,
            "feature_csv_ready": False,
            "warnings": ["wilds package is not installed"],
        }
    try:
        dataset = get_dataset(
            dataset="camelyon17",
            root_dir=str(wilds_root),
            download=allow_download,
        )
    except Exception as exc:  # pragma: no cover - depends on local data/package
        return {
            "mode": "wilds",
            "status": "wilds_probe_failed",
            "message": str(exc),
            "wilds_root": str(wilds_root),
            "allow_download": allow_download,
            "feature_csv_ready": False,
            "warnings": ["local WILDS Camelyon17 data could not be opened"],
        }
    metadata_fields = [
        str(field) for field in getattr(dataset, "metadata_fields", []) or []
    ]
    return {
        "mode": "wilds",
        "status": "wilds_dataset_opened",
        "wilds_root": str(wilds_root),
        "allow_download": allow_download,
        "dataset_name": "camelyon17",
        "sample_count": int(len(dataset)),
        "metadata_fields": metadata_fields,
        "feature_csv_ready": False,
        "warnings": [
            "export local WILDS metadata to CSV before feature conversion",
        ],
    }


def render_probe_report(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Camelyon17-WILDS Metadata Probe",
            "",
            f"- Mode: `{summary.get('mode')}`",
            f"- Row count: `{summary.get('row_count', summary.get('sample_count', 0))}`",
            f"- Center counts: `{summary.get('center_counts', {})}`",
            f"- Label prevalence by center: "
            f"`{summary.get('label_prevalence_by_center', {})}`",
            f"- Split counts: `{summary.get('split_counts', {})}`",
            f"- Has patient/group ID: `{summary.get('has_patient_or_group_id', False)}`",
            f"- Candidate leakage risks: `{summary.get('candidate_leakage_risks', [])}`",
            f"- Feature CSV ready: `{summary.get('feature_csv_ready', False)}`",
            f"- Warnings: `{summary.get('warnings', [])}`",
            "",
            "No data was downloaded or written outside the requested output directory.",
            "",
        ]
    )


def _candidate_leakage_risks(validation: dict[str, object]) -> list[str]:
    risks = []
    if not validation["has_patient_or_group_id"]:
        risks.append("missing patient_or_group_id prevents grouping leak audit")
    if validation["center_count"] < 3:
        risks.append("fewer than three centers prevents leave-center-out benchmark")
    if validation["missing_patient_or_group_id_count"]:
        risks.append("some rows lack patient_or_group_id")
    return risks


if __name__ == "__main__":
    main()
