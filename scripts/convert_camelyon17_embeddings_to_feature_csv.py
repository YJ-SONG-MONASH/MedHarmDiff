from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medharmdiff.io import validate_real_feature_contract
from medharmdiff.real_pilots.camelyon17_wilds import (
    normalize_camelyon17_metadata_frame,
    validate_camelyon17_rows,
    write_camelyon17_feature_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert local Camelyon17 metadata and embeddings to feature CSV."
    )
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--embeddings-file", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = convert_embeddings_to_feature_csv(
        metadata_csv=args.metadata_csv,
        embeddings_file=args.embeddings_file,
        output_csv=args.output_csv,
    )
    (args.output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "conversion_report.md").write_text(
        render_conversion_report(summary),
        encoding="utf-8",
    )
    print(f"feature_csv: {args.output_csv}")


def convert_embeddings_to_feature_csv(
    *,
    metadata_csv: Path,
    embeddings_file: Path,
    output_csv: Path,
) -> dict[str, object]:
    metadata_df = pd.read_csv(metadata_csv)
    metadata_rows = normalize_camelyon17_metadata_frame(metadata_df)
    metadata_validation = validate_camelyon17_rows(metadata_rows)
    embeddings_by_sample_id = load_embeddings(embeddings_file)
    embedding_metadata = load_embedding_metadata(embeddings_file)
    write_summary = write_camelyon17_feature_csv(
        metadata_rows,
        embeddings_by_sample_id,
        output_csv,
    )
    contract_validation = validate_real_feature_contract(output_csv)
    return {
        "metadata_csv": str(metadata_csv),
        "embeddings_file": str(embeddings_file),
        "output_csv": str(output_csv),
        "embedding_metadata": embedding_metadata,
        "metadata_validation": metadata_validation,
        "write_summary": write_summary,
        "contract_validation": contract_validation,
    }


def load_embeddings(embeddings_file: Path) -> dict[str, np.ndarray]:
    suffix = embeddings_file.suffix.lower()
    if suffix == ".csv":
        return _load_csv_embeddings(embeddings_file)
    if suffix == ".npz":
        return _load_npz_embeddings(embeddings_file)
    raise ValueError("Supported embedding formats are CSV and NPZ")


def load_embedding_metadata(embeddings_file: Path) -> dict[str, object]:
    suffix = embeddings_file.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(embeddings_file, nrows=1)
        embedding_columns = [
            column
            for column in df.columns
            if column != "sample_id" and pd.api.types.is_numeric_dtype(df[column])
        ]
        return {
            "embeddings_file": str(embeddings_file),
            "format": "csv",
            "extractor_name": None,
            "feature_dim": len(embedding_columns),
            "feature_names": [str(column) for column in embedding_columns],
        }
    if suffix == ".npz":
        data = np.load(embeddings_file, allow_pickle=False)
        extractor = None
        if "extractor" in data:
            extractor_values = data["extractor"].astype(str).reshape(-1).tolist()
            extractor = extractor_values[0] if extractor_values else None
        feature_names = (
            data["feature_names"].astype(str).tolist()
            if "feature_names" in data
            else []
        )
        feature_dim = (
            int(np.asarray(data["embeddings"]).shape[1])
            if "embeddings" in data and np.asarray(data["embeddings"]).ndim == 2
            else len(feature_names)
        )
        return {
            "embeddings_file": str(embeddings_file),
            "format": "npz",
            "extractor_name": extractor,
            "feature_dim": feature_dim,
            "feature_names": feature_names,
        }
    raise ValueError("Supported embedding formats are CSV and NPZ")


def _load_csv_embeddings(embeddings_file: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(embeddings_file)
    if "sample_id" not in df.columns:
        raise ValueError("CSV embeddings must include sample_id")
    embedding_columns = [
        column
        for column in df.columns
        if column != "sample_id" and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not embedding_columns:
        raise ValueError("CSV embeddings must include numeric embedding columns")
    return {
        str(row["sample_id"]): row[embedding_columns].to_numpy(dtype=float)
        for _, row in df.iterrows()
    }


def _load_npz_embeddings(embeddings_file: Path) -> dict[str, np.ndarray]:
    data = np.load(embeddings_file, allow_pickle=False)
    if "sample_id" not in data or "embeddings" not in data:
        raise ValueError("NPZ embeddings must contain sample_id and embeddings arrays")
    sample_ids = data["sample_id"].astype(str)
    embeddings = np.asarray(data["embeddings"], dtype=float)
    if embeddings.ndim != 2:
        raise ValueError("NPZ embeddings array must be 2D")
    if sample_ids.shape[0] != embeddings.shape[0]:
        raise ValueError("sample_id and embeddings row counts do not match")
    return {
        str(sample_id): embeddings[index]
        for index, sample_id in enumerate(sample_ids.tolist())
    }


def render_conversion_report(summary: dict[str, object]) -> str:
    contract = summary["contract_validation"]
    metadata = summary["metadata_validation"]
    return "\n".join(
        [
            "# Camelyon17 Feature CSV Conversion",
            "",
            f"- Metadata CSV: `{summary['metadata_csv']}`",
            f"- Embeddings file: `{summary['embeddings_file']}`",
            f"- Output CSV: `{summary['output_csv']}`",
            f"- Sample count: `{contract['sample_count']}`",
            f"- Center count: `{contract['center_count']}`",
            f"- Feature count: `{contract['feature_count']}`",
            f"- Embedding metadata: `{summary['embedding_metadata']}`",
            f"- Metadata warnings: `{metadata['warnings']}`",
            f"- Contract warnings: `{contract['warnings']}`",
            "",
            "The output uses only `feature_*` columns as model features.",
            "",
        ]
    )


if __name__ == "__main__":
    main()
