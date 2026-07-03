from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


EXTRACTOR_NAME = "color_stats_v0"
RGB_STATS = ("mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90")
GRAY_STATS = ("mean", "std", "min", "max", "p10", "p50", "p90")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract deterministic frozen Camelyon17 patch embeddings."
    )
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--image-path-column", default="image_path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25_000)
    args = parser.parse_args()

    summary = extract_patch_embeddings(
        metadata_csv=args.metadata_csv,
        image_root=args.image_root,
        output_npz=args.output_npz,
        sample_id_column=args.sample_id_column,
        image_path_column=args.image_path_column,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "embedding_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "embedding_report.md").write_text(
        render_embedding_report(summary),
        encoding="utf-8",
    )
    print(f"embeddings_npz: {args.output_npz}")


def extract_patch_embeddings(
    *,
    metadata_csv: Path,
    image_root: Path,
    output_npz: Path,
    sample_id_column: str = "sample_id",
    image_path_column: str = "image_path",
    limit: int | None = None,
    progress_every: int = 25_000,
) -> dict[str, object]:
    metadata = pd.read_csv(metadata_csv)
    for column in (sample_id_column, image_path_column):
        if column not in metadata.columns:
            raise ValueError(f"metadata CSV must include {column!r}")
    if limit is not None:
        metadata = metadata.head(limit).copy()
    if metadata.empty:
        raise ValueError("metadata CSV has no rows to embed")

    feature_names = color_stats_feature_names()
    embeddings = np.empty((len(metadata), len(feature_names)), dtype=np.float32)
    sample_ids: list[str] = []
    missing_paths: list[str] = []

    for row_index, row in enumerate(
        metadata[[sample_id_column, image_path_column]].itertuples(index=False)
    ):
        sample_id = str(row[0])
        image_path = image_root / str(row[1])
        if not image_path.exists():
            missing_paths.append(str(image_path))
            if len(missing_paths) >= 5:
                break
            continue
        embeddings[row_index] = color_stats_embedding(image_path)
        sample_ids.append(sample_id)
        if progress_every and (row_index + 1) % progress_every == 0:
            print(f"embedded {row_index + 1}/{len(metadata)}", flush=True)

    if missing_paths:
        raise FileNotFoundError(
            "Missing image path(s): " + ", ".join(missing_paths)
        )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        sample_id=np.asarray(sample_ids, dtype=str),
        embeddings=embeddings,
        feature_names=np.asarray(feature_names, dtype=str),
        extractor=np.asarray([EXTRACTOR_NAME], dtype=str),
    )
    return {
        "metadata_csv": str(metadata_csv),
        "image_root": str(image_root),
        "output_npz": str(output_npz),
        "extractor": EXTRACTOR_NAME,
        "sample_count": int(len(sample_ids)),
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "limit": limit,
    }


def color_stats_embedding(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = (
        0.2989 * rgb[:, :, 0]
        + 0.5870 * rgb[:, :, 1]
        + 0.1140 * rgb[:, :, 2]
    )
    features: list[float] = []
    for channel in range(3):
        features.extend(_stats(rgb[:, :, channel], RGB_STATS))
    features.extend(_stats(gray, GRAY_STATS))
    grad_y, grad_x = np.gradient(gray)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    features.extend(
        [
            float(np.mean(grad_mag)),
            float(np.std(grad_mag)),
            float(np.percentile(grad_mag, 90)),
        ]
    )
    return np.asarray(features, dtype=np.float32)


def color_stats_feature_names() -> list[str]:
    names: list[str] = []
    for channel_name in ("r", "g", "b"):
        names.extend([f"{channel_name}_{stat}" for stat in RGB_STATS])
    names.extend([f"gray_{stat}" for stat in GRAY_STATS])
    names.extend(["gradient_mean", "gradient_std", "gradient_p90"])
    return names


def _stats(values: np.ndarray, stats: tuple[str, ...]) -> list[float]:
    resolved: list[float] = []
    for stat in stats:
        if stat == "mean":
            resolved.append(float(np.mean(values)))
        elif stat == "std":
            resolved.append(float(np.std(values)))
        elif stat == "min":
            resolved.append(float(np.min(values)))
        elif stat == "max":
            resolved.append(float(np.max(values)))
        elif stat.startswith("p"):
            resolved.append(float(np.percentile(values, float(stat[1:]))))
        else:  # pragma: no cover - developer error
            raise ValueError(f"Unknown statistic {stat!r}")
    return resolved


def render_embedding_report(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Camelyon17 Patch Embedding Extraction",
            "",
            f"- Metadata CSV: `{summary['metadata_csv']}`",
            f"- Image root: `{summary['image_root']}`",
            f"- Output NPZ: `{summary['output_npz']}`",
            f"- Extractor: `{summary['extractor']}`",
            f"- Sample count: `{summary['sample_count']}`",
            f"- Feature count: `{summary['feature_count']}`",
            f"- Limit: `{summary['limit']}`",
            "",
            "These are deterministic frozen color/statistics features, "
            "not a deep pathology encoder.",
            "",
        ]
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
