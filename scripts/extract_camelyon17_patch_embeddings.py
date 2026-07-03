from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image


EXTRACTOR_NAME = "color_stats_v0"
RGB_STATS = ("mean", "std", "min", "max", "p10", "p25", "p50", "p75", "p90")
GRAY_STATS = ("mean", "std", "min", "max", "p10", "p50", "p90")
SCRIPT_NAME = "scripts/extract_camelyon17_patch_embeddings.py"
EXTRACTOR_REGISTRY = {
    "color_stats_v0": "deterministic 37-feature RGB/gray/gradient statistics",
    "resnet18_imagenet_v0": "torchvision ResNet-18 ImageNet frozen features",
    "resnet50_imagenet_v0": "torchvision ResNet-50 ImageNet frozen features",
    "pathology_encoder_placeholder": "reserved pathology encoder slot",
}


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
    parser.add_argument("--extractor", choices=sorted(EXTRACTOR_REGISTRY), default=EXTRACTOR_NAME)
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--allow-download-weights", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25_000)
    args = parser.parse_args()

    summary = extract_patch_embeddings(
        metadata_csv=args.metadata_csv,
        image_root=args.image_root,
        output_npz=args.output_npz,
        sample_id_column=args.sample_id_column,
        image_path_column=args.image_path_column,
        extractor=args.extractor,
        weights_path=args.weights_path,
        allow_download_weights=args.allow_download_weights,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "embedding_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "embedding_manifest.json").write_text(
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
    extractor: str = EXTRACTOR_NAME,
    weights_path: Path | None = None,
    allow_download_weights: bool = False,
    batch_size: int = 32,
    device: str = "auto",
    limit: int | None = None,
    progress_every: int = 25_000,
) -> dict[str, object]:
    if extractor not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown extractor {extractor!r}")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    metadata = pd.read_csv(metadata_csv)
    for column in (sample_id_column, image_path_column):
        if column not in metadata.columns:
            raise ValueError(f"metadata CSV must include {column!r}")
    if limit is not None:
        metadata = metadata.head(limit).copy()
    if metadata.empty:
        raise ValueError("metadata CSV has no rows to embed")

    sample_ids, image_paths = _resolve_rows(
        metadata,
        image_root=image_root,
        sample_id_column=sample_id_column,
        image_path_column=image_path_column,
    )
    device_name = _resolve_device(device)
    if extractor == "color_stats_v0":
        embeddings = _extract_color_stats(image_paths, progress_every=progress_every)
        feature_names = color_stats_feature_names()
        model_family = "handcrafted_color_statistics"
        weights_source = "none"
        weights_downloaded = False
    elif extractor in {"resnet18_imagenet_v0", "resnet50_imagenet_v0"}:
        embeddings, feature_names, weights_source, weights_downloaded = _extract_resnet(
            image_paths,
            extractor=extractor,
            weights_path=weights_path,
            allow_download_weights=allow_download_weights,
            batch_size=batch_size,
            device=device_name,
            progress_every=progress_every,
        )
        model_family = "torchvision_imagenet"
    else:
        raise RuntimeError(
            "pathology_encoder_placeholder is a scaffold only; provide a concrete "
            "pathology encoder implementation and local weights before use"
        )

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        sample_id=np.asarray(sample_ids, dtype=str),
        embeddings=embeddings,
        feature_names=np.asarray(feature_names, dtype=str),
        extractor=np.asarray([extractor], dtype=str),
    )
    return {
        "extractor_name": extractor,
        "model_family": model_family,
        "weights_source": weights_source,
        "weights_downloaded": weights_downloaded,
        "feature_dim": int(embeddings.shape[1]),
        "sample_count": int(len(sample_ids)),
        "image_root": str(image_root),
        "metadata_csv": str(metadata_csv),
        "output_npz": str(output_npz),
        "limit": limit,
        "batch_size": batch_size,
        "device": device_name,
        "created_by_script": SCRIPT_NAME,
        "extractor": extractor,
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
    }


def _resolve_rows(
    metadata: pd.DataFrame,
    *,
    image_root: Path,
    sample_id_column: str,
    image_path_column: str,
) -> tuple[list[str], list[Path]]:
    sample_ids: list[str] = []
    image_paths: list[Path] = []
    missing_paths: list[str] = []
    for row in metadata[[sample_id_column, image_path_column]].itertuples(index=False):
        sample_id = str(row[0])
        image_path = image_root / str(row[1])
        if not image_path.exists():
            missing_paths.append(str(image_path))
            if len(missing_paths) >= 5:
                break
            continue
        sample_ids.append(sample_id)
        image_paths.append(image_path)
    if missing_paths:
        raise FileNotFoundError("Missing image path(s): " + ", ".join(missing_paths))
    return sample_ids, image_paths


def _extract_color_stats(
    image_paths: list[Path],
    *,
    progress_every: int,
) -> np.ndarray:
    feature_names = color_stats_feature_names()
    embeddings = np.empty((len(image_paths), len(feature_names)), dtype=np.float32)
    for row_index, image_path in enumerate(image_paths):
        embeddings[row_index] = color_stats_embedding(image_path)
        if progress_every and (row_index + 1) % progress_every == 0:
            print(f"embedded {row_index + 1}/{len(image_paths)}", flush=True)
    return embeddings


def _extract_resnet(
    image_paths: list[Path],
    *,
    extractor: str,
    weights_path: Path | None,
    allow_download_weights: bool,
    batch_size: int,
    device: str,
    progress_every: int,
) -> tuple[np.ndarray, list[str], str, bool]:
    if weights_path is None and not allow_download_weights:
        raise RuntimeError(
            f"{extractor} requires ImageNet weights. Pass --weights-path for local "
            "weights or --allow-download-weights to let torchvision download them."
        )
    try:
        import torch
        from torch import nn
        from torchvision import models, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for ImageNet extractors") from exc

    model_factory, weights_enum, feature_dim = _resnet_factory(extractor, models)
    weights = weights_enum.DEFAULT if allow_download_weights and weights_path is None else None
    model = model_factory(weights=weights)
    weights_source = "torchvision_default_download" if weights is not None else str(weights_path)
    weights_downloaded = bool(weights is not None)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch = torch.stack(
                [preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
            )
            features = model(batch.to(device)).detach().cpu().numpy().astype(np.float32)
            chunks.append(features.reshape(features.shape[0], -1))
            if progress_every and (start + len(batch_paths)) % progress_every == 0:
                print(f"embedded {start + len(batch_paths)}/{len(image_paths)}", flush=True)
    embeddings = np.vstack(chunks) if chunks else np.empty((0, feature_dim), dtype=np.float32)
    feature_names = [f"{extractor}_feature_{index}" for index in range(embeddings.shape[1])]
    return embeddings, feature_names, weights_source, weights_downloaded


def _resnet_factory(extractor: str, models: object) -> tuple[Callable, object, int]:
    if extractor == "resnet18_imagenet_v0":
        return models.resnet18, models.ResNet18_Weights, 512
    if extractor == "resnet50_imagenet_v0":
        return models.resnet50, models.ResNet50_Weights, 2048
    raise ValueError(f"Unsupported ResNet extractor {extractor!r}")


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


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
            f"- Model family: `{summary['model_family']}`",
            f"- Weights source: `{summary['weights_source']}`",
            f"- Sample count: `{summary['sample_count']}`",
            f"- Feature count: `{summary['feature_count']}`",
            f"- Limit: `{summary['limit']}`",
            f"- Batch size: `{summary['batch_size']}`",
            f"- Device: `{summary['device']}`",
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
