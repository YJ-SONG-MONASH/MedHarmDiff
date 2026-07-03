import json
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path("scripts/extract_camelyon17_patch_embeddings.py")
_SPEC = importlib.util.spec_from_file_location("extract_camelyon17_patch_embeddings", _SCRIPT_PATH)
extractors = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(extractors)


def _write_fake_images_and_metadata(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.fromarray(np.full((8, 8, 3), [255, 0, 0], dtype=np.uint8)).save(
        image_root / "red.png"
    )
    Image.fromarray(np.full((8, 8, 3), [0, 255, 0], dtype=np.uint8)).save(
        image_root / "green.png"
    )
    metadata_csv = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "image_path": ["red.png", "green.png"],
        }
    ).to_csv(metadata_csv, index=False)
    return image_root, metadata_csv


def test_extractor_registry_lists_supported_embedding_families() -> None:
    assert set(extractors.EXTRACTOR_REGISTRY) == {
        "color_stats_v0",
        "resnet18_imagenet_v0",
        "resnet50_imagenet_v0",
        "pathology_encoder_placeholder",
    }
    assert len(extractors.color_stats_feature_names()) == 37


def test_color_stats_extractor_writes_embedding_manifest(tmp_path) -> None:
    image_root, metadata_csv = _write_fake_images_and_metadata(tmp_path)
    output_npz = tmp_path / "embeddings.npz"
    output_dir = tmp_path / "summary"

    subprocess.run(
        [
            sys.executable,
            "scripts/extract_camelyon17_patch_embeddings.py",
            "--metadata-csv",
            str(metadata_csv),
            "--image-root",
            str(image_root),
            "--output-npz",
            str(output_npz),
            "--output-dir",
            str(output_dir),
            "--extractor",
            "color_stats_v0",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ],
        check=True,
        cwd=".",
        capture_output=True,
        text=True,
    )

    data = np.load(output_npz, allow_pickle=False)
    manifest = json.loads((output_dir / "embedding_manifest.json").read_text("utf-8"))

    assert data["embeddings"].shape == (2, 37)
    assert manifest["extractor_name"] == "color_stats_v0"
    assert manifest["model_family"] == "handcrafted_color_statistics"
    assert manifest["weights_downloaded"] is False
    assert manifest["feature_dim"] == 37
    assert manifest["sample_count"] == 2
    assert manifest["batch_size"] == 2
    assert manifest["device"] == "cpu"
    assert manifest["created_by_script"] == "scripts/extract_camelyon17_patch_embeddings.py"


def test_resnet_extractor_without_weights_fails_clearly(tmp_path) -> None:
    image_root, metadata_csv = _write_fake_images_and_metadata(tmp_path)
    output_npz = tmp_path / "resnet.npz"
    output_dir = tmp_path / "summary"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/extract_camelyon17_patch_embeddings.py",
            "--metadata-csv",
            str(metadata_csv),
            "--image-root",
            str(image_root),
            "--output-npz",
            str(output_npz),
            "--output-dir",
            str(output_dir),
            "--extractor",
            "resnet18_imagenet_v0",
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--limit",
            "1",
        ],
        cwd=".",
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "allow-download-weights" in completed.stderr or "weights-path" in completed.stderr
    assert not output_npz.exists()
