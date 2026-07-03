import json
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image


def test_extract_camelyon17_patch_embeddings_writes_npz_and_summary(tmp_path) -> None:
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
        ],
        check=True,
        cwd=".",
        capture_output=True,
        text=True,
    )

    data = np.load(output_npz, allow_pickle=False)
    summary = json.loads((output_dir / "embedding_summary.json").read_text("utf-8"))

    assert data["sample_id"].astype(str).tolist() == ["s1", "s2"]
    assert data["embeddings"].shape == (2, summary["feature_count"])
    assert summary["sample_count"] == 2
    assert summary["extractor"] == "color_stats_v0"
    assert (output_dir / "embedding_report.md").exists()
    assert not np.allclose(data["embeddings"][0], data["embeddings"][1])
