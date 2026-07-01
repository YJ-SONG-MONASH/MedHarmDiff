from __future__ import annotations

import string

import numpy as np
import pandas as pd


def generate_synthetic_feature_dataset(
    *,
    n_centers: int = 3,
    samples_per_center: int = 80,
    n_features: int = 8,
    clinical_signal_strength: float = 1.0,
    site_shift_strength: float = 1.0,
    site_label_confounding_strength: float = 0.0,
    noise_strength: float = 1.0,
    random_seed: int = 13,
) -> pd.DataFrame:
    """Generate a synthetic multi-center binary classification feature dataset.

    The generator intentionally separates three knobs that can fool harmonizers:
    clinical signal, center-specific feature shift, and label/site confounding.
    """

    if n_centers < 3:
        raise ValueError("synthetic benchmark requires at least three centers")
    if samples_per_center < 2:
        raise ValueError("samples_per_center must be at least 2")
    if n_features < 1:
        raise ValueError("n_features must be at least 1")

    rng = np.random.default_rng(random_seed)
    centers = _center_names(n_centers)
    center_positions = np.linspace(-1.0, 1.0, n_centers)

    clinical_direction = rng.normal(size=n_features)
    clinical_direction /= max(float(np.linalg.norm(clinical_direction)), 1e-12)
    clinical_direction[0] = abs(clinical_direction[0]) + 0.5
    clinical_direction /= max(float(np.linalg.norm(clinical_direction)), 1e-12)

    site_offsets = rng.normal(scale=0.15, size=(n_centers, n_features))
    site_offsets += center_positions[:, None] * site_shift_strength
    site_offsets[:, 0] = center_positions * site_shift_strength

    rows: list[dict[str, float | int | str]] = []
    for center_index, center_id in enumerate(centers):
        label_logit = center_positions[center_index] * site_label_confounding_strength * 4.0
        label_probability = _sigmoid(label_logit)
        labels = rng.binomial(1, label_probability, samples_per_center)
        noise = rng.normal(scale=noise_strength, size=(samples_per_center, n_features))
        clinical_component = (
            (labels[:, None] * 2.0 - 1.0) * clinical_signal_strength * clinical_direction
        )
        features = noise + clinical_component + site_offsets[center_index]

        for sample_index, (label, feature_row) in enumerate(zip(labels, features, strict=True)):
            row: dict[str, float | int | str] = {
                "sample_id": f"{center_id}_{sample_index:04d}",
                "center_id": center_id,
                "label": int(label),
            }
            row.update({f"feature_{idx}": float(value) for idx, value in enumerate(feature_row)})
            rows.append(row)

    return pd.DataFrame(rows)


def _center_names(n_centers: int) -> list[str]:
    if n_centers <= len(string.ascii_uppercase):
        return list(string.ascii_uppercase[:n_centers])
    return [f"C{idx:02d}" for idx in range(n_centers)]


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))
