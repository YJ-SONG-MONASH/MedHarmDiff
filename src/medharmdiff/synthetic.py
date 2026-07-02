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
    noise_std: float | None = None,
    noise_strength: float | None = None,
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
    noise_scale = 1.0 if noise_std is None else noise_std
    if noise_strength is not None:
        noise_scale = noise_strength

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
        noise = rng.normal(scale=noise_scale, size=(samples_per_center, n_features))
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


def generate_synthetic_nonlinear_multicenter_features(
    *,
    n_centers: int = 3,
    samples_per_center: int = 80,
    n_features: int = 16,
    additive_shift_strength: float = 0.0,
    covariance_shift_strength: float = 0.0,
    rotation_shift_strength: float = 0.0,
    heteroskedastic_noise_strength: float = 0.0,
    nonlinear_warp_strength: float = 0.0,
    class_conditional_style_strength: float = 0.0,
    site_specific_feature_interaction_strength: float = 0.0,
    label_site_confounding_strength: float = 0.0,
    site_label_confounding_strength: float | None = None,
    clinical_signal_strength: float = 1.0,
    noise_strength: float = 1.0,
    include_oracle_columns: bool = False,
    random_seed: int = 13,
) -> pd.DataFrame:
    """Generate nonlinear multi-center feature shifts for stress testing.

    The output keeps the benchmark schema used by ``load_feature_csv``. Optional
    ``oracle_*`` columns expose hidden clinical and style components for
    diagnostics only; they intentionally do not use the ``feature_`` prefix.
    """

    if n_centers < 3:
        raise ValueError("synthetic benchmark requires at least three centers")
    if samples_per_center < 2:
        raise ValueError("samples_per_center must be at least 2")
    if n_features < 2:
        raise ValueError("n_features must be at least 2 for nonlinear shifts")
    if site_label_confounding_strength is not None:
        label_site_confounding_strength = site_label_confounding_strength

    rng = np.random.default_rng(random_seed)
    centers = _center_names(n_centers)
    center_positions = np.linspace(-1.0, 1.0, n_centers)
    feature_axis = np.linspace(-1.0, 1.0, n_features)

    clinical_direction = _unit_vector(rng.normal(size=n_features))
    clinical_direction[0] = abs(clinical_direction[0]) + 0.5
    clinical_direction = _unit_vector(clinical_direction)
    additive_direction = _unit_vector(rng.normal(size=n_features))
    class_style_direction = _unit_vector(rng.normal(size=n_features))
    interaction_pairs = [(idx, (idx + 1) % n_features) for idx in range(n_features)]

    rows: list[dict[str, float | int | str]] = []
    for center_index, center_id in enumerate(centers):
        position = float(center_positions[center_index])
        label_logit = position * label_site_confounding_strength * 4.0
        labels = _sample_binary_labels(rng, _sigmoid(label_logit), samples_per_center)
        base_noise = rng.normal(
            scale=noise_strength,
            size=(samples_per_center, n_features),
        )
        clinical_latent = (
            base_noise
            + (labels[:, None] * 2.0 - 1.0)
            * clinical_signal_strength
            * clinical_direction
        )

        rotation = _rotation_matrix(n_features, position, rotation_shift_strength)
        rotated = clinical_latent @ rotation.T
        scales = np.exp(covariance_shift_strength * position * feature_axis)
        covariance_shifted = rotated * scales

        nonlinear_component = nonlinear_warp_strength * position * np.tanh(
            covariance_shifted * (1.0 + np.abs(feature_axis))
        )
        class_style = (
            class_conditional_style_strength
            * position
            * (labels[:, None] * 2.0 - 1.0)
            * class_style_direction
        )
        interactions = np.zeros_like(covariance_shifted)
        if site_specific_feature_interaction_strength:
            for out_idx, (left_idx, right_idx) in enumerate(interaction_pairs):
                interactions[:, out_idx] = (
                    covariance_shifted[:, left_idx]
                    * covariance_shifted[:, right_idx]
                    * position
                    * site_specific_feature_interaction_strength
                    / max(1, n_features)
                )

        additive_style = additive_shift_strength * position * additive_direction
        noise_scale = noise_strength * (
            1.0
            + heteroskedastic_noise_strength
            * (0.35 + abs(position))
            * (1.0 + np.abs(feature_axis))
        )
        heteroskedastic_noise = rng.normal(
            scale=noise_scale,
            size=(samples_per_center, n_features),
        )
        features = (
            covariance_shifted
            + nonlinear_component
            + class_style
            + interactions
            + additive_style
            + heteroskedastic_noise
        )
        site_style = features - clinical_latent

        for sample_index, (label, feature_row) in enumerate(
            zip(labels, features, strict=True)
        ):
            row: dict[str, float | int | str] = {
                "sample_id": f"{center_id}_{sample_index:04d}",
                "center_id": center_id,
                "label": int(label),
            }
            row.update({f"feature_{idx}": float(value) for idx, value in enumerate(feature_row)})
            if include_oracle_columns:
                row.update(
                    {
                        f"oracle_clinical_latent_{idx}": float(value)
                        for idx, value in enumerate(clinical_latent[sample_index])
                    }
                )
                row.update(
                    {
                        f"oracle_site_style_{idx}": float(value)
                        for idx, value in enumerate(site_style[sample_index])
                    }
                )
            rows.append(row)

    return pd.DataFrame(rows)


def generate_synthetic_multicenter_features(
    *,
    n_centers: int = 3,
    samples_per_center: int = 200,
    n_features: int = 32,
    clinical_signal_strength: float = 1.0,
    site_shift_strength: float = 1.0,
    label_site_confounding: float = 0.0,
    noise_strength: float = 0.5,
    seed: int = 13,
) -> pd.DataFrame:
    """Generate the GitHub-prompt synthetic schema.

    This wrapper keeps the original A/B/C smoke-test generator intact while
    exposing the prompt's center IDs and padded feature names.
    """

    df = generate_synthetic_feature_dataset(
        n_centers=n_centers,
        samples_per_center=samples_per_center,
        n_features=n_features,
        clinical_signal_strength=clinical_signal_strength,
        site_shift_strength=site_shift_strength,
        site_label_confounding_strength=label_site_confounding,
        noise_strength=noise_strength,
        random_seed=seed,
    )
    centers = sorted(df["center_id"].unique().tolist())
    center_map = {center: f"center_{idx}" for idx, center in enumerate(centers)}
    rename_map = {
        f"feature_{idx}": f"feature_{idx:03d}"
        for idx in range(n_features)
    }
    df = df.replace({"center_id": center_map}).rename(columns=rename_map)
    sample_idx = df.groupby("center_id").cumcount().map(lambda idx: f"{idx:04d}")
    df["sample_id"] = df["center_id"].astype(str) + "_" + sample_idx
    return df


def _center_names(n_centers: int) -> list[str]:
    if n_centers <= len(string.ascii_uppercase):
        return list(string.ascii_uppercase[:n_centers])
    return [f"C{idx:02d}" for idx in range(n_centers)]


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _unit_vector(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        return np.ones_like(values) / np.sqrt(values.size)
    return values / norm


def _rotation_matrix(
    n_features: int,
    center_position: float,
    rotation_shift_strength: float,
) -> np.ndarray:
    rotation = np.eye(n_features)
    if rotation_shift_strength == 0:
        return rotation
    for start in range(0, n_features - 1, 2):
        angle = center_position * rotation_shift_strength * (0.35 + 0.05 * start)
        cos_angle = float(np.cos(angle))
        sin_angle = float(np.sin(angle))
        pair_rotation = np.eye(n_features)
        pair_rotation[start, start] = cos_angle
        pair_rotation[start, start + 1] = -sin_angle
        pair_rotation[start + 1, start] = sin_angle
        pair_rotation[start + 1, start + 1] = cos_angle
        rotation = pair_rotation @ rotation
    return rotation


def _sample_binary_labels(
    rng: np.random.Generator,
    probability: float,
    sample_count: int,
) -> np.ndarray:
    labels = rng.binomial(1, probability, sample_count)
    if labels.min() == labels.max():
        labels[0] = 0
        labels[-1] = 1
    return labels
