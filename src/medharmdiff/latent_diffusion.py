from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class TimeConditionedRidgeDiffusionHarmonizer:
    """Torch-free latent diffusion-style harmonizer for feature-level benchmarks.

    This is a time-conditioned ridge denoising baseline. It is diffusion-style
    because it trains on a noise schedule and applies a deterministic reverse
    denoising loop, but it is still a lightweight v0 rather than a neural DDPM.
    """

    num_steps: int = 10
    n_augments: int = 8
    noise_strength: float = 0.2
    alpha: float = 1.0
    clinical_preservation_strength: float = 0.0
    random_seed: int = 13
    is_fitted: bool = False
    uses_target_unlabeled: bool = False
    uses_target_labels: bool = False
    model_: Ridge | None = None
    global_mean_: np.ndarray | None = None
    site_offsets_: dict[str, np.ndarray] = field(default_factory=dict)
    feature_scale_: np.ndarray | None = None
    clinical_direction_: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        site: np.ndarray,
        y: np.ndarray | None = None,
        *,
        target_x: np.ndarray | None = None,
        target_site: np.ndarray | None = None,
    ) -> "TimeConditionedRidgeDiffusionHarmonizer":
        del target_x, target_site
        source_x = np.asarray(x, dtype=float)
        source_site = np.asarray(site).astype(str)
        if source_x.ndim != 2:
            raise ValueError("x must be a 2D feature matrix")
        if source_x.shape[0] != source_site.shape[0]:
            raise ValueError("x and site must have the same number of rows")
        if self.num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        if self.n_augments < 1:
            raise ValueError("n_augments must be at least 1")

        self.global_mean_ = source_x.mean(axis=0)
        self.feature_scale_ = np.where(source_x.std(axis=0) == 0, 1.0, source_x.std(axis=0))
        self.site_offsets_ = {
            site_id: source_x[source_site == site_id].mean(axis=0) - self.global_mean_
            for site_id in sorted(set(source_site.tolist()))
        }
        canonical_x = self._canonicalize_source(source_x, source_site)
        self._fit_clinical_direction(canonical_x, y)
        train_x, train_y = self._diffusion_training_pairs(canonical_x)
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(train_x, train_y)
        self.is_fitted = True
        return self

    def transform(self, x: np.ndarray, site: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted or self.model_ is None:
            raise RuntimeError("fit must be called before transform")
        current = np.asarray(x, dtype=float).copy()
        original = current.copy()
        site_offsets = self._offsets_for(site, current)
        for step in range(self.num_steps, 0, -1):
            t_value = step / self.num_steps
            predicted_clean = self.model_.predict(
                self._design_matrix(
                    current,
                    np.full(current.shape[0], t_value, dtype=float),
                    site_offsets,
                )
            )
            step_size = 1.0 / step
            current = (1.0 - step_size) * current + step_size * predicted_clean
        return self._restore_clinical_projection(original, np.asarray(current, dtype=float))

    def _canonicalize_source(self, x: np.ndarray, site: np.ndarray) -> np.ndarray:
        canonical = x.copy()
        for idx, site_id in enumerate(site):
            canonical[idx] = x[idx] - self.site_offsets_[site_id]
        return canonical

    def _diffusion_training_pairs(self, canonical_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.feature_scale_ is None:
            raise RuntimeError("fit must be called before creating training pairs")
        rng = np.random.default_rng(self.random_seed)
        offsets = list(self.site_offsets_.values()) or [np.zeros(canonical_x.shape[1])]
        train_inputs = []
        train_targets = []
        for _ in range(self.n_augments):
            t_values = rng.uniform(0.0, 1.0, size=canonical_x.shape[0])
            offset_indices = rng.integers(0, len(offsets), size=canonical_x.shape[0])
            sampled_offsets = np.vstack([offsets[idx] for idx in offset_indices])
            sigma = self.noise_strength * np.maximum(t_values[:, None], 1e-3)
            noise = rng.normal(scale=sigma * self.feature_scale_, size=canonical_x.shape)
            noisy_shifted = canonical_x + sampled_offsets * t_values[:, None] + noise
            train_inputs.append(self._design_matrix(noisy_shifted, t_values, sampled_offsets))
            train_targets.append(canonical_x)
        return np.vstack(train_inputs), np.vstack(train_targets)

    def _design_matrix(
        self,
        x: np.ndarray,
        t_values: np.ndarray,
        offsets: np.ndarray,
    ) -> np.ndarray:
        return np.hstack([x, t_values.reshape(-1, 1), offsets])

    def _offsets_for(self, site: np.ndarray | None, x: np.ndarray) -> np.ndarray:
        if site is None:
            return np.zeros_like(x)
        site_values = np.asarray(site).astype(str)
        offsets = np.zeros_like(x)
        for idx, site_id in enumerate(site_values):
            if site_id in self.site_offsets_:
                offsets[idx] = self.site_offsets_[site_id]
        return offsets

    def _fit_clinical_direction(self, canonical_x: np.ndarray, y: np.ndarray | None) -> None:
        self.clinical_direction_ = None
        if y is None:
            return
        labels = np.asarray(y).astype(int)
        if set(labels.tolist()) != {0, 1}:
            return
        direction = canonical_x[labels == 1].mean(axis=0) - canonical_x[labels == 0].mean(axis=0)
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            self.clinical_direction_ = direction / norm

    def _restore_clinical_projection(
        self,
        original: np.ndarray,
        denoised: np.ndarray,
    ) -> np.ndarray:
        if self.clinical_direction_ is None or self.clinical_preservation_strength <= 0:
            return denoised
        strength = float(np.clip(self.clinical_preservation_strength, 0.0, 1.0))
        delta = original - denoised
        projection = delta @ self.clinical_direction_
        return denoised + strength * np.outer(projection, self.clinical_direction_)


@dataclass
class ClinicalPreservingTimeConditionedRidgeDiffusionHarmonizer(
    TimeConditionedRidgeDiffusionHarmonizer
):
    """Clinical-preserving latent diffusion v0 variant using source labels only."""

    clinical_preservation_strength: float = 0.5
