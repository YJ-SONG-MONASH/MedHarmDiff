from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class RidgeDenoisingHarmonizer:
    """Source-only learned denoising baseline for feature-level harmonization.

    This is an intentionally small baseline. It learns to map synthetic
    site-shifted/noisy source features back to source-canonicalized features.
    It is not a diffusion model and must not be reported as one.
    """

    noise_strength: float = 0.1
    n_augments: int = 4
    alpha: float = 1.0
    random_seed: int = 13
    model_: Ridge | None = None
    global_mean_: np.ndarray | None = None
    site_offsets_: dict[str, np.ndarray] = field(default_factory=dict)
    feature_scale_: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        site: np.ndarray,
        y: np.ndarray | None = None,
        *,
        target_x: np.ndarray | None = None,
        target_site: np.ndarray | None = None,
    ) -> "RidgeDenoisingHarmonizer":
        del y, target_x, target_site
        source_x = np.asarray(x, dtype=float)
        source_site = np.asarray(site).astype(str)
        if source_x.ndim != 2:
            raise ValueError("x must be a 2D feature matrix")
        if source_x.shape[0] != source_site.shape[0]:
            raise ValueError("x and site must have the same number of rows")
        if self.n_augments < 1:
            raise ValueError("n_augments must be at least 1")

        self.global_mean_ = source_x.mean(axis=0)
        self.site_offsets_ = {
            site_id: source_x[source_site == site_id].mean(axis=0) - self.global_mean_
            for site_id in sorted(set(source_site.tolist()))
        }
        self.feature_scale_ = np.where(source_x.std(axis=0) == 0, 1.0, source_x.std(axis=0))
        canonical_x = self._canonicalize_source(source_x, source_site)
        train_x, train_y = self._augmented_training_pairs(canonical_x)
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(train_x, train_y)
        return self

    def transform(self, x: np.ndarray, site: np.ndarray | None = None) -> np.ndarray:
        del site
        if self.model_ is None:
            raise RuntimeError("fit must be called before transform")
        return np.asarray(self.model_.predict(np.asarray(x, dtype=float)), dtype=float)

    def _canonicalize_source(self, x: np.ndarray, site: np.ndarray) -> np.ndarray:
        canonical = x.copy()
        for idx, site_id in enumerate(site):
            canonical[idx] = x[idx] - self.site_offsets_[site_id]
        return canonical

    def _augmented_training_pairs(self, canonical_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.feature_scale_ is None:
            raise RuntimeError("fit must be called before creating training pairs")
        rng = np.random.default_rng(self.random_seed)
        offsets = list(self.site_offsets_.values()) or [np.zeros(canonical_x.shape[1])]
        train_inputs = [canonical_x]
        train_targets = [canonical_x]
        for _ in range(self.n_augments):
            offset_indices = rng.integers(0, len(offsets), size=canonical_x.shape[0])
            sampled_offsets = np.vstack([offsets[idx] for idx in offset_indices])
            noise = rng.normal(
                scale=self.noise_strength * self.feature_scale_,
                size=canonical_x.shape,
            )
            train_inputs.append(canonical_x + sampled_offsets + noise)
            train_targets.append(canonical_x)
        return np.vstack(train_inputs), np.vstack(train_targets)


@dataclass
class ClinicalPreservingRidgeDenoisingHarmonizer(RidgeDenoisingHarmonizer):
    """Ridge denoising variant that preserves source label-predictive direction."""

    clinical_preservation_strength: float = 0.5
    clinical_direction_: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        site: np.ndarray,
        y: np.ndarray | None = None,
        *,
        target_x: np.ndarray | None = None,
        target_site: np.ndarray | None = None,
    ) -> "ClinicalPreservingRidgeDenoisingHarmonizer":
        super().fit(x, site, y, target_x=target_x, target_site=target_site)
        if y is not None and self.global_mean_ is not None:
            source_x = np.asarray(x, dtype=float)
            source_site = np.asarray(site).astype(str)
            labels = np.asarray(y).astype(int)
            if set(labels.tolist()) == {0, 1}:
                canonical_x = self._canonicalize_source(source_x, source_site)
                direction = canonical_x[labels == 1].mean(axis=0) - canonical_x[
                    labels == 0
                ].mean(axis=0)
                norm = float(np.linalg.norm(direction))
                if norm > 1e-12:
                    self.clinical_direction_ = direction / norm
        return self

    def transform(self, x: np.ndarray, site: np.ndarray | None = None) -> np.ndarray:
        original = np.asarray(x, dtype=float)
        denoised = super().transform(original, site)
        if self.clinical_direction_ is None:
            return denoised
        strength = float(np.clip(self.clinical_preservation_strength, 0.0, 1.0))
        delta = original - denoised
        projection = delta @ self.clinical_direction_
        preserved = np.outer(projection, self.clinical_direction_)
        return denoised + strength * preserved
