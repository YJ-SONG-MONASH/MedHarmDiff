from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .latent_denoising import (
    ClinicalPreservingRidgeDenoisingHarmonizer,
    RidgeDenoisingHarmonizer,
)
from .latent_diffusion import (
    ClinicalPreservingTimeConditionedRidgeDiffusionHarmonizer,
    TimeConditionedRidgeDiffusionHarmonizer,
)


class Harmonizer(Protocol):
    def fit(
        self,
        x: np.ndarray,
        site: np.ndarray,
        y: np.ndarray | None = None,
        *,
        target_x: np.ndarray | None = None,
        target_site: np.ndarray | None = None,
    ) -> "Harmonizer": ...

    def transform(self, x: np.ndarray, site: np.ndarray | None = None) -> np.ndarray: ...


@dataclass
class IdentityHarmonizer:
    """No-harmonization baseline."""

    def fit(
        self,
        x: np.ndarray,
        site: np.ndarray,
        y: np.ndarray | None = None,
        *,
        target_x: np.ndarray | None = None,
        target_site: np.ndarray | None = None,
    ) -> "IdentityHarmonizer":
        return self

    def transform(self, x: np.ndarray, site: np.ndarray | None = None) -> np.ndarray:
        return np.asarray(x, dtype=float)


@dataclass
class StandardizeBySourceHarmonizer:
    """Simple source-only standardization baseline.

    This is not a replacement for ComBat/CORAL/MMD. It is a lightweight smoke-test
    harmonizer used before adding full baselines.
    """

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(
        self,
        x: np.ndarray,
        site: np.ndarray,
        y: np.ndarray | None = None,
        *,
        target_x: np.ndarray | None = None,
        target_site: np.ndarray | None = None,
    ) -> "StandardizeBySourceHarmonizer":
        arr = np.asarray(x, dtype=float)
        self.mean_ = arr.mean(axis=0)
        self.std_ = np.where(arr.std(axis=0) == 0, 1.0, arr.std(axis=0))
        return self

    def transform(self, x: np.ndarray, site: np.ndarray | None = None) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("fit must be called before transform")
        return (np.asarray(x, dtype=float) - self.mean_) / self.std_


@dataclass
class CenterMeanHarmonizer:
    """ComBat-style fallback that removes center mean offsets.

    This is a documented lightweight fallback, not a full empirical-Bayes ComBat
    implementation. If a ComBat dependency is added later, keep this class as the
    no-extra-dependency smoke-test baseline.
    """

    global_mean_: np.ndarray | None = None
    center_means_: dict[str, np.ndarray] = field(default_factory=dict)

    def fit(
        self,
        x: np.ndarray,
        site: np.ndarray,
        y: np.ndarray | None = None,
        *,
        target_x: np.ndarray | None = None,
        target_site: np.ndarray | None = None,
    ) -> "CenterMeanHarmonizer":
        source_x = np.asarray(x, dtype=float)
        source_site = np.asarray(site).astype(str)
        self.global_mean_ = source_x.mean(axis=0)
        self.center_means_ = _means_by_site(source_x, source_site)
        if target_x is not None and target_site is not None:
            self.center_means_.update(
                _means_by_site(
                    np.asarray(target_x, dtype=float),
                    np.asarray(target_site).astype(str),
                )
            )
        return self

    def transform(self, x: np.ndarray, site: np.ndarray | None = None) -> np.ndarray:
        if self.global_mean_ is None:
            raise RuntimeError("fit must be called before transform")
        arr = np.asarray(x, dtype=float)
        if site is None:
            return arr
        transformed = arr.copy()
        for idx, site_id in enumerate(np.asarray(site).astype(str)):
            if site_id in self.center_means_:
                transformed[idx] = arr[idx] - self.center_means_[site_id] + self.global_mean_
        return transformed


@dataclass
class CoralHarmonizer:
    """Align target-center covariance to the source-center covariance."""

    source_mean_: np.ndarray | None = None
    target_mean_: np.ndarray | None = None
    transform_: np.ndarray | None = None
    target_sites_: set[str] = field(default_factory=set)

    def fit(
        self,
        x: np.ndarray,
        site: np.ndarray,
        y: np.ndarray | None = None,
        *,
        target_x: np.ndarray | None = None,
        target_site: np.ndarray | None = None,
    ) -> "CoralHarmonizer":
        if target_x is None or target_site is None:
            raise ValueError("CORAL requires target_x and target_site without target labels")
        source_x = np.asarray(x, dtype=float)
        target_arr = np.asarray(target_x, dtype=float)
        self.source_mean_ = source_x.mean(axis=0)
        self.target_mean_ = target_arr.mean(axis=0)
        source_cov = np.cov(source_x, rowvar=False) + np.eye(source_x.shape[1]) * 1e-6
        target_cov = np.cov(target_arr, rowvar=False) + np.eye(target_arr.shape[1]) * 1e-6
        self.transform_ = _matrix_inv_sqrt(target_cov) @ _matrix_sqrt(source_cov)
        self.target_sites_ = set(np.asarray(target_site).astype(str).tolist())
        return self

    def transform(self, x: np.ndarray, site: np.ndarray | None = None) -> np.ndarray:
        if self.source_mean_ is None or self.target_mean_ is None or self.transform_ is None:
            raise RuntimeError("fit must be called before transform")
        arr = np.asarray(x, dtype=float)
        if site is None:
            return (arr - self.target_mean_) @ self.transform_ + self.source_mean_
        site_values = np.asarray(site).astype(str)
        transformed = arr.copy()
        mask = np.array([site_id in self.target_sites_ for site_id in site_values], dtype=bool)
        transformed[mask] = (arr[mask] - self.target_mean_) @ self.transform_ + self.source_mean_
        return transformed


@dataclass
class MMDMeanAlignmentHarmonizer:
    """Simple distribution-matching baseline that shifts target means to source means."""

    source_mean_: np.ndarray | None = None
    target_mean_: np.ndarray | None = None
    target_sites_: set[str] = field(default_factory=set)

    def fit(
        self,
        x: np.ndarray,
        site: np.ndarray,
        y: np.ndarray | None = None,
        *,
        target_x: np.ndarray | None = None,
        target_site: np.ndarray | None = None,
    ) -> "MMDMeanAlignmentHarmonizer":
        if target_x is None or target_site is None:
            raise ValueError(
                "MMD alignment requires target_x and target_site without target labels"
            )
        self.source_mean_ = np.asarray(x, dtype=float).mean(axis=0)
        self.target_mean_ = np.asarray(target_x, dtype=float).mean(axis=0)
        self.target_sites_ = set(np.asarray(target_site).astype(str).tolist())
        return self

    def transform(self, x: np.ndarray, site: np.ndarray | None = None) -> np.ndarray:
        if self.source_mean_ is None or self.target_mean_ is None:
            raise RuntimeError("fit must be called before transform")
        arr = np.asarray(x, dtype=float)
        if site is None:
            return arr - self.target_mean_ + self.source_mean_
        site_values = np.asarray(site).astype(str)
        transformed = arr.copy()
        mask = np.array([site_id in self.target_sites_ for site_id in site_values], dtype=bool)
        transformed[mask] = arr[mask] - self.target_mean_ + self.source_mean_
        return transformed


@dataclass
class DiffusionPlaceholderHarmonizer(StandardizeBySourceHarmonizer):
    """Explicit non-claimable placeholder until a real diffusion model exists."""


def _means_by_site(x: np.ndarray, site: np.ndarray) -> dict[str, np.ndarray]:
    return {site_id: x[site == site_id].mean(axis=0) for site_id in sorted(set(site.tolist()))}


def _matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 1e-8, None)
    return vectors @ np.diag(np.sqrt(values)) @ vectors.T


def _matrix_inv_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 1e-8, None)
    return vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
