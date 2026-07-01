from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DiffusionHarmonizerConfig:
    latent_dim: int
    num_steps: int = 100
    site_conditioned: bool = True
    clinical_preservation_weight: float = 1.0
    domain_confusion_weight: float = 0.1
    content_preservation_weight: float = 0.1


class DiffusionHarmonizer:
    """Placeholder interface for future latent diffusion harmonizer.

    This class intentionally does not implement a fake diffusion model. Agents should
    add a real PyTorch implementation only after baseline and claim-gate experiments
    are in place.
    """

    def __init__(self, config: DiffusionHarmonizerConfig) -> None:
        self.config = config
        self.is_fitted = False

    def fit(
        self,
        x: np.ndarray,
        site: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "DiffusionHarmonizer":
        raise NotImplementedError("Implement PyTorch latent diffusion training here.")

    def transform(self, x: np.ndarray, site: np.ndarray | None = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("fit must be called before transform")
        raise NotImplementedError("Implement latent denoising / harmonization here.")
