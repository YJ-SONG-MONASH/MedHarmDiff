"""MedHarmDiff: diffusion harmonization scaffolding for multi-center medical data."""

import os

for _thread_env_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_var, "1")

__all__ = [
    "data_schema",
    "metrics",
    "claim_gate",
    "protocol",
    "synthetic",
    "benchmark",
    "io",
    "confounding",
    "latent_denoising",
    "latent_diffusion",
]
