from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


@dataclass(frozen=True)
class ClassificationMetrics:
    auc: float
    accuracy: float


@dataclass(frozen=True)
class HarmonizationMetrics:
    site_accuracy: float
    site_auc: float | None
    mmd_rbf: float
    coral_distance: float


def binary_classification_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
) -> ClassificationMetrics:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(y_score), dtype=float)
    if y.size == 0:
        return ClassificationMetrics(auc=0.0, accuracy=0.0)
    auc = 0.5 if len(set(y.tolist())) < 2 else float(roc_auc_score(y, s))
    pred = (s >= 0.5).astype(int)
    return ClassificationMetrics(
        auc=round(auc, 4),
        accuracy=round(float(accuracy_score(y, pred)), 4),
    )


def coral_distance(source: np.ndarray, target: np.ndarray) -> float:
    source = _as_2d(source)
    target = _as_2d(target)
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target must have the same feature dimension")
    cs = np.cov(source, rowvar=False)
    ct = np.cov(target, rowvar=False)
    return round(float(np.linalg.norm(cs - ct, ord="fro") / (4 * source.shape[1] ** 2)), 6)


def mmd_rbf(
    source: np.ndarray,
    target: np.ndarray,
    *,
    gamma: float | None = None,
    max_samples: int | None = 2048,
) -> float:
    source = _as_2d(source)
    target = _as_2d(target)
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target must have the same feature dimension")
    source = _cap_rows(source, max_samples)
    target = _cap_rows(target, max_samples)
    if gamma is None:
        gamma = 1.0 / max(1, source.shape[1])
    k_xx = _rbf_kernel(source, source, gamma).mean()
    k_yy = _rbf_kernel(target, target, gamma).mean()
    k_xy = _rbf_kernel(source, target, gamma).mean()
    return round(float(k_xx + k_yy - 2 * k_xy), 6)


def _rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    a2 = np.sum(a * a, axis=1, keepdims=True)
    b2 = np.sum(b * b, axis=1, keepdims=True).T
    dist = np.maximum(a2 + b2 - 2 * a @ b.T, 0.0)
    return np.exp(-gamma * dist)


def _cap_rows(array: np.ndarray, max_samples: int | None) -> np.ndarray:
    if max_samples is None or array.shape[0] <= max_samples:
        return array
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")
    indices = np.linspace(0, array.shape[0] - 1, num=max_samples, dtype=int)
    return array[indices]


def _as_2d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim != 2:
        raise ValueError("expected a 2D feature array")
    if arr.shape[0] < 2:
        raise ValueError("at least two rows are required")
    return arr
