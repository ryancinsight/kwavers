from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel

from .data import normalize_unit


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None and np.any(mask):
        av = a[mask].astype(np.float64)
        bv = b[mask].astype(np.float64)
    else:
        av = a.astype(np.float64).ravel()
        bv = b.astype(np.float64).ravel()
    av = av - float(av.mean())
    bv = bv - float(bv.mean())
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1.0e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


def normalized_mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64, mask: np.ndarray | None = None) -> float:
    au = normalize_unit(a)
    bu = normalize_unit(b)
    if mask is not None and np.any(mask):
        av = au[mask].ravel()
        bv = bu[mask].ravel()
    else:
        av = au.ravel()
        bv = bu.ravel()
    hist, _, _ = np.histogram2d(av, bv, bins=bins, range=((0.0, 1.0), (0.0, 1.0)))
    probability = hist / float(np.sum(hist))
    pa = probability.sum(axis=1)
    pb = probability.sum(axis=0)
    pab = probability[probability > 0.0]
    pa = pa[pa > 0.0]
    pb = pb[pb > 0.0]
    ha = -float(np.sum(pa * np.log(pa)))
    hb = -float(np.sum(pb * np.log(pb)))
    hab = -float(np.sum(pab * np.log(pab)))
    if hab <= 1.0e-12:
        return 0.0
    return (ha + hb) / hab


def normalized_mean_squared_error(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    au = normalize_unit(a)
    bu = normalize_unit(b)
    if mask is not None and np.any(mask):
        diff = au[mask].astype(np.float64) - bu[mask].astype(np.float64)
    else:
        diff = au.astype(np.float64).ravel() - bu.astype(np.float64).ravel()
    return float(np.mean(diff * diff))


def registration_quality(ncc: float, nmi: float, mse: float) -> float:
    return float(nmi + 0.25 * ncc - 0.10 * mse)


def multimodal_affine_score(
    fixed: np.ndarray,
    moving: np.ndarray,
    metric_mask: np.ndarray,
    fixed_mask: np.ndarray,
    moving_mask: np.ndarray,
) -> float:
    nmi = normalized_mutual_information(fixed, moving, mask=metric_mask)
    mse = normalized_mean_squared_error(fixed, moving, metric_mask)
    edge = binary_edge_overlap(fixed, moving)
    dice = dice_overlap(fixed_mask, moving_mask)
    return float(nmi + 0.25 * edge + 0.10 * dice - 0.10 * mse)


def binary_edge_overlap(a: np.ndarray, b: np.ndarray, quantile: float = 0.85) -> float:
    edge_a = _edge_mask(a, quantile)
    edge_b = _edge_mask(b, quantile)
    union = edge_a | edge_b
    if not np.any(union):
        return 0.0
    return float(np.count_nonzero(edge_a & edge_b) / np.count_nonzero(union))


def dice_overlap(a: np.ndarray, b: np.ndarray) -> float:
    av = a.astype(bool)
    bv = b.astype(bool)
    denom = np.count_nonzero(av) + np.count_nonzero(bv)
    if denom == 0:
        return 0.0
    return float(2.0 * np.count_nonzero(av & bv) / denom)


def foreground_mask(data: np.ndarray) -> np.ndarray:
    normalized = normalize_unit(data)
    return normalized > 0.05


def _edge_mask(data: np.ndarray, quantile: float) -> np.ndarray:
    normalized = normalize_unit(data)
    grad = np.sqrt(
        sobel(normalized, axis=0, mode="nearest") ** 2
        + sobel(normalized, axis=1, mode="nearest") ** 2
        + sobel(normalized, axis=2, mode="nearest") ** 2
    )
    threshold = float(np.quantile(grad, quantile))
    return grad > threshold
