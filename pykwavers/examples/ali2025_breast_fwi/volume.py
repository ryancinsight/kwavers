"""Volume validation and slicing helpers for Ali 2025 replication."""

from __future__ import annotations

import numpy as np


def require_volume(volume: np.ndarray) -> np.ndarray:
    array = np.asarray(volume, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("volume must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("volume contains nonfinite sound-speed values")
    if np.any(array <= 0.0):
        raise ValueError("sound-speed volume must be strictly positive")
    return array


def paired_arrays(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = require_volume(np.asarray(reference, dtype=np.float64))
    est = require_volume(np.asarray(estimate, dtype=np.float64))
    if ref.shape != est.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape}, estimate {est.shape}")
    return ref.ravel(), est.ravel()


def orthographic_slices(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    volume = require_volume(volume)
    cx, cy, cz = (axis // 2 for axis in volume.shape)
    return volume[cx, :, :], volume[:, cy, :], volume[:, :, cz]
