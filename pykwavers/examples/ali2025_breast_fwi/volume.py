"""Plotting volume validation and slicing helpers for Ali 2025 replication."""

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


def orthographic_slices(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    volume = require_volume(volume)
    cx, cy, cz = (axis // 2 for axis in volume.shape)
    return volume[cx, :, :], volume[:, cy, :], volume[:, :, cz]
