"""Graph regularization for CT-derived abdominal FWI."""

from __future__ import annotations

import numpy as np


def active_degree(mask: np.ndarray) -> np.ndarray:
    """Return four-neighbor graph degree for every active voxel."""

    degree = np.zeros(mask.shape, dtype=np.float32)
    degree[1:, :] += mask[:-1, :]
    degree[:-1, :] += mask[1:, :]
    degree[:, 1:] += mask[:, :-1]
    degree[:, :-1] += mask[:, 1:]
    return degree[mask].astype(np.float32)


def laplacian_apply(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Apply the positive semidefinite four-neighbor graph Laplacian."""

    full = np.zeros(mask.shape, dtype=np.float32)
    full[mask] = values
    out = active_degree(mask) * values
    yy, xx = np.where(mask)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        y = yy + dy
        x = xx + dx
        valid = (y >= 0) & (x >= 0) & (y < mask.shape[0]) & (x < mask.shape[1])
        out[valid] -= full[y[valid], x[valid]]
    return out.astype(np.float32)
