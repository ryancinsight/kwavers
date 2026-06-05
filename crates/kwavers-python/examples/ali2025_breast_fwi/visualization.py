"""Plotting helpers for Ali 2025 replication artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .volume import orthographic_slices, require_volume


def write_orthographic_plot(reference: np.ndarray, estimate: np.ndarray, path: Path) -> None:
    reference = require_volume(reference)
    estimate = require_volume(estimate)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    ref_slices = orthographic_slices(reference)
    est_slices = orthographic_slices(estimate)
    vmin = float(min(np.min(reference), np.min(estimate)))
    vmax = float(max(np.max(reference), np.max(estimate)))

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for row, slices in enumerate((ref_slices, est_slices)):
        for col, image in enumerate(slices):
            ax = axes[row][col]
            im = ax.imshow(image.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0][0].set_ylabel("phantom")
    axes[1][0].set_ylabel("reconstruction")
    for col, title in enumerate(("x", "y", "z")):
        axes[0][col].set_title(f"{title}-center")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="sound speed [m/s]")
    fig.savefig(path, dpi=160)
    plt.close(fig)
