"""Centroid-region reconstruction views for Chapter 27.

The ROI is a reproducible centroid crop over the CT-derived brain mask.  It is
an anatomical proxy for inspecting deep midline slices and is not a segmented
pons/thalamus label map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RoiBounds:
    """Inclusive-exclusive voxel bounds for an x/y crop."""

    x_start: int
    x_stop: int
    y_start: int
    y_stop: int

    def as_slices(self) -> tuple[slice, slice]:
        return slice(self.x_start, self.x_stop), slice(self.y_start, self.y_stop)

    def as_list(self) -> list[int]:
        return [self.x_start, self.x_stop, self.y_start, self.y_stop]


def brain_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Return the voxel centroid of a nonempty brain mask."""

    points = np.argwhere(np.asarray(mask, dtype=bool))
    if points.size == 0:
        raise ValueError("brain centroid requires a nonempty mask")
    centroid = points.mean(axis=0)
    return float(centroid[0]), float(centroid[1])


def centroid_bounds(mask: np.ndarray, spacing_m: float, half_width_mm: float) -> RoiBounds:
    """Build a clamped square crop centered on the brain-mask centroid."""

    if not np.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("spacing_m must be finite and positive")
    if not np.isfinite(half_width_mm) or half_width_mm <= 0.0:
        raise ValueError("half_width_mm must be finite and positive")
    nx, ny = np.asarray(mask).shape
    half_voxels = max(1, int(np.ceil(half_width_mm * 1.0e-3 / spacing_m)))
    cx, cy = brain_centroid(mask)
    ix = int(np.floor(cx + 0.5))
    iy = int(np.floor(cy + 0.5))
    return RoiBounds(
        max(0, ix - half_voxels),
        min(nx, ix + half_voxels + 1),
        max(0, iy - half_voxels),
        min(ny, iy + half_voxels + 1),
    )


def crop_roi(image: np.ndarray, bounds: RoiBounds) -> np.ndarray:
    """Return an x/y crop using inclusive-exclusive ROI bounds."""

    return np.asarray(image)[bounds.as_slices()]


def roi_extent_mm(shape: tuple[int, int], spacing_m: float, bounds: RoiBounds) -> tuple[float, float, float, float]:
    """Return imshow extents for a crop within a centered full image grid."""

    nx, ny = shape
    spacing_mm = 1.0e3 * spacing_m
    return (
        (-0.5 * nx + bounds.x_start) * spacing_mm,
        (-0.5 * nx + bounds.x_stop) * spacing_mm,
        (-0.5 * ny + bounds.y_start) * spacing_mm,
        (-0.5 * ny + bounds.y_stop) * spacing_mm,
    )


def stack_roi_metadata(stack: list[tuple[dict[str, Any], dict[str, float | int]]], half_width_mm: float) -> dict[str, Any]:
    """Return JSON-serializable centroid ROI metadata for a slice stack."""

    regions = []
    skipped = []
    for result, _ in stack:
        mask = np.asarray(result["brain_mask"], dtype=bool)
        if not mask.any():
            skipped.append(int(result["source_slice_index"]))
            continue
        spacing_m = float(result["spacing_m"])
        bounds = centroid_bounds(mask, spacing_m, half_width_mm)
        cx, cy = brain_centroid(mask)
        regions.append(
            {
                "source_slice_index": int(result["source_slice_index"]),
                "centroid_voxel": [cx, cy],
                "bounds_voxels": bounds.as_list(),
                "half_width_mm": float(half_width_mm),
            }
        )
    return {
        "half_width_mm": float(half_width_mm),
        "slice_indices": [record["source_slice_index"] for record in regions],
        "skipped_empty_slice_indices": skipped,
        "regions": regions,
    }


def plot_centroid_roi_stack(
    stack: list[tuple[dict[str, Any], dict[str, float | int]]],
    half_width_mm: float,
    out_dir: Path,
) -> dict[str, Any]:
    """Write a centroid-cropped target/reconstruction/error stack figure."""

    metadata = stack_roi_metadata(stack, half_width_mm)
    valid_stack = [
        (result, metrics)
        for result, metrics in stack
        if np.asarray(result["brain_mask"], dtype=bool).any()
    ]
    if not valid_stack:
        raise ValueError("centroid ROI requires at least one nonempty brain-mask slice")
    targets = [np.asarray(result["target_sound_speed_m_s"], dtype=float) for result, _ in valid_stack]
    enhanced = [
        np.asarray(result["enhanced_reconstruction_sound_speed_m_s"], dtype=float)
        for result, _ in valid_stack
    ]
    masks = [np.asarray(result["brain_mask"], dtype=bool) for result, _ in valid_stack]
    bounds = [
        centroid_bounds(mask, float(result["spacing_m"]), half_width_mm)
        for (result, _), mask in zip(valid_stack, masks)
    ]
    target_rois = [crop_roi(image, box) for image, box in zip(targets, bounds)]
    enhanced_rois = [crop_roi(image, box) for image, box in zip(enhanced, bounds)]
    mask_rois = [crop_roi(mask, box) for mask, box in zip(masks, bounds)]
    speed_values = np.concatenate([roi[mask] for roi, mask in zip(target_rois, mask_rois)])
    vmin = np.percentile(speed_values, 2)
    vmax = np.percentile(speed_values, 98)
    errors = [
        np.where(mask, recon - target, np.nan)
        for target, recon, mask in zip(target_rois, enhanced_rois, mask_rois)
    ]
    emax = max(np.nanpercentile(np.abs(error), 98) for error in errors)
    n_slices = len(valid_stack)

    fig, axes = plt.subplots(3, n_slices, figsize=(1.8 * n_slices, 6.3), constrained_layout=True)
    if n_slices == 1:
        axes = axes.reshape(3, 1)

    speed_im = None
    error_im = None
    for col, ((result, metrics), target, recon, error, mask, box) in enumerate(
        zip(valid_stack, target_rois, enhanced_rois, errors, mask_rois, bounds)
    ):
        extent = roi_extent_mm(masks[col].shape, float(result["spacing_m"]), box)
        for row, image, label in ((0, target, "target"), (1, recon, "enhanced FWI")):
            ax = axes[row, col]
            speed_im = ax.imshow(image.T, cmap="viridis", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
            _contour_mask(ax, mask, extent)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(
                    f"slice {int(result['source_slice_index'])}\nr={float(metrics['pearson_correlation']):.3f}",
                    fontsize=8,
                )
            if col == 0:
                ax.set_ylabel(label)

        ax = axes[2, col]
        error_im = ax.imshow(error.T, cmap="coolwarm", origin="lower", extent=extent, vmin=-emax, vmax=emax)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("error")

    if speed_im is not None:
        fig.colorbar(speed_im, ax=axes[:2, :].ravel().tolist(), label="m/s", shrink=0.82)
    if error_im is not None:
        fig.colorbar(error_im, ax=axes[2, :].ravel().tolist(), label="m/s", shrink=0.82)

    _savefig(fig, out_dir, "fig07_centroid_pons_thalamus_roi")
    plt.close(fig)
    return metadata


def _contour_mask(ax: plt.Axes, mask: np.ndarray, extent: tuple[float, float, float, float]) -> None:
    x = np.linspace(extent[0], extent[1], mask.shape[0])
    y = np.linspace(extent[2], extent[3], mask.shape[1])
    ax.contour(x, y, mask.T.astype(float), levels=[0.5], colors=["white"], linewidths=0.6)


def _savefig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    print(f"  saved: docs/book/figures/ch27/{name}.{{pdf,png}}")
