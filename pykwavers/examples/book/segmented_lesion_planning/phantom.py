"""Deterministic segmentation fixture for Chapter 32."""

from __future__ import annotations

import numpy as np

from .types import SegmentationGrid, Tissue


def build_segmented_therapy_phantom(
    n: int = 96,
    spacing_m: float = 1.2e-3,
) -> SegmentationGrid:
    """Create a tissue-label planning slice with tumor and protected anatomy.

    The phantom encodes the planning problem instead of a display-only image:
    outside-body air, an internal bowel-gas pocket, a subcutaneous fat shell,
    rib-like bone arcs, an elliptical tumor, and a protected structure adjacent
    to the tumor.  All masks derive analytically from ellipses and angular
    sectors, so tests can assert exact planning invariants without random data.
    """

    if n < 48:
        raise ValueError("n must be at least 48 for the analytic phantom")
    if spacing_m <= 0.0:
        raise ValueError("spacing_m must be positive")

    axis = (np.arange(n, dtype=float) - 0.5 * (n - 1)) * spacing_m
    x, y = np.meshgrid(axis, axis, indexing="ij")
    labels = np.full((n, n), int(Tissue.AIR), dtype=np.uint8)

    body_level = (x / 0.050) ** 2 + (y / 0.040) ** 2
    body = body_level <= 1.0
    labels[body] = int(Tissue.NORMAL)

    fat = body & (body_level >= 0.72)
    labels[fat] = int(Tissue.FAT)

    theta = np.arctan2(y, x)
    radius = np.sqrt((x / 0.050) ** 2 + (y / 0.040) ** 2)
    anterior_ribs = body & (radius > 0.62) & (radius < 0.86) & (y > 0.016) & (np.abs(x) > 0.012)
    posterior_spine = body & (x < -0.032) & (np.abs(y) < 0.010)
    oblique_rib = body & (theta > 2.20) & (theta < 2.72) & (radius > 0.45) & (radius < 0.74)
    labels[anterior_ribs | posterior_spine | oblique_rib] = int(Tissue.BONE)

    internal_air = ((x + 0.017) / 0.0065) ** 2 + ((y + 0.008) / 0.010) ** 2 <= 1.0
    labels[internal_air] = int(Tissue.AIR)

    tumor = ((x - 0.012) / 0.0105) ** 2 + ((y + 0.007) / 0.0072) ** 2 <= 1.0
    labels[tumor] = int(Tissue.TUMOR)

    protected = ((x - 0.003) / 0.006) ** 2 + ((y - 0.004) / 0.017) ** 2 <= 1.0
    protected &= body & ~tumor
    labels[protected] = int(Tissue.AVOID)

    body_mask = body & ~internal_air
    return SegmentationGrid(labels=labels, body_mask=body_mask, spacing_m=float(spacing_m))
