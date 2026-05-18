"""Dense rendered-field acceptance metrics for Chapter 32 planning."""

from __future__ import annotations

import numpy as np

from .types import HybridPlanConfig, SegmentationGrid, Tissue


def normalized_intensity_from_pressure(
    grid: SegmentationGrid,
    pressure_map: np.ndarray,
) -> np.ndarray:
    intensity = np.abs(pressure_map) ** 2
    tumor_peak = max(float(np.max(intensity[grid.mask(Tissue.TUMOR)])), 1e-12)
    return intensity / tumor_peak


def dense_field_metrics(
    grid: SegmentationGrid,
    normalized_intensity: np.ndarray,
    target_centroid_m: np.ndarray,
    config: HybridPlanConfig,
) -> dict[str, float | bool]:
    tumor = grid.mask(Tissue.TUMOR)
    avoid = grid.mask(Tissue.AVOID)
    normal = (grid.labels == int(Tissue.NORMAL)) | (grid.labels == int(Tissue.FAT))
    sidelobe = body_sidelobe_mask(grid, target_centroid_m, config)
    protected_peak = float(np.max(normalized_intensity[avoid])) if np.any(avoid) else 0.0
    normal_mean = float(np.mean(normalized_intensity[normal])) if np.any(normal) else 0.0
    sidelobe_peak = float(np.max(normalized_intensity[sidelobe])) if np.any(sidelobe) else 0.0
    sidelobe_p99 = float(np.quantile(normalized_intensity[sidelobe], 0.99)) if np.any(sidelobe) else 0.0
    coverage = float(np.count_nonzero(normalized_intensity[tumor] >= 0.35) / np.count_nonzero(tumor))
    tumor_mean = float(np.mean(normalized_intensity[tumor]))
    return {
        "tumor_mean_intensity": tumor_mean,
        "tumor_coverage_fraction": coverage,
        "protected_peak_ratio": protected_peak,
        "normal_mean_ratio": normal_mean,
        "body_sidelobe_peak_ratio": sidelobe_peak,
        "body_sidelobe_p99_ratio": sidelobe_p99,
        "target_dominant": sidelobe_peak <= config.target_dominance_ratio,
    }


def dense_acceptance_key(metrics: dict[str, float | bool]) -> tuple[float, ...]:
    target_dominant = 1.0 if bool(metrics["target_dominant"]) else 0.0
    return (
        target_dominant,
        -float(metrics["body_sidelobe_peak_ratio"]),
        -float(metrics["body_sidelobe_p99_ratio"]),
        float(metrics["tumor_coverage_fraction"]),
        float(metrics["tumor_mean_intensity"]),
        -float(metrics["protected_peak_ratio"]),
        -float(metrics["normal_mean_ratio"]),
    )


def body_sidelobe_mask(
    grid: SegmentationGrid,
    target_centroid_m: np.ndarray,
    config: HybridPlanConfig | None,
) -> np.ndarray:
    cfg = config or HybridPlanConfig()
    x, y = grid.coordinates_m()
    distance = np.sqrt((x - target_centroid_m[0]) ** 2 + (y - target_centroid_m[1]) ** 2)
    return grid.body_mask & ~grid.mask(Tissue.TUMOR) & (distance >= cfg.sidelobe_exclusion_radius_m)
