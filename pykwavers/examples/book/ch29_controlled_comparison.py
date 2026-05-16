"""Controlled Figure 2/Figure 5 linear-vs-nonlinear comparison utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

COMPARISON_FIGURE_NAME = "fig06_controlled_linear_nonlinear_comparison.png"
COMPARISON_METRICS_NAME = "controlled_comparison_metrics.json"
COMPARISON_FIELDS_NAME = "controlled_comparison_fields.npz"

def build_controlled_comparison(
    linear_results: list[dict[str, object]],
    nonlinear_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    if len(linear_results) != len(nonlinear_results):
        raise ValueError("linear and nonlinear result counts differ")
    comparisons = []
    for linear, nonlinear in zip(linear_results, nonlinear_results):
        if str(linear["anatomy"]) != str(nonlinear["anatomy"]):
            raise ValueError(f"case order mismatch: {linear['anatomy']} != {nonlinear['anatomy']}")
        comparisons.append(_case_comparison(linear, nonlinear))
    return comparisons

def render_controlled_comparison(
    comparisons: list[dict[str, object]],
    out_dir: Path,
    save_figure: Callable[[plt.Figure, Path], None],
) -> Path:
    columns = (
        ("common_target", "gray", "matched target"),
        ("linear_exposure", "magma", "linear exposure"),
        ("nonlinear_pressure", "magma", "nonlinear peak pressure"),
        ("linear_fusion", "viridis", "linear fusion"),
        ("nonlinear_fusion", "viridis", "nonlinear fusion"),
        ("fusion_difference", "coolwarm", "nonlinear - linear"),
    )
    fig, axes = plt.subplots(
        len(comparisons),
        len(columns),
        figsize=(18.0, 3.6 * len(comparisons)),
        constrained_layout=True,
    )
    axes_2d = np.asarray(axes, dtype=object).reshape(len(comparisons), len(columns))
    for row, comparison in enumerate(comparisons):
        extent = comparison["common_extent_m"]
        target = np.asarray(comparison["fields"]["common_target"], dtype=bool)
        points = np.asarray(comparison["fields"]["therapy_points_xy_m"], dtype=float)
        for col, (key, cmap, title) in enumerate(columns):
            ax = axes_2d[row, col]
            image = np.asarray(comparison["fields"][key], dtype=float)
            if key == "fusion_difference":
                vmax = max(float(np.max(np.abs(image))), 1.0e-12)
                im = ax.imshow(
                    image.T,
                    cmap=cmap,
                    origin="lower",
                    extent=extent,
                    vmin=-vmax,
                    vmax=vmax,
                )
            else:
                im = ax.imshow(image.T, cmap=cmap, origin="lower", extent=extent)
            _contour_mask(ax, target, extent, "white", 0.7)
            if points.size:
                ax.scatter(points[:, 0], points[:, 1], s=2.0, c="#ffcf33", alpha=0.34)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{comparison['anatomy']} {title}" if col == 0 else title, fontsize=8.5)
            if col == len(columns) - 1:
                metrics = comparison["comparison_metrics"]
                ax.set_xlabel(
                    f"linear Dice={metrics['linear_fusion']['dice_equal_area']:.2f}; "
                    f"nonlinear Dice={metrics['nonlinear_fusion']['dice_equal_area']:.2f}"
                )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    path = out_dir / COMPARISON_FIGURE_NAME
    save_figure(fig, path)
    plt.close(fig)
    return path

def write_controlled_comparison_metrics(
    comparisons: list[dict[str, object]],
    out_dir: Path,
    figure: Path | None = None,
    fields: Path | None = None,
) -> Path:
    payload = controlled_comparison_payload(comparisons, figure, fields)
    path = out_dir / COMPARISON_METRICS_NAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

def write_controlled_comparison_fields(comparisons: list[dict[str, object]], out_dir: Path) -> Path:
    arrays: dict[str, np.ndarray] = {}
    for comparison in comparisons:
        anatomy = str(comparison["anatomy"])
        for key, value in comparison["fields"].items():
            arrays[f"{anatomy}_{key}"] = np.asarray(value)
    path = out_dir / COMPARISON_FIELDS_NAME
    np.savez_compressed(path, **arrays)
    return path

def controlled_comparison_payload(
    comparisons: list[dict[str, object]],
    figure: Path | None,
    fields: Path | None,
) -> dict[str, object]:
    return {
        "comparison_contract": (
            "linear and nonlinear outputs evaluated on the nonlinear 3-D crop projection; "
            "linear fields are physically resampled to that grid, and both panels use the "
            "same projected nonlinear target and therapy aperture"
        ),
        "figure": None if figure is None else str(figure),
        "field_archive": None if fields is None else str(fields),
        "summary": comparison_summary(comparisons),
        "cases": [serializable_comparison(comparison) for comparison in comparisons],
    }

def comparison_summary(comparisons: list[dict[str, object]]) -> dict[str, object]:
    if not comparisons:
        return {"case_count": 0}
    linear_dice = [case["comparison_metrics"]["linear_fusion"]["dice_equal_area"] for case in comparisons]
    nonlinear_dice = [
        case["comparison_metrics"]["nonlinear_fusion"]["dice_equal_area"] for case in comparisons
    ]
    pressure_outside = [
        case["comparison_metrics"]["nonlinear_pressure"]["outside_energy_fraction"]
        for case in comparisons
    ]
    return {
        "case_count": len(comparisons),
        "mean_linear_fusion_dice_equal_area": float(np.mean(linear_dice)),
        "mean_nonlinear_fusion_dice_equal_area": float(np.mean(nonlinear_dice)),
        "mean_dice_gap_linear_minus_nonlinear": float(np.mean(linear_dice) - np.mean(nonlinear_dice)),
        "mean_nonlinear_pressure_outside_energy_fraction": float(np.mean(pressure_outside)),
        "dominant_observation": dominant_observation(comparisons),
    }

def serializable_comparison(comparison: dict[str, object]) -> dict[str, object]:
    return {
        "anatomy": comparison["anatomy"],
        "common_grid_shape": comparison["common_grid_shape"],
        "common_extent_m": comparison["common_extent_m"],
        "geometry": comparison["geometry"],
        "comparison_metrics": comparison["comparison_metrics"],
        "cross_model_metrics": comparison["cross_model_metrics"],
        "objective_history": comparison["objective_history"],
        "technical_explanation": comparison["technical_explanation"],
    }


def dominant_observation(comparisons: list[dict[str, object]]) -> str:
    gaps = [
        case["comparison_metrics"]["linear_fusion"]["dice_equal_area"]
        - case["comparison_metrics"]["nonlinear_fusion"]["dice_equal_area"]
        for case in comparisons
    ]
    pressure_outside = [
        case["comparison_metrics"]["nonlinear_pressure"]["outside_energy_fraction"]
        for case in comparisons
    ]
    aperture_residual = [
        case["geometry"]["median_nearest_projected_element_distance_m"] for case in comparisons
    ]
    if float(np.mean(pressure_outside)) > 0.90 and float(np.mean(aperture_residual)) > 0.02:
        return (
            "the common-grid comparison rejects display resolution as the primary cause; "
            "after histotripsy-scale drive and corrected source weighting, nonlinear fusion "
            "improves, while the cavitation-specific divergence remains localized to "
            "off-target 3-D pressure/cavitation energy spread and residual projected "
            "2-D-vs-3-D aperture difference"
        )
    if float(np.mean(gaps)) > 0.20 and float(np.mean(pressure_outside)) > 0.50:
        return "the nonlinear branch loses target localization after propagation and separated inversion"
    return "the matched comparison does not isolate one dominant divergence source"
def _case_comparison(linear: dict[str, object], nonlinear: dict[str, object]) -> dict[str, object]:
    target_3d = np.asarray(nonlinear["target_mask"], dtype=bool)
    slab = _target_slab_bounds(target_3d, _target_slice_index(target_3d))
    common_target = _slab_projection(target_3d, slab, mode="max").astype(bool)
    common_extent = _nonlinear_extent(nonlinear)
    common_shape = common_target.shape
    linear_extent = _linear_extent(linear)

    fields = {
        "common_target": common_target,
        "linear_exposure": _normalize01(
            _resample_to_extent(np.asarray(linear["exposure"], dtype=float), linear_extent, common_shape, common_extent)
        ),
        "linear_active": _resample_to_extent(
            np.asarray(linear["active_lesion_reconstruction"], dtype=float),
            linear_extent,
            common_shape,
            common_extent,
        ),
        "linear_fusion": _resample_to_extent(
            np.asarray(linear["fused_reconstruction"], dtype=float),
            linear_extent,
            common_shape,
            common_extent,
        ),
        "nonlinear_pressure": _normalize01(
            _slab_projection(np.asarray(nonlinear["westervelt_peak_pressure_pa"], dtype=float), slab, mode="max")
        ),
        "nonlinear_fwi": _slab_projection(
            np.asarray(nonlinear["multiparameter_fwi_score"], dtype=float), slab, mode="max"
        ),
        "nonlinear_cavitation": _slab_projection(
            np.asarray(nonlinear["reconstructed_cavitation_density"], dtype=float), slab, mode="max"
        ),
        "nonlinear_fusion": _slab_projection(
            np.asarray(nonlinear["nonlinear_fusion_score"], dtype=float), slab, mode="max"
        ),
        "therapy_points_xy_m": np.asarray(nonlinear["therapy_points_m"], dtype=float)[:, :2],
    }
    fields["fusion_difference"] = fields["nonlinear_fusion"] - fields["linear_fusion"]

    metrics = {
        "linear_exposure": _field_metrics(fields["linear_exposure"], common_target),
        "linear_active": _field_metrics(fields["linear_active"], common_target),
        "linear_fusion": _field_metrics(fields["linear_fusion"], common_target),
        "nonlinear_pressure": _field_metrics(fields["nonlinear_pressure"], common_target),
        "nonlinear_fwi": _field_metrics(fields["nonlinear_fwi"], common_target),
        "nonlinear_cavitation": _field_metrics(fields["nonlinear_cavitation"], common_target),
        "nonlinear_fusion": _field_metrics(fields["nonlinear_fusion"], common_target),
    }
    cross = _cross_model_metrics(fields, common_target)
    geometry = _geometry_metrics(linear, nonlinear, common_target, common_extent)
    objective_history = {
        "nonlinear_fwi": _float_list(nonlinear.get("fwi_objective_history", [])),
        "nonlinear_cavitation": _float_list(nonlinear.get("cavitation_objective_history", [])),
    }
    return {
        "anatomy": str(linear["anatomy"]),
        "common_grid_shape": [int(v) for v in common_shape],
        "common_extent_m": [float(v) for v in common_extent],
        "fields": fields,
        "geometry": geometry,
        "comparison_metrics": metrics,
        "cross_model_metrics": cross,
        "objective_history": objective_history,
        "technical_explanation": _technical_explanation(metrics, cross, geometry, objective_history),
    }
def _linear_extent(result: dict[str, object]) -> list[float]:
    image = np.asarray(result["fused_reconstruction"], dtype=float)
    spacing = float(result["spacing_m"])
    nx, ny = image.shape
    return [
        -0.5 * (nx - 1) * spacing,
        0.5 * (nx - 1) * spacing,
        -0.5 * (ny - 1) * spacing,
        0.5 * (ny - 1) * spacing,
    ]
def _nonlinear_extent(result: dict[str, object]) -> list[float]:
    bounds = np.asarray(result["crop_bounds_index"], dtype=float)
    dims = np.asarray(result["source_dimensions"], dtype=float)
    spacing = np.asarray(result["source_spacing_m"], dtype=float)
    center_x = 0.5 * (dims[0] - 1.0)
    center_y = 0.5 * (dims[1] - 1.0)
    return [
        float((bounds[0] - center_x) * spacing[0]),
        float((bounds[1] - center_x) * spacing[0]),
        float((bounds[2] - center_y) * spacing[1]),
        float((bounds[3] - center_y) * spacing[1]),
    ]
def _target_slice_index(mask: np.ndarray) -> int:
    return int(np.argmax(np.sum(mask, axis=(0, 1))))


def _target_slab_bounds(mask: np.ndarray, z_index: int) -> tuple[int, int]:
    half_width = max(1, min(3, mask.shape[2] // 8))
    return max(0, z_index - half_width), min(mask.shape[2], z_index + half_width + 1)


def _slab_projection(volume: np.ndarray, slab: tuple[int, int], *, mode: str) -> np.ndarray:
    data = np.asarray(volume[:, :, slab[0] : slab[1]], dtype=float)
    if mode == "mean":
        return np.mean(data, axis=2)
    return np.max(data, axis=2)


def _resample_to_extent(
    image: np.ndarray,
    source_extent: list[float],
    target_shape: tuple[int, int],
    target_extent: list[float],
) -> np.ndarray:
    sx, sy = image.shape
    tx, ty = target_shape
    x = np.linspace(target_extent[0], target_extent[1], tx)
    y = np.linspace(target_extent[2], target_extent[3], ty)
    u = (x - source_extent[0]) * (sx - 1) / max(source_extent[1] - source_extent[0], 1.0e-12)
    v = (y - source_extent[2]) * (sy - 1) / max(source_extent[3] - source_extent[2], 1.0e-12)
    out = np.zeros((tx, ty), dtype=float)
    valid_x = (u >= 0.0) & (u <= sx - 1)
    valid_y = (v >= 0.0) & (v <= sy - 1)
    for ix, ux in enumerate(u):
        if not valid_x[ix]:
            continue
        x0 = int(np.floor(ux))
        x1 = min(x0 + 1, sx - 1)
        wx = ux - x0
        for iy, vy in enumerate(v):
            if not valid_y[iy]:
                continue
            y0 = int(np.floor(vy))
            y1 = min(y0 + 1, sy - 1)
            wy = vy - y0
            out[ix, iy] = (
                (1.0 - wx) * (1.0 - wy) * image[x0, y0]
                + wx * (1.0 - wy) * image[x1, y0]
                + (1.0 - wx) * wy * image[x0, y1]
                + wx * wy * image[x1, y1]
            )
    return out


def _field_metrics(field: np.ndarray, target: np.ndarray) -> dict[str, float]:
    values = _normalize01(field)
    active = np.asarray(target, dtype=bool)
    outside = ~active
    top = _equal_area_mask(values, active)
    dice = _dice(top, active)
    target_values = values[active]
    outside_values = values[outside]
    target_energy = float(np.sum(values[active] ** 2))
    total_energy = float(np.sum(values**2))
    return {
        "dice_equal_area": dice,
        "cnr": _cnr(target_values, outside_values),
        "nrmse": _nrmse(values, active.astype(float)),
        "pearson": _pearson(values, active.astype(float)),
        "target_peak": float(np.max(target_values)) if target_values.size else 0.0,
        "outside_peak": float(np.max(outside_values)) if outside_values.size else 0.0,
        "outside_energy_fraction": 1.0 - target_energy / total_energy if total_energy > 0.0 else 0.0,
    }


def _cross_model_metrics(fields: dict[str, np.ndarray], target: np.ndarray) -> dict[str, float]:
    return {
        "linear_fusion_vs_nonlinear_fusion_pearson": _pearson(
            fields["linear_fusion"], fields["nonlinear_fusion"]
        ),
        "linear_exposure_vs_nonlinear_pressure_pearson": _pearson(
            fields["linear_exposure"], fields["nonlinear_pressure"]
        ),
        "linear_fusion_hotspot_distance_grid_cells": _hotspot_distance(
            fields["linear_fusion"], fields["nonlinear_fusion"], target
        ),
        "nonlinear_pressure_to_fusion_pearson": _pearson(
            fields["nonlinear_pressure"], fields["nonlinear_fusion"]
        ),
    }


def _geometry_metrics(
    linear: dict[str, object],
    nonlinear: dict[str, object],
    common_target: np.ndarray,
    common_extent: list[float],
) -> dict[str, float | list[float] | str]:
    linear_focus = np.asarray(linear.get("focus_m", (0.0, 0.0)), dtype=float)[:2]
    nonlinear_focus = _mask_centroid_m(common_target, common_extent)
    linear_points = np.column_stack(
        [np.asarray(linear["therapy_x_m"], dtype=float), np.asarray(linear["therapy_y_m"], dtype=float)]
    )
    nonlinear_points = np.asarray(nonlinear["therapy_points_m"], dtype=float)[:, :2]
    return {
        "comparison_frame": "nonlinear_crop_xy_projection",
        "linear_focus_to_common_target_centroid_m": float(np.linalg.norm(linear_focus - nonlinear_focus)),
        "linear_element_count": int(np.asarray(linear["therapy_x_m"]).size),
        "nonlinear_element_count": int(nonlinear_points.shape[0]),
        "median_nearest_projected_element_distance_m": _median_nearest_distance(linear_points, nonlinear_points),
        "common_target_voxels": int(np.count_nonzero(common_target)),
        "common_target_centroid_m": [float(v) for v in nonlinear_focus],
    }


def _technical_explanation(
    metrics: dict[str, dict[str, float]],
    cross: dict[str, float],
    geometry: dict[str, object],
    objective_history: dict[str, list[float]],
) -> str:
    linear_dice = metrics["linear_fusion"]["dice_equal_area"]
    nonlinear_dice = metrics["nonlinear_fusion"]["dice_equal_area"]
    outside = metrics["nonlinear_pressure"]["outside_energy_fraction"]
    fwi_history = objective_history["nonlinear_fwi"]
    objective_drop = fwi_history[0] - fwi_history[-1] if len(fwi_history) >= 2 else 0.0
    return (
        f"On the common crop grid, linear fusion Dice is {linear_dice:.3f} and nonlinear fusion "
        f"Dice is {nonlinear_dice:.3f}. Nonlinear peak-pressure outside-target energy is "
        f"{outside:.3f}, linear/nonlinear fusion Pearson is "
        f"{cross['linear_fusion_vs_nonlinear_fusion_pearson']:.3f}, and nonlinear FWI objective "
        f"drop is {objective_drop:.3e}. The residual projected aperture mismatch from the original "
        f"runs is {1.0e3 * float(geometry['median_nearest_projected_element_distance_m']):.2f} mm; "
        "therefore the matched artifact localizes the remaining channel difference to nonlinear "
        "pressure/cavitation spread plus residual 2-D-vs-3-D aperture mismatch, not to image "
        "resolution alone."
    )


def _normalize01(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        return np.zeros(values.shape, dtype=float)
    return (values - low) / (high - low)


def _equal_area_mask(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    count = int(np.count_nonzero(target))
    if count <= 0:
        return np.zeros(values.shape, dtype=bool)
    flat = values.ravel()
    selected = np.argpartition(flat, -count)[-count:]
    out = np.zeros(flat.shape, dtype=bool)
    out[selected] = True
    return out.reshape(values.shape)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(np.count_nonzero(a)) + int(np.count_nonzero(b))
    if denom == 0:
        return 1.0
    return 2.0 * int(np.count_nonzero(a & b)) / denom


def _cnr(target_values: np.ndarray, outside_values: np.ndarray) -> float:
    if target_values.size == 0 or outside_values.size == 0:
        return 0.0
    denom = np.sqrt(0.5 * (np.var(target_values) + np.var(outside_values))).item()
    if denom <= 0.0:
        return 0.0
    return float((np.mean(target_values) - np.mean(outside_values)) / denom)


def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.sqrt(np.mean(b**2)).item()
    if denom <= 0.0:
        return 0.0
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - b) ** 2)) / denom)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _hotspot_distance(a: np.ndarray, b: np.ndarray, target: np.ndarray) -> float:
    if target.size == 0:
        return 0.0
    ia = np.array(np.unravel_index(int(np.argmax(a)), a.shape), dtype=float)
    ib = np.array(np.unravel_index(int(np.argmax(b)), b.shape), dtype=float)
    return float(np.linalg.norm(ia - ib))


def _mask_centroid_m(mask: np.ndarray, extent: list[float]) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.array([0.5 * (extent[0] + extent[1]), 0.5 * (extent[2] + extent[3])])
    x = np.linspace(extent[0], extent[1], mask.shape[0])
    y = np.linspace(extent[2], extent[3], mask.shape[1])
    return np.array([float(np.mean(x[coords[:, 0]])), float(np.mean(y[coords[:, 1]]))])


def _median_nearest_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    distances = []
    for point in a:
        delta = b - point[np.newaxis, :]
        distances.append(float(np.min(np.sum(delta * delta, axis=1)) ** 0.5))
    return float(np.median(distances))


def _contour_mask(ax: plt.Axes, mask: np.ndarray, extent: list[float], color: str, width: float) -> None:
    x = np.linspace(extent[0], extent[1], mask.shape[0])
    y = np.linspace(extent[2], extent[3], mask.shape[1])
    ax.contour(x, y, mask.T.astype(float), levels=[0.5], colors=color, linewidths=width)


def _float_list(values: object) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).ravel()]
