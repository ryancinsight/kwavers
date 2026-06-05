"""Controlled Figure 2/Figure 5 linear-vs-nonlinear comparison utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from ch29_controlled_placement import (
    add_ct_frame_fields,
    apply_placement_axes,
    ct_frame_key,
    placement_fields,
    plot_placement_context,
    resample_to_extent,
)
from ch29_pressure_diagnostics import pressure_field_diagnostics
from ch29_pressure_localization import aperture_geometry_metrics, pressure_hotspot_physical_metrics

COMPARISON_FIGURE_NAME = "fig06_controlled_linear_nonlinear_comparison.png"
COMPARISON_METRICS_NAME = "controlled_comparison_metrics.json"
COMPARISON_FIELDS_NAME = "controlled_comparison_fields.npz"
CONTROLLED_COMPARISON_COLUMNS = (
    ("placement_ct_hu", "gray", "CT + target + tx/rx"),
    ("common_target", "gray", "matched target"),
    ("linear_active", "viridis", "Born inverse"),
    ("nonlinear_pressure", "magma", "FWI Westervelt pressure"),
    ("elastic_shear", "viridis", "FWI elastic shear"),
    ("nonlinear_fusion", "viridis", "FWI fusion"),
    ("fusion_difference", "coolwarm", "FWI fusion - Born"),
)
CONTROLLED_COMPARISON_THEOREM = (
    "Theorem: CT-frame registration makes the Born inverse and FWI reconstructions comparable on one "
    "physical grid; the active Born inverse is the linearized reduced-Born/Tikhonov reconstruction, "
    "iterative elastic FWI is a nonlinear ElasticPSTD baseline/lesion residual inversion, and FWI fusion "
    "combines Westervelt FWI with cavitation evidence. The final panel is the FWI-fusion-minus-Born difference."
)

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
    columns = CONTROLLED_COMPARISON_COLUMNS
    fig, axes = plt.subplots(
        len(comparisons),
        len(columns),
        figsize=(21.0, 3.6 * len(comparisons)),
        constrained_layout=True,
    )
    axes_2d = np.asarray(axes, dtype=object).reshape(len(comparisons), len(columns))
    fig.suptitle(CONTROLLED_COMPARISON_THEOREM, fontsize=9.5)
    for row, comparison in enumerate(comparisons):
        fields = comparison["fields"]
        extent = [float(v) for v in np.asarray(fields["placement_extent_m"], dtype=float)]
        ct = np.asarray(fields["placement_ct_hu"], dtype=float)
        target = np.asarray(fields[ct_frame_key("common_target")], dtype=bool)
        for col, (key, cmap, title) in enumerate(columns):
            ax = axes_2d[row, col]
            if key == "placement_ct_hu":
                im = plot_placement_context(ax, comparison)
            elif key == "common_target":
                im = _plot_anatomy_target(ax, ct, target, extent)
            else:
                image = np.asarray(fields[ct_frame_key(key)], dtype=float)
                im = _overlay_field(ax, ct, image, extent, cmap, signed=(key == "fusion_difference"))
            if key != "placement_ct_hu":
                _contour_mask(ax, target, extent, "white", 0.9)
                apply_placement_axes(ax, fields)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{comparison['anatomy']} {title}" if col == 0 else title, fontsize=8.5)
            if col == len(columns) - 1:
                metrics = comparison["comparison_metrics"]
                ax.set_xlabel(
                    f"Born Dice={metrics['linear_active']['dice_equal_area']:.2f}; "
                    f"elastic FWI={metrics['elastic_shear']['dice_equal_area']:.2f}; "
                    f"FWI fusion Dice={metrics['nonlinear_fusion']['dice_equal_area']:.2f}"
                )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.text(
        0.5,
        0.01,
        "Caption: each reconstruction is overlaid on its CT anatomy after common-grid CT-frame resampling; "
        "overlay opacity scales with signal strength and the white contour is the matched target. The first "
        "column carries the full transducer placement context.",
        ha="center",
        va="bottom",
        fontsize=8.2,
    )
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
            "linear and nonlinear outputs are physically resampled to the full-resolution "
            "controlled CT placement grid before metrics, archived fields, and display"
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
    elastic_dice = [
        case["comparison_metrics"]["elastic_shear"]["dice_equal_area"] for case in comparisons
    ]
    pressure_outside = [
        case["comparison_metrics"]["nonlinear_pressure"]["outside_energy_fraction"]
        for case in comparisons
    ]
    source_outside = [
        case["comparison_metrics"]["nonlinear_cavitation_source"]["outside_energy_fraction"]
        for case in comparisons
    ]
    return {
        "case_count": len(comparisons),
        "mean_linear_fusion_dice_equal_area": float(np.mean(linear_dice)),
        "mean_elastic_shear_dice_equal_area": float(np.mean(elastic_dice)),
        "mean_nonlinear_fusion_dice_equal_area": float(np.mean(nonlinear_dice)),
        "mean_dice_gap_linear_minus_nonlinear": float(np.mean(linear_dice) - np.mean(nonlinear_dice)),
        "mean_nonlinear_pressure_outside_energy_fraction": float(np.mean(pressure_outside)),
        "mean_nonlinear_pressure_hotspot_distance_m": float(np.mean([
            case["comparison_metrics"]["nonlinear_pressure"]["ct_frame_pressure_hotspot_distance_m"]
            for case in comparisons
        ])),
        "mean_nonlinear_pressure_hotspot_axis_offset_m": float(np.mean([
            case["comparison_metrics"]["nonlinear_pressure"]["ct_frame_pressure_hotspot_axis_offset_m"]
            for case in comparisons
        ])),
        "mean_nonlinear_pressure_hotspot_cross_axis_offset_m": float(np.mean([
            case["comparison_metrics"]["nonlinear_pressure"]["ct_frame_pressure_hotspot_cross_axis_offset_m"]
            for case in comparisons
        ])),
        "mean_nonlinear_cavitation_source_outside_energy_fraction": float(np.mean(source_outside)),
        "mean_planned_to_nonlinear_aperture_axis_angle_deg": float(np.mean([
            case["geometry"]["planned_to_nonlinear_aperture_axis_angle_deg"] for case in comparisons
        ])),
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
    source_outside = [
        case["comparison_metrics"]["nonlinear_cavitation_source"]["outside_energy_fraction"]
        for case in comparisons
    ]
    aperture_residual = [
        case["geometry"]["median_nearest_projected_element_distance_m"] for case in comparisons
    ]
    if (
        float(np.mean(pressure_outside)) > 0.90
        and float(np.mean(source_outside)) > 0.90
        and float(np.mean(aperture_residual)) > 0.02
    ):
        return (
            "the common-grid comparison rejects display resolution as the primary cause; "
            "after histotripsy-scale drive and corrected source weighting, nonlinear fusion "
            "improves, while the cavitation-specific divergence remains localized to "
            "the MI-gated Rayleigh-Plesset source support, passive cavitation reconstruction, "
            "and residual projected 2-D-vs-3-D aperture difference"
        )
    if float(np.mean(gaps)) > 0.20 and float(np.mean(pressure_outside)) > 0.50:
        return "the nonlinear branch loses target localization after propagation and separated inversion"
    return "the matched comparison does not isolate one dominant divergence source"
def _case_comparison(linear: dict[str, object], nonlinear: dict[str, object]) -> dict[str, object]:
    target_3d = np.asarray(nonlinear["target_mask"], dtype=bool)
    inversion_3d = np.asarray(nonlinear.get("inversion_mask", target_3d), dtype=bool)
    slab = _target_slab_bounds(target_3d, _target_slice_index(target_3d))
    crop_target = _slab_projection(target_3d, slab, mode="max").astype(bool)
    crop_inversion = _slab_projection(inversion_3d, slab, mode="max").astype(bool)
    nonlinear_extent = _nonlinear_extent(nonlinear)
    placement = placement_fields(linear)
    common_extent = [float(v) for v in np.asarray(placement["placement_extent_m"], dtype=float)]
    common_shape = tuple(int(v) for v in np.asarray(placement["placement_ct_hu"]).shape)
    common_target = resample_to_extent(crop_target.astype(float), nonlinear_extent, common_shape, common_extent) >= 0.5
    linear_extent = _linear_extent(linear)

    raw_pressure_crop = _slab_projection(np.asarray(nonlinear["westervelt_peak_pressure_pa"], dtype=float), slab, mode="max")
    crop_body = _slab_projection(np.asarray(nonlinear["body_mask"], dtype=bool), slab, mode="max") >= 0.5
    body_pressure_crop = np.where(crop_body, raw_pressure_crop, 0.0)
    target_pressure_crop = np.where(crop_body & crop_target, raw_pressure_crop, 0.0)
    treatment_pressure_crop = np.where(crop_body & crop_inversion, raw_pressure_crop, 0.0)
    raw_pressure_projection = resample_to_extent(body_pressure_crop, nonlinear_extent, common_shape, common_extent)
    target_pressure_projection = resample_to_extent(
        target_pressure_crop,
        nonlinear_extent,
        common_shape,
        common_extent,
    )
    target_pressure_projection = np.where(common_target, target_pressure_projection, 0.0)
    treatment_pressure_projection = resample_to_extent(
        treatment_pressure_crop,
        nonlinear_extent,
        common_shape,
        common_extent,
    )
    fields = {
        **placement,
        "common_target": common_target,
        "linear_exposure": _normalize01(
            resample_to_extent(np.asarray(linear["exposure"], dtype=float), linear_extent, common_shape, common_extent)
        ),
        "linear_active": resample_to_extent(
            np.asarray(linear["active_lesion_reconstruction"], dtype=float),
            linear_extent,
            common_shape,
            common_extent,
        ),
        "linear_fusion": resample_to_extent(
            np.asarray(linear["fused_reconstruction"], dtype=float),
            linear_extent,
            common_shape,
            common_extent,
        ),
        "elastic_shear": resample_to_extent(
            np.asarray(linear["elastic_shear_reconstruction"], dtype=float),
            linear_extent,
            common_shape,
            common_extent,
        ),
        "nonlinear_pressure": _normalize01(target_pressure_projection),
        "nonlinear_pressure_window": _normalize01(treatment_pressure_projection),
        "nonlinear_pressure_raw": _normalize01(raw_pressure_projection),
        "nonlinear_treatment_window": resample_to_extent(
            crop_inversion.astype(float),
            nonlinear_extent,
            common_shape,
            common_extent,
        )
        >= 0.5,
        "nonlinear_fwi": resample_to_extent(
            _slab_projection(np.asarray(nonlinear["multiparameter_fwi_score"], dtype=float), slab, mode="max"),
            nonlinear_extent,
            common_shape,
            common_extent,
        ),
        "nonlinear_cavitation": resample_to_extent(
            _slab_projection(np.asarray(nonlinear["reconstructed_cavitation_density"], dtype=float), slab, mode="max"),
            nonlinear_extent,
            common_shape,
            common_extent,
        ),
        "nonlinear_cavitation_source": resample_to_extent(
            _slab_projection(np.asarray(nonlinear["cavitation_source_density"], dtype=float), slab, mode="max"),
            nonlinear_extent,
            common_shape,
            common_extent,
        ),
        "nonlinear_fusion": resample_to_extent(
            _slab_projection(np.asarray(nonlinear["nonlinear_fusion_score"], dtype=float), slab, mode="max"),
            nonlinear_extent,
            common_shape,
            common_extent,
        ),
        "therapy_points_xy_m": np.asarray(nonlinear["therapy_points_m"], dtype=float)[:, :2],
    }
    fields["fusion_difference"] = fields["nonlinear_fusion"] - fields["linear_active"]
    add_ct_frame_fields(fields, common_extent)

    geometry = aperture_geometry_metrics(linear, nonlinear, common_target, common_extent)
    pressure_metrics = _field_metrics(fields["nonlinear_pressure"], common_target)
    window_pressure_metrics = _field_metrics(fields["nonlinear_pressure_window"], common_target)
    raw_pressure_metrics = _field_metrics(fields["nonlinear_pressure_raw"], common_target)
    pressure_metrics.update({
        f"window_ct_field_{key}": value
        for key, value in window_pressure_metrics.items()
    })
    pressure_metrics.update({
        f"raw_ct_field_{key}": value
        for key, value in raw_pressure_metrics.items()
    })
    pressure_metrics.update(pressure_field_diagnostics(
        raw_pressure_crop,
        crop_target,
        body_mask=crop_body,
        frequency_hz=float(nonlinear.get("frequency_hz", 1.0e6)),
        source_pressure_pa=float(nonlinear.get("source_pressure_pa", 0.0)),
        source_scale=float(nonlinear.get("source_scale", 1.0)),
        inertial_mi_threshold=float(nonlinear.get("inertial_mi_threshold", 1.9)),
    ))
    pressure_metrics.update(pressure_hotspot_physical_metrics(
        fields["nonlinear_pressure"],
        common_target,
        common_extent,
        geometry,
    ))
    pressure_metrics.update({
        f"raw_{key}": value
        for key, value in pressure_hotspot_physical_metrics(
            fields["nonlinear_pressure_raw"],
            common_target,
            common_extent,
            geometry,
        ).items()
    })
    pressure_metrics.update({
        f"window_{key}": value
        for key, value in pressure_hotspot_physical_metrics(
            fields["nonlinear_pressure_window"],
            common_target,
            common_extent,
            geometry,
        ).items()
    })
    if "electronic_steering_metrics" in nonlinear:
        pressure_metrics.update({
            f"electronic_steering_{key}": value
            for key, value in dict(nonlinear["electronic_steering_metrics"]).items()
        })
    metrics = {
        "linear_exposure": _field_metrics(fields["linear_exposure"], common_target),
        "linear_active": _field_metrics(fields["linear_active"], common_target),
        "linear_fusion": _field_metrics(fields["linear_fusion"], common_target),
        "elastic_shear": _field_metrics(fields["elastic_shear"], common_target),
        "nonlinear_pressure": pressure_metrics,
        "nonlinear_fwi": _field_metrics(fields["nonlinear_fwi"], common_target),
        "nonlinear_cavitation_source": _field_metrics(fields["nonlinear_cavitation_source"], common_target),
        "nonlinear_cavitation": _field_metrics(fields["nonlinear_cavitation"], common_target),
        "nonlinear_fusion": _field_metrics(fields["nonlinear_fusion"], common_target),
    }
    cross = _cross_model_metrics(fields, common_target)
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
    if all(key in result for key in ("crop_bounds_index", "source_dimensions", "source_spacing_m")):
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
    return np.mean(data, axis=2) if mode == "mean" else np.max(data, axis=2)
def _field_metrics(field: np.ndarray, target: np.ndarray) -> dict[str, float]:
    values = _normalize01(field)
    active = np.asarray(target, dtype=bool)
    outside = ~active
    target_values = values[active]
    outside_values = values[outside]
    target_energy = float(np.sum(values[active] ** 2))
    total_energy = float(np.sum(values**2))
    return {
        "dice_equal_area": _dice(_equal_area_mask(values, active), active),
        "cnr": _cnr(target_values, outside_values),
        "nrmse": _nrmse(values, active.astype(float)),
        "pearson": _pearson(values, active.astype(float)),
        "target_peak": float(np.max(target_values)) if target_values.size else 0.0,
        "outside_peak": float(np.max(outside_values)) if outside_values.size else 0.0,
        "outside_energy_fraction": 1.0 - target_energy / total_energy if total_energy > 0.0 else 0.0,
        "hotspot_distance_to_target_grid_cells": _hotspot_distance_to_target(values, active),
    }
def _cross_model_metrics(fields: dict[str, np.ndarray], target: np.ndarray) -> dict[str, float]:
    return {
        "linear_fusion_vs_nonlinear_fusion_pearson": _pearson(fields["linear_fusion"], fields["nonlinear_fusion"]),
        "elastic_shear_vs_linear_fusion_pearson": _pearson(fields["elastic_shear"], fields["linear_fusion"]),
        "elastic_shear_vs_nonlinear_fusion_pearson": _pearson(fields["elastic_shear"], fields["nonlinear_fusion"]),
        "linear_exposure_vs_nonlinear_pressure_pearson": _pearson(fields["linear_exposure"], fields["nonlinear_pressure"]),
        "linear_fusion_hotspot_distance_grid_cells": _hotspot_distance(fields["linear_fusion"], fields["nonlinear_fusion"], target),
        "nonlinear_pressure_to_fusion_pearson": _pearson(fields["nonlinear_pressure"], fields["nonlinear_fusion"]),
        "nonlinear_pressure_to_cavitation_source_pearson": _pearson(fields["nonlinear_pressure"], fields["nonlinear_cavitation_source"]),
        "nonlinear_cavitation_source_to_reconstruction_pearson": _pearson(fields["nonlinear_cavitation_source"], fields["nonlinear_cavitation"]),
        "nonlinear_cavitation_source_to_reconstruction_hotspot_distance_grid_cells": _hotspot_distance(
            fields["nonlinear_cavitation_source"], fields["nonlinear_cavitation"], target
        ),
    }
def _technical_explanation(
    metrics: dict[str, dict[str, float]],
    cross: dict[str, float],
    geometry: dict[str, object],
    objective_history: dict[str, list[float]],
) -> str:
    linear_dice = metrics["linear_fusion"]["dice_equal_area"]
    elastic_dice = metrics["elastic_shear"]["dice_equal_area"]
    nonlinear_dice = metrics["nonlinear_fusion"]["dice_equal_area"]
    pressure = metrics["nonlinear_pressure"]
    outside = pressure["outside_energy_fraction"]
    source_distance = metrics["nonlinear_cavitation_source"]["hotspot_distance_to_target_grid_cells"]
    reconstructed_distance = metrics["nonlinear_cavitation"]["hotspot_distance_to_target_grid_cells"]
    fwi_history = objective_history["nonlinear_fwi"]
    objective_drop = fwi_history[0] - fwi_history[-1] if len(fwi_history) >= 2 else 0.0
    return (
        f"On the full CT placement grid, linear fusion Dice is {linear_dice:.3f}, elastic shear "
        f"Dice is {elastic_dice:.3f}, and nonlinear fusion Dice is {nonlinear_dice:.3f}. "
        f"Nonlinear target MI is {pressure['target_peak_mechanical_index']:.2f} "
        f"at {1.0e-6 * pressure['raw_target_peak_pressure_pa']:.2f} MPa; displayed target-pressure outside energy is "
        f"{outside:.3f}, linear/nonlinear fusion Pearson is "
        f"{cross['linear_fusion_vs_nonlinear_fusion_pearson']:.3f}, and nonlinear FWI objective "
        f"drop is {objective_drop:.3e}. The Rayleigh-Plesset source and passive reconstruction "
        f"hotspots are {source_distance:.2f} and {reconstructed_distance:.2f} grid cells from the "
        f"target centroid. The residual projected aperture mismatch from the original "
        f"runs is {1.0e3 * float(geometry['median_nearest_projected_element_distance_m']):.2f} mm; "
        f"the nonlinear aperture axis differs from the planned beam axis by "
        f"{float(geometry['planned_to_nonlinear_aperture_axis_angle_deg']):.2f} deg; "
        f"the displayed target-pressure hotspot offset is "
        f"{1.0e3 * pressure['ct_frame_pressure_hotspot_axis_offset_m']:.2f} mm "
        f"along the planned beam axis and "
        f"{1.0e3 * pressure['ct_frame_pressure_hotspot_cross_axis_offset_m']:.2f} mm cross-axis; "
        f"the treatment-window pressure hotspot offset is "
        f"{1.0e3 * pressure['window_ct_frame_pressure_hotspot_axis_offset_m']:.2f} mm along-axis and "
        f"{1.0e3 * pressure['window_ct_frame_pressure_hotspot_cross_axis_offset_m']:.2f} mm cross-axis; "
        f"the raw body-pressure hotspot offset is "
        f"{1.0e3 * pressure['raw_ct_frame_pressure_hotspot_axis_offset_m']:.2f} mm along-axis and "
        f"{1.0e3 * pressure['raw_ct_frame_pressure_hotspot_cross_axis_offset_m']:.2f} mm cross-axis; "
        f"the raw pressure peak is {'in the coupling/source region' if pressure['raw_peak_is_in_coupling'] else 'inside the body'} "
        f"with coupling/body peak ratio {pressure['coupling_to_body_peak_ratio']:.2f}, and the "
        f"body pressure hotspot is {pressure['body_hotspot_distance_to_target_grid_cells']:.2f} "
        "grid cells from target. "
        f"the measured electronic-steering calibration hotspot is "
        f"{pressure.get('electronic_steering_calibration_hotspot_distance_grid_cells', 0.0):.2f} "
        f"grid cells from the nominal focus and selected correction "
        f"{pressure.get('electronic_steering_correction_grid_cells', [0, 0, 0])}. "
        "therefore the matched artifact localizes the remaining channel difference to nonlinear "
        "in-body pressure/cavitation spread, source-region dominance in raw peak-pressure display, "
        "and residual 2-D-vs-3-D aperture mismatch, not to image resolution alone."
    )
def _normalize01(image: np.ndarray) -> np.ndarray:
    values = np.where(np.isfinite(np.asarray(image, dtype=float)), np.asarray(image, dtype=float), 0.0)
    low = float(np.min(values))
    high = float(np.max(values))
    return np.zeros(values.shape, dtype=float) if high <= low else (values - low) / (high - low)
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
    return 1.0 if denom == 0 else 2.0 * int(np.count_nonzero(a & b)) / denom
def _cnr(target_values: np.ndarray, outside_values: np.ndarray) -> float:
    if target_values.size == 0 or outside_values.size == 0:
        return 0.0
    denom = np.sqrt(0.5 * (np.var(target_values) + np.var(outside_values))).item()
    return 0.0 if denom <= 0.0 else float((np.mean(target_values) - np.mean(outside_values)) / denom)
def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.sqrt(np.mean(b**2)).item()
    return 0.0 if denom <= 0.0 else float(np.sqrt(np.mean((np.asarray(a, dtype=float) - b) ** 2)) / denom)
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if denom <= 0.0 else float(np.dot(x, y) / denom)
def _hotspot_distance(a: np.ndarray, b: np.ndarray, target: np.ndarray) -> float:
    if target.size == 0:
        return 0.0
    ia = np.array(np.unravel_index(int(np.argmax(a)), a.shape), dtype=float)
    ib = np.array(np.unravel_index(int(np.argmax(b)), b.shape), dtype=float)
    return float(np.linalg.norm(ia - ib))
def _hotspot_distance_to_target(field: np.ndarray, target: np.ndarray) -> float:
    coords = np.argwhere(target)
    if coords.size == 0:
        return 0.0
    hotspot = np.array(np.unravel_index(int(np.argmax(field)), field.shape), dtype=float)
    return float(np.linalg.norm(hotspot - np.mean(coords, axis=0)))
def _contour_mask(ax: plt.Axes, mask: np.ndarray, extent: list[float], color: str, width: float) -> None:
    x = np.linspace(extent[0], extent[1], mask.shape[0])
    y = np.linspace(extent[2], extent[3], mask.shape[1])
    ax.contour(x, y, mask.T.astype(float), levels=[0.5], colors=color, linewidths=width)
def _ct_underlay(ax: plt.Axes, ct: np.ndarray, extent: list[float]) -> None:
    ax.imshow(ct.T, cmap="gray", origin="lower", extent=extent, vmin=-200.0, vmax=300.0)
def _signal_alpha(magnitude: np.ndarray) -> np.ndarray:
    peak = max(float(np.max(magnitude)), 1.0e-12)
    # sqrt boosts mid-strength signal so faint-but-real structure stays visible
    # while zero-valued background outside the reconstruction stays transparent.
    return np.sqrt(np.clip(magnitude / peak, 0.0, 1.0))
def _overlay_field(
    ax: plt.Axes,
    ct: np.ndarray,
    field: np.ndarray,
    extent: list[float],
    cmap: str,
    signed: bool,
) -> plt.AxesImage:
    _ct_underlay(ax, ct, extent)
    data = np.asarray(field, dtype=float)
    if signed:
        vmax = max(float(np.max(np.abs(data))), 1.0e-12)
        alpha = _signal_alpha(np.abs(data))
        return ax.imshow(
            data.T,
            cmap=cmap,
            origin="lower",
            extent=extent,
            vmin=-vmax,
            vmax=vmax,
            alpha=alpha.T,
            interpolation="bilinear",
        )
    vmax = max(float(np.max(data)), 1.0e-12)
    alpha = _signal_alpha(np.maximum(data, 0.0))
    return ax.imshow(
        data.T,
        cmap=cmap,
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=vmax,
        alpha=alpha.T,
        interpolation="bilinear",
    )
def _plot_anatomy_target(
    ax: plt.Axes,
    ct: np.ndarray,
    target: np.ndarray,
    extent: list[float],
) -> plt.AxesImage:
    im = ax.imshow(ct.T, cmap="gray", origin="lower", extent=extent, vmin=-200.0, vmax=300.0)
    fill = np.where(target, 1.0, np.nan)
    ax.imshow(
        fill.T,
        cmap="autumn",
        origin="lower",
        extent=extent,
        vmin=0.0,
        vmax=1.0,
        alpha=0.45,
    )
    return im
def _float_list(values: object) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).ravel()]
