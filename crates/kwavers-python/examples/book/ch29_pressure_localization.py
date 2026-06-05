"""Chapter 29 CT-frame aperture and pressure-localization diagnostics."""

from __future__ import annotations

import numpy as np


def aperture_geometry_metrics(
    linear: dict[str, object],
    nonlinear: dict[str, object],
    common_target: np.ndarray,
    common_extent: list[float],
) -> dict[str, float | list[float] | str]:
    linear_focus = np.asarray(linear.get("focus_m", (0.0, 0.0)), dtype=float)[:2]
    nonlinear_focus = _mask_centroid_m(common_target, common_extent)
    planned_skin = np.asarray(linear.get("placement_skin_contact_m", linear_focus), dtype=float)[:2]
    linear_points = np.column_stack(
        [np.asarray(linear["therapy_x_m"], dtype=float), np.asarray(linear["therapy_y_m"], dtype=float)]
    )
    nonlinear_points = np.asarray(nonlinear["therapy_points_m"], dtype=float)[:, :2]
    nonlinear_centroid = _point_centroid(nonlinear_points)
    planned_axis = _beam_axis(planned_skin, nonlinear_focus, linear_points)
    nonlinear_axis = _beam_axis(nonlinear_centroid, nonlinear_focus, nonlinear_points)
    nonlinear_distances = np.linalg.norm(nonlinear_points - nonlinear_focus[np.newaxis, :], axis=1)
    return {
        "comparison_frame": "full_ct_placement_xy_projection",
        "linear_focus_to_common_target_centroid_m": float(np.linalg.norm(linear_focus - nonlinear_focus)),
        "linear_element_count": int(np.asarray(linear["therapy_x_m"]).size),
        "nonlinear_element_count": int(nonlinear_points.shape[0]),
        "median_nearest_projected_element_distance_m": _median_nearest_distance(linear_points, nonlinear_points),
        "common_target_voxels": int(np.count_nonzero(common_target)),
        "common_target_centroid_m": [float(v) for v in nonlinear_focus],
        "planned_skin_contact_m": [float(v) for v in planned_skin],
        "planned_skin_to_target_m": float(np.linalg.norm(nonlinear_focus - planned_skin)),
        "planned_beam_axis_unit": [float(v) for v in planned_axis],
        "nonlinear_aperture_centroid_m": [float(v) for v in nonlinear_centroid],
        "nonlinear_aperture_centroid_to_target_m": float(np.linalg.norm(nonlinear_focus - nonlinear_centroid)),
        "nonlinear_aperture_axis_unit": [float(v) for v in nonlinear_axis],
        "planned_to_nonlinear_aperture_axis_angle_deg": _angle_deg(planned_axis, nonlinear_axis),
        "nonlinear_source_to_target_distance_median_m": float(np.median(nonlinear_distances))
        if nonlinear_distances.size
        else 0.0,
        "nonlinear_source_to_target_distance_min_m": float(np.min(nonlinear_distances))
        if nonlinear_distances.size
        else 0.0,
        "nonlinear_source_to_target_distance_max_m": float(np.max(nonlinear_distances))
        if nonlinear_distances.size
        else 0.0,
    }


def pressure_hotspot_physical_metrics(
    field: np.ndarray,
    target: np.ndarray,
    extent: list[float],
    geometry: dict[str, object],
) -> dict[str, float | bool | list[float]]:
    values = _normalize01(field)
    target_centroid = _mask_centroid_m(target, extent)
    hotspot = _hotspot_m(values, extent)
    offset = hotspot - target_centroid
    axis = _unit(np.asarray(geometry.get("planned_beam_axis_unit", [0.0, 0.0]), dtype=float))
    axis_offset = float(np.dot(offset, axis))
    cross = offset - axis_offset * axis
    cross_offset = float(np.linalg.norm(cross))
    distance = float(np.linalg.norm(offset))
    return {
        "ct_frame_target_centroid_m": [float(v) for v in target_centroid],
        "ct_frame_pressure_hotspot_m": [float(v) for v in hotspot],
        "ct_frame_pressure_hotspot_distance_m": distance,
        "ct_frame_pressure_hotspot_axis_offset_m": axis_offset,
        "ct_frame_pressure_hotspot_cross_axis_offset_m": cross_offset,
        "pressure_hotspot_is_prefocal": bool(axis_offset < 0.0),
        "pressure_hotspot_is_postfocal": bool(axis_offset > 0.0),
    }


def _normalize01(image: np.ndarray) -> np.ndarray:
    values = np.where(np.isfinite(np.asarray(image, dtype=float)), np.asarray(image, dtype=float), 0.0)
    low = float(np.min(values))
    high = float(np.max(values))
    return np.zeros(values.shape, dtype=float) if high <= low else (values - low) / (high - low)


def _hotspot_m(field: np.ndarray, extent: list[float]) -> np.ndarray:
    values = np.asarray(field, dtype=float)
    if values.size == 0:
        return np.array([0.5 * (extent[0] + extent[1]), 0.5 * (extent[2] + extent[3])])
    i, j = np.unravel_index(int(np.argmax(values)), values.shape)
    x = np.linspace(extent[0], extent[1], values.shape[0])
    y = np.linspace(extent[2], extent[3], values.shape[1])
    return np.array([float(x[i]), float(y[j])])


def _mask_centroid_m(mask: np.ndarray, extent: list[float]) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.array([0.5 * (extent[0] + extent[1]), 0.5 * (extent[2] + extent[3])])
    x = np.linspace(extent[0], extent[1], mask.shape[0])
    y = np.linspace(extent[2], extent[3], mask.shape[1])
    return np.array([float(np.mean(x[coords[:, 0]])), float(np.mean(y[coords[:, 1]]))])


def _point_centroid(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.size == 0:
        return np.zeros(2, dtype=float)
    return np.mean(values[:, :2], axis=0)


def _beam_axis(anchor: np.ndarray, target: np.ndarray, fallback_points: np.ndarray) -> np.ndarray:
    axis = _unit(np.asarray(target, dtype=float) - np.asarray(anchor, dtype=float))
    if np.linalg.norm(axis) > 0.0:
        return axis
    fallback = _point_centroid(fallback_points)
    return _unit(np.asarray(target, dtype=float) - fallback)


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    return np.zeros(values.shape, dtype=float) if norm <= 0.0 else values / norm


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    au = _unit(a)
    bu = _unit(b)
    if np.linalg.norm(au) <= 0.0 or np.linalg.norm(bu) <= 0.0:
        return 0.0
    cosine = float(np.clip(np.dot(au, bu), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _median_nearest_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    distances = []
    for point in a:
        delta = b - point[np.newaxis, :]
        distances.append(float(np.min(np.sum(delta * delta, axis=1)) ** 0.5))
    return float(np.median(distances))
