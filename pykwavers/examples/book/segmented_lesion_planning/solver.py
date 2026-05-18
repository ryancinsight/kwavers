"""Hybrid transducer-placement and focal-field optimizer."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from .crossfire import aperture_element_inward_units, build_composite_aperture, select_crossfire_components
from .dense_field import (
    body_sidelobe_mask,
    dense_acceptance_key,
    dense_field_metrics,
    normalized_intensity_from_pressure,
)
from .types import HybridPlanConfig, SegmentationGrid, TISSUE_PROPERTIES, Tissue, TransducerAperture


@dataclass(frozen=True)
class ControlSet:
    points_m: np.ndarray
    desired_pressure: np.ndarray
    weights: np.ndarray
    target_count: int
    avoid_count: int
    normal_count: int
    sidelobe_count: int


def optimize_transducer_layout(
    grid: SegmentationGrid,
    config: HybridPlanConfig | None = None,
) -> dict[str, object]:
    """Optimize aperture location and complex drive weights from segmentation.

    The hybrid solve has two layers:
    1. A geometric ray model scores whether each candidate aperture reaches the
       tumor without traversing segmented air, bone, or excess fat.
    2. For each geometrically valid aperture, a weighted complex ridge solve
       computes per-element phase/amplitude weights that match the requested
       tumor spot shape and null protected anatomy.
    """

    cfg = config or HybridPlanConfig()
    target = grid.centroid(Tissue.TUMOR)
    controls = build_control_set(grid, target, cfg)
    apertures = build_candidate_apertures(grid, target, cfg)
    if not apertures:
        raise ValueError("no candidate apertures were generated")

    candidates: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_index = -1
    for aperture in apertures:
        record = solve_aperture_record(grid, aperture, controls, cfg)
        metrics = record["metrics"]
        candidates.append(record)
        if best is None or metrics["score"] > best["metrics"]["score"]:
            best = record
            best_index = len(candidates) - 1

    assert best is not None
    crossfire = solve_crossfire_record(grid, candidates, best, controls, cfg)
    if crossfire is not None:
        candidates.append(crossfire)
        best = crossfire
        best_index = len(candidates) - 1
    best = refine_candidate_hotspots(grid, best, controls, target, cfg)
    candidates[best_index] = best
    pressure_map = synthesize_pressure_map(
        grid,
        best["aperture"],
        best["drive_weights"],
        cfg,
        best.get("_field_matrix"),
    )
    normalized_intensity = normalized_intensity_from_pressure(grid, pressure_map)
    return {
        "config": cfg,
        "target_centroid_m": target,
        "best": best,
        "candidates": candidates,
        "pressure_map": pressure_map,
        "normalized_intensity": normalized_intensity,
        "summary": result_summary(grid, normalized_intensity, best, candidates, target, cfg),
    }


def solve_aperture_record(
    grid: SegmentationGrid,
    aperture: TransducerAperture,
    controls: ControlSet,
    config: HybridPlanConfig,
) -> dict[str, object]:
    matrix, path = propagation_matrix(grid, aperture, controls.points_m, config)
    drive = solve_complex_drive(matrix, controls.desired_pressure, controls.weights, config.ridge)
    metrics = score_candidate(matrix, drive, controls, path, config)
    return {
        "aperture": aperture,
        "drive_weights": drive,
        "path_summary": path,
        "metrics": metrics,
    }


def solve_crossfire_record(
    grid: SegmentationGrid,
    candidates: list[dict[str, object]],
    best: dict[str, object],
    controls: ControlSet,
    config: HybridPlanConfig,
) -> dict[str, object] | None:
    components = select_crossfire_components(candidates, best, config)
    if len(components) <= 1:
        return None
    aperture = build_composite_aperture(components)
    record = solve_aperture_record(grid, aperture, controls, config)
    record["component_angles_deg"] = [float(item.angle_deg) for item in components]
    return record


def build_control_set(
    grid: SegmentationGrid,
    target_centroid_m: np.ndarray,
    config: HybridPlanConfig,
) -> ControlSet:
    target_points = grid.sample_points(grid.mask(Tissue.TUMOR), config.max_target_points)
    avoid_points = grid.sample_points(grid.mask(Tissue.AVOID), config.max_avoid_points)
    near_target = near_target_normal_mask(grid, target_centroid_m)
    normal_points = grid.sample_points(near_target, config.max_normal_points)
    sidelobe_points = grid.sample_points(
        sidelobe_control_mask(grid, target_centroid_m, config),
        config.max_sidelobe_points,
    )
    points = np.vstack([target_points, avoid_points, normal_points, sidelobe_points])

    target_profile = desired_spot_profile(target_points, target_centroid_m, config)
    desired = np.concatenate(
        [
            target_profile,
            np.zeros(avoid_points.shape[0] + normal_points.shape[0] + sidelobe_points.shape[0], dtype=float),
        ]
    )
    weights = np.concatenate(
        [
            np.full(target_points.shape[0], config.target_weight, dtype=float),
            np.full(avoid_points.shape[0], config.avoid_weight, dtype=float),
            np.full(normal_points.shape[0], config.normal_weight, dtype=float),
            np.full(sidelobe_points.shape[0], config.sidelobe_weight, dtype=float),
        ]
    )
    return ControlSet(
        points_m=points,
        desired_pressure=desired.astype(complex),
        weights=weights,
        target_count=int(target_points.shape[0]),
        avoid_count=int(avoid_points.shape[0]),
        normal_count=int(normal_points.shape[0]),
        sidelobe_count=int(sidelobe_points.shape[0]),
    )


def near_target_normal_mask(grid: SegmentationGrid, target_centroid_m: np.ndarray) -> np.ndarray:
    x, y = grid.coordinates_m()
    distance = np.sqrt((x - target_centroid_m[0]) ** 2 + (y - target_centroid_m[1]) ** 2)
    normal_tissues = (grid.labels == int(Tissue.NORMAL)) | (grid.labels == int(Tissue.FAT))
    return normal_tissues & (distance < 0.026)


def sidelobe_control_mask(
    grid: SegmentationGrid,
    target_centroid_m: np.ndarray,
    config: HybridPlanConfig,
) -> np.ndarray:
    x, y = grid.coordinates_m()
    distance = np.sqrt((x - target_centroid_m[0]) ** 2 + (y - target_centroid_m[1]) ** 2)
    target = grid.mask(Tissue.TUMOR)
    avoid = grid.mask(Tissue.AVOID)
    return grid.body_mask & ~target & ~avoid & (distance >= config.sidelobe_exclusion_radius_m)


def desired_spot_profile(
    points_m: np.ndarray,
    target_centroid_m: np.ndarray,
    config: HybridPlanConfig,
) -> np.ndarray:
    if points_m.size == 0:
        return np.zeros(0, dtype=float)
    angle = np.deg2rad(config.spot_angle_deg)
    axis = np.asarray([np.cos(angle), np.sin(angle)], dtype=float)
    cross = np.asarray([-axis[1], axis[0]], dtype=float)
    rel = points_m - target_centroid_m
    major = rel @ axis
    minor = rel @ cross
    r2 = (major / config.spot_major_axis_m) ** 2 + (minor / config.spot_minor_axis_m) ** 2
    profile = np.exp(-0.5 * r2)
    return profile / max(float(np.max(profile)), 1e-12)


def build_candidate_apertures(
    grid: SegmentationGrid,
    focus_m: np.ndarray,
    config: HybridPlanConfig,
) -> list[TransducerAperture]:
    x, y = grid.coordinates_m()
    body_points = np.column_stack([x[grid.body_mask], y[grid.body_mask]])
    offsets = np.linspace(-0.5, 0.5, config.element_count) * config.aperture_width_m
    apertures: list[TransducerAperture] = []
    for angle_deg in config.candidate_angles_deg:
        angle = np.deg2rad(angle_deg)
        radial = np.asarray([np.cos(angle), np.sin(angle)], dtype=float)
        extent = float(np.max(body_points @ radial))
        center = radial * (extent + config.standoff_m)
        inward = focus_m - center
        inward /= max(float(np.linalg.norm(inward)), 1e-12)
        tangent = np.asarray([-inward[1], inward[0]], dtype=float)
        elements = center[None, :] + offsets[:, None] * tangent[None, :]
        apertures.append(
            TransducerAperture(
                angle_deg=float(angle_deg),
                center_m=center,
                inward_unit=inward,
                element_positions_m=elements,
                element_inward_units_m=np.repeat(inward[None, :], config.element_count, axis=0),
                source_angles_deg=(float(angle_deg),),
            )
        )
    return apertures


def propagation_matrix(
    grid: SegmentationGrid,
    aperture: TransducerAperture,
    control_points_m: np.ndarray,
    config: HybridPlanConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    n_points = int(control_points_m.shape[0])
    n_elements = int(aperture.element_positions_m.shape[0])
    matrix = np.zeros((n_points, n_elements), dtype=complex)
    inward_units = aperture_element_inward_units(aperture)
    path_totals = {tissue: 0.0 for tissue in Tissue}
    path_count = 0
    for i, point in enumerate(control_points_m):
        for j, element in enumerate(aperture.element_positions_m):
            fractions = ray_tissue_fractions(grid, element, point, config.ray_samples)
            for tissue, value in fractions.items():
                path_totals[tissue] += value
            path_count += 1
            matrix[i, j] = ray_transfer(element, point, inward_units[j], fractions, config)
    denom = max(float(path_count), 1.0)
    return matrix, {
        tissue.name.lower(): float(total / denom)
        for tissue, total in path_totals.items()
    }


def ray_tissue_fractions(
    grid: SegmentationGrid,
    start_m: np.ndarray,
    end_m: np.ndarray,
    samples: int,
) -> dict[Tissue, float]:
    t = np.linspace(0.0, 1.0, int(samples), dtype=float)
    points = start_m[None, :] + t[:, None] * (end_m - start_m)[None, :]
    labels = []
    for point in points:
        i, j = grid.index_from_point(point)
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
            labels.append(int(grid.labels[i, j]))
        else:
            labels.append(int(Tissue.AIR))
    arr = np.asarray(labels, dtype=np.uint8)
    non_air = np.flatnonzero(arr != int(Tissue.AIR))
    segment = arr[non_air[0] :] if non_air.size else arr
    denom = max(float(segment.size), 1.0)
    return {
        tissue: float(np.count_nonzero(segment == int(tissue)) / denom)
        for tissue in Tissue
    }


def ray_transfer(
    element_m: np.ndarray,
    point_m: np.ndarray,
    inward_unit: np.ndarray,
    fractions: dict[Tissue, float],
    config: HybridPlanConfig,
) -> complex:
    vector = point_m - element_m
    distance = max(float(np.linalg.norm(vector)), 1e-9)
    direction = vector / distance
    directivity = max(float(np.dot(direction, inward_unit)), 0.0) ** 2
    if directivity == 0.0:
        return 0.0 + 0.0j
    slowness = sum(fractions[t] / TISSUE_PROPERTIES[t].sound_speed_m_s for t in Tissue)
    attenuation = sum(fractions[t] * TISSUE_PROPERTIES[t].attenuation_np_m for t in Tissue)
    phase = 2.0 * pi * config.frequency_hz * distance * slowness
    amplitude = directivity * np.exp(-attenuation * distance) / np.sqrt(distance)
    return complex(amplitude * np.cos(-phase), amplitude * np.sin(-phase))


def solve_complex_drive(
    matrix: np.ndarray,
    desired: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> np.ndarray:
    weighted = matrix * weights[:, None]
    rhs = desired * weights
    gram = weighted.conj().T @ weighted
    normal_rhs = weighted.conj().T @ rhs
    drive = np.linalg.solve(gram + ridge * np.eye(gram.shape[0], dtype=complex), normal_rhs)
    max_abs = max(float(np.max(np.abs(drive))), 1e-12)
    return drive / max_abs


def score_candidate(
    matrix: np.ndarray,
    drive: np.ndarray,
    controls: ControlSet,
    path: dict[str, float],
    config: HybridPlanConfig,
) -> dict[str, float]:
    pressure = matrix @ drive
    intensity = np.abs(pressure) ** 2
    target = intensity[: controls.target_count]
    avoid_start = controls.target_count
    avoid_end = avoid_start + controls.avoid_count
    normal_end = avoid_end + controls.normal_count
    avoid = intensity[avoid_start:avoid_end]
    normal = intensity[avoid_end:normal_end]
    sidelobe = intensity[normal_end:]
    target_peak = max(float(np.max(target)), 1e-12)
    target_norm = target / target_peak
    avoid_norm = avoid / target_peak if avoid.size else np.zeros(0, dtype=float)
    normal_norm = normal / target_peak if normal.size else np.zeros(0, dtype=float)
    sidelobe_norm = sidelobe / target_peak if sidelobe.size else np.zeros(0, dtype=float)
    target_mean = float(np.mean(target_norm))
    protected_peak = float(np.max(avoid_norm)) if avoid_norm.size else 0.0
    off_target_mean = float(np.mean(normal_norm)) if normal_norm.size else 0.0
    sidelobe_peak = float(np.max(sidelobe_norm)) if sidelobe_norm.size else 0.0
    sidelobe_mean = float(np.mean(sidelobe_norm)) if sidelobe_norm.size else 0.0
    coverage = float(np.count_nonzero(target_norm >= 0.35) / max(target_norm.size, 1))
    score = (
        target_mean
        + 0.35 * coverage
        - config.protected_peak_penalty * protected_peak
        - config.off_target_penalty * off_target_mean
        - config.sidelobe_peak_penalty * sidelobe_peak
        - config.air_path_penalty * path["air"]
        - config.bone_path_penalty * path["bone"]
        - config.fat_path_penalty * path["fat"]
    )
    return {
        "score": float(score),
        "target_mean": target_mean,
        "target_coverage_fraction": coverage,
        "protected_peak_ratio": protected_peak,
        "normal_mean_ratio": off_target_mean,
        "sidelobe_peak_ratio": sidelobe_peak,
        "sidelobe_mean_ratio": sidelobe_mean,
        "air_path_fraction": float(path["air"]),
        "bone_path_fraction": float(path["bone"]),
        "fat_path_fraction": float(path["fat"]),
        "drive_nonuniformity": float(np.std(np.abs(drive))),
    }


def refine_candidate_hotspots(
    grid: SegmentationGrid,
    record: dict[str, object],
    controls: ControlSet,
    target_centroid_m: np.ndarray,
    config: HybridPlanConfig,
) -> dict[str, object]:
    refined_controls = controls
    refined_record = record
    field_matrix = pressure_field_matrix(grid, refined_record["aperture"], config)
    best_record = attach_dense_metrics(grid, refined_record, target_centroid_m, config, field_matrix)
    best_key = dense_acceptance_key(best_record["dense_metrics"])
    rounds = max(int(config.hotspot_refinement_rounds), 0)
    for _ in range(rounds):
        pressure_map = synthesize_pressure_map(
            grid,
            refined_record["aperture"],
            refined_record["drive_weights"],
            config,
            field_matrix,
        )
        normalized = normalized_intensity_from_pressure(grid, pressure_map)
        hotspots = hotspot_control_points(grid, normalized, target_centroid_m, config)
        if hotspots.size == 0:
            break
        refined_controls = append_hotspot_controls(refined_controls, hotspots, config)
        matrix, path = propagation_matrix(grid, refined_record["aperture"], refined_controls.points_m, config)
        drive = solve_complex_drive(matrix, refined_controls.desired_pressure, refined_controls.weights, config.ridge)
        metrics = score_candidate(matrix, drive, refined_controls, path, config)
        refined_record = {
            "aperture": refined_record["aperture"],
            "drive_weights": drive,
            "path_summary": path,
            "metrics": metrics,
            "hotspot_control_count": int(hotspots.shape[0]),
        }
        dense_record = attach_dense_metrics(grid, refined_record, target_centroid_m, config, field_matrix)
        dense_key = dense_acceptance_key(dense_record["dense_metrics"])
        if dense_key > best_key:
            best_record = dense_record
            best_key = dense_key
    return best_record


def attach_dense_metrics(
    grid: SegmentationGrid,
    record: dict[str, object],
    target_centroid_m: np.ndarray,
    config: HybridPlanConfig,
    field_matrix: np.ndarray | None = None,
) -> dict[str, object]:
    pressure_map = synthesize_pressure_map(
        grid,
        record["aperture"],
        record["drive_weights"],
        config,
        field_matrix,
    )
    normalized = normalized_intensity_from_pressure(grid, pressure_map)
    merged = dict(record)
    merged["dense_metrics"] = dense_field_metrics(grid, normalized, target_centroid_m, config)
    if field_matrix is not None:
        merged["_field_matrix"] = field_matrix
    return merged


def append_hotspot_controls(
    controls: ControlSet,
    hotspots_m: np.ndarray,
    config: HybridPlanConfig,
) -> ControlSet:
    if hotspots_m.size == 0:
        return controls
    return ControlSet(
        points_m=np.vstack([controls.points_m, hotspots_m]),
        desired_pressure=np.concatenate(
            [controls.desired_pressure, np.zeros(hotspots_m.shape[0], dtype=complex)]
        ),
        weights=np.concatenate(
            [controls.weights, np.full(hotspots_m.shape[0], config.hotspot_weight, dtype=float)]
        ),
        target_count=controls.target_count,
        avoid_count=controls.avoid_count,
        normal_count=controls.normal_count,
        sidelobe_count=controls.sidelobe_count + int(hotspots_m.shape[0]),
    )


def hotspot_control_points(
    grid: SegmentationGrid,
    normalized_intensity: np.ndarray,
    target_centroid_m: np.ndarray,
    config: HybridPlanConfig,
) -> np.ndarray:
    mask = body_sidelobe_mask(grid, target_centroid_m, config)
    indices = np.argwhere(mask)
    if indices.size == 0 or config.max_hotspot_points <= 0:
        return np.zeros((0, 2), dtype=float)
    values = normalized_intensity[mask]
    order = np.argsort(values)[::-1]
    chosen: list[np.ndarray] = []
    min_distance = float(config.hotspot_min_spacing_m)
    for idx in indices[order]:
        point = grid.point_from_index(tuple(idx))
        if all(float(np.linalg.norm(point - prior)) >= min_distance for prior in chosen):
            chosen.append(point)
        if len(chosen) >= int(config.max_hotspot_points):
            break
    if not chosen:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(chosen)


def synthesize_pressure_map(
    grid: SegmentationGrid,
    aperture: TransducerAperture,
    drive: np.ndarray,
    config: HybridPlanConfig,
    field_matrix: np.ndarray | None = None,
) -> np.ndarray:
    matrix = field_matrix
    if matrix is None:
        matrix = pressure_field_matrix(grid, aperture, config)
    return (matrix @ drive).reshape(grid.shape)


def pressure_field_matrix(
    grid: SegmentationGrid,
    aperture: TransducerAperture,
    config: HybridPlanConfig,
) -> np.ndarray:
    x, y = grid.coordinates_m()
    points = np.column_stack([x.ravel(), y.ravel()])
    matrix, _ = propagation_matrix(grid, aperture, points, config)
    return matrix


def result_summary(
    grid: SegmentationGrid,
    normalized_intensity: np.ndarray,
    best: dict[str, object],
    candidates: list[dict[str, object]],
    target_centroid_m: np.ndarray,
    config: HybridPlanConfig,
) -> dict[str, object]:
    metrics = best["metrics"]
    dense = dense_field_metrics(grid, normalized_intensity, target_centroid_m, config)
    return {
        "candidate_count": len(candidates),
        "selected_angle_deg": float(best["aperture"].angle_deg),
        "selected_source_angles_deg": list(best["aperture"].source_angles_deg),
        "selected_score": float(metrics["score"]),
        **dense,
        "air_path_fraction": float(metrics["air_path_fraction"]),
        "bone_path_fraction": float(metrics["bone_path_fraction"]),
        "fat_path_fraction": float(metrics["fat_path_fraction"]),
    }
