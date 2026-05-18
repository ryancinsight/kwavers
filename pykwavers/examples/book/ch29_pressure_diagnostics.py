"""Chapter 29 raw Westervelt pressure diagnostics."""

from __future__ import annotations

import numpy as np


def pressure_diagnostics(result: dict[str, object]) -> dict[str, float | bool]:
    pressure = np.asarray(result["westervelt_peak_pressure_pa"], dtype=float)
    target = np.asarray(result.get("target_mask", np.zeros_like(pressure, dtype=bool)), dtype=bool)
    body = np.asarray(result.get("body_mask", np.ones_like(pressure, dtype=bool)), dtype=bool)
    diagnostics = pressure_field_diagnostics(
        pressure,
        target,
        body_mask=body,
        frequency_hz=float(result.get("frequency_hz", 1.0e6)),
        source_pressure_pa=float(result.get("source_pressure_pa", 0.0)),
        source_scale=float(result.get("source_scale", 1.0)),
        inertial_mi_threshold=float(result.get("inertial_mi_threshold", 1.9)),
    )
    diagnostics["points_per_wavelength_min"] = float(result.get("points_per_wavelength_min", 0.0))
    diagnostics["resolution_meets_min_ppw"] = bool(result.get("resolution_meets_min_ppw", False))
    return diagnostics


def pressure_field_diagnostics(
    pressure: np.ndarray,
    target: np.ndarray,
    *,
    body_mask: np.ndarray | None = None,
    frequency_hz: float,
    source_pressure_pa: float,
    source_scale: float,
    inertial_mi_threshold: float,
) -> dict[str, float | bool]:
    values = np.asarray(pressure, dtype=float)
    finite = np.isfinite(values)
    finite_values = np.where(finite, np.abs(values), 0.0)
    active = np.asarray(target, dtype=bool)
    body = np.ones(values.shape, dtype=bool) if body_mask is None else np.asarray(body_mask, dtype=bool)
    if body.shape != values.shape:
        body = np.ones(values.shape, dtype=bool)
    target_values = finite_values[active]
    body_values = finite_values[body]
    coupling_values = finite_values[~body]
    peak = float(np.max(finite_values)) if finite_values.size else 0.0
    target_peak = float(np.max(target_values)) if target_values.size else 0.0
    body_peak = float(np.max(body_values)) if body_values.size else 0.0
    coupling_peak = float(np.max(coupling_values)) if coupling_values.size else 0.0
    target_centroid = _mask_centroid(active)
    raw_hotspot = _hotspot_index(finite_values)
    body_hotspot = _hotspot_index(np.where(body, finite_values, 0.0))
    frequency_mhz = max(float(frequency_hz) * 1.0e-6, 0.0)
    peak_mi = _mechanical_index(peak, frequency_mhz)
    target_mi = _mechanical_index(target_peak, frequency_mhz)
    body_mi = _mechanical_index(body_peak, frequency_mhz)
    coupling_mi = _mechanical_index(coupling_peak, frequency_mhz)
    effective_source_pressure = float(source_pressure_pa) * float(source_scale)
    return {
        "raw_peak_pressure_pa": peak,
        "raw_target_peak_pressure_pa": target_peak,
        "body_peak_pressure_pa": body_peak,
        "coupling_peak_pressure_pa": coupling_peak,
        "peak_mechanical_index": peak_mi,
        "target_peak_mechanical_index": target_mi,
        "body_peak_mechanical_index": body_mi,
        "coupling_peak_mechanical_index": coupling_mi,
        "inertial_mi_threshold": float(inertial_mi_threshold),
        "peak_exceeds_inertial_threshold": bool(peak_mi >= inertial_mi_threshold),
        "target_exceeds_inertial_threshold": bool(target_mi >= inertial_mi_threshold),
        "body_peak_exceeds_inertial_threshold": bool(body_mi >= inertial_mi_threshold),
        "coupling_peak_exceeds_inertial_threshold": bool(coupling_mi >= inertial_mi_threshold),
        "raw_peak_is_in_coupling": bool(coupling_peak >= body_peak and coupling_peak > 0.0),
        "target_centroid_x_index": _component(target_centroid, 0),
        "target_centroid_y_index": _component(target_centroid, 1),
        "target_centroid_z_index": _component(target_centroid, 2),
        "raw_hotspot_x_index": _component(raw_hotspot, 0),
        "raw_hotspot_y_index": _component(raw_hotspot, 1),
        "raw_hotspot_z_index": _component(raw_hotspot, 2),
        "body_hotspot_x_index": _component(body_hotspot, 0),
        "body_hotspot_y_index": _component(body_hotspot, 1),
        "body_hotspot_z_index": _component(body_hotspot, 2),
        "raw_hotspot_distance_to_target_grid_cells": _distance(raw_hotspot, target_centroid),
        "body_hotspot_distance_to_target_grid_cells": _distance(body_hotspot, target_centroid),
        "finite_pressure_fraction": float(np.count_nonzero(finite) / finite.size) if finite.size else 0.0,
        "source_pressure_pa": float(source_pressure_pa),
        "source_scale": float(source_scale),
        "effective_source_pressure_pa": effective_source_pressure,
        "raw_peak_to_effective_source_ratio": peak / effective_source_pressure
        if effective_source_pressure > 0.0
        else 0.0,
        "target_to_body_peak_ratio": target_peak / body_peak if body_peak > 0.0 else 0.0,
        "coupling_to_body_peak_ratio": coupling_peak / body_peak if body_peak > 0.0 else 0.0,
    }


def _mask_centroid(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size:
        return coords.mean(axis=0)
    shape = np.asarray(mask.shape, dtype=float)
    return 0.5 * np.maximum(shape - 1.0, 0.0)


def _hotspot_index(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(3, dtype=float)
    return np.asarray(np.unravel_index(int(np.argmax(values)), values.shape), dtype=float)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def _component(values: np.ndarray, index: int) -> float:
    array = np.asarray(values, dtype=float)
    return float(array[index]) if index < array.size else 0.0


def _mechanical_index(pressure_pa: float, frequency_mhz: float) -> float:
    if pressure_pa <= 0.0 or frequency_mhz <= 0.0:
        return 0.0
    return pressure_pa * 1.0e-6 / float(np.sqrt(frequency_mhz))
