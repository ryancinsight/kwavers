"""Chapter 29 raw Westervelt pressure diagnostics."""

from __future__ import annotations

import numpy as np


def pressure_diagnostics(result: dict[str, object]) -> dict[str, float | bool]:
    pressure = np.asarray(result["westervelt_peak_pressure_pa"], dtype=float)
    target = np.asarray(result.get("target_mask", np.zeros_like(pressure, dtype=bool)), dtype=bool)
    return pressure_field_diagnostics(
        pressure,
        target,
        frequency_hz=float(result.get("frequency_hz", 1.0e6)),
        source_pressure_pa=float(result.get("source_pressure_pa", 0.0)),
        source_scale=float(result.get("source_scale", 1.0)),
        inertial_mi_threshold=float(result.get("inertial_mi_threshold", 1.9)),
    )


def pressure_field_diagnostics(
    pressure: np.ndarray,
    target: np.ndarray,
    *,
    frequency_hz: float,
    source_pressure_pa: float,
    source_scale: float,
    inertial_mi_threshold: float,
) -> dict[str, float | bool]:
    values = np.asarray(pressure, dtype=float)
    finite = np.isfinite(values)
    finite_values = np.where(finite, np.abs(values), 0.0)
    active = np.asarray(target, dtype=bool)
    target_values = finite_values[active]
    peak = float(np.max(finite_values)) if finite_values.size else 0.0
    target_peak = float(np.max(target_values)) if target_values.size else 0.0
    frequency_mhz = max(float(frequency_hz) * 1.0e-6, 0.0)
    peak_mi = _mechanical_index(peak, frequency_mhz)
    target_mi = _mechanical_index(target_peak, frequency_mhz)
    effective_source_pressure = float(source_pressure_pa) * float(source_scale)
    return {
        "raw_peak_pressure_pa": peak,
        "raw_target_peak_pressure_pa": target_peak,
        "peak_mechanical_index": peak_mi,
        "target_peak_mechanical_index": target_mi,
        "inertial_mi_threshold": float(inertial_mi_threshold),
        "peak_exceeds_inertial_threshold": bool(peak_mi >= inertial_mi_threshold),
        "target_exceeds_inertial_threshold": bool(target_mi >= inertial_mi_threshold),
        "finite_pressure_fraction": float(np.count_nonzero(finite) / finite.size) if finite.size else 0.0,
        "source_pressure_pa": float(source_pressure_pa),
        "source_scale": float(source_scale),
        "effective_source_pressure_pa": effective_source_pressure,
        "raw_peak_to_effective_source_ratio": peak / effective_source_pressure
        if effective_source_pressure > 0.0
        else 0.0,
    }


def _mechanical_index(pressure_pa: float, frequency_mhz: float) -> float:
    if pressure_pa <= 0.0 or frequency_mhz <= 0.0:
        return 0.0
    return pressure_pa * 1.0e-6 / float(np.sqrt(frequency_mhz))
