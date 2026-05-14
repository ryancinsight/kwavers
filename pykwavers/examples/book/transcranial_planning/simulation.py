from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from .transducer import PhaseCorrection, TransducerConfig, index_to_point


@dataclass(frozen=True)
class AcousticResult:
    pressure_pa: np.ndarray
    intensity_w_m2: np.ndarray
    mechanical_index: np.ndarray
    cavitation_probability: np.ndarray


@dataclass(frozen=True)
class ThermalResult:
    peak_temperature_c: np.ndarray
    final_temperature_c: np.ndarray
    cem43_min: np.ndarray
    lesion_mask: np.ndarray


@dataclass(frozen=True)
class SubspotPlan:
    indices: np.ndarray
    covered_fraction: float
    pitch_m: float


@dataclass(frozen=True)
class BbbOpeningResult:
    acoustic_dose: np.ndarray
    permeability: np.ndarray
    stable_cavitation_probability: np.ndarray
    inertial_cavitation_risk: np.ndarray
    opened_mask: np.ndarray


def simulation_coordinates(
    shape: tuple[int, int, int],
    spacing_m: tuple[float, float, float],
    target_index: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = [
        (np.arange(n, dtype=np.float64) - target_index[axis]) * spacing_m[axis]
        for axis, n in enumerate(shape)
    ]
    return np.meshgrid(*axes, indexing="ij")


def rayleigh_pressure_field(
    phase: PhaseCorrection,
    shape: tuple[int, int, int],
    spacing_m: tuple[float, float, float],
    target_index: tuple[int, int, int],
    config: TransducerConfig,
    target_peak_pressure_pa: float,
    chunk_points: int = 768,
) -> np.ndarray:
    grid = simulation_coordinates(shape, spacing_m, target_index)
    points = np.column_stack([axis.ravel() for axis in grid])
    field = np.empty(points.shape[0], dtype=np.complex128)
    k = 2.0 * np.pi * config.frequency_hz / config.brain_sound_speed_m_s
    element_weight = phase.active.astype(np.float64) * phase.amplitude_weights.astype(np.float64)
    weight_sum = float(np.sum(element_weight))
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        raise ValueError("phase correction produced non-positive element weights")
    element_weight = element_weight / weight_sum
    source = phase.element_positions_m

    for start in range(0, points.shape[0], chunk_points):
        stop = min(start + chunk_points, points.shape[0])
        diff = points[start:stop, None, :] - source[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist = np.maximum(dist, 1.0e-6)
        contribution = element_weight[None, :] * np.exp(1j * (k * dist + phase.phases_rad[None, :])) / dist
        field[start:stop] = np.sum(contribution, axis=1)

    pressure = np.abs(field).reshape(shape)
    peak = float(np.max(pressure))
    if peak <= 0.0 or not np.isfinite(peak):
        raise ValueError("Rayleigh field produced non-positive pressure")
    return (pressure * (target_peak_pressure_pa / peak)).astype(np.float32)


def acoustic_observables(
    pressure_pa: np.ndarray,
    frequency_hz: float,
    density_kg_m3: float = 1040.0,
    sound_speed_m_s: float = 1540.0,
    inertial_mi_threshold: float = 1.9,
) -> AcousticResult:
    intensity = pressure_pa.astype(np.float64) ** 2 / (2.0 * density_kg_m3 * sound_speed_m_s)
    frequency_mhz = frequency_hz / 1.0e6
    mi = pressure_pa.astype(np.float64) / 1.0e6 / np.sqrt(frequency_mhz)
    cavitation = 1.0 / (1.0 + np.exp(-(mi - inertial_mi_threshold) / 0.10))
    return AcousticResult(
        pressure_pa.astype(np.float32),
        intensity.astype(np.float32),
        mi.astype(np.float32),
        cavitation.astype(np.float32),
    )


def pennes_thermal_dose(
    intensity_w_m2: np.ndarray,
    skull_mask: np.ndarray,
    brain_mask: np.ndarray,
    spacing_m: tuple[float, float, float],
    sonication_s: float,
    dt_s: float,
    baseline_c: float = 37.0,
) -> ThermalResult:
    rho = np.where(skull_mask, 1908.0, np.where(brain_mask, 1040.0, 998.0))
    cp = np.where(skull_mask, 1313.0, np.where(brain_mask, 3650.0, 4182.0))
    k_th = np.where(skull_mask, 0.32, np.where(brain_mask, 0.51, 0.598))
    perf = np.where(brain_mask & ~skull_mask, 0.009, 0.0)
    alpha_np = np.where(skull_mask, 15.0, np.where(brain_mask, 3.5, 0.002)) * 0.650 * 11.516
    heat = 2.0 * alpha_np * intensity_w_m2.astype(np.float64)

    temp = np.full(intensity_w_m2.shape, baseline_c, dtype=np.float64)
    peak = temp.copy()
    cem43 = np.zeros_like(temp)
    steps = int(np.ceil(sonication_s / dt_s))
    for _ in range(steps):
        lap = anisotropic_laplacian(temp, spacing_m)
        dtemp = (
            k_th * lap
            - perf * rho * cp * (temp - baseline_c)
            + heat
        ) / (rho * cp)
        temp = temp + dt_s * dtemp
        peak = np.maximum(peak, temp)
        cem43 += cem43_increment(temp, dt_s)
    lesion = (cem43 >= 240.0) & brain_mask & ~skull_mask
    return ThermalResult(peak.astype(np.float32), temp.astype(np.float32), cem43.astype(np.float32), lesion)


def anisotropic_laplacian(values: np.ndarray, spacing_m: tuple[float, float, float]) -> np.ndarray:
    padded = np.pad(values, 1, mode="edge")
    center = padded[1:-1, 1:-1, 1:-1]
    lap = np.zeros_like(values, dtype=np.float64)
    for axis, spacing in enumerate(spacing_m):
        plus = [slice(1, -1), slice(1, -1), slice(1, -1)]
        minus = [slice(1, -1), slice(1, -1), slice(1, -1)]
        plus[axis] = slice(2, None)
        minus[axis] = slice(None, -2)
        lap += (padded[tuple(plus)] - 2.0 * center + padded[tuple(minus)]) / (spacing * spacing)
    return lap


def cem43_increment(temp_c: np.ndarray, dt_s: float) -> np.ndarray:
    r = np.where(temp_c >= 43.0, 0.5, 0.25)
    return (dt_s / 60.0) * np.power(r, 43.0 - temp_c)


def gbm_subspot_plan(
    tumor_mask: np.ndarray,
    spacing_m: tuple[float, float, float],
    pitch_m: float = 3.0e-3,
) -> SubspotPlan:
    if not np.any(tumor_mask):
        raise ValueError("tumor mask is empty")
    stride = np.maximum(np.rint(pitch_m / np.asarray(spacing_m)).astype(int), 1)
    coords = np.argwhere(tumor_mask)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    grid_axes = [np.arange(lo[axis], hi[axis], stride[axis]) for axis in range(3)]
    candidates = np.array(np.meshgrid(*grid_axes, indexing="ij")).reshape(3, -1).T
    inside = candidates[tumor_mask[candidates[:, 0], candidates[:, 1], candidates[:, 2]]]
    centroid = np.rint(coords.mean(axis=0)).astype(int)
    if inside.size == 0:
        inside = centroid[None, :]
    else:
        inside = np.unique(np.vstack([inside, centroid]), axis=0)

    dist_vox = distance_transform_edt(~tumor_mask, sampling=spacing_m)
    covered = np.zeros_like(tumor_mask, dtype=bool)
    radius_m = 0.5 * pitch_m
    for idx in inside:
        axes_m = index_to_physical_index_grid(tumor_mask.shape, spacing_m, tuple(idx))
        d2 = axes_m[0] * axes_m[0] + axes_m[1] * axes_m[1] + axes_m[2] * axes_m[2]
        covered |= d2 <= radius_m * radius_m
    tumor_count = int(np.count_nonzero(tumor_mask))
    covered_fraction = float(np.count_nonzero(covered & tumor_mask) / max(tumor_count, 1))
    finite_guard = float(np.max(dist_vox[tumor_mask]))
    if not np.isfinite(finite_guard):
        raise ValueError("tumor distance transform is non-finite")
    return SubspotPlan(inside.astype(int), covered_fraction, pitch_m)


def bbb_opening_from_subspots(
    tumor_mask: np.ndarray,
    plan: SubspotPlan,
    spacing_m: tuple[float, float, float],
    mechanical_index: float = 0.45,
    sonication_s: float = 60.0,
    duty_cycle: float = 0.02,
    focal_radius_m: float = 2.0e-3,
    d50: float = 0.40,
    hill_n: float = 2.5,
) -> BbbOpeningResult:
    """Compute BBB-opening dose from planned tumor subspots.

    Dose follows the Chapter 24 convention:

        D = MI^2 * t_on

    with Gaussian focal weighting around each subspot.  The stable operating
    window is the BBB-opening interval 0.20 <= MI <= 0.55 used in Chapter 24.
    """
    if not np.any(tumor_mask):
        raise ValueError("tumor mask is empty")
    if plan.indices.ndim != 2 or plan.indices.shape[1] != 3:
        raise ValueError("subspot plan indices must have shape (N, 3)")
    if mechanical_index <= 0.0:
        raise ValueError("mechanical index must be positive")
    if not 0.0 < duty_cycle <= 1.0:
        raise ValueError("duty cycle must be in (0, 1]")

    on_time_s = sonication_s * duty_cycle
    subspot_dose = mechanical_index * mechanical_index * on_time_s
    dose = np.zeros(tumor_mask.shape, dtype=np.float64)
    radius2 = focal_radius_m * focal_radius_m
    for center in plan.indices:
        axes_m = index_to_physical_index_grid(tumor_mask.shape, spacing_m, tuple(center))
        d2 = axes_m[0] * axes_m[0] + axes_m[1] * axes_m[1] + axes_m[2] * axes_m[2]
        dose += subspot_dose * np.exp(-0.5 * d2 / radius2)

    permeability = np.power(dose, hill_n) / (np.power(d50, hill_n) + np.power(dose, hill_n))
    stable_low = logistic((mechanical_index - 0.20) / 0.04)
    stable_high = logistic((0.55 - mechanical_index) / 0.04)
    stable_probability = permeability * stable_low * stable_high
    inertial_risk = logistic((mechanical_index - 0.55) / 0.04) * permeability
    opened = (permeability >= 0.50) & tumor_mask
    return BbbOpeningResult(
        dose.astype(np.float32),
        permeability.astype(np.float32),
        stable_probability.astype(np.float32),
        inertial_risk.astype(np.float32),
        opened,
    )


def logistic(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def index_to_physical_index_grid(
    shape: tuple[int, int, int],
    spacing_m: tuple[float, float, float],
    center: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = [
        (np.arange(n, dtype=np.float64) - center[axis]) * spacing_m[axis]
        for axis, n in enumerate(shape)
    ]
    return np.meshgrid(*axes, indexing="ij")
