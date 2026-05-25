"""Transcranial FUS simulation orchestration.

All wave physics (Rayleigh-Sommerfeld integration, skull ray tracing, Pennes
bioheat PDE) execute in Rust via ``pykwavers``.  This module contains only:

1. Python dataclasses that carry the Rust outputs.
2. Thin marshalling wrappers that convert triplet/transducer objects to Rust
   call arguments and package the returned dicts as typed dataclasses.
3. Analytical / planning functions that have no Rust equivalent (acceptable in
   Python): ``acoustic_observables``, ``gbm_subspot_plan``,
   ``bbb_opening_from_subspots``.

No NumPy wave simulation loops, no finite-difference PDE steppers, and no
Rayleigh-Sommerfeld summations appear in this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from .transducer import PhaseCorrection, TransducerConfig, index_to_point


# ── Output dataclasses ────────────────────────────────────────────────────────

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


# ── Rust-delegating wrappers ──────────────────────────────────────────────────

def run_transcranial_fus_from_arrays(
    triplet: object,
    transducer: TransducerConfig,
    samples_per_ray: int = 192,
    target_peak_pressure_pa: float = 1.0e6,
    chunk_size: int = 512,
) -> tuple[PhaseCorrection, AcousticResult]:
    """Run skull ray tracing + Rayleigh field synthesis via Rust.

    Delegates the full transcranial FUS planning pipeline to
    ``pykwavers.run_transcranial_fus_planning_from_arrays``.  The returned
    ``PhaseCorrection`` object is reconstructed from the Rust plan dict via
    pure data marshalling (no physics in Python).

    Parameters
    ----------
    triplet:
        ``BrainTriplet`` with ``ct_hu``, ``skull_mask``, ``brain_mask``,
        ``target_index``.
    transducer:
        ``TransducerConfig`` carrying element count, frequency, geometry.
    samples_per_ray:
        Number of ray-marching samples per transducer element for skull delay
        estimation.
    target_peak_pressure_pa:
        Desired peak positive pressure at the focus [Pa].
    chunk_size:
        Number of grid points evaluated per Rayleigh chunk (memory control).

    Returns
    -------
    (phase, acoustic):
        ``PhaseCorrection`` with per-element phases, delays, skull lengths, and
        amplitude weights; ``AcousticResult`` with pressure, intensity, MI, and
        cavitation probability fields.
    """
    import pykwavers as kw

    ct = triplet.ct_hu.data.astype(np.float64, copy=False)
    sp = triplet.ct_hu.spacing_m
    # Empty tumor mask: ET VIM planning does not involve GBM subspot raster.
    tumor = np.zeros(ct.shape, dtype=bool)

    plan = kw.run_transcranial_fus_planning_from_arrays(
        ct_hu=ct,
        skull_mask=triplet.skull_mask,
        brain_mask=triplet.brain_mask,
        tumor_mask=tumor,
        spacing_m=sp,
        target_index=triplet.target_index,
        element_count=transducer.element_count,
        frequency_hz=transducer.frequency_hz,
        radius_m=transducer.radius_m,
        cap_min_polar_rad=transducer.cap_min_polar_rad,
        cap_max_polar_rad=transducer.cap_max_polar_rad,
        brain_sound_speed=transducer.brain_sound_speed_m_s,
        skull_sound_speed=transducer.skull_sound_speed_m_s,
        target_peak_pa=target_peak_pressure_pa,
        samples_per_ray=samples_per_ray,
        chunk_size=chunk_size,
    )

    # Reconstruct PhaseCorrection from plan dict — data marshalling only, no physics.
    phase = PhaseCorrection(
        element_positions_m=plan["element_positions_m"],
        phases_rad=plan["phases_rad"],
        delays_s=plan["delays_s"],
        skull_lengths_m=plan["skull_lengths_m"],
        amplitude_weights=plan["amplitude_weights"],
        active=np.ones(transducer.element_count, dtype=bool),
    )

    acoustic = AcousticResult(
        pressure_pa=plan["pressure_pa"],
        intensity_w_m2=plan["intensity_w_m2"],
        mechanical_index=plan["mechanical_index"],
        cavitation_probability=plan["cavitation_probability"],
    )

    return phase, acoustic


def pennes_thermal_dose(
    intensity_w_m2: np.ndarray,
    skull_mask: np.ndarray,
    brain_mask: np.ndarray,
    spacing_m: tuple[float, float, float],
    sonication_s: float,
    dt_s: float,
    baseline_c: float = 37.0,
    frequency_hz: float = 650_000.0,
) -> ThermalResult:
    """Heterogeneous Pennes bioheat thermal dose via Rust.

    Delegates all PDE stepping to
    ``pykwavers.transcranial_pennes_thermal_dose_py``.  Skull and brain tissue
    properties (density, specific heat, thermal conductivity, perfusion,
    acoustic attenuation) are resolved per voxel inside the Rust implementation.

    Parameters
    ----------
    intensity_w_m2:
        Steady-state time-averaged acoustic intensity field [W/m²],
        shape (nx, ny, nz).
    skull_mask, brain_mask:
        Boolean masks of shape (nx, ny, nz).
    spacing_m:
        Voxel edge lengths [m].
    sonication_s:
        Total sonication duration [s].
    dt_s:
        Explicit Euler time step [s].
    baseline_c:
        Initial and arterial blood temperature [°C].
    frequency_hz:
        Operating frequency [Hz]; controls α→Q conversion.

    Returns
    -------
    ThermalResult
    """
    import pykwavers as kw

    result = kw.transcranial_pennes_thermal_dose_py(
        intensity_w_m2=intensity_w_m2.astype(np.float32, copy=False),
        skull_mask=skull_mask,
        brain_mask=brain_mask,
        spacing_m=spacing_m,
        frequency_hz=frequency_hz,
        sonication_s=sonication_s,
        dt_s=dt_s,
        baseline_c=baseline_c,
    )
    return ThermalResult(
        peak_temperature_c=result["peak_temperature_c"],
        final_temperature_c=result["final_temperature_c"],
        cem43_min=result["cem43_min"],
        lesion_mask=result["lesion_mask"],
    )


# ── Analytical / planning functions (acceptable Python) ──────────────────────

def acoustic_observables(
    pressure_pa: np.ndarray,
    frequency_hz: float,
    density_kg_m3: float = 1040.0,
    sound_speed_m_s: float = 1540.0,
    inertial_mi_threshold: float = 1.9,
) -> AcousticResult:
    """Derive acoustic observables from a pressure field.

    Analytical relations only — no wave simulation:

    - Intensity: I = p² / (2·ρ·c)   [W/m²]
    - MI:        MI = p_peak / (1 MPa·√f_MHz)
    - Cavitation probability: logistic in MI.

    Used for ad-hoc pressure field analysis when the Rust pipeline output is
    not available.
    """
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


def gbm_subspot_plan(
    tumor_mask: np.ndarray,
    spacing_m: tuple[float, float, float],
    pitch_m: float = 3.0e-3,
) -> SubspotPlan:
    """Compute a raster grid of sonication subspots inside the tumour mask.

    Geometric planning only — no wave physics.
    """
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
    """Compute BBB-opening dose from planned tumour subspots.

    Dose follows the Chapter 24 convention:

        D = MI^2 * t_on

    with Gaussian focal weighting around each subspot.  The stable operating
    window is the BBB-opening interval 0.20 <= MI <= 0.55 used in Chapter 24.

    Semi-empirical Hill model — no wave physics.
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


# ── Utility ───────────────────────────────────────────────────────────────────

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
