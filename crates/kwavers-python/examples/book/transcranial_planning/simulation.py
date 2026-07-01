"""Transcranial FUS simulation orchestration.

All wave physics (Rayleigh-Sommerfeld integration, skull ray tracing, Pennes
bioheat PDE) execute in Rust via ``pykwavers``.  This module contains only:

1. Python dataclasses that carry the Rust outputs.
2. Thin marshalling wrappers that convert triplet/transducer objects to Rust
   call arguments and package the returned dicts as typed dataclasses.
3. Thin planning adapters around Rust/PyO3 kernels:
   ``acoustic_observables``, ``gbm_subspot_plan``,
   ``bbb_opening_from_subspots``.

No NumPy wave simulation loops, no finite-difference PDE steppers, and no
Rayleigh-Sommerfeld summations appear in this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import pykwavers as kw

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
    pressure_flat = np.ascontiguousarray(pressure_pa, dtype=np.float64).ravel()
    mi = np.asarray(kw.mechanical_index_field(pressure_flat, frequency_hz)).reshape(pressure_pa.shape)
    cavitation = np.asarray(
        kw.mechanical_index_cavitation_risk(
            np.ascontiguousarray(mi.ravel(), dtype=np.float64),
            inertial_mi_threshold,
            10.0,
        )
    ).reshape(pressure_pa.shape)
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
    """Compute a Rust-owned raster grid of sonication subspots inside a tumour."""
    if not np.any(tumor_mask):
        raise ValueError("tumor mask is empty")
    result = kw.gbm_subspot_raster_py(
        np.ascontiguousarray(tumor_mask, dtype=bool),
        spacing_m,
        pitch_m,
    )
    inside = np.asarray(result["indices"], dtype=int)
    covered_fraction = float(result["covered_fraction"])
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
    """Compute Rust-owned BBB-opening dose from planned tumour subspots."""
    if not np.any(tumor_mask):
        raise ValueError("tumor mask is empty")
    if plan.indices.ndim != 2 or plan.indices.shape[1] != 3:
        raise ValueError("subspot plan indices must have shape (N, 3)")
    if mechanical_index <= 0.0:
        raise ValueError("mechanical index must be positive")
    if not 0.0 < duty_cycle <= 1.0:
        raise ValueError("duty cycle must be in (0, 1]")

    result = kw.bbb_opening_from_subspots_py(
        np.ascontiguousarray(tumor_mask, dtype=bool),
        np.ascontiguousarray(plan.indices, dtype=np.uintp),
        spacing_m,
        mechanical_index,
        sonication_s,
        duty_cycle,
        focal_radius_m,
        d50,
        hill_n,
    )
    return BbbOpeningResult(
        np.asarray(result["dose"], dtype=np.float32),
        np.asarray(result["permeability"], dtype=np.float32),
        np.asarray(result["stable_cavitation_probability"], dtype=np.float32),
        np.asarray(result["inertial_cavitation_risk"], dtype=np.float32),
        np.asarray(result["opened_mask"], dtype=bool),
    )
