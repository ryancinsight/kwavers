//! Lesion-state and histotripsy-dose PyO3 wrappers.

use kwavers_physics::analytical::cavitation;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Backscatter coefficient of partially fractionated tissue (lesion B-mode).
///
/// σ_bsc(f) = σ_liquefied + (σ_intact − σ_liquefied)·(1 − f)^γ. As the
/// fractionation fraction `f` rises the lesion loses speckle scatterers and
/// becomes hypoechoic. Thin wrapper over
/// `kwavers_physics::analytical::cavitation::fractionation_backscatter_coefficient`.
///
/// Args:
///     fractionation: per-voxel fractionation/kill fraction [0, 1].
///     sigma_intact: intact-tissue backscatter coefficient (arb. units).
///     sigma_liquefied: liquefied-homogenate floor backscatter coefficient.
///     gamma: scatterer-loss exponent (≥ 1; 2 = quadratic).
///
/// Returns:
///     Backscatter-coefficient array, same length as `fractionation`.
#[pyfunction]
#[pyo3(signature = (fractionation, sigma_intact, sigma_liquefied, gamma))]
pub fn fractionation_backscatter_coefficient(
    py: Python<'_>,
    fractionation: PyReadonlyArray1<f64>,
    sigma_intact: f64,
    sigma_liquefied: f64,
    gamma: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f = fractionation
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out =
        cavitation::fractionation_backscatter_coefficient(f, sigma_intact, sigma_liquefied, gamma);
    Ok(out.to_pyarray(py).unbind())
}

/// Acoustic impedance of partially fractionated tissue (lesion-rim echo).
///
/// Z(f) = z_intact·(1 − f) + z_liquefied·f (linear volume mixing). The spatial
/// gradient of this map produces the specular bright rim at the lesion boundary.
/// Thin wrapper over
/// `kwavers_physics::analytical::cavitation::fractionation_acoustic_impedance`.
///
/// Args:
///     fractionation: per-voxel fractionation/kill fraction [0, 1].
///     z_intact: intact-tissue acoustic impedance [Rayl].
///     z_liquefied: liquefied-homogenate acoustic impedance [Rayl].
///
/// Returns:
///     Acoustic-impedance array, same length as `fractionation`.
#[pyfunction]
#[pyo3(signature = (fractionation, z_intact, z_liquefied))]
pub fn fractionation_acoustic_impedance(
    py: Python<'_>,
    fractionation: PyReadonlyArray1<f64>,
    z_intact: f64,
    z_liquefied: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let f = fractionation
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = cavitation::fractionation_acoustic_impedance(f, z_intact, z_liquefied);
    Ok(out.to_pyarray(py).unbind())
}

/// Size boiling-histotripsy lesion and pulse count from a resolved pressure
/// profile. Returns `(pulses, lateral_radius_m, axial_radius_m, pulse_ms)`, or
/// `None` when the focus does not boil within the pulse limit.
#[pyfunction]
#[pyo3(signature = (
    radius_m, normalized_pressure, focal_pressure_pa, focal_depth_m, freq_hz,
    c_m_s, rho_kg_m3, beta_nonlinearity, alpha_np_m, heat_capacity_j_kg_k,
    delta_t_k, tau_max_s, axial_to_lateral_ratio, clearance_m, coverage_target
))]
#[allow(clippy::too_many_arguments)]
pub fn boiling_lesion_from_pressure_profile(
    radius_m: PyReadonlyArray1<f64>,
    normalized_pressure: PyReadonlyArray1<f64>,
    focal_pressure_pa: f64,
    focal_depth_m: f64,
    freq_hz: f64,
    c_m_s: f64,
    rho_kg_m3: f64,
    beta_nonlinearity: f64,
    alpha_np_m: f64,
    heat_capacity_j_kg_k: f64,
    delta_t_k: f64,
    tau_max_s: f64,
    axial_to_lateral_ratio: f64,
    clearance_m: f64,
    coverage_target: f64,
) -> PyResult<Option<(usize, f64, f64, f64)>> {
    let r = radius_m
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let b = normalized_pressure
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(cavitation::boiling_lesion_from_pressure_profile(
        r,
        b,
        focal_pressure_pa,
        focal_depth_m,
        freq_hz,
        c_m_s,
        rho_kg_m3,
        beta_nonlinearity,
        alpha_np_m,
        heat_capacity_j_kg_k,
        delta_t_k,
        tau_max_s,
        axial_to_lateral_ratio,
        clearance_m,
        coverage_target,
    )
    .map(|p| (p.pulses, p.lateral_radius_m, p.axial_radius_m, p.pulse_ms)))
}

/// Boiling-onset time samples from normalized pressure samples.
#[pyfunction]
#[pyo3(signature = (
    normalized_pressure, focal_pressure_pa, focal_depth_m, freq_hz, c_m_s,
    rho_kg_m3, beta_nonlinearity, alpha_np_m, heat_capacity_j_kg_k, delta_t_k
))]
#[allow(clippy::too_many_arguments)]
pub fn boiling_time_profile_from_pressure(
    py: Python<'_>,
    normalized_pressure: PyReadonlyArray1<f64>,
    focal_pressure_pa: f64,
    focal_depth_m: f64,
    freq_hz: f64,
    c_m_s: f64,
    rho_kg_m3: f64,
    beta_nonlinearity: f64,
    alpha_np_m: f64,
    heat_capacity_j_kg_k: f64,
    delta_t_k: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let b = normalized_pressure
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = py.detach(|| {
        cavitation::boiling_time_profile_from_pressure(
            b,
            focal_pressure_pa,
            focal_depth_m,
            freq_hz,
            c_m_s,
            rho_kg_m3,
            beta_nonlinearity,
            alpha_np_m,
            heat_capacity_j_kg_k,
            delta_t_k,
        )
    });
    Ok(out.to_pyarray(py).unbind())
}

/// Lacuna gas void fraction in fractionated tissue from first-order gas-evolution
/// growth: `β = β_max·f·(1 − exp(−t_since/τ_lacuna))`. Feeds the Wood/Commander–
/// Prosperetti medium coupling so the growing lacuna geometry shields and aberrates
/// subsequent pulses (the persistent gas cavity, distinct from the fast residual
/// bubble-cloud dissolution).
///
/// Args:
///     fractionation, time_since_lesion_s, tau_lacuna_s, beta_max.
#[pyfunction]
#[pyo3(signature = (fractionation, time_since_lesion_s, tau_lacuna_s, beta_max))]
pub fn lacuna_void_fraction(
    fractionation: f64,
    time_since_lesion_s: f64,
    tau_lacuna_s: f64,
    beta_max: f64,
) -> PyResult<f64> {
    Ok(cavitation::lacuna_void_fraction(
        fractionation,
        time_since_lesion_s,
        tau_lacuna_s,
        beta_max,
    ))
}

/// Pulse count needed to grow a histotripsy lesion to `target_radius_m` via the
/// cavitation energy-balance model `R_L = R₀·(P₀·N·icd_per_pulse/σ_y)^(1/3)`.
///
/// Used to size the per-spot dose for full tumour coverage and to cap it so the
/// expanding lesion keeps a safe margin from a sensitive structure.
///
/// Args:
///     target_radius_m, r0_m, p0_pa, tissue_yield_stress_pa, icd_per_pulse.
#[pyfunction]
#[pyo3(signature = (target_radius_m, r0_m, p0_pa, tissue_yield_stress_pa, icd_per_pulse))]
pub fn histotripsy_pulses_for_lesion_radius(
    target_radius_m: f64,
    r0_m: f64,
    p0_pa: f64,
    tissue_yield_stress_pa: f64,
    icd_per_pulse: f64,
) -> PyResult<f64> {
    Ok(cavitation::histotripsy_pulses_for_lesion_radius(
        target_radius_m,
        r0_m,
        p0_pa,
        tissue_yield_stress_pa,
        icd_per_pulse,
    ))
}

/// Histotripsy lesion radius `m` from accumulated inertial cavitation dose via
/// the cavitation energy-balance model `R_L = R₀·(P₀·icd/σ_y)^(1/3)` (forward
/// of [`histotripsy_pulses_for_lesion_radius`]).
///
/// Args:
///     icd (total dimensionless inertial cavitation dose), r0_m, p0_pa,
///     tissue_yield_stress_pa.
#[pyfunction]
#[pyo3(signature = (icd, r0_m, p0_pa, tissue_yield_stress_pa))]
pub fn histotripsy_lesion_radius_m(
    icd: f64,
    r0_m: f64,
    p0_pa: f64,
    tissue_yield_stress_pa: f64,
) -> PyResult<f64> {
    Ok(cavitation::histotripsy_lesion_radius_m(
        icd,
        r0_m,
        p0_pa,
        tissue_yield_stress_pa,
    ))
}

/// Inertial cavitation dose (ICD) from a bubble radius/wall-velocity trajectory:
/// the sum of `(R_max/R₀)³` over detected inertial collapse events (Duryea 2015).
/// Dimensionless, O(1–1000); feeds the lesion energy-balance model.
///
/// Args:
///     r_arr (radius `m`), rdot_arr (wall velocity [m/s]), r0_m (equilibrium `m`).
#[pyfunction]
#[pyo3(signature = (r_arr, rdot_arr, r0_m))]
pub fn inertial_cavitation_dose(
    r_arr: PyReadonlyArray1<f64>,
    rdot_arr: PyReadonlyArray1<f64>,
    r0_m: f64,
) -> PyResult<f64> {
    let r = r_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let rd = rdot_arr
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(cavitation::inertial_cavitation_dose(r, rd, r0_m))
}
