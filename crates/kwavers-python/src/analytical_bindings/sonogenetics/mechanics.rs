//! Acoustic force and streaming bindings for sonogenetics.

use kwavers_physics::analytical::sonogenetics;
use numpy::{ToPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Compute the 1-D acoustic radiation force density.
///
/// F = 2 * alpha * I / c
///
/// Args:
///     intensity_w_m2: Intensity array [W/m²].
///     alpha_np_m: Attenuation [Np/m].
///     c: Sound speed [m/s].
///
/// Returns:
///     Radiation force density array [N/m³].
#[pyfunction]
#[pyo3(signature = (intensity_w_m2, alpha_np_m, c))]
pub fn radiation_force_1d(
    py: Python<'_>,
    intensity_w_m2: PyReadonlyArray1<f64>,
    alpha_np_m: f64,
    c: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let i_s = intensity_w_m2
        .as_slice()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = sonogenetics::radiation_force_1d(i_s, alpha_np_m, c);
    Ok(result.to_pyarray(py).unbind())
}

/// Gor'kov monopole (compressibility) contrast factor `f₁ = 1 − κ̃` (Eq. 17.3),
/// where `κ̃ = κ_cell/κ_medium`.
#[pyfunction]
#[pyo3(signature = (compressibility_ratio))]
pub fn acoustic_monopole_contrast(compressibility_ratio: f64) -> f64 {
    sonogenetics::acoustic_monopole_contrast(compressibility_ratio)
}

/// Gor'kov dipole (density) contrast factor `f₂ = 2(ρ̃ − 1)/(2ρ̃ + 1)` (Eq. 17.3),
/// where `ρ̃ = ρ_cell/ρ_medium`.
#[pyfunction]
#[pyo3(signature = (density_ratio))]
pub fn acoustic_dipole_contrast(density_ratio: f64) -> f64 {
    sonogenetics::acoustic_dipole_contrast(density_ratio)
}

/// One-dimensional Gor'kov primary radiation force `F = −dU/dx` [N] on a small
/// sphere (Eq. 17.2), from the spatial gradients of `⟨p²⟩` and `⟨v²⟩`:
///
/// `F = −(2π r³/3)·[ f₁·∂⟨p²⟩/∂x /(ρc²) − (3/2)·f₂·ρ·∂⟨v²⟩/∂x ]`.
///
/// Args:
///     radius_m: Sphere radius r [m] (r ≪ λ).
///     grad_pressure_sq: ∂⟨p²⟩/∂x [Pa²/m].
///     grad_velocity_sq: ∂⟨v²⟩/∂x [(m/s)²/m].
///     rho_medium: Medium density ρ [kg/m³].
///     c_medium: Medium sound speed c [m/s].
///     density_ratio: ρ̃ = ρ_sphere/ρ_medium.
///     compressibility_ratio: κ̃ = κ_sphere/κ_medium.
///
/// Returns:
///     Radiation force [N].
#[pyfunction]
#[pyo3(signature = (
    radius_m, grad_pressure_sq, grad_velocity_sq,
    rho_medium, c_medium, density_ratio, compressibility_ratio
))]
pub fn gorkov_radiation_force_1d(
    radius_m: f64,
    grad_pressure_sq: f64,
    grad_velocity_sq: f64,
    rho_medium: f64,
    c_medium: f64,
    density_ratio: f64,
    compressibility_ratio: f64,
) -> f64 {
    sonogenetics::gorkov_radiation_force_1d(
        radius_m,
        grad_pressure_sq,
        grad_velocity_sq,
        rho_medium,
        c_medium,
        density_ratio,
        compressibility_ratio,
    )
}

/// Compute the steady acoustic streaming velocity (Eckart streaming).
///
/// Args:
///     i_w_m2: Beam intensity [W/m²].
///     mu_pa_s: Dynamic viscosity [Pa·s].
///     alpha_np_m: Attenuation [Np/m].
///     c: Sound speed [m/s].
///     l_m: Beam propagation length [m].
///
/// Returns:
///     Streaming velocity [m/s].
#[pyfunction]
#[pyo3(signature = (i_w_m2, mu_pa_s, alpha_np_m, c, l_m))]
pub fn acoustic_streaming_velocity(
    i_w_m2: f64,
    mu_pa_s: f64,
    alpha_np_m: f64,
    c: f64,
    l_m: f64,
) -> PyResult<f64> {
    Ok(sonogenetics::acoustic_streaming_velocity(
        i_w_m2, mu_pa_s, alpha_np_m, c, l_m,
    ))
}

