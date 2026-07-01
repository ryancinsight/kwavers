//! Acousto-optic angle and frequency-shift bindings.

use kwavers_physics::analytical::acousto_optics;
use pyo3::prelude::*;

/// Diffraction angle [rad] of order m from sin θₘ = m λ₀/(n Λ); NaN if evanescent.
#[pyfunction]
#[pyo3(signature = (order, optical_wavelength_m, refractive_index, acoustic_wavelength_m))]
pub fn diffraction_angle_rad(
    order: i32,
    optical_wavelength_m: f64,
    refractive_index: f64,
    acoustic_wavelength_m: f64,
) -> f64 {
    acousto_optics::diffraction_angle_rad(
        order,
        optical_wavelength_m,
        refractive_index,
        acoustic_wavelength_m,
    )
    .unwrap_or(f64::NAN)
}

/// Frequency shift Δf = m·f_acoustic of the m-th diffracted order (AOM principle).
#[pyfunction]
#[pyo3(signature = (order, acoustic_frequency_hz))]
pub fn diffraction_frequency_shift_hz(order: i32, acoustic_frequency_hz: f64) -> f64 {
    acousto_optics::diffraction_frequency_shift_hz(order, acoustic_frequency_hz)
}

/// Bragg angle θ_B = arcsin(λ₀/(2nΛ)) [rad]; NaN if no Bragg solution.
#[pyfunction]
#[pyo3(signature = (optical_wavelength_m, refractive_index, acoustic_wavelength_m))]
pub fn bragg_angle_rad(
    optical_wavelength_m: f64,
    refractive_index: f64,
    acoustic_wavelength_m: f64,
) -> f64 {
    acousto_optics::bragg_angle_rad(
        optical_wavelength_m,
        refractive_index,
        acoustic_wavelength_m,
    )
    .unwrap_or(f64::NAN)
}
