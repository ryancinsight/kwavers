//! Acousto-optic regime parameter bindings.

use kwavers_physics::analytical::acousto_optics;
use pyo3::prelude::*;

/// Klein–Cook parameter Q = 2π λ₀ L / (n Λ²).
///
/// Args:
///     optical_wavelength_m: Vacuum optical wavelength λ₀ `m`.
///     interaction_length_m: Sound-column width L `m`.
///     refractive_index: Medium refractive index n.
///     acoustic_wavelength_m: Acoustic wavelength Λ `m`.
///
/// Returns:
///     Klein–Cook parameter Q (dimensionless).
#[pyfunction]
#[pyo3(signature = (optical_wavelength_m, interaction_length_m, refractive_index, acoustic_wavelength_m))]
pub fn klein_cook_parameter(
    optical_wavelength_m: f64,
    interaction_length_m: f64,
    refractive_index: f64,
    acoustic_wavelength_m: f64,
) -> f64 {
    acousto_optics::klein_cook_parameter(
        optical_wavelength_m,
        interaction_length_m,
        refractive_index,
        acoustic_wavelength_m,
    )
}

/// Raman–Nath phase parameter ν = 2π Δn L / λ₀.
#[pyfunction]
#[pyo3(signature = (delta_n, interaction_length_m, optical_wavelength_m))]
pub fn raman_nath_parameter(
    delta_n: f64,
    interaction_length_m: f64,
    optical_wavelength_m: f64,
) -> f64 {
    acousto_optics::raman_nath_parameter(delta_n, interaction_length_m, optical_wavelength_m)
}

/// Bragg (thick-grating) first-order diffraction efficiency η = sin²(ν/2).
#[pyfunction]
#[pyo3(signature = (nu,))]
pub fn bragg_diffraction_efficiency(nu: f64) -> f64 {
    acousto_optics::bragg_diffraction_efficiency(nu)
}
