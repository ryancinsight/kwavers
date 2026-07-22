//! Single-bubble scalar PyO3 wrappers.

use kwavers_physics::analytical::cavitation;
use pyo3::prelude::*;

/// Compute the Minnaert resonance frequency of a free bubble.
///
/// f_r = (1/(2*pi*r0)) * sqrt(3*gamma*p0 / rho)
///
/// Args:
///     r0_m: Equilibrium bubble radius `m`.
///     gamma: Polytropic index.
///     p0_pa: Ambient pressure `Pa`.
///     rho: Liquid density [kg/m³].
///
/// Returns:
///     Resonance frequency `Hz`.
#[pyfunction]
#[pyo3(signature = (r0_m, gamma, p0_pa, rho))]
pub fn minnaert_resonance_hz(r0_m: f64, gamma: f64, p0_pa: f64, rho: f64) -> PyResult<f64> {
    Ok(cavitation::minnaert_resonance_hz(r0_m, gamma, p0_pa, rho))
}

/// Compute the bubble radius whose Minnaert resonance matches a frequency.
///
/// R0 = (1/(2*pi*f_r)) * sqrt(3*gamma*p0 / rho)
///
/// Args:
///     frequency_hz: Resonance frequency `Hz`.
///     gamma: Polytropic index.
///     p0_pa: Ambient pressure `Pa`.
///     rho: Liquid density [kg/m³].
///
/// Returns:
///     Equilibrium bubble radius `m`, or 0 if inputs are invalid.
#[pyfunction]
#[pyo3(signature = (frequency_hz, gamma, p0_pa, rho))]
pub fn minnaert_radius_for_frequency_m(
    frequency_hz: f64,
    gamma: f64,
    p0_pa: f64,
    rho: f64,
) -> PyResult<f64> {
    Ok(cavitation::minnaert_radius_for_frequency_m(
        frequency_hz,
        gamma,
        p0_pa,
        rho,
    ))
}

/// Minnaert resonance frequency with the surface-tension correction (Eq. 5.6).
///
/// f₀ = 1/(2πR₀)·√([3γP₀ + (3γ−1)·2σ/R₀]/ρ); reduces to `minnaert_resonance_hz`
/// as σ→0. Required for sub-micron bubbles (R₀ ≲ 1 µm), where it exceeds the
/// uncorrected value by >10%.
///
/// Args:
///     r0_m: Equilibrium bubble radius `m`.
///     gamma: Polytropic index.
///     p0_pa: Ambient pressure `Pa`.
///     rho: Liquid density [kg/m³].
///     sigma_n_m: Surface-tension coefficient [N/m] (water ≈ 0.0725).
///
/// Returns:
///     Resonance frequency `Hz` (0 if inputs are invalid or the bubble is
///     surface-tension destabilised).
#[pyfunction]
#[pyo3(signature = (r0_m, gamma, p0_pa, rho, sigma_n_m))]
pub fn minnaert_resonance_corrected_hz(
    r0_m: f64,
    gamma: f64,
    p0_pa: f64,
    rho: f64,
    sigma_n_m: f64,
) -> PyResult<f64> {
    Ok(cavitation::minnaert_resonance_corrected_hz(
        r0_m, gamma, p0_pa, rho, sigma_n_m,
    ))
}

/// Compute the Blake cavitation threshold pressure.
///
/// Args:
///     r0_m: Initial bubble radius `m`.
///     p0_pa: Ambient pressure `Pa`.
///     sigma_n_m: Surface tension [N/m].
///
/// Returns:
///     Blake threshold negative pressure `Pa`.
#[pyfunction]
#[pyo3(signature = (r0_m, p0_pa, sigma_n_m))]
pub fn blake_threshold_pa(r0_m: f64, p0_pa: f64, sigma_n_m: f64) -> PyResult<f64> {
    Ok(cavitation::blake_threshold_pa(r0_m, p0_pa, sigma_n_m))
}

/// Compute the Rayleigh collapse time of an empty spherical cavity.
///
/// t_c = 0.9147 * r_max * sqrt(rho / p_inf)
///
/// Args:
///     rmax_m: Maximum bubble radius `m`.
///     p_inf_pa: Ambient pressure `Pa`.
///     rho: Liquid density [kg/m³].
///
/// Returns:
///     Collapse time `s`.
#[pyfunction]
#[pyo3(signature = (rmax_m, p_inf_pa, rho))]
pub fn rayleigh_collapse_time_s(rmax_m: f64, p_inf_pa: f64, rho: f64) -> PyResult<f64> {
    Ok(cavitation::rayleigh_collapse_time_s(rmax_m, p_inf_pa, rho))
}
