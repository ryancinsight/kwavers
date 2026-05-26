//! Derived constants for bremsstrahlung and Saha kinetics.

use std::f64::consts::PI;

use crate::core::constants::fundamental::{
    BOLTZMANN as BOLTZMANN_CONSTANT, ELECTRON_MASS, ELEMENTARY_CHARGE as ELECTRON_CHARGE,
    PLANCK as PLANCK_CONSTANT, SPEED_OF_LIGHT, VACUUM_PERMITTIVITY,
};
use crate::core::constants::numerical::{FOUR_PI, TWO_PI};

/// Pre-factor constant `C_ff` for the free-free emission coefficient **per steradian**.
///
/// # Theorem
///
/// The total (4π-integrated) free-free emissivity in SI units is
/// (Rybicki & Lightman 1979, eq. 5.14b):
///
/// ```text
/// eps_nu = 32 pi e^6 / (3 m_e c^3 (4 pi eps_0)^3)
///          * sqrt(2 pi / (3 k m_e))
///          * Z^2 n_e n_i T^{-1/2} g_ff exp(-h nu / k T)   [W m^{-3} Hz^{-1}]
/// ```
///
/// For an isotropic source the per-steradian emission coefficient is:
/// ```text
/// j_nu = eps_nu / (4 pi)
/// C_ff_per_sr = eps_nu_coefficient / (4 pi)
///             = 8 e^6 / (3 m_e c^3 (4 pi eps_0)^3)
///               * sqrt(2 pi / (3 k m_e))                   [W m^3 Hz^{-1} sr^{-1} K^{1/2}]
/// ```
///
/// Numerical check: `C_ff_per_sr ≈ 5.44e-52`;
/// `C_ff_per_sr * 4pi ≈ 6.84e-51` matches R&L tabulated 6.8e-51 [W m^3 Hz^{-1} K^{1/2}].
pub(super) fn c_ff_per_sr() -> f64 {
    let e6 = ELECTRON_CHARGE.powi(6);
    let four_pi_eps0 = FOUR_PI * VACUUM_PERMITTIVITY;
    // 32π e^6 / (3 m_e c^3 (4πε₀)^3) — total emissivity prefactor (Rybicki & Lightman 5.14b)
    let prefactor =
        32.0 * PI * e6 / (3.0 * ELECTRON_MASS * SPEED_OF_LIGHT.powi(3) * four_pi_eps0.powi(3));
    let thermal_coeff = (TWO_PI / (3.0 * BOLTZMANN_CONSTANT * ELECTRON_MASS)).sqrt();
    // Divide by 4π to convert total emissivity constant → per-steradian emission coefficient
    (prefactor * thermal_coeff) / (FOUR_PI)
}

/// Saha base constant `(2 pi m_e k / h^2)^(3/2)`.
pub(super) fn saha_k0() -> f64 {
    (TWO_PI * ELECTRON_MASS * BOLTZMANN_CONSTANT / PLANCK_CONSTANT.powi(2)).powf(1.5)
}
