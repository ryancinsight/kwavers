//! Derived constants for bremsstrahlung and Saha kinetics.

use std::f64::consts::PI;

use crate::core::constants::fundamental::{
    BOLTZMANN as BOLTZMANN_CONSTANT, ELECTRON_MASS, ELEMENTARY_CHARGE as ELECTRON_CHARGE,
    PLANCK as PLANCK_CONSTANT, SPEED_OF_LIGHT, VACUUM_PERMITTIVITY,
};

/// Pre-factor constant `C_ff` for the free-free emission coefficient.
///
/// # Theorem
///
/// In SI units per steradian,
///
/// ```text
/// C_ff = 32 pi e^6 / (3 m_e c^3 (4 pi eps_0)^3)
///        * sqrt(2 pi / (3 k m_e)).
/// ```
///
/// Proof: combine the Coulomb acceleration radiation prefactor with the
/// Maxwellian thermal velocity integral. The mass appears once from the
/// Coulomb cross-section and once under the square root from the velocity
/// average, giving the required `m_e^(-3/2)` scaling.
pub(super) fn c_ff_per_sr() -> f64 {
    let e6 = ELECTRON_CHARGE.powi(6);
    let four_pi_eps0 = 4.0 * PI * VACUUM_PERMITTIVITY;
    let prefactor =
        32.0 * PI * e6 / (3.0 * ELECTRON_MASS * SPEED_OF_LIGHT.powi(3) * four_pi_eps0.powi(3));
    let thermal_coeff = (2.0 * PI / (3.0 * BOLTZMANN_CONSTANT * ELECTRON_MASS)).sqrt();
    prefactor * thermal_coeff
}

/// Saha base constant `(2 pi m_e k / h^2)^(3/2)`.
pub(super) fn saha_k0() -> f64 {
    (2.0 * PI * ELECTRON_MASS * BOLTZMANN_CONSTANT / PLANCK_CONSTANT.powi(2)).powf(1.5)
}
