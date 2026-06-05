//! Free-free Gaunt factor model.

use std::f64::consts::PI;

use kwavers_core::constants::fundamental::{
    BOLTZMANN as BOLTZMANN_CONSTANT, PLANCK as PLANCK_CONSTANT,
};

/// Thermally averaged free-free Gaunt factor `g_ff(nu, T)`.
///
/// # Theorem
///
/// For optical and ultraviolet photons in a hot plasma, the Elwert Coulomb-log
/// approximation is
///
/// ```text
/// g_ff = sqrt(3)/pi * ln(2 kT / h nu),      h nu < kT.
/// ```
///
/// For `h nu >= kT`, the hard-photon limit tends to one. The implementation
/// clamps the optical branch to `[1, 10]`, preserving positivity and bounded
/// numerical behavior while retaining the analytic lower hard-photon limit.
#[must_use]
pub fn gaunt_factor_thermal(frequency: f64, temperature: f64) -> f64 {
    if frequency <= 0.0 || temperature <= 0.0 {
        return 1.0;
    }

    let h_nu = PLANCK_CONSTANT * frequency;
    let k_t = BOLTZMANN_CONSTANT * temperature;
    if h_nu >= k_t {
        1.0
    } else {
        let g = (3.0_f64).sqrt() / PI * (2.0 * k_t / h_nu).ln();
        g.clamp(1.0, 10.0)
    }
}
