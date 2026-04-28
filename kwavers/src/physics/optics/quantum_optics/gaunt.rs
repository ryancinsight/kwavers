//! Free-free Gaunt factors for thermal plasma emission.

use std::f64::consts::PI;

use super::constants::{H_PLANCK, KB};
use super::special::bessel_k0;

/// Frequency- and temperature-dependent free-free Gaunt factor.
///
/// # Formula
///
/// For non-degenerate hydrogen-like plasma in the thermally averaged
/// nonrelativistic Born limit,
///
/// ```text
/// g_ff(nu, T) = (sqrt(3) / pi) exp(u/2) K0(u/2)
/// u = h nu / (k_B T)
/// ```
///
/// The formula is positive for positive `nu` and `T` because `K0(x) > 0` and
/// `exp(x) > 0` for `x > 0`. It decreases with frequency at fixed temperature
/// over visible/UV SBSL conditions.
///
/// # Domain
///
/// Positive finite frequency and temperature are required. Invalid values
/// return `NaN`; no constant fallback is substituted.
#[must_use]
pub fn gaunt_factor_ff(freq_hz: f64, temperature_k: f64, _z: f64) -> f64 {
    if !(freq_hz.is_finite() && temperature_k.is_finite() && freq_hz > 0.0 && temperature_k > 0.0) {
        return f64::NAN;
    }

    let u = H_PLANCK * freq_hz / (KB * temperature_k);
    let x = 0.5 * u;
    (3.0_f64.sqrt() / PI) * x.exp() * bessel_k0(x)
}
