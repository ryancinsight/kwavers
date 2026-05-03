//! Analytical prediction functions for SBSL benchmarks.

use super::constants::{C_LIGHT, H_PLANCK, KB, WIEN_CONST};
use std::f64::consts::PI;

/// Compute the Minnaert resonance radius for a given driving frequency.
///
/// ## Algorithm (Minnaert 1933)
///
/// The natural frequency of a spherical bubble oscillating in a liquid is:
/// ```text
/// f₀ = (1 / 2πR₀) · √(3γ p₀ / ρ_L)
/// ```
/// Solving for the equilibrium radius at resonance:
/// ```text
/// R₀_res = (1 / 2π f₀) · √(3γ p₀ / ρ_L)
/// ```
///
/// ## Arguments
/// * `freq_hz` — driving frequency [Hz]
/// * `gamma`   — polytropic index
/// * `p0`      — ambient pressure [Pa]
/// * `rho_l`   — liquid density [kg/m³]
#[must_use]
pub fn minnaert_resonance_radius(freq_hz: f64, gamma: f64, p0: f64, rho_l: f64) -> f64 {
    if freq_hz < 1.0 || rho_l < 1.0 || p0 < 1.0 {
        return 0.0;
    }
    (1.0 / (2.0 * PI * freq_hz)) * (3.0 * gamma * p0 / rho_l).sqrt()
}

/// Compute the Blake threshold pressure for inertial cavitation nucleation.
///
/// ## Algorithm (Apfel 1981)
///
/// A stable bubble at equilibrium radius R₀ nucleates inertial cavitation
/// when the acoustic pressure exceeds the Blake threshold:
/// ```text
/// p_B = p_v − (4σ / 3R₀) · √(3 p₀ R₀ / (2σ))
/// ```
///
/// The threshold is negative (tensile) and represents the critical
/// underpressure at which a stable bubble begins inertial growth.
///
/// ## Arguments
/// * `p0`    — ambient pressure [Pa]
/// * `p_v`   — vapour pressure [Pa]
/// * `r0`    — equilibrium bubble radius [m]
/// * `sigma` — surface tension [N/m]
#[must_use]
pub fn blake_threshold(p0: f64, p_v: f64, r0: f64, sigma: f64) -> f64 {
    if r0 < 1e-15 || sigma < 1e-15 {
        return p_v - p0;
    }
    let factor = (3.0 * p0 * r0 / (2.0 * sigma)).sqrt();
    p_v - (4.0 * sigma / (3.0 * r0)) * factor
}

/// Estimate the Rayleigh collapse time fraction of the acoustic period.
///
/// ## Algorithm (Rayleigh 1917)
///
/// The collapse time for a void bubble from maximum radius R_max:
/// ```text
/// t_c = 0.9147 · R_max · √(ρ_L / p_collapse)
/// ```
/// Normalised by acoustic period T = 1/f:
/// ```text
/// t_c / T = 0.9147 · f · R_max · √(ρ_L / p_∞)
/// ```
///
/// For SBSL with R_max ≈ 8 R₀:  `t_c/T ≈ 0.5–1 %` (Brenner 2002 §III).
///
/// ## Arguments
/// * `r_max`   — maximum bubble radius [m]
/// * `freq_hz` — driving frequency [Hz]
/// * `p0`      — ambient pressure [Pa]
/// * `rho_l`   — liquid density [kg/m³]
#[must_use]
pub fn collapse_time_fraction(r_max: f64, freq_hz: f64, p0: f64, rho_l: f64) -> f64 {
    if p0 < 1.0 || rho_l < 1.0 || freq_hz < 1.0 {
        return 0.0;
    }
    let t_c = 0.9147 * r_max * (rho_l / p0).sqrt();
    t_c * freq_hz
}

/// Compute Wien's displacement law peak emission wavelength.
///
/// ## Algorithm (Wien 1893)
///
/// At temperature T, the blackbody spectrum peaks at:
/// ```text
/// λ_max = b / T,     b = 2.897771955 × 10⁻³ m·K
/// ```
///
/// For T = 10,000 K: λ_max = 290 nm (Brenner 2002 §IV.C).
/// Putterman & Weninger (2000) observe UV peak at ~310 nm, consistent
/// with T ≈ 9,000–10,000 K (accounting for liquid absorption).
///
/// ## Arguments
/// * `temperature_k` — blackbody temperature [K]
#[must_use]
pub fn wien_peak_wavelength_m(temperature_k: f64) -> f64 {
    if temperature_k < 1.0 {
        return f64::INFINITY;
    }
    WIEN_CONST / temperature_k
}

/// Compute relative Planck spectral radiance at wavelength λ for temperature T.
///
/// ## Algorithm (Planck 1900)
///
/// ```text
/// B(λ, T) = (2hc² / λ⁵) × 1 / (exp(hc/(λkT)) − 1)
/// ```
///
/// Returns the radiance normalised by the peak value at this temperature.
///
/// ## Arguments
/// * `wavelength_m` — wavelength [m]
/// * `temperature_k` — blackbody temperature [K]
#[must_use]
pub fn planck_radiance_relative(wavelength_m: f64, temperature_k: f64) -> f64 {
    if wavelength_m < 1e-12 || temperature_k < 1.0 {
        return 0.0;
    }
    let x = H_PLANCK * C_LIGHT / (wavelength_m * KB * temperature_k);
    if x > 700.0 {
        return 0.0; // underflow guard: exp(x) >> 1
    }
    let prefactor = 2.0 * H_PLANCK * C_LIGHT * C_LIGHT / wavelength_m.powi(5);
    let bose = (x.exp() - 1.0).recip();
    // normalise by peak value using Wien's law
    let lambda_peak = wien_peak_wavelength_m(temperature_k);
    let x_peak = H_PLANCK * C_LIGHT / (lambda_peak * KB * temperature_k);
    let bose_peak = (x_peak.exp() - 1.0).recip();
    let prefactor_peak = 2.0 * H_PLANCK * C_LIGHT * C_LIGHT / lambda_peak.powi(5);
    let b = prefactor * bose;
    let b_peak = prefactor_peak * bose_peak;
    b / b_peak
}

/// Yasui (1997) emission intensity scaling exponent.
///
/// ## Algorithm (Yasui 1997, §III)
///
/// The time-integrated SBSL light intensity scales approximately as:
/// ```text
/// I_emit ∝ T_max^n,     n ≈ 8–12
/// ```
/// Estimated via the integrated Planck spectrum in the visible/UV window
/// 200–700 nm, which is extremely sensitive to T_max.
///
/// Returns the ratio `I(T1) / I(T2)` for two maximum temperatures.
///
/// ## Arguments
/// * `t1_k`, `t2_k` — maximum collapse temperatures [K]
#[must_use]
pub fn yasui_intensity_ratio(t1_k: f64, t2_k: f64) -> f64 {
    // Integrate Planck spectrum from 200 nm to 700 nm at each temperature.
    let integrate_visible = |temp: f64| -> f64 {
        let n_pts = 500;
        let lam_min = 200e-9;
        let lam_max = 700e-9;
        let dlam = (lam_max - lam_min) / n_pts as f64;
        let mut integral = 0.0;
        for k in 0..n_pts {
            let lam = lam_min + (k as f64 + 0.5) * dlam;
            let x = H_PLANCK * C_LIGHT / (lam * KB * temp);
            if x < 700.0 {
                let b = 2.0 * H_PLANCK * C_LIGHT * C_LIGHT / lam.powi(5) / (x.exp() - 1.0);
                integral += b * dlam;
            }
        }
        integral
    };
    let i1 = integrate_visible(t1_k);
    let i2 = integrate_visible(t2_k);
    if i2 < f64::EPSILON {
        return f64::INFINITY;
    }
    i1 / i2
}
