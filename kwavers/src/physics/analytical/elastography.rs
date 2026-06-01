//! Elastography and shear-wave physics for book chapter ch10.
//!
//! Covers: shear wave speed, Voigt–Kelvin and springpot (fractional)
//! viscoelastic complex moduli, Voigt dispersion relation, and a synthetic
//! 2-D MRE displacement field.

use crate::core::constants::numerical::TWO_PI;
use num_complex::Complex64;
use std::f64::consts::PI;

// ─── Shear wave speed ─────────────────────────────────────────────────────────

/// Shear wave propagation speed in a purely elastic medium.
///
/// ```text
/// c_s = √(μ/ρ)   [m/s]
/// ```
///
/// # Arguments
/// * `mu_pa` – shear modulus μ [Pa]
/// * `rho` – tissue density [kg/m³]
///
/// # Reference
/// Sarvazyan et al. (1998), *Ultrasound Med. Biol.* 24, 1419.
#[must_use]
#[inline]
pub fn shear_wave_speed(mu_pa: f64, rho: f64) -> f64 {
    (mu_pa / rho).sqrt()
}

// ─── Viscoelastic models ──────────────────────────────────────────────────────

/// Voigt–Kelvin complex shear modulus.
///
/// ```text
/// G*(ω) = μ + iω·η   [Pa]
/// ```
///
/// # Arguments
/// * `omega_arr` – angular frequencies [rad/s]
/// * `mu_pa` – elastic shear modulus μ [Pa]
/// * `eta_pa_s` – viscosity η [Pa·s]
///
/// # Reference
/// Lakes (2009) *Viscoelastic Materials*, ch. 2.
#[must_use]
pub fn voigt_complex_modulus(omega_arr: &[f64], mu_pa: f64, eta_pa_s: f64) -> Vec<Complex64> {
    omega_arr
        .iter()
        .map(|&w| Complex64::new(mu_pa, w * eta_pa_s))
        .collect()
}

/// Springpot (fractional Kelvin) complex shear modulus.
///
/// Interpolates between elastic (α = 0) and viscous (α = 1) behaviour:
/// ```text
/// G*(ω) = G₀·(iω)^α   [Pa]
/// ```
///
/// # Arguments
/// * `omega_arr` – angular frequencies [rad/s]
/// * `g0` – scale modulus G₀ [Pa]
/// * `alpha_exp` – power-law exponent α ∈ (0, 1)
///
/// # Reference
/// Koeller (1984), *J. Appl. Mech.* 51, 299.
#[must_use]
pub fn springpot_complex_modulus(omega_arr: &[f64], g0: f64, alpha_exp: f64) -> Vec<Complex64> {
    // (iω)^α = |ω|^α · exp(i·α·π/2)
    let phase = alpha_exp * PI / 2.0;
    let cos_ph = phase.cos();
    let sin_ph = phase.sin();
    omega_arr
        .iter()
        .map(|&w| {
            let mag = g0 * w.abs().powf(alpha_exp);
            Complex64::new(mag * cos_ph, mag * sin_ph)
        })
        .collect()
}

// ─── Voigt dispersion ─────────────────────────────────────────────────────────

/// Phase speed of shear waves in a Voigt–Kelvin medium vs frequency.
///
/// Derived from `c_s(ω) = √(2·(μ² + ω²η²) / (ρ·(μ + √(μ² + ω²η²))))`.
///
/// # Arguments
/// * `f_arr` – frequencies [Hz]
/// * `mu_pa` – elastic shear modulus [Pa]
/// * `eta_pa_s` – viscosity [Pa·s]
/// * `rho` – density [kg/m³]
///
/// # Reference
/// Catheline et al. (2004), *J. Acoust. Soc. Am.* 116, 3736.
#[must_use]
pub fn voigt_shear_wave_dispersion(f_arr: &[f64], mu_pa: f64, eta_pa_s: f64, rho: f64) -> Vec<f64> {
    f_arr
        .iter()
        .map(|&f| {
            let omega = TWO_PI * f;
            let mu2 = mu_pa * mu_pa;
            let ve2 = omega * omega * eta_pa_s * eta_pa_s;
            let norm = (mu2 + ve2).sqrt();
            (2.0 * norm * norm / (rho * (mu_pa + norm))).sqrt()
        })
        .collect()
}

// ─── 2-D MRE displacement field ───────────────────────────────────────────────

/// Synthetic 2-D MRE displacement field with exponential depth attenuation.
///
/// Models a plane shear wave propagating along z with sinusoidal variation
/// and exponential penetration decay:
/// ```text
/// u(x, z) = A · sin(k_s·z) · exp(−z/d_pen)
/// ```
/// where `k_s = 2π·f / c_s`. The field is uniform in x (1-D shear wave).
///
/// Output is a flattened row-major Vec of size `NX × NZ`.
///
/// # Arguments
/// * `x_arr` – lateral positions [m]
/// * `z_arr` – depth positions [m]
/// * `shear_speed` – c_s [m/s]
/// * `freq_hz` – vibration frequency [Hz]
/// * `amplitude` – displacement amplitude [m]
/// * `penetration_depth_m` – 1/e decay depth d_pen [m]
///
/// # Reference
/// Muthupillai et al. (1995), *Science* 269, 1854.
#[must_use]
pub fn mre_displacement_field(
    x_arr: &[f64],
    z_arr: &[f64],
    shear_speed: f64,
    freq_hz: f64,
    amplitude: f64,
    penetration_depth_m: f64,
) -> Vec<f64> {
    let k_s = TWO_PI * freq_hz / shear_speed;
    let nx = x_arr.len();
    let nz = z_arr.len();
    let mut out = vec![0.0_f64; nx * nz];
    for (ix, _x) in x_arr.iter().enumerate() {
        for (iz, &z) in z_arr.iter().enumerate() {
            let u = amplitude * (k_s * z).sin() * (-z / penetration_depth_m).exp();
            out[ix * nz + iz] = u;
        }
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::tissue_acoustics::DENSITY_BLOOD;

    #[test]
    fn shear_wave_speed_elastic() {
        // Liver-like: μ = 2000 Pa, ρ = 1060 kg/m³ → c_s ≈ 1.37 m/s
        let cs = shear_wave_speed(2000.0, DENSITY_BLOOD);
        assert!((cs - (2000.0_f64 / DENSITY_BLOOD).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn voigt_at_zero_freq_is_elastic() {
        let g = voigt_complex_modulus(&[0.0], 1000.0, 1.0);
        assert!((g[0].re - 1000.0).abs() < 1e-10);
        assert!((g[0].im).abs() < 1e-10);
    }

    #[test]
    fn springpot_pure_elastic_at_alpha0() {
        // α = 0 → G*(ω) = G₀·(iω)⁰ = G₀  (real, independent of ω)
        let g = springpot_complex_modulus(&[1.0, 100.0], 500.0, 0.0);
        for gi in &g {
            assert!((gi.re - 500.0).abs() < 1e-8, "re={}", gi.re);
            assert!(gi.im.abs() < 1e-8, "im={}", gi.im);
        }
    }

    #[test]
    fn voigt_dispersion_increases_with_frequency() {
        let f = vec![10.0, 100.0, 1000.0];
        let cs = voigt_shear_wave_dispersion(&f, 2000.0, 5.0, DENSITY_BLOOD);
        assert!(cs[0] <= cs[1] && cs[1] <= cs[2]);
    }

    #[test]
    fn mre_field_size() {
        let x = vec![0.0, 0.01, 0.02];
        let z = vec![0.0, 0.005, 0.01, 0.015];
        let u = mre_displacement_field(&x, &z, 1.5, 100.0, 1e-5, 0.02);
        assert_eq!(u.len(), 12);
    }
}
