use std::f64::consts::PI;

use super::bessel::jn;

/// Evaluate the normalised amplitude of the nth harmonic at nonlinear parameter σ.
///
/// Fubini (1935) showed that for a lossless plane wave in the pre-shock
/// regime (σ < 1):
/// ```text
/// Bₙ(σ) = 2/(n·σ) · Jₙ(n·σ)
/// ```
/// where Jₙ is the Bessel function of the first kind of order n.
///
/// # Arguments
/// * `n` – harmonic number (n ≥ 1)
/// * `sigma` – Fubini–Euler parameter (0 ≤ σ < 1)
///
/// # Reference
/// Hamilton & Blackstock (1998) *Nonlinear Acoustics*, §3.3.
pub fn fubini_harmonic_amplitude(n: u32, sigma: f64) -> f64 {
    let n_f = n as f64;
    let x = n_f * sigma;
    if x.abs() < 1e-15 {
        return if n == 1 { 1.0 } else { 0.0 };
    }
    2.0 / x * jn(n, x)
}

/// Compute the Fubini harmonic spectrum for harmonics n = 1..=n_max at parameter σ.
///
/// Returns a `Vec<f64>` of length `n_max` where index 0 corresponds to n = 1.
pub fn fubini_harmonic_spectrum(n_max: u32, sigma: f64) -> Vec<f64> {
    (1..=n_max)
        .map(|n| fubini_harmonic_amplitude(n, sigma))
        .collect()
}

/// Shock-formation distance for a sinusoidal plane wave (Fubini–Euler criterion).
///
/// ```text
/// x_s = ρ₀·c₀³ / (β·p₀·ω)   [m]
/// ```
///
/// # Arguments
/// * `p0_pa` – source pressure amplitude [Pa]
/// * `f0_hz` – fundamental frequency [Hz]
/// * `c0` – small-signal sound speed [m/s]
/// * `rho0` – ambient density [kg/m³]
/// * `beta` – nonlinearity parameter β = 1 + B/(2A)
///
/// # Reference
/// Blackstock (1966), *J. Acoust. Soc. Am.* 39, 1019.
#[inline]
pub fn shock_formation_distance(p0_pa: f64, f0_hz: f64, c0: f64, rho0: f64, beta: f64) -> f64 {
    let omega = 2.0 * PI * f0_hz;
    rho0 * c0.powi(3) / (beta * p0_pa * omega)
}

/// Compute harmonic evolution along propagation axis using the Westervelt / KZK
/// plane-wave solution with linear absorption (perturbation theory, first-order
/// successive-approximation for n = 2 harmonics, exact Fubini for higher
/// harmonics scaled by exponential absorption).
///
/// For the nth harmonic:
/// ```text
/// pₙ(z) = p₀ · Bₙ(σ(z)) · exp(−n²·α·z)
/// ```
/// where σ(z) = z / x_s and α is the absorption at the fundamental.
///
/// # Arguments
/// * `z_arr` – propagation distances [m]
/// * `p0` – source pressure [Pa]
/// * `f0` – fundamental frequency [Hz]
/// * `c0` – sound speed [m/s]
/// * `rho0` – density [kg/m³]
/// * `beta` – nonlinearity parameter β
/// * `alpha_np_m` – attenuation at fundamental [Np/m]
/// * `n_max` – highest harmonic to compute
///
/// Returns a 2-D Vec of shape `[n_z][n_harmonic]` (n_harmonic = n_max).
///
/// # Reference
/// Hamilton & Blackstock (1998) *Nonlinear Acoustics*, ch. 4.
pub fn westervelt_harmonic_evolution(
    z_arr: &[f64],
    p0: f64,
    f0: f64,
    c0: f64,
    rho0: f64,
    beta: f64,
    alpha_np_m: f64,
    n_max: usize,
) -> Vec<Vec<f64>> {
    let omega = 2.0 * PI * f0;
    let x_s = rho0 * c0.powi(3) / (beta * p0 * omega);

    z_arr
        .iter()
        .map(|&z| {
            let sigma = (z / x_s).min(0.99);
            (1..=n_max)
                .map(|n| {
                    let b_n = fubini_harmonic_amplitude(n as u32, sigma);
                    // n-th harmonic is at n·f₀; for power-law α∝f², α_n = n²·α₁
                    // (Hamilton & Blackstock 1998 §4.3 eq. 4.3.9; Aanonsen et al. 1984 eq. 6)
                    let absorption = (-(n as f64).powi(2) * alpha_np_m * z).exp();
                    p0 * b_n * absorption
                })
                .collect()
        })
        .collect()
}
