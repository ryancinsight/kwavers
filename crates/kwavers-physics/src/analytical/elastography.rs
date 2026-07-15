//! Elastography and shear-wave physics for book chapter ch10.
//!
//! Covers: shear wave speed, Voigt–Kelvin and springpot (fractional)
//! viscoelastic complex moduli, Voigt dispersion relation, and a synthetic
//! 2-D MRE displacement field.

use eunomia::Complex64;
use kwavers_core::constants::numerical::TWO_PI;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
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

/// Ratio of compressional (P) to shear (S) wave speed as a function of Poisson's
/// ratio (book §11.2):
///
/// ```text
/// c_P/c_S = √( (λ + 2μ)/μ ) = √( 2(1 − ν)/(1 − 2ν) )
/// ```
///
/// using the isotropic identity `λ/μ = 2ν/(1 − 2ν)`. The ratio diverges as the
/// medium approaches incompressibility (`ν → ½⁻`), which is why soft tissue —
/// nearly incompressible (`ν ≈ 0.4999`) — has `c_P ≫ c_S` and SWE can treat the
/// P-wave background as instantaneous. Returns `f64::INFINITY` at `ν ≥ ½`.
///
/// # Reference
/// Achenbach (1973), *Wave Propagation in Elastic Solids*, §4.
#[must_use]
#[inline]
pub fn pwave_to_swave_velocity_ratio(poisson_ratio: f64) -> f64 {
    let denom = 1.0 - 2.0 * poisson_ratio;
    if denom <= 0.0 {
        return f64::INFINITY;
    }
    (2.0 * (1.0 - poisson_ratio) / denom).sqrt()
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

/// Exponential MRE displacement envelope `A · exp(-z / d_pen)`.
///
/// The returned vector has the same length as `z_arr` and contains the
/// positive envelope in meters. The caller plots both signs when showing the
/// symmetric displacement bound.
///
/// # Errors
/// Returns an error when any coordinate is non-finite, the amplitude is
/// non-finite, or the penetration depth is not positive and finite.
pub fn mre_displacement_envelope(
    z_arr: &[f64],
    amplitude_m: f64,
    penetration_depth_m: f64,
) -> Result<Vec<f64>, String> {
    if !amplitude_m.is_finite() {
        return Err("amplitude_m must be finite".to_owned());
    }
    if !penetration_depth_m.is_finite() || penetration_depth_m <= 0.0 {
        return Err("penetration_depth_m must be positive and finite".to_owned());
    }
    if z_arr.iter().any(|z| !z.is_finite()) {
        return Err("z_arr must contain only finite values".to_owned());
    }

    Ok(z_arr
        .iter()
        .map(|&z| amplitude_m * (-z / penetration_depth_m).exp())
        .collect())
}

/// Deterministic RF fixture for thermal-strain thermometry validation.
///
/// Generates `n_lines` independent axial RF lines, applies a sinusoidal carrier
/// to smoothed speckle reflectivity, then warps each post-heating line by the
/// apparent displacement `u(i) = k_T · ΔT · i` in sample coordinates.  The
/// returned pair is row-major with shape `n_lines × nz`; PyO3 reshapes it to
/// `[n_lines, 1, nz]` for [`crate::acoustics::imaging::modalities::elastography`].
///
/// # Errors
/// Returns an error if dimensions are zero, `samples_per_carrier` is not
/// positive and finite, or either thermal scalar is non-finite.
pub fn thermal_strain_rf_fixture(
    n_lines: usize,
    nz: usize,
    k_t: f64,
    delta_t_c: f64,
    samples_per_carrier: f64,
    seed: u64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if n_lines == 0 {
        return Err("n_lines must be positive".to_owned());
    }
    if nz < 2 {
        return Err("nz must be at least two".to_owned());
    }
    if !(k_t.is_finite() && delta_t_c.is_finite()) {
        return Err("k_t and delta_t_c must be finite".to_owned());
    }
    if !(samples_per_carrier.is_finite() && samples_per_carrier > 0.0) {
        return Err("samples_per_carrier must be positive and finite".to_owned());
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut reference = Vec::with_capacity(n_lines * nz);
    let mut tracked = Vec::with_capacity(n_lines * nz);
    let mut reflectivity = vec![0.0; nz];
    let mut smoothed = vec![0.0; nz];
    let mut rf_line = vec![0.0; nz];

    for _ in 0..n_lines {
        for value in &mut reflectivity {
            *value = rng.gen_range(-1.0..1.0);
        }
        for idx in 0..nz {
            let previous = if idx == 0 {
                reflectivity[nz - 1]
            } else {
                reflectivity[idx - 1]
            };
            smoothed[idx] = 0.5 * (reflectivity[idx] + previous);
        }
        for idx in 0..nz {
            let carrier = (TWO_PI * idx as f64 / samples_per_carrier).cos();
            rf_line[idx] = smoothed[idx] * carrier;
            reference.push(rf_line[idx]);
        }
        for idx in 0..nz {
            let src = (idx as f64 - k_t * delta_t_c * idx as f64).clamp(0.0, (nz - 1) as f64);
            let lo = src.floor() as usize;
            let hi = (lo + 1).min(nz - 1);
            let frac = src - lo as f64;
            tracked.push(rf_line[lo] * (1.0 - frac) + rf_line[hi] * frac);
        }
    }

    Ok((reference, tracked))
}

/// Acousto-elastic sensitivity coefficient `A = (m + n) / (2(λ + μ))` relating the
/// shear-modulus shift to applied uniaxial pre-stress (Murnaghan constants `m, n`;
/// Lamé `λ, μ`). Units: Pa·shift per Pa·stress (dimensionless slope of `ρc_S²` vs `σ₀`).
///
/// Elastography §11.9 (Hughes & Kelly 1953; Murnaghan 1951).
#[must_use]
pub fn acoustoelastic_sensitivity(m_const: f64, n_const: f64, lambda: f64, mu: f64) -> f64 {
    (m_const + n_const) / (2.0 * (lambda + mu))
}

/// Stress-dependent shear-wave speed under uniaxial pre-stress `σ₀`:
///
/// ```text
/// ρ c_S²(σ₀) = μ + A·σ₀,   A = (m+n)/(2(λ+μ));   c_S = √((μ + A σ₀)/ρ)
/// ```
///
/// First-order acousto-elastic relation (§11.9). Returns `0.0` if the effective
/// modulus `μ + A σ₀` is non-positive (pre-stress beyond the linear regime).
#[must_use]
pub fn acoustoelastic_shear_speed(
    mu: f64,
    lambda: f64,
    m_const: f64,
    n_const: f64,
    rho: f64,
    sigma0: f64,
) -> f64 {
    let a = acoustoelastic_sensitivity(m_const, n_const, lambda, mu);
    let eff = mu + a * sigma0;
    if eff <= 0.0 || rho <= 0.0 {
        return 0.0;
    }
    (eff / rho).sqrt()
}

/// Recover the uniaxial pre-stress `σ₀` from a shear-speed measurement relative to
/// an unstressed reference, by inverting the slope of `ρc_S²` vs `σ₀`:
///
/// ```text
/// σ₀ = ρ (c_S² − c_S0²) / A
/// ```
///
/// `a_sensitivity` is [`acoustoelastic_sensitivity`]. Returns `0.0` if `A` or `ρ`
/// is zero. (Algorithm "Pre-Stress Estimation", §11.9.)
#[must_use]
pub fn estimate_prestress(c_s: f64, c_s0: f64, rho: f64, a_sensitivity: f64) -> f64 {
    if a_sensitivity == 0.0 || rho == 0.0 {
        return 0.0;
    }
    rho * (c_s * c_s - c_s0 * c_s0) / a_sensitivity
}

/// Pre-stress estimate over a sequence of shear-speed measurements (e.g. a cardiac
/// cycle), against a fixed unstressed reference `c_s0`.
#[must_use]
pub fn estimate_prestress_sequence(
    c_s_seq: &[f64],
    c_s0: f64,
    rho: f64,
    a_sensitivity: f64,
) -> Vec<f64> {
    c_s_seq
        .iter()
        .map(|&c| estimate_prestress(c, c_s0, rho, a_sensitivity))
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::tissue_acoustics::DENSITY_BLOOD;

    #[test]
    fn acoustoelastic_zero_prestress_matches_elastic_speed() {
        let (mu, lambda, m, n, rho) = (3000.0, 2.0e9, -3000.0, -2000.0, 1060.0);
        let c0 = acoustoelastic_shear_speed(mu, lambda, m, n, rho, 0.0);
        assert!((c0 - shear_wave_speed(mu, rho)).abs() < 1e-9);
    }

    #[test]
    fn acoustoelastic_sensitivity_formula() {
        let a = acoustoelastic_sensitivity(-3000.0, -2000.0, 2.0e9, 3000.0);
        assert!((a - (-5000.0) / (2.0 * (2.0e9 + 3000.0))).abs() < 1e-18);
    }

    #[test]
    fn prestress_round_trip() {
        // softer skull/tissue-like Lamé to give a non-negligible sensitivity
        let (mu, lambda, m, n, rho) = (3000.0, 5000.0, -3.0e4, -2.0e4, 1060.0);
        let a = acoustoelastic_sensitivity(m, n, lambda, mu);
        assert!(a.abs() > 0.0);
        let c0 = acoustoelastic_shear_speed(mu, lambda, m, n, rho, 0.0);
        for &sigma0 in &[-500.0, 0.0, 300.0, 800.0] {
            let cs = acoustoelastic_shear_speed(mu, lambda, m, n, rho, sigma0);
            let recovered = estimate_prestress(cs, c0, rho, a);
            assert!(
                (recovered - sigma0).abs() < 1e-6,
                "σ₀ round-trip: in {sigma0}, out {recovered}"
            );
        }
    }

    #[test]
    fn prestress_sequence_recovers_each_frame() {
        let (mu, lambda, m, n, rho) = (3000.0, 5000.0, -3.0e4, -2.0e4, 1060.0);
        let a = acoustoelastic_sensitivity(m, n, lambda, mu);
        let c0 = acoustoelastic_shear_speed(mu, lambda, m, n, rho, 0.0);
        let truth = [0.0, 200.0, 400.0, 200.0, -100.0];
        let seq: Vec<f64> = truth
            .iter()
            .map(|&s| acoustoelastic_shear_speed(mu, lambda, m, n, rho, s))
            .collect();
        let est = estimate_prestress_sequence(&seq, c0, rho, a);
        for (e, t) in est.iter().zip(truth) {
            assert!((e - t).abs() < 1e-6, "frame σ₀ {e} vs {t}");
        }
    }

    #[test]
    fn shear_wave_speed_elastic() {
        // Liver-like: μ = 2000 Pa, ρ = 1060 kg/m³ → c_s ≈ 1.37 m/s
        let cs = shear_wave_speed(2000.0, DENSITY_BLOOD);
        assert!((cs - (2000.0_f64 / DENSITY_BLOOD).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn pwave_to_swave_ratio_matches_closed_form_and_diverges_at_incompressible() {
        // ν = 0 (no lateral coupling): c_P/c_S = √2.
        assert!((pwave_to_swave_velocity_ratio(0.0) - 2.0_f64.sqrt()).abs() < 1e-12);
        // ν = 1/3: √(2·(2/3)/(1/3)) = √4 = 2.
        assert!((pwave_to_swave_velocity_ratio(1.0 / 3.0) - 2.0).abs() < 1e-12);
        // Soft tissue ν → ½⁻: ratio is large (≈ √(2·0.5001/1e-4) ≈ 100).
        assert!(pwave_to_swave_velocity_ratio(0.49995) > 50.0);
        // ν ≥ ½ (incompressible / non-physical): diverges.
        assert!(pwave_to_swave_velocity_ratio(0.5).is_infinite());
        // Monotone increasing toward incompressibility.
        assert!(pwave_to_swave_velocity_ratio(0.45) > pwave_to_swave_velocity_ratio(0.30));
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

    #[test]
    fn mre_displacement_envelope_matches_decay_law() {
        let z = [0.0, 0.035, 0.070];
        let envelope =
            mre_displacement_envelope(&z, 25.0e-6, 0.035).expect("valid MRE envelope parameters");

        assert_eq!(envelope.len(), z.len());
        assert_eq!(envelope[0], 25.0e-6);
        assert!(
            (envelope[1] - 25.0e-6 / std::f64::consts::E).abs() <= 1.0e-20,
            "one-depth envelope={} expected {}",
            envelope[1],
            25.0e-6 / std::f64::consts::E
        );
        assert!(envelope[2] < envelope[1]);
    }

    #[test]
    fn mre_displacement_envelope_rejects_invalid_inputs() {
        assert!(mre_displacement_envelope(&[0.0], f64::NAN, 0.035).is_err());
        assert!(mre_displacement_envelope(&[0.0], 25.0e-6, 0.0).is_err());
        assert!(mre_displacement_envelope(&[f64::NAN], 25.0e-6, 0.035).is_err());
    }

    #[test]
    fn thermal_strain_rf_fixture_is_seeded_and_input_sensitive() {
        let (ref_a, tracked_a) = thermal_strain_rf_fixture(3, 32, -1.0e-3, 6.0, 5.0, 2024)
            .expect("invariant: valid fixture parameters generate RF");
        let (ref_b, tracked_b) = thermal_strain_rf_fixture(3, 32, -1.0e-3, 6.0, 5.0, 2024)
            .expect("invariant: valid fixture parameters generate RF");
        let (ref_c, _) = thermal_strain_rf_fixture(3, 32, -1.0e-3, 6.0, 5.0, 2025)
            .expect("invariant: valid fixture parameters generate RF");

        assert_eq!(ref_a.len(), 96);
        assert_eq!(tracked_a.len(), 96);
        assert_eq!(ref_a, ref_b);
        assert_eq!(tracked_a, tracked_b);
        assert_ne!(ref_a, ref_c);
        assert_ne!(ref_a, tracked_a);
    }

    #[test]
    fn thermal_strain_rf_fixture_zero_shift_preserves_reference() {
        let (reference, tracked) = thermal_strain_rf_fixture(2, 24, -1.0e-3, 0.0, 5.0, 7)
            .expect("invariant: zero temperature shift is valid");

        assert_eq!(reference, tracked);
    }

    #[test]
    fn thermal_strain_rf_fixture_rejects_invalid_inputs() {
        assert!(thermal_strain_rf_fixture(0, 24, -1.0e-3, 6.0, 5.0, 1).is_err());
        assert!(thermal_strain_rf_fixture(1, 1, -1.0e-3, 6.0, 5.0, 1).is_err());
        assert!(thermal_strain_rf_fixture(1, 24, f64::NAN, 6.0, 5.0, 1).is_err());
        assert!(thermal_strain_rf_fixture(1, 24, -1.0e-3, 6.0, 0.0, 1).is_err());
    }
}
