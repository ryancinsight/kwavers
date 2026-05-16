//! Skull and transcranial ultrasound physics for book chapters ch16, ch25.
//!
//! Covers: two-way skull insertion loss, random Gaussian phase screens,
//! Hounsfield unit conversions (Schneider 1996), Strehl ratio, semi-infinite
//! solid surface temperature rise, and transfer-matrix skull transmission.

use num_complex::Complex64;
use std::f64::consts::PI;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use rand_chacha::ChaCha8Rng;

// ─── Insertion loss ───────────────────────────────────────────────────────────

/// Two-way skull insertion loss [dB] using power-law attenuation.
///
/// ```text
/// IL(f) = α₀ · f_MHz^1.2 · 2 · d_cm   [dB]
/// ```
///
/// # Arguments
/// * `f_mhz` – frequencies [MHz]
/// * `thickness_cm` – skull thickness [cm]
/// * `alpha0` – skull attenuation prefactor [dB/(cm·MHz^1.2)]
///
/// # Reference
/// Deffieux & Konofagou (2010), *Ultrasound Med. Biol.* 36, 1718.
pub fn skull_insertion_loss_two_way_db(
    f_mhz: &[f64],
    thickness_cm: f64,
    alpha0: f64,
) -> Vec<f64> {
    f_mhz
        .iter()
        .map(|&f| alpha0 * f.powf(1.2) * 2.0 * thickness_cm)
        .collect()
}

// ─── Phase screen ─────────────────────────────────────────────────────────────

/// Generate a reproducible random Gaussian phase screen.
///
/// Phase samples are drawn from N(0, σ²_φ) and represent the aberration
/// phase introduced by skull heterogeneity at each transducer element.
///
/// # Arguments
/// * `n` – number of samples
/// * `sigma_phi_rad` – standard deviation of phase aberration [rad]
/// * `seed` – 64-bit seed for ChaCha8Rng
///
/// # Reference
/// Tanter et al. (1998), *J. Acoust. Soc. Am.* 103, 2403.
pub fn skull_phase_screen(n: usize, sigma_phi_rad: f64, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dist = Normal::new(0.0, sigma_phi_rad).expect("sigma must be finite and positive");
    (0..n).map(|_| dist.sample(&mut rng)).collect()
}

// ─── CT conversions ───────────────────────────────────────────────────────────

/// Convert Hounsfield Units to acoustic sound speed (Schneider 1996).
///
/// ```text
/// c(HU) = 1500 + 0.50·HU   for HU < 0
/// c(HU) = 1500 + 0.76·HU   for HU ≥ 0   [m/s]
/// ```
///
/// # Reference
/// Schneider et al. (1996), *Phys. Med. Biol.* 41, 111.
pub fn hu_to_sound_speed_schneider(hu: &[f64]) -> Vec<f64> {
    hu.iter()
        .map(|&h| if h < 0.0 { 1500.0 + 0.50 * h } else { 1500.0 + 0.76 * h })
        .collect()
}

/// Convert Hounsfield Units to density (Schneider 1996).
///
/// ```text
/// ρ(HU) = 1000 + 0.96·HU   [kg/m³]
/// ```
///
/// # Reference
/// Schneider et al. (1996), *Phys. Med. Biol.* 41, 111.
pub fn hu_to_density_schneider(hu: &[f64]) -> Vec<f64> {
    hu.iter().map(|&h| 1000.0 + 0.96 * h).collect()
}

// ─── Strehl ratio ─────────────────────────────────────────────────────────────

/// Strehl ratio for a random-phase aberrator.
///
/// ```text
/// S = exp(−σ²_φ)   (Maréchal 1947)
/// ```
///
/// # Arguments
/// * `sigma_phi_rad` – rms phase aberration [rad]
///
/// # Reference
/// Maréchal (1947), *Rev. Opt.* 26, 257.
#[inline]
pub fn strehl_ratio(sigma_phi_rad: f64) -> f64 {
    (-sigma_phi_rad * sigma_phi_rad).exp()
}

// ─── Surface temperature rise ─────────────────────────────────────────────────

/// Temperature rise at the surface of a semi-infinite solid under constant
/// heat flux (Green's function solution, step heating).
///
/// ```text
/// ΔT(t) = 2·q″ / k · √(α_th·t / π)
/// ```
/// where `α_th = k / (ρ·cₚ)` is the thermal diffusivity [m²/s].
///
/// # Arguments
/// * `t_arr` – times [s] (t > 0)
/// * `heat_flux_w_m2` – surface heat flux q″ [W/m²]
/// * `k_skull` – thermal conductivity k [W/(m·K)]
/// * `rho_skull` – skull density [kg/m³]
/// * `cp_skull` – specific heat capacity [J/(kg·K)]
///
/// # Reference
/// Carslaw & Jaeger (1959) *Conduction of Heat in Solids*, §2.6.
pub fn skull_surface_temperature_rise(
    t_arr: &[f64],
    heat_flux_w_m2: f64,
    k_skull: f64,
    rho_skull: f64,
    cp_skull: f64,
) -> Vec<f64> {
    let alpha_th = k_skull / (rho_skull * cp_skull);
    t_arr
        .iter()
        .map(|&t| {
            if t <= 0.0 {
                return 0.0;
            }
            2.0 * heat_flux_w_m2 / k_skull * (alpha_th * t / PI).sqrt()
        })
        .collect()
}

// ─── Transfer matrix ──────────────────────────────────────────────────────────

/// Scalar skull transmission coefficient via the transfer-matrix method.
///
/// Models the skull as a homogeneous layer of impedance Z₂ and sound speed
/// c₂ between water (Z₁) and brain (Z₃):
/// ```text
/// T = 2 / [(1 + Z₃/Z₁)·cos(k₂·d) + i·(Z₃/Z₂ + Z₂/Z₁)·sin(k₂·d)]
/// ```
///
/// # Arguments
/// * `f_hz` – frequency [Hz]
/// * `z_water` – acoustic impedance of water [Pa·s/m]
/// * `z_skull` – acoustic impedance of skull [Pa·s/m]
/// * `z_brain` – acoustic impedance of brain [Pa·s/m]
/// * `c_skull` – sound speed in skull [m/s]
/// * `d_skull_m` – skull thickness [m]
///
/// # Reference
/// Brekhovskikh (1980) *Waves in Layered Media*, §1.5.
pub fn skull_transfer_matrix_transmission(
    f_hz: f64,
    z_water: f64,
    z_skull: f64,
    z_brain: f64,
    c_skull: f64,
    d_skull_m: f64,
) -> Complex64 {
    let k2 = 2.0 * PI * f_hz / c_skull;
    let phase = k2 * d_skull_m;
    let cos_ph = phase.cos();
    let sin_ph = phase.sin();
    let a = (1.0 + z_brain / z_water) * cos_ph;
    let b = z_brain / z_skull + z_skull / z_water;
    let denom = Complex64::new(a, b * sin_ph);
    Complex64::new(2.0, 0.0) / denom
}

/// Compute |T| and ∠T for an array of frequencies via the transfer matrix.
///
/// Returns `(magnitude_arr, phase_rad_arr)` parallel to `f_hz`.
pub fn skull_transmission_spectrum(
    f_hz: &[f64],
    z_water: f64,
    z_skull: f64,
    z_brain: f64,
    c_skull: f64,
    d_skull_m: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut mags = Vec::with_capacity(f_hz.len());
    let mut phases = Vec::with_capacity(f_hz.len());
    for &f in f_hz {
        let t = skull_transfer_matrix_transmission(f, z_water, z_skull, z_brain, c_skull, d_skull_m);
        mags.push(t.norm());
        phases.push(t.arg());
    }
    (mags, phases)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insertion_loss_positive() {
        let il = skull_insertion_loss_two_way_db(&[1.0, 2.0], 0.7, 13.0);
        assert!(il.iter().all(|&v| v > 0.0));
        assert!(il[1] > il[0]); // higher frequency → more loss
    }

    #[test]
    fn phase_screen_reproducible() {
        let a = skull_phase_screen(100, 1.0, 42);
        let b = skull_phase_screen(100, 1.0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn phase_screen_mean_approx_zero() {
        let s = skull_phase_screen(10000, 1.0, 7);
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        assert!(mean.abs() < 0.05, "mean={}", mean);
    }

    #[test]
    fn hu_water_is_1500() {
        let c = hu_to_sound_speed_schneider(&[0.0]);
        assert!((c[0] - 1500.0).abs() < 1e-10);
    }

    #[test]
    fn strehl_zero_aberration() {
        assert!((strehl_ratio(0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn strehl_monotone_decreasing() {
        let s0 = strehl_ratio(0.1);
        let s1 = strehl_ratio(0.5);
        let s2 = strehl_ratio(1.0);
        assert!(s0 > s1 && s1 > s2 && s2 > 0.0);
    }

    #[test]
    fn transfer_matrix_at_matching_impedance() {
        // When Z_skull = Z_water = Z_brain: T → 1 regardless of frequency
        let t = skull_transfer_matrix_transmission(1e6, 1.5e6, 1.5e6, 1.5e6, 2900.0, 7e-3);
        // With equal impedances, the 1+Z3/Z1 = 2 and Z3/Z2+Z2/Z1 = 2,
        // so T = 2/(2·cos + i·2·sin) = 1/(cos+i·sin) = exp(-iφ), |T| = 1
        assert!((t.norm() - 1.0).abs() < 1e-8, "|T|={}", t.norm());
    }

    #[test]
    fn surface_temperature_zero_at_t0() {
        let dt = skull_surface_temperature_rise(&[0.0], 1000.0, 0.5, 1900.0, 1300.0);
        assert!((dt[0]).abs() < 1e-15);
    }
}
