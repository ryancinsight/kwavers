//! Sonogenetics physics for book chapter ch18.
//!
//! Covers: Hill activation probability, acoustic radiation force,
//! acoustic streaming velocity, and ISPTA calculation.

// ─── Hill activation ──────────────────────────────────────────────────────────

/// Hill equation activation probability for mechanosensitive channels.
///
/// Models the probability of channel activation as a function of acoustic
/// pressure amplitude:
/// ```text
/// p(P) = P^n / (P^n + P_thresh^n)   ∈ [0, 1]
/// ```
///
/// # Arguments
/// * `pressure_arr` – acoustic pressure amplitudes [Pa]
/// * `p_threshold_pa` – half-activation threshold pressure [Pa]
/// * `hill_n` – Hill coefficient (cooperativity exponent)
///
/// # Reference
/// Ibsen et al. (2015), *Nat. Commun.* 6, 8264.
#[must_use]
pub fn hill_activation_probability(
    pressure_arr: &[f64],
    p_threshold_pa: f64,
    hill_n: f64,
) -> Vec<f64> {
    if !positive_finite(p_threshold_pa) || !positive_finite(hill_n) {
        return vec![0.0; pressure_arr.len()];
    }

    let pt_n = p_threshold_pa.powf(hill_n);
    if !positive_finite(pt_n) {
        return vec![0.0; pressure_arr.len()];
    }

    pressure_arr
        .iter()
        .map(|&p| {
            if !p.is_finite() {
                return 0.0;
            }

            let pn = p.abs().powf(hill_n);
            let denominator = pn + pt_n;
            if pn.is_finite() && positive_finite(denominator) {
                pn / denominator
            } else {
                0.0
            }
        })
        .collect()
}

// ─── Acoustic radiation force ─────────────────────────────────────────────────

/// Acoustic radiation force density from a travelling plane wave.
///
/// ```text
/// F = 2·α·I / c   [N/m³]
/// ```
/// where I = intensity [W/m²] at each spatial position, α is the absorption
/// coefficient [Np/m], and c is sound speed.
///
/// # Arguments
/// * `intensity_w_m2` – intensity profile [W/m²]
/// * `alpha_np_m` – absorption coefficient [Np/m]
/// * `c` – sound speed [m/s]
///
/// # Reference
/// Nyborg (1965), *Physical Acoustics* Vol. 2, ch. 11.
#[must_use]
#[inline]
pub fn radiation_force_1d(intensity_w_m2: &[f64], alpha_np_m: f64, c: f64) -> Vec<f64> {
    if !nonnegative_finite(alpha_np_m) || !positive_finite(c) {
        return vec![0.0; intensity_w_m2.len()];
    }

    let scale = 2.0 * alpha_np_m / c;
    intensity_w_m2
        .iter()
        .map(|&i| {
            if nonnegative_finite(i) {
                scale * i
            } else {
                0.0
            }
        })
        .collect()
}

// ─── Acoustic streaming ───────────────────────────────────────────────────────

/// Acoustic streaming (Eckart) velocity in an absorbing fluid column.
///
/// ```text
/// u_s = α·I·L² / (2·μ·c)   [m/s]
/// ```
/// This is the Eckart (1948) approximation for a cylindrical fluid column of
/// length L driven by a Gaussian beam with intensity I.
///
/// # Arguments
/// * `i_w_m2` – beam intensity [W/m²]
/// * `mu_pa_s` – dynamic viscosity [Pa·s]
/// * `alpha_np_m` – absorption coefficient [Np/m]
/// * `c` – sound speed [m/s]
/// * `l_m` – streaming path length [m]
///
/// # Reference
/// Eckart (1948), *Phys. Rev.* 73, 68.
#[must_use]
#[inline]
pub fn acoustic_streaming_velocity(
    i_w_m2: f64,
    mu_pa_s: f64,
    alpha_np_m: f64,
    c: f64,
    l_m: f64,
) -> f64 {
    if !nonnegative_finite(i_w_m2)
        || !positive_finite(mu_pa_s)
        || !nonnegative_finite(alpha_np_m)
        || !positive_finite(c)
        || !nonnegative_finite(l_m)
    {
        return 0.0;
    }

    alpha_np_m * i_w_m2 * l_m * l_m / (2.0 * mu_pa_s * c)
}

// ─── Safety metric ────────────────────────────────────────────────────────────

/// Spatial-peak time-average intensity (ISPTA) from a pressure waveform.
///
/// ```text
/// ISPTA = (1/T) · ∫ p²(t) dt / (ρ·c)   [W/m²]  →  converted to W/cm²
/// ```
/// Integrated by the rectangle rule.
///
/// # Arguments
/// * `p_pa` – pressure waveform at the spatial peak [Pa]
/// * `dt_s` – time step [s]
/// * `rho` – density [kg/m³]
/// * `c` – sound speed [m/s]
///
/// Returns ISPTA in W/cm².
///
/// # Reference
/// NCRP Report 74 (1983), §4.
#[must_use]
pub fn ispta_w_cm2(p_pa: &[f64], dt_s: f64, rho: f64, c: f64) -> f64 {
    if p_pa.is_empty() || !positive_finite(dt_s) || !positive_finite(rho) || !positive_finite(c) {
        return 0.0;
    }

    let n = p_pa.len() as f64;
    let total_time = n * dt_s;
    let integral: f64 = p_pa
        .iter()
        .map(|&p| if p.is_finite() { p * p * dt_s } else { 0.0 })
        .sum();
    let ispta_w_m2 = integral / (rho * c * total_time);
    if ispta_w_m2.is_finite() {
        ispta_w_m2 * 1e-4 // convert W/m² → W/cm²
    } else {
        0.0
    }
}

#[inline]
fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

#[inline]
fn nonnegative_finite(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};

    #[test]
    fn hill_at_threshold_is_half() {
        let p = hill_activation_probability(&[100.0], 100.0, 2.0);
        assert!((p[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn hill_zero_pressure_is_zero() {
        let p = hill_activation_probability(&[0.0], 100.0, 2.0);
        assert!((p[0]).abs() < 1e-15);
    }

    #[test]
    fn hill_saturates_at_high_pressure() {
        let p = hill_activation_probability(&[1e10], 100.0, 2.0);
        assert!((p[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn hill_rejects_invalid_domains_and_samples() {
        let invalid_threshold = hill_activation_probability(&[100.0], 0.0, 2.0);
        assert_eq!(invalid_threshold, vec![0.0]);

        let invalid_hill = hill_activation_probability(&[100.0], 100.0, 0.0);
        assert_eq!(invalid_hill, vec![0.0]);

        let nonfinite_pressure = hill_activation_probability(&[f64::NAN, 100.0], 100.0, 2.0);
        assert_eq!(nonfinite_pressure[0], 0.0);
        assert!((nonfinite_pressure[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn radiation_force_proportional_to_intensity() {
        let f = radiation_force_1d(&[1.0, 2.0, 4.0], 1.0, SOUND_SPEED_WATER_SIM);
        assert!((f[1] / f[0] - 2.0).abs() < 1e-10);
        assert!((f[2] / f[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn radiation_force_rejects_negative_or_nonfinite_domains() {
        assert_eq!(
            radiation_force_1d(&[1.0, 2.0], -1.0, SOUND_SPEED_WATER_SIM),
            vec![0.0, 0.0]
        );
        assert_eq!(radiation_force_1d(&[1.0], 1.0, 0.0), vec![0.0]);

        let f = radiation_force_1d(&[1.0, -2.0, f64::INFINITY], 1.0, SOUND_SPEED_WATER_SIM);
        assert!((f[0] - 2.0 / SOUND_SPEED_WATER_SIM).abs() < 1e-15);
        assert_eq!(f[1], 0.0);
        assert_eq!(f[2], 0.0);
    }

    #[test]
    fn streaming_velocity_positive() {
        let v = acoustic_streaming_velocity(
            DENSITY_WATER_NOMINAL,
            1e-3,
            0.5,
            SOUND_SPEED_WATER_SIM,
            0.05,
        );
        assert!(v > 0.0);
    }

    #[test]
    fn streaming_velocity_rejects_invalid_domains() {
        assert_eq!(
            acoustic_streaming_velocity(-1.0, 1e-3, 0.5, SOUND_SPEED_WATER_SIM, 0.05),
            0.0
        );
        assert_eq!(
            acoustic_streaming_velocity(1.0, 0.0, 0.5, SOUND_SPEED_WATER_SIM, 0.05),
            0.0
        );
        assert_eq!(
            acoustic_streaming_velocity(1.0, 1e-3, -0.5, SOUND_SPEED_WATER_SIM, 0.05),
            0.0
        );
        assert_eq!(acoustic_streaming_velocity(1.0, 1e-3, 0.5, 0.0, 0.05), 0.0);
        assert_eq!(
            acoustic_streaming_velocity(1.0, 1e-3, 0.5, SOUND_SPEED_WATER_SIM, f64::NAN),
            0.0
        );
    }

    #[test]
    fn ispta_constant_pressure() {
        // Constant pressure p0 → ISPTA = p0²/(rho*c) in W/m² → W/cm²
        let p0 = 1e5_f64;
        let rho = DENSITY_WATER_NOMINAL;
        let c = SOUND_SPEED_WATER_SIM;
        let n = 1000;
        let dt = 1e-7_f64;
        let p = vec![p0; n];
        let ispta = ispta_w_cm2(&p, dt, rho, c);
        let expected = p0 * p0 / (rho * c) * 1e-4;
        assert!(
            (ispta - expected).abs() / expected < 1e-10,
            "got={} expected={}",
            ispta,
            expected
        );
    }

    #[test]
    fn ispta_rejects_empty_invalid_and_nonfinite_domains() {
        assert_eq!(
            ispta_w_cm2(&[], 1e-7, DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM),
            0.0
        );
        assert_eq!(
            ispta_w_cm2(&[1.0], 0.0, DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM),
            0.0
        );
        assert_eq!(ispta_w_cm2(&[1.0], 1e-7, 0.0, SOUND_SPEED_WATER_SIM), 0.0);
        assert_eq!(
            ispta_w_cm2(&[1.0], 1e-7, DENSITY_WATER_NOMINAL, -SOUND_SPEED_WATER_SIM),
            0.0
        );

        let ispta = ispta_w_cm2(&[1.0, f64::NAN, 1.0], 1.0, 1.0, 1.0);
        assert!((ispta - (2.0 / 3.0) * 1e-4).abs() < 1e-16);
    }
}
