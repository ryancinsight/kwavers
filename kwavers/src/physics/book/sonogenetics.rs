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
pub fn hill_activation_probability(
    pressure_arr: &[f64],
    p_threshold_pa: f64,
    hill_n: f64,
) -> Vec<f64> {
    pressure_arr
        .iter()
        .map(|&p| {
            let pn = p.abs().powf(hill_n);
            let pt_n = p_threshold_pa.powf(hill_n);
            pn / (pn + pt_n)
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
#[inline]
pub fn radiation_force_1d(intensity_w_m2: &[f64], alpha_np_m: f64, c: f64) -> Vec<f64> {
    let scale = 2.0 * alpha_np_m / c;
    intensity_w_m2.iter().map(|&i| scale * i).collect()
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
#[inline]
pub fn acoustic_streaming_velocity(
    i_w_m2: f64,
    mu_pa_s: f64,
    alpha_np_m: f64,
    c: f64,
    l_m: f64,
) -> f64 {
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
pub fn ispta_w_cm2(p_pa: &[f64], dt_s: f64, rho: f64, c: f64) -> f64 {
    let n = p_pa.len() as f64;
    let total_time = n * dt_s;
    let integral: f64 = p_pa.iter().map(|&p| p * p * dt_s).sum();
    let ispta_w_m2 = integral / (rho * c * total_time);
    ispta_w_m2 * 1e-4 // convert W/m² → W/cm²
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
    fn radiation_force_proportional_to_intensity() {
        let f = radiation_force_1d(&[1.0, 2.0, 4.0], 1.0, 1500.0);
        assert!((f[1] / f[0] - 2.0).abs() < 1e-10);
        assert!((f[2] / f[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn streaming_velocity_positive() {
        let v = acoustic_streaming_velocity(1000.0, 1e-3, 0.5, 1500.0, 0.05);
        assert!(v > 0.0);
    }

    #[test]
    fn ispta_constant_pressure() {
        // Constant pressure p0 → ISPTA = p0²/(rho*c) in W/m² → W/cm²
        let p0 = 1e5_f64;
        let rho = 1000.0_f64;
        let c = 1500.0_f64;
        let n = 1000;
        let dt = 1e-7_f64;
        let p = vec![p0; n];
        let ispta = ispta_w_cm2(&p, dt, rho, c);
        let expected = p0 * p0 / (rho * c) * 1e-4;
        assert!((ispta - expected).abs() / expected < 1e-10, "got={} expected={}", ispta, expected);
    }
}
