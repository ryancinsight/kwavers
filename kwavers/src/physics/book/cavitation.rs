//! Bubble dynamics and cavitation physics for book chapters ch07, ch09.
//!
//! Covers: Minnaert resonance, Blake threshold, Rayleigh collapse time,
//! Rayleigh–Plesset and Keller–Miksis ODE integrators (RK4), and a DFT-based
//! power spectrum estimator for bubble radius time series.

use std::f64::consts::PI;

// ─── Scalar estimates ─────────────────────────────────────────────────────────

/// Minnaert resonance frequency of a spherical gas bubble.
///
/// ```text
/// f_M = 1/(2π·R₀) · √(3γP₀/ρ)   [Hz]
/// ```
///
/// # Arguments
/// * `r0_m` – equilibrium bubble radius [m]
/// * `gamma` – polytropic exponent of the gas (1.4 for air)
/// * `p0_pa` – ambient pressure [Pa]
/// * `rho` – liquid density [kg/m³]
///
/// # Reference
/// Minnaert (1933), *Philos. Mag.* 16, 235.
#[inline]
pub fn minnaert_resonance_hz(r0_m: f64, gamma: f64, p0_pa: f64, rho: f64) -> f64 {
    1.0 / (2.0 * PI * r0_m) * (3.0 * gamma * p0_pa / rho).sqrt()
}

/// Blake threshold pressure (inertial cavitation onset).
///
/// Approximate closed-form expression derived from the static equilibrium of a
/// bubble under surface tension σ and ambient pressure P₀:
/// ```text
/// P_B = P₀ + (2σ)/(3R₀) · √(3/(P₀·R₀/(2σ) + 1))   (Blake 1949)
/// ```
///
/// # Arguments
/// * `r0_m` – equilibrium bubble radius [m]
/// * `p0_pa` – ambient pressure [Pa]
/// * `sigma_n_m` – surface tension coefficient [N/m] (water ≈ 0.0725)
///
/// # Reference
/// Blake (1949) *Appendix to: Cavitation*, HMSO Report.
#[inline]
pub fn blake_threshold_pa(r0_m: f64, p0_pa: f64, sigma_n_m: f64) -> f64 {
    let ratio = sigma_n_m / (r0_m * p0_pa); // (σ/R₀P₀)
    let inner = (1.0 + 2.0 * ratio / 3.0) * (1.0 + 1.0 / (1.5 * ratio + 1.0)).sqrt();
    p0_pa * inner
}

/// Rayleigh collapse time for a spherical cavity.
///
/// ```text
/// t_c = 0.9147 · R_max · √(ρ/P_∞)   [s]
/// ```
///
/// # Arguments
/// * `rmax_m` – maximum bubble radius [m]
/// * `p_inf_pa` – driving pressure at infinity (usually ambient) [Pa]
/// * `rho` – liquid density [kg/m³]
///
/// # Reference
/// Rayleigh (1917), *Philos. Mag.* 34, 94.
#[inline]
pub fn rayleigh_collapse_time_s(rmax_m: f64, p_inf_pa: f64, rho: f64) -> f64 {
    0.9147 * rmax_m * (rho / p_inf_pa).sqrt()
}

// ─── Rayleigh–Plesset ODE ─────────────────────────────────────────────────────

/// Integrate the Rayleigh–Plesset equation using RK4.
///
/// The RP equation in state form, with state `y = [R, Ṙ]`:
/// ```text
/// R·R̈ + (3/2)·Ṙ² = [P_gas(R) − P₀ − P_ac(t) − 2σ/R − 4μ·Ṙ/R] / ρ
/// P_gas(R) = (P₀ + 2σ/R₀)·(R₀/R)^(3κ) − P_v
/// ```
///
/// # Arguments
/// * `r0_m` – equilibrium radius [m]
/// * `rdot0` – initial radial velocity [m/s]
/// * `p_ac_pa` – acoustic pressure amplitude [Pa]
/// * `freq_hz` – acoustic driving frequency [Hz]
/// * `t_arr` – time array (must be uniformly spaced) [s]
/// * `p0_pa` – ambient pressure [Pa]
/// * `rho` – liquid density [kg/m³]
/// * `sigma` – surface tension [N/m]
/// * `mu` – dynamic viscosity [Pa·s]
/// * `kappa` – polytropic exponent
/// * `p_v_pa` – vapour pressure [Pa]
///
/// Returns `(R_arr [m], Ṙ_arr [m/s])`.
///
/// # Reference
/// Plesset & Prosperetti (1977), *Annu. Rev. Fluid Mech.* 9, 145.
pub fn rayleigh_plesset_rk4(
    r0_m: f64,
    rdot0: f64,
    p_ac_pa: f64,
    freq_hz: f64,
    t_arr: &[f64],
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    p_v_pa: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = t_arr.len();
    let mut r_out = vec![0.0_f64; n];
    let mut rdot_out = vec![0.0_f64; n];

    r_out[0] = r0_m;
    rdot_out[0] = rdot0;

    let omega = 2.0 * PI * freq_hz;
    let p_gas0 = p0_pa + 2.0 * sigma / r0_m; // gas pressure at equilibrium

    // RP right-hand side: returns [dR/dt, d²R/dt²] = [Ṙ, R̈]
    let rhs = |r: f64, rdot: f64, t: f64| -> (f64, f64) {
        let r_clamped = r.max(1e-15); // guard against collapse to zero
        let p_gas = p_gas0 * (r0_m / r_clamped).powf(3.0 * kappa) - p_v_pa;
        let p_ac = p_ac_pa * (omega * t).sin();
        let rddot = (p_gas - p0_pa - p_ac - 2.0 * sigma / r_clamped - 4.0 * mu * rdot / r_clamped)
            / (rho * r_clamped)
            - 1.5 * rdot * rdot / r_clamped;
        (rdot, rddot)
    };

    for i in 1..n {
        let dt = t_arr[i] - t_arr[i - 1];
        let t = t_arr[i - 1];
        let r = r_out[i - 1];
        let v = rdot_out[i - 1];

        let (k1r, k1v) = rhs(r, v, t);
        let (k2r, k2v) = rhs(r + 0.5 * dt * k1r, v + 0.5 * dt * k1v, t + 0.5 * dt);
        let (k3r, k3v) = rhs(r + 0.5 * dt * k2r, v + 0.5 * dt * k2v, t + 0.5 * dt);
        let (k4r, k4v) = rhs(r + dt * k3r, v + dt * k3v, t + dt);

        r_out[i] = r + dt / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
        rdot_out[i] = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        // Guard against unphysical collapse
        r_out[i] = r_out[i].max(1e-15);
    }

    (r_out, rdot_out)
}

// ─── Keller–Miksis ODE ────────────────────────────────────────────────────────

/// Integrate the Keller–Miksis equation (compressible-liquid correction) using RK4.
///
/// ```text
/// (1 − Ṙ/c_L)·R·R̈ + (3/2)·Ṙ²·(1 − Ṙ/(3c_L))
///   = (1 + Ṙ/c_L)/ρ · (P_gas − P₀ − P_ac(t+R/c_L)) + Ṙ·dP_gas/dt/(ρ·c_L)
///   − 2σ/(ρ·R) − 4μ·Ṙ/(ρ·R)
/// ```
///
/// # Arguments
/// Same as [`rayleigh_plesset_rk4`] with the addition of:
/// * `c_liquid` – sound speed in the liquid [m/s]
///
/// # Reference
/// Keller & Miksis (1980), *J. Acoust. Soc. Am.* 68, 628.
pub fn keller_miksis_rk4(
    r0_m: f64,
    rdot0: f64,
    p_ac_pa: f64,
    freq_hz: f64,
    t_arr: &[f64],
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    p_v_pa: f64,
    c_liquid: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = t_arr.len();
    let mut r_out = vec![0.0_f64; n];
    let mut rdot_out = vec![0.0_f64; n];

    r_out[0] = r0_m;
    rdot_out[0] = rdot0;

    let omega = 2.0 * PI * freq_hz;
    let p_gas0 = p0_pa + 2.0 * sigma / r0_m;

    let rhs = |r: f64, rdot: f64, t: f64| -> (f64, f64) {
        let r_c = r.max(1e-15);
        let p_gas = p_gas0 * (r0_m / r_c).powf(3.0 * kappa) - p_v_pa;
        // Retarded acoustic pressure at t + R/c_L
        let t_ret = t + r_c / c_liquid;
        let p_ac_now = p_ac_pa * (omega * t).sin();
        let p_ac_ret = p_ac_pa * (omega * t_ret).sin();
        // dP_gas/dt = −3κ·P_gas · Ṙ/R  (via chain rule from (R₀/R)^{3κ})
        let dp_gas_dt = -3.0 * kappa * p_gas * rdot / r_c;

        let rdot_cl = rdot / c_liquid;
        let lhs_coeff = (1.0 - rdot_cl) * r_c;
        if lhs_coeff.abs() < 1e-20 {
            return (rdot, 0.0);
        }
        let rhs_val = (1.0 + rdot_cl) / rho * (p_gas - p0_pa - p_ac_ret)
            + r_c / (rho * c_liquid) * dp_gas_dt
            - 2.0 * sigma / (rho * r_c)
            - 4.0 * mu * rdot / (rho * r_c)
            - p_ac_now * rdot_cl / rho;
        let rddot = (rhs_val - 1.5 * rdot * rdot * (1.0 - rdot_cl / 3.0)) / lhs_coeff;
        (rdot, rddot)
    };

    for i in 1..n {
        let dt = t_arr[i] - t_arr[i - 1];
        let t = t_arr[i - 1];
        let r = r_out[i - 1];
        let v = rdot_out[i - 1];

        let (k1r, k1v) = rhs(r, v, t);
        let (k2r, k2v) = rhs(r + 0.5 * dt * k1r, v + 0.5 * dt * k1v, t + 0.5 * dt);
        let (k3r, k3v) = rhs(r + 0.5 * dt * k2r, v + 0.5 * dt * k2v, t + 0.5 * dt);
        let (k4r, k4v) = rhs(r + dt * k3r, v + dt * k3v, t + dt);

        r_out[i] = (r + dt / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)).max(1e-15);
        rdot_out[i] = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    }

    (r_out, rdot_out)
}

// ─── Power spectrum ───────────────────────────────────────────────────────────

/// Estimate the power spectrum of the bubble radius time series via DFT.
///
/// Computes the single-sided power spectral density:
/// ```text
/// S(f) = |DFT(R)|² / (N² · Δt)   [m²/Hz],  f ≥ 0
/// ```
/// using a rectangular (no) window, zero-padded to `n_fft` points.
///
/// # Arguments
/// * `r_arr` – radius time series [m]
/// * `dt_s` – time step [s]
/// * `n_fft` – DFT length (should be ≥ `r_arr.len()`, preferably a power of 2)
///
/// Returns `(f_arr [Hz], power_arr [m²/Hz])` for non-negative frequencies.
///
/// # Note
/// This implements a direct O(N²) DFT, which is exact but slow for large N.
/// For production use, prefer a dedicated FFT; here correctness takes priority.
pub fn bubble_power_spectrum(r_arr: &[f64], dt_s: f64, n_fft: usize) -> (Vec<f64>, Vec<f64>) {
    let n = n_fft;
    let n_f = n as f64;
    // Zero-pad signal
    let mut padded = vec![0.0_f64; n];
    for (i, &v) in r_arr.iter().enumerate().take(n) {
        padded[i] = v;
    }
    // Compute mean to remove DC bias before PSD
    let mean: f64 = padded.iter().sum::<f64>() / n_f;
    padded.iter_mut().for_each(|x| *x -= mean);

    let n_pos = n / 2 + 1;
    let mut f_arr = vec![0.0_f64; n_pos];
    let mut power = vec![0.0_f64; n_pos];

    for k in 0..n_pos {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        let phase_step = 2.0 * PI * k as f64 / n_f;
        for (j, &rj) in padded.iter().enumerate() {
            let phi = phase_step * j as f64;
            re += rj * phi.cos();
            im -= rj * phi.sin();
        }
        let mag_sq = re * re + im * im;
        // One-sided PSD: double for k > 0 and k < N/2
        let scale = if k == 0 || k == n / 2 { 1.0 } else { 2.0 };
        power[k] = scale * mag_sq / (n_f * n_f * dt_s.recip()); // [m²/Hz]... wait: /= fs
                                                                // Correct: S[k] = |X[k]|²·dt / N  for one-sided
        power[k] = scale * mag_sq * dt_s / n_f;
        f_arr[k] = k as f64 / (n_f * dt_s);
    }

    (f_arr, power)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minnaert_water_air_bubble() {
        // 10 μm air bubble in water at 101 325 Pa
        let f = minnaert_resonance_hz(10e-6, 1.4, 101_325.0, 998.0);
        // Expected ≈ 327 kHz  (literature value ~325–330 kHz for 10 μm)
        assert!((f - 327_000.0).abs() / 327_000.0 < 0.05, "f={}", f);
    }

    #[test]
    fn rayleigh_collapse_positive() {
        let tc = rayleigh_collapse_time_s(100e-6, 101_325.0, 998.0);
        assert!(tc > 0.0 && tc < 1e-4);
    }

    #[test]
    fn rp_rk4_initial_condition() {
        let t: Vec<f64> = (0..10).map(|i| i as f64 * 1e-9).collect();
        let (r, _) = rayleigh_plesset_rk4(
            10e-6, 0.0, 0.0, 1e6, &t, 101_325.0, 998.0, 0.0725, 0.001, 1.4, 2_330.0,
        );
        // Zero driving: bubble should remain near R₀
        assert!((r[0] - 10e-6).abs() < 1e-15);
        assert!((r[9] - 10e-6).abs() / 10e-6 < 0.01);
    }

    #[test]
    fn km_rk4_length_matches() {
        let t: Vec<f64> = (0..5).map(|i| i as f64 * 1e-9).collect();
        let (r, v) = keller_miksis_rk4(
            10e-6, 0.0, 0.0, 1e6, &t, 101_325.0, 998.0, 0.0725, 0.001, 1.4, 2_330.0, 1500.0,
        );
        assert_eq!(r.len(), 5);
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn bubble_spectrum_length() {
        let r: Vec<f64> = (0..64)
            .map(|i| 10e-6 + 1e-7 * (i as f64 * 0.1).sin())
            .collect();
        let (f, p) = bubble_power_spectrum(&r, 1e-9, 64);
        assert_eq!(f.len(), 33);
        assert_eq!(p.len(), 33);
        assert!(f[0] == 0.0);
        assert!(p.iter().all(|&v| v >= 0.0));
    }
}
