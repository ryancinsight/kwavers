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
/// Coefficient = B(5/6, 1/2)·√(3/2)/3 where B is the beta function;
/// evaluated as Γ(5/6)Γ(1/2)/Γ(4/3) ≈ 2.241, giving 2.241·√(3/2)/3 ≈ 0.9147.
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
        // One-sided PSD: double for interior bins; S[k] = |X[k]|²·Δt/N [m²/Hz]
        let scale = if k == 0 || k == n / 2 { 1.0 } else { 2.0 };
        power[k] = scale * mag_sq * dt_s / n_f;
        f_arr[k] = k as f64 / (n_f * dt_s);
    }

    (f_arr, power)
}

// ─── Mechanical index ─────────────────────────────────────────────────────────

/// FDA mechanical index: peak negative pressure normalised by √(frequency).
///
/// ```text
/// MI = |P_neg| [MPa] / √(f [MHz])
/// ```
/// FDA safety guideline for diagnostic imaging: MI < 1.9.
/// Histotripsy (intrinsic threshold) requires MI > 3 for microsecond pulses.
///
/// # Reference
/// Apfel & Holland (1991), *Ultrasound Med. Biol.* 17, 179.
#[inline]
pub fn mechanical_index(p_neg_pa: f64, freq_hz: f64) -> f64 {
    let p_neg_mpa = p_neg_pa.abs() * 1e-6;
    let f_mhz = freq_hz * 1e-6;
    p_neg_mpa / f_mhz.sqrt()
}

// ─── Inertial cavitation dose ─────────────────────────────────────────────────

/// Inertial cavitation dose (ICD) from a bubble radius time series.
///
/// Accumulates the normalised collapse strength over all detected inertial
/// collapse events.  A collapse event is a local minimum of `R` below `R₀`
/// coinciding with a sign change of `Ṙ` from negative to non-negative:
/// ```text
/// ICD = Σ_{collapse events i} (R_max_i / R₀)³   [dimensionless]
/// ```
/// `R_max_i` is the maximum bubble radius reached during the expansion phase
/// immediately preceding the ith collapse.  The cubic weighting is proportional
/// to the maximum volume ratio and therefore to the inertial energy.
///
/// # Reference
/// Duryea et al. (2015), *Ultrasound Med. Biol.* 41, 1937.
pub fn inertial_cavitation_dose(r_arr: &[f64], rdot_arr: &[f64], r0_m: f64) -> f64 {
    let r0 = r0_m.max(1e-15);
    let n = r_arr.len().min(rdot_arr.len());
    if n < 2 {
        return 0.0;
    }
    let mut dose = 0.0_f64;
    let mut r_max = r0;
    for i in 0..n - 1 {
        let r = r_arr[i].max(1e-15);
        r_max = r_max.max(r);
        // Inertial collapse: velocity reversal from negative to non-negative below R₀
        let is_min = rdot_arr[i] < 0.0 && rdot_arr[i + 1] >= 0.0 && r < r0;
        if is_min {
            dose += (r_max / r0).powi(3);
            r_max = r0; // reset for next expansion–collapse cycle
        }
    }
    dose
}

// ─── Histotripsy lesion radius ────────────────────────────────────────────────

/// Estimated histotripsy lesion radius from cavitation energy balance.
///
/// ## Derivation
/// The Rayleigh–Plesset energy released during inertial collapse equals the
/// PdV work against ambient pressure over the bubble volume excursion:
/// ```text
/// E_collapse ≈ (4π/3) · P₀ · R_max³    [per event]
/// ```
/// Summing over all ICD events: `E_total = (4π/3) · P₀ · R₀³ · ICD`.
/// Setting `E_total = σ_y · (4π/3) · R_L³` and solving for `R_L`:
/// ```text
/// R_L = R₀ · (P₀ · ICD / σ_y)^(1/3)   [m]
/// ```
///
/// # Arguments
/// * `icd` – dimensionless inertial cavitation dose (from `inertial_cavitation_dose`)
/// * `r0_m` – equilibrium bubble radius [m]
/// * `p0_pa` – ambient pressure [Pa]
/// * `tissue_yield_stress_pa` – tensile yield stress of tissue [Pa]
///   (brain white matter: 1–4 kPa; Vlaisavljevich et al. 2015)
///
/// # Reference
/// Maxwell et al. (2011), *J. Acoust. Soc. Am.* 130, 2012.
/// Vlaisavljevich et al. (2015), *Ultrasound Med. Biol.* 41, 2896.
#[inline]
pub fn histotripsy_lesion_radius_m(
    icd: f64,
    r0_m: f64,
    p0_pa: f64,
    tissue_yield_stress_pa: f64,
) -> f64 {
    let sigma_y = tissue_yield_stress_pa.max(1.0);
    // R_L = R₀ · (P₀ · ICD / σ_y)^(1/3)
    r0_m * (p0_pa * icd / sigma_y).cbrt()
}

// ─── Period-doubling ratio ────────────────────────────────────────────────────

/// Subharmonic period-doubling ratio from a bubble power spectrum.
///
/// Computes the spectral energy ratio of the half-harmonic (f₀/2) to the
/// fundamental (f₀) — a passive acoustic marker of inertial cavitation:
/// ```text
/// PD_ratio = S(f₀/2) / S(f₀)
/// ```
/// Each band is integrated over a ±2-bin window. Values above ~0.1 indicate
/// onset of subharmonic emission consistent with histotripsy bubble activity.
///
/// # Reference
/// Cramer et al. (2021), *Ultrasound Med. Biol.* 47, 2102.
pub fn period_doubling_ratio(f_arr: &[f64], power_arr: &[f64], freq_hz: f64) -> f64 {
    if f_arr.len() < 2 || power_arr.is_empty() {
        return 0.0;
    }
    let df = f_arr[1] - f_arr[0];
    let band_energy = |target_hz: f64| -> f64 {
        let idx = f_arr
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (**a - target_hz).abs().total_cmp(&(**b - target_hz).abs()))
            .map(|(i, _)| i)
            .unwrap_or(0);
        // ±1 bin window (3 bins) reduces cross-contamination between
        // fundamental and sub-harmonic bins while accommodating minor leakage.
        let lo = idx.saturating_sub(1);
        let hi = (idx + 2).min(power_arr.len());
        power_arr[lo..hi].iter().sum::<f64>() * df
    };
    let s_fund = band_energy(freq_hz);
    let s_sub = band_energy(freq_hz * 0.5);
    if s_fund > 0.0 {
        s_sub / s_fund
    } else {
        0.0
    }
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
    fn mechanical_index_known_value() {
        // 1 MPa at 1 MHz → MI = 1.0
        let mi = mechanical_index(1e6, 1e6);
        assert!((mi - 1.0).abs() < 1e-9, "mi={}", mi);
    }

    #[test]
    fn mechanical_index_scales_inversely_with_sqrt_freq() {
        // MI at 4 MHz should be half of MI at 1 MHz for same pressure
        let mi_1 = mechanical_index(1e6, 1e6);
        let mi_4 = mechanical_index(1e6, 4e6);
        assert!((mi_1 / mi_4 - 2.0).abs() < 1e-9, "ratio={}", mi_1 / mi_4);
    }

    #[test]
    fn icd_zero_driving_gives_zero() {
        // Zero acoustic drive → bubble stays at R₀ → no collapse events → ICD = 0
        let t: Vec<f64> = (0..100).map(|i| i as f64 * 1e-9).collect();
        let (r, rdot) = rayleigh_plesset_rk4(
            10e-6, 0.0, 0.0, 1e6, &t, 101_325.0, 998.0, 0.0725, 0.001, 1.4, 2_330.0,
        );
        let icd = inertial_cavitation_dose(&r, &rdot, 10e-6);
        assert_eq!(icd, 0.0, "icd={}", icd);
    }

    #[test]
    fn icd_strong_driving_nonzero() {
        // Strong driving (10× P₀) over 5 acoustic cycles should trigger at least one collapse
        let f0 = 500e3_f64;
        let n_pts = 2000usize;
        let dt = 1.0 / (20.0 * f0);
        let t: Vec<f64> = (0..n_pts).map(|i| i as f64 * dt).collect();
        let r0 = 5e-6;
        let (r, rdot) = rayleigh_plesset_rk4(
            r0, 0.0, 5e6, f0, &t, 101_325.0, 998.0, 0.0725, 0.001, 1.4, 2_330.0,
        );
        let icd = inertial_cavitation_dose(&r, &rdot, r0);
        assert!(
            icd > 0.0,
            "expected ICD > 0 for strong driving, got {}",
            icd
        );
    }

    #[test]
    fn lesion_radius_scales_with_icd_cube_root() {
        // R_L ∝ ICD^(1/3) when all other parameters are fixed
        let r0 = 5e-6;
        let p0 = 101_325.0;
        let sigma_y = 2000.0; // 2 kPa — brain white matter
        let r1 = histotripsy_lesion_radius_m(1.0, r0, p0, sigma_y);
        let r8 = histotripsy_lesion_radius_m(8.0, r0, p0, sigma_y);
        // ICD 8× → radius 2× (cube root)
        assert!((r8 / r1 - 2.0).abs() < 1e-9, "r8/r1={}", r8 / r1);
    }

    #[test]
    fn lesion_radius_dimensional_consistency() {
        // With ICD = 1.0, R_L = R₀ · (P₀/σ_y)^(1/3)
        let r0 = 5e-6;
        let p0 = 101_325.0;
        let sigma_y = 101_325.0; // same as P₀ → R_L = R₀
        let r_l = histotripsy_lesion_radius_m(1.0, r0, p0, sigma_y);
        assert!((r_l - r0).abs() < 1e-18, "r_l={}, r0={}", r_l, r0);
    }

    #[test]
    fn period_doubling_ratio_no_subharmonic_is_small() {
        // Pure fundamental with no half-harmonic → PD ratio near 0.
        // Place f0 at DFT bin 8 and f0/2 at bin 4 so ±1-bin windows [7,9] and [3,5]
        // do not overlap: separation = 3 bins, window width = 3 bins.
        let n = 512usize;
        let f0 = 1e6_f64;
        let fs = f0 * n as f64 / 8.0; // fs = 64 MHz; df = 125 kHz; f0 = bin 8
        let dt = 1.0 / fs;
        let r: Vec<f64> = (0..n)
            .map(|i| 1e-7 * (2.0 * PI * f0 * i as f64 * dt).sin())
            .collect();
        let (f_arr, p_arr) = bubble_power_spectrum(&r, dt, n);
        let pd = period_doubling_ratio(&f_arr, &p_arr, f0);
        assert!(pd < 0.1, "expected near-zero PD ratio, got {}", pd);
    }

    #[test]
    fn period_doubling_ratio_dominant_subharmonic_exceeds_one() {
        // Subharmonic amplitude 3× fundamental → power ratio ~9 → PD ratio >> 1.
        // Same bin placement as no-subharmonic test: f0=bin 8, f0/2=bin 4.
        let n = 512usize;
        let f0 = 1e6_f64;
        let fs = f0 * n as f64 / 8.0;
        let dt = 1.0 / fs;
        let r: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 * dt;
                1e-7 * (2.0 * PI * f0 * t).sin()             // fundamental
                    + 3e-7 * (2.0 * PI * f0 * 0.5 * t).sin() // dominant subharmonic
            })
            .collect();
        let (f_arr, p_arr) = bubble_power_spectrum(&r, dt, n);
        let pd = period_doubling_ratio(&f_arr, &p_arr, f0);
        assert!(
            pd > 1.0,
            "expected PD ratio > 1 for dominant subharmonic, got {}",
            pd
        );
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
