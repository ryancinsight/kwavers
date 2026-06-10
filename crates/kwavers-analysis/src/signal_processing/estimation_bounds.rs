//! Cramér–Rao lower bounds (CRLB) for ultrasound displacement / speed estimators.
//!
//! These are analytical performance bounds — the minimum achievable variance of
//! any unbiased estimator given the signal model. They are used for uncertainty
//! quantification in elastography (displacement → strain → shear-wave speed) and
//! to size acquisition parameters (window length, aperture, averaging).
//!
//! All functions are pure and panic-free: a non-positive denominator (degenerate
//! acquisition — zero bandwidth, zero window, zero SNR) yields `f64::INFINITY`,
//! i.e. "no information / unbounded variance", rather than a NaN or panic.
//!
//! # References
//! - Walker, W. F., & Trahey, G. E. (1995). "A fundamental limit on delay
//!   estimation using partially correlated speckle signals." *IEEE TUFFC*,
//!   42(2), 301–308.
//! - Céspedes, I., Huang, Y., Ophir, J., Spratt, S. (1995). "Methods for
//!   estimation of subsample time delays of digitized echo signals."
//!   *Ultrason. Imaging*, 17(2), 142–171.

/// Guard: return `true` when every argument is finite and strictly positive.
#[inline]
fn all_positive(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite() && *v > 0.0)
}

use core::f64::consts::PI;

/// CRLB on the **variance** of a time-delay (jitter) estimate from
/// cross-correlation of band-limited signals (narrowband form):
///
/// ```text
/// Var[τ̂] ≥ 1 / (8 π² f₀² T_w · SNR)
/// ```
///
/// - `f0_hz`: signal centre frequency [Hz]
/// - `window_s`: correlation window duration `T_w` [s]
/// - `snr_linear`: echo signal-to-noise ratio (linear power ratio, **not** dB)
///
/// Returns the variance [s²]. Larger bandwidth-time-SNR product → tighter bound.
#[must_use]
pub fn time_delay_crlb_variance(f0_hz: f64, window_s: f64, snr_linear: f64) -> f64 {
    if !all_positive(&[f0_hz, window_s, snr_linear]) {
        return f64::INFINITY;
    }
    let denom = 8.0 * PI * PI * f0_hz * f0_hz * window_s * snr_linear;
    1.0 / denom
}

/// CRLB on the **standard deviation** of a time-delay estimate [s].
///
/// `√(Var[τ̂])` from [`time_delay_crlb_variance`].
#[must_use]
pub fn time_delay_crlb_std(f0_hz: f64, window_s: f64, snr_linear: f64) -> f64 {
    time_delay_crlb_variance(f0_hz, window_s, snr_linear).sqrt()
}

/// CRLB on the **standard deviation of axial strain** in strain elastography.
///
/// Propagating the delay bound through displacement `δ = c_P τ̂ / 2` and strain
/// `ε = δ / Δz`:
///
/// ```text
/// σ_ε ≥ c_P / (4 π f₀ √(T_w · SNR) · Δz)
/// ```
///
/// - `c_p`: longitudinal (compressional) wave speed [m/s]
/// - `f0_hz`: centre frequency [Hz]
/// - `window_s`: correlation window `T_w` [s]
/// - `snr_linear`: echo SNR (linear)
/// - `axial_window_m`: axial gradient baseline `Δz` [m]
///
/// Returns the dimensionless strain standard deviation.
#[must_use]
pub fn strain_crlb_std(
    c_p: f64,
    f0_hz: f64,
    window_s: f64,
    snr_linear: f64,
    axial_window_m: f64,
) -> f64 {
    if !all_positive(&[c_p, f0_hz, window_s, snr_linear, axial_window_m]) {
        return f64::INFINITY;
    }
    let denom = 4.0 * PI * f0_hz * (window_s * snr_linear).sqrt() * axial_window_m;
    c_p / denom
}

/// CRLB-style **standard deviation of shear-wave speed** for the phase-gradient
/// estimator (Elastography §10.12.2):
///
/// ```text
/// σ_{c_s} ≈ c_s² / (ω · L_x · √(N_t · SNR_v))
/// ```
///
/// - `c_s`: shear-wave speed [m/s]
/// - `omega_rad_s`: angular drive frequency `ω = 2πf` [rad/s]
/// - `aperture_x_m`: lateral aperture `L_x` over which the phase gradient is taken [m]
/// - `n_temporal`: number of temporal samples `N_t` in the time-frequency analysis
/// - `snr_v_linear`: tracking (velocity) SNR (linear)
///
/// Returns the shear-wave-speed standard deviation [m/s].
#[must_use]
pub fn shear_wave_speed_crlb_std(
    c_s: f64,
    omega_rad_s: f64,
    aperture_x_m: f64,
    n_temporal: f64,
    snr_v_linear: f64,
) -> f64 {
    if !all_positive(&[c_s, omega_rad_s, aperture_x_m, n_temporal, snr_v_linear]) {
        return f64::INFINITY;
    }
    let denom = omega_rad_s * aperture_x_m * (n_temporal * snr_v_linear).sqrt();
    (c_s * c_s) / denom
}

/// A bootstrap confidence interval for a scalar estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapCi {
    /// Point estimate (the statistic on the full sample) — here the sample mean.
    pub point: f64,
    /// Lower confidence bound (the `(1−level)/2` percentile of the resamples).
    pub lower: f64,
    /// Upper confidence bound (the `(1+level)/2` percentile of the resamples).
    pub upper: f64,
}

/// One `splitmix64` step — a small, fully-deterministic PRNG so bootstrap CIs
/// are reproducible from a seed (no global RNG state, no `rand` dependency).
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The `q`-quantile (`q ∈ [0, 1]`) of `sorted` (ascending) by linear
/// interpolation between order statistics.
#[inline]
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted[hi].mul_add(frac, sorted[lo] * (1.0 - frac))
}

/// **Percentile bootstrap confidence interval** for the *mean* of `samples` at
/// confidence `level` (e.g. `0.95`), from `n_resamples` resamples with
/// replacement, using a deterministic seeded PRNG (`seed`).
///
/// Each resample draws `samples.len()` values with replacement and records its
/// mean; the CI is the `[(1−level)/2, (1+level)/2]` percentiles of those
/// resample means (Efron 1979). The point estimate is the full-sample mean. The
/// CI width tracks the standard error `σ/√N`, so it narrows with more samples
/// and widens with sample spread — without assuming a distributional form.
///
/// Returns `None` for an empty sample, `level ∉ (0, 1)`, or `n_resamples == 0`.
/// A single sample yields a degenerate (zero-width) interval at that value.
///
/// # References
/// - Efron, B. (1979). "Bootstrap methods: another look at the jackknife."
///   *Ann. Statist.* 7(1), 1–26.
#[must_use]
pub fn bootstrap_ci_mean(
    samples: &[f64],
    level: f64,
    n_resamples: usize,
    seed: u64,
) -> Option<BootstrapCi> {
    let n = samples.len();
    if n == 0 || n_resamples == 0 || !(level > 0.0 && level < 1.0) || !all_finite(samples) {
        return None;
    }
    let point = samples.iter().sum::<f64>() / n as f64;
    if n == 1 {
        return Some(BootstrapCi {
            point,
            lower: point,
            upper: point,
        });
    }

    let mut state = seed ^ 0xD1B5_4A32_D192_ED03; // de-bias all-zero seeds
    let mut means = Vec::with_capacity(n_resamples);
    for _ in 0..n_resamples {
        let mut acc = 0.0;
        for _ in 0..n {
            let idx = (splitmix64(&mut state) % n as u64) as usize;
            acc += samples[idx];
        }
        means.push(acc / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    let lower = quantile_sorted(&means, 0.5 * (1.0 - level));
    let upper = quantile_sorted(&means, 0.5 * (1.0 + level));
    Some(BootstrapCi {
        point,
        lower,
        upper,
    })
}

/// Guard: every value finite (no NaN/Inf).
#[inline]
fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_delay_crlb_matches_closed_form() {
        let (f0, tw, snr) = (5.0e6, 1.0e-6, 100.0);
        let var = time_delay_crlb_variance(f0, tw, snr);
        let expected = 1.0 / (8.0 * PI * PI * f0 * f0 * tw * snr);
        assert!(
            (var - expected).abs() / expected < 1e-12,
            "var {var} must equal closed form {expected}"
        );
        // std is the square root of the variance and is finite/positive
        let std = time_delay_crlb_std(f0, tw, snr);
        assert!((std - var.sqrt()).abs() < 1e-30);
        assert!(std.is_finite() && std > 0.0);
        // raising SNR by 40 dB (×10⁴ power) lowers the std by ×100
        let std_hi = time_delay_crlb_std(f0, tw, snr * 1.0e4);
        assert!(
            (std / std_hi - 100.0).abs() < 1e-6,
            "100× SNR-amplitude improvement expected, got {}",
            std / std_hi
        );
    }

    #[test]
    fn time_delay_crlb_is_monotone_in_snr_and_bandwidth() {
        let base = time_delay_crlb_variance(5.0e6, 1.0e-6, 100.0);
        // higher SNR → strictly smaller variance
        assert!(time_delay_crlb_variance(5.0e6, 1.0e-6, 400.0) < base);
        // higher centre frequency → strictly smaller variance
        assert!(time_delay_crlb_variance(1.0e7, 1.0e-6, 100.0) < base);
        // longer window → strictly smaller variance
        assert!(time_delay_crlb_variance(5.0e6, 4.0e-6, 100.0) < base);
    }

    #[test]
    fn strain_crlb_matches_closed_form() {
        let (cp, f0, tw, snr, dz) = (1540.0, 5.0e6, 1.0e-6, 100.0, 1.0e-3);
        let got = strain_crlb_std(cp, f0, tw, snr, dz);
        let expected = cp / (4.0 * PI * f0 * (tw * snr).sqrt() * dz);
        assert!(
            (got - expected).abs() / expected < 1e-12,
            "strain σ {got} must equal closed form {expected}"
        );
    }

    #[test]
    fn shear_speed_crlb_matches_closed_form() {
        let (cs, omega, lx, nt, snrv) = (3.0, 2.0 * PI * 200.0, 0.02, 64.0, 50.0);
        let got = shear_wave_speed_crlb_std(cs, omega, lx, nt, snrv);
        let expected = (cs * cs) / (omega * lx * (nt * snrv).sqrt());
        assert!(
            (got - expected).abs() / expected < 1e-12,
            "shear-speed σ {got} must equal closed form {expected}"
        );
        // larger aperture and more averaging both reduce the bound
        assert!(shear_wave_speed_crlb_std(cs, omega, 0.04, nt, snrv) < got);
        assert!(shear_wave_speed_crlb_std(cs, omega, lx, 256.0, snrv) < got);
    }

    #[test]
    fn degenerate_inputs_give_infinite_bound() {
        assert!(time_delay_crlb_variance(0.0, 1.0e-6, 100.0).is_infinite());
        assert!(strain_crlb_std(1540.0, 5.0e6, 1.0e-6, 0.0, 1.0e-3).is_infinite());
        assert!(shear_wave_speed_crlb_std(3.0, 0.0, 0.02, 64.0, 50.0).is_infinite());
    }

    // ── Bootstrap confidence intervals ──────────────────────────────────────

    /// Synthetic σ₀ estimates over a cardiac cycle (kPa), mean ≈ 12.
    fn sigma_samples() -> Vec<f64> {
        vec![
            10.2, 11.8, 12.5, 13.1, 11.0, 12.9, 10.7, 13.4, 12.1, 11.5, 12.7, 13.0, 10.9, 12.3,
            11.7, 12.6,
        ]
    }

    /// The CI brackets the point estimate, is ordered, and is reproducible from
    /// a fixed seed.
    #[test]
    fn bootstrap_ci_brackets_point_and_is_deterministic() {
        let s = sigma_samples();
        let ci = bootstrap_ci_mean(&s, 0.95, 2000, 42).expect("ci");
        let mean = s.iter().sum::<f64>() / s.len() as f64;
        assert!((ci.point - mean).abs() < 1e-12, "point = sample mean");
        assert!(ci.lower <= ci.point && ci.point <= ci.upper, "CI brackets the point");
        // Same seed ⇒ bit-identical CI (deterministic PRNG).
        let ci2 = bootstrap_ci_mean(&s, 0.95, 2000, 42).expect("ci");
        assert_eq!(ci, ci2);
    }

    /// The 95% bootstrap CI half-width of the mean tracks the standard error:
    /// it is within a modest factor of the analytical `1.96·σ/√N`.
    #[test]
    fn bootstrap_ci_halfwidth_tracks_standard_error() {
        let s = sigma_samples();
        let n = s.len() as f64;
        let mean = s.iter().sum::<f64>() / n;
        let var = s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let se = (var / n).sqrt();

        let ci = bootstrap_ci_mean(&s, 0.95, 5000, 7).expect("ci");
        let half = 0.5 * (ci.upper - ci.lower);
        let normal_half = 1.96 * se;
        assert!(
            (half / normal_half - 1.0).abs() < 0.35,
            "bootstrap half-width {half} must track 1.96·SE {normal_half}"
        );
    }

    /// A higher-spread sample (same mean) yields a wider CI; a higher confidence
    /// level widens the CI for the same data.
    #[test]
    fn bootstrap_ci_widens_with_spread_and_confidence() {
        let tight: Vec<f64> = vec![11.9, 12.0, 12.1, 11.95, 12.05, 12.0, 11.98, 12.02];
        let wide: Vec<f64> = vec![8.0, 16.0, 9.0, 15.0, 10.0, 14.0, 11.0, 13.0]; // same mean 12
        let tw = bootstrap_ci_mean(&tight, 0.95, 3000, 3).unwrap();
        let wd = bootstrap_ci_mean(&wide, 0.95, 3000, 3).unwrap();
        assert!(
            (wd.upper - wd.lower) > (tw.upper - tw.lower),
            "more spread ⇒ wider CI"
        );

        let s = sigma_samples();
        let ci95 = bootstrap_ci_mean(&s, 0.95, 3000, 9).unwrap();
        let ci99 = bootstrap_ci_mean(&s, 0.99, 3000, 9).unwrap();
        assert!(
            (ci99.upper - ci99.lower) > (ci95.upper - ci95.lower),
            "higher confidence ⇒ wider CI"
        );
    }

    /// Degenerate cases: empty → None; single sample → zero-width CI; invalid
    /// level/resamples → None.
    #[test]
    fn bootstrap_ci_degenerate_cases() {
        assert!(bootstrap_ci_mean(&[], 0.95, 1000, 1).is_none());
        assert!(bootstrap_ci_mean(&[1.0, 2.0], 0.0, 1000, 1).is_none());
        assert!(bootstrap_ci_mean(&[1.0, 2.0], 1.0, 1000, 1).is_none());
        assert!(bootstrap_ci_mean(&[1.0, 2.0], 0.95, 0, 1).is_none());
        let single = bootstrap_ci_mean(&[7.5], 0.95, 1000, 1).unwrap();
        assert_eq!(single.point, 7.5);
        assert_eq!(single.lower, 7.5);
        assert_eq!(single.upper, 7.5);
    }
}
