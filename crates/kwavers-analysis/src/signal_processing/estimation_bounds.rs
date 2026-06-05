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
}
