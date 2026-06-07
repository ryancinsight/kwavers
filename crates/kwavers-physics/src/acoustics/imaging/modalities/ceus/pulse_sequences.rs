//! Contrast pulse sequences (CPS): pulse inversion and amplitude modulation.
//!
//! Microbubbles scatter nonlinearly; tissue (at low MI) scatters linearly.
//! Multi-pulse transmit sequences combine the echoes so the **linear** response
//! cancels and the **nonlinear** (microbubble) response survives, suppressing
//! tissue clutter:
//!
//! - **Pulse inversion (PI)** — transmit a pulse and its 180°-inverted copy and
//!   *sum* the echoes. A linear scatterer gives `e + (−e) = 0`; even harmonics
//!   (2f, 4f, …) are unchanged by inversion and add coherently.
//! - **Amplitude modulation (AM)** — transmit two half-amplitude pulses and one
//!   full-amplitude pulse; form `e_full − e_½a − e_½b`. The linear part cancels
//!   (½+½ = 1); the super-linear microbubble residual remains.
//! - **CPS** — a general weighted combination of N pulse echoes (Phillips 2001).
//!
//! These operate on received echo time series; the nonlinear scattering physics
//! is in [`super::scattering`] / [`super::microbubble`].
//!
//! # References
//! - Simpson, D. H., et al. (1999). "Pulse inversion Doppler." *IEEE TUFFC*, 46(2).
//! - Phillips, P. (2001). "Contrast pulse sequences (CPS)." *IEEE Ultrason. Symp.*

/// Pulse-inversion combine: `e_positive + e_inverted` (element-wise, truncated to
/// the shorter length). Linear echoes cancel; even harmonics reinforce.
#[must_use]
pub fn pulse_inversion(echo_positive: &[f64], echo_inverted: &[f64]) -> Vec<f64> {
    echo_positive
        .iter()
        .zip(echo_inverted)
        .map(|(a, b)| a + b)
        .collect()
}

/// Amplitude-modulation combine: `e_full − e_half_a − e_half_b`. The linear
/// response cancels; the nonlinear residual survives.
#[must_use]
pub fn amplitude_modulation(
    echo_half_a: &[f64],
    echo_half_b: &[f64],
    echo_full: &[f64],
) -> Vec<f64> {
    echo_full
        .iter()
        .zip(echo_half_a)
        .zip(echo_half_b)
        .map(|((f, a), b)| f - a - b)
        .collect()
}

/// General contrast-pulse-sequence combine: `Σ_k w_k · e_k` over N pulse echoes.
///
/// Returns an empty vector if `echoes` and `weights` differ in length or no
/// echoes are given. Echoes are combined up to the shortest sample length.
#[must_use]
pub fn cps_combine(echoes: &[&[f64]], weights: &[f64]) -> Vec<f64> {
    if echoes.is_empty() || echoes.len() != weights.len() {
        return Vec::new();
    }
    let n = echoes.iter().map(|e| e.len()).min().unwrap_or(0);
    let mut out = vec![0.0_f64; n];
    for (echo, &w) in echoes.iter().zip(weights) {
        for (o, &e) in out.iter_mut().zip(echo.iter()) {
            *o += w * e;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::TAU;

    /// Instantaneous quadratic scatterer response `r(p) = a·p + b·p²`:
    /// `a·p` is the linear (tissue) term, `b·p²` the microbubble nonlinearity
    /// (a 2nd-harmonic + DC generator).
    fn scatter(drive: &[f64], a: f64, b: f64) -> Vec<f64> {
        drive.iter().map(|&p| a * p + b * p * p).collect()
    }

    /// |Σ s(t) e^{-i2πf t}| — single-frequency (Goertzel-style) magnitude.
    fn band_mag(sig: &[f64], f: f64, fs: f64) -> f64 {
        let (mut re, mut im) = (0.0, 0.0);
        for (n, &s) in sig.iter().enumerate() {
            let ph = TAU * f * n as f64 / fs;
            re += s * ph.cos();
            im -= s * ph.sin();
        }
        (re * re + im * im).sqrt()
    }

    fn drive(amp: f64, f: f64, fs: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (TAU * f * i as f64 / fs).sin())
            .collect()
    }

    #[test]
    fn pulse_inversion_cancels_linear_keeps_2nd_harmonic() {
        // n = 480 → exactly 12 cycles of f (and 24 of 2f) land on clean DFT bins,
        // so the single-frequency magnitudes are leakage-free.
        let (f, fs, n) = (1.0e6, 40.0e6, 480);
        let xp = drive(1.0, f, fs, n);
        let xn: Vec<f64> = xp.iter().map(|v| -v).collect();

        // pure tissue (linear): a=1, b=0 → PI sum must vanish
        let tissue = pulse_inversion(&scatter(&xp, 1.0, 0.0), &scatter(&xn, 1.0, 0.0));
        let tissue_e: f64 = tissue.iter().map(|v| v * v).sum();
        assert!(
            tissue_e < 1e-12,
            "linear PI residual energy {tissue_e} not ~0"
        );

        // microbubble (nonlinear): a=1, b=0.3
        let bub = pulse_inversion(&scatter(&xp, 1.0, 0.3), &scatter(&xn, 1.0, 0.3));
        let fund = band_mag(&bub, f, fs); // should be ~0 (fundamental cancelled)
        let second = band_mag(&bub, 2.0 * f, fs); // should dominate
        assert!(
            second > 50.0 * (fund + 1e-9),
            "PI: 2nd harmonic {second} should dominate fundamental {fund}"
        );
    }

    #[test]
    fn amplitude_modulation_cancels_linear_response() {
        let (f, fs, n) = (1.0e6, 40.0e6, 512);
        let full = drive(1.0, f, fs, n);
        let half = drive(0.5, f, fs, n);

        // linear: e_full = e_½a + e_½b exactly → AM residual ~0
        let lin = amplitude_modulation(
            &scatter(&half, 1.0, 0.0),
            &scatter(&half, 1.0, 0.0),
            &scatter(&full, 1.0, 0.0),
        );
        let lin_e: f64 = lin.iter().map(|v| v * v).sum();
        assert!(lin_e < 1e-12, "linear AM residual {lin_e} not ~0");

        // nonlinear: residual survives (b·(A² − 2·(A/2)²) = b·A²/2 ≠ 0)
        let nl = amplitude_modulation(
            &scatter(&half, 1.0, 0.3),
            &scatter(&half, 1.0, 0.3),
            &scatter(&full, 1.0, 0.3),
        );
        let nl_e: f64 = nl.iter().map(|v| v * v).sum();
        assert!(
            nl_e > 1.0,
            "nonlinear AM residual {nl_e} should be substantial"
        );
    }

    #[test]
    fn cps_combine_matches_pulse_inversion() {
        let xp = drive(1.0, 1.0e6, 40.0e6, 64);
        let xn: Vec<f64> = xp.iter().map(|v| -v).collect();
        let ep = scatter(&xp, 1.0, 0.3);
        let en = scatter(&xn, 1.0, 0.3);
        let pi = pulse_inversion(&ep, &en);
        let cps = cps_combine(&[&ep, &en], &[1.0, 1.0]);
        assert_eq!(pi.len(), cps.len());
        assert!(pi.iter().zip(&cps).all(|(a, b)| (a - b).abs() < 1e-12));
        // guard rails
        assert!(cps_combine(&[&ep], &[1.0, 2.0]).is_empty());
    }
}
