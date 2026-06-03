//! Hann-windowed single-sided power spectral density for emission analysis.
//!
//! The rectangular-window DFT of a strongly harmonic emission leaks energy from
//! the dominant harmonic lines into every other bin, which would corrupt the
//! broadband (inertial) band of a passive-cavitation-dose decomposition. A Hann
//! window suppresses that spectral leakage (>30 dB sidelobe attenuation) so the
//! inter-line floor reflects genuine inharmonic emission. This is the standard
//! estimator used for passive-cavitation spectroscopy (Gyöngy & Coussios 2010).

use kwavers_core::constants::numerical::TWO_PI;

/// Single-sided Hann-windowed power spectral density of an emission series.
///
/// ```text
///   w[j]  = ½·(1 − cos(2πj/(N−1)))                 (Hann window)
///   X[k]  = Σ_j (x[j] − x̄)·w[j]·exp(−i2πkj/N)
///   S[k]  = c_k · |X[k]|² · Δt / Σ_j w[j]²          c_k = 1 (DC/Nyquist) else 2
/// ```
/// The `Σ w²` normalisation makes `S` a true power-preserving PSD comparable to
/// the rectangular-window [`super::super::bubble_power_spectrum`]; the band
/// energies that [`super::decompose_emission_spectrum`] integrates from it are
/// therefore leakage-suppressed.
///
/// # Arguments
/// * `signal` – emission time series (e.g. from [`super::bubble_acoustic_emission_pressure`])
/// * `dt_s`   – uniform sample interval [s]
/// * `n_fft`  – DFT length (≥ `signal.len()`; zero-padded)
///
/// Returns `(f_arr [Hz], psd)` over non-negative frequencies. Returns a pair of
/// empty vectors if `n_fft < 2` or `dt_s ≤ 0`.
///
/// # Note
/// Direct O(N²) DFT — exact, slow for large N. Callers decimate to a few
/// thousand samples before invoking.
#[must_use]
pub fn hann_windowed_power_spectrum(
    signal: &[f64],
    dt_s: f64,
    n_fft: usize,
) -> (Vec<f64>, Vec<f64>) {
    if n_fft < 2 || !(dt_s.is_finite() && dt_s > 0.0) {
        return (Vec::new(), Vec::new());
    }
    let n = n_fft;
    let n_f = n as f64;

    // Windowed, mean-removed, zero-padded record.
    let mut x = vec![0.0_f64; n];
    let count = signal.len().min(n);
    let mean: f64 = signal.iter().take(count).sum::<f64>() / count.max(1) as f64;
    let mut w_power = 0.0_f64; // Σ w² over the *populated* samples
    for j in 0..count {
        // Hann window across the populated record length `count`.
        let w = if count > 1 {
            0.5 * (1.0 - (TWO_PI * j as f64 / (count as f64 - 1.0)).cos())
        } else {
            1.0
        };
        x[j] = (signal[j] - mean) * w;
        w_power += w * w;
    }
    if w_power <= 0.0 {
        return (vec![0.0; n / 2 + 1], vec![0.0; n / 2 + 1]);
    }

    let n_pos = n / 2 + 1;
    let mut f_arr = vec![0.0_f64; n_pos];
    let mut psd = vec![0.0_f64; n_pos];
    for k in 0..n_pos {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        let phase_step = TWO_PI * k as f64 / n_f;
        for (j, &xj) in x.iter().enumerate() {
            let phi = phase_step * j as f64;
            re += xj * phi.cos();
            im -= xj * phi.sin();
        }
        let mag_sq = re * re + im * im;
        let scale = if k == 0 || k == n / 2 { 1.0 } else { 2.0 };
        psd[k] = scale * mag_sq * dt_s / w_power;
        f_arr[k] = k as f64 / (n_f * dt_s);
    }
    (f_arr, psd)
}
