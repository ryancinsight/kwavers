use std::f64::consts::PI;

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
#[must_use]
pub fn bubble_power_spectrum(r_arr: &[f64], dt_s: f64, n_fft: usize) -> (Vec<f64>, Vec<f64>) {
    let n = n_fft;
    let n_f = n as f64;
    let mut padded = vec![0.0_f64; n];
    for (i, &v) in r_arr.iter().enumerate().take(n) {
        padded[i] = v;
    }
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
        let scale = if k == 0 || k == n / 2 { 1.0 } else { 2.0 };
        power[k] = scale * mag_sq * dt_s / n_f;
        f_arr[k] = k as f64 / (n_f * dt_s);
    }

    (f_arr, power)
}

/// Subharmonic period-doubling ratio from a bubble power spectrum.
///
/// Computes the spectral energy ratio of the half-harmonic (f₀/2) to the
/// fundamental (f₀) — a passive acoustic marker of inertial cavitation:
/// ```text
/// PD_ratio = S(f₀/2) / S(f₀)
/// ```
/// Each band is integrated over a ±1-bin window. Values above ~0.1 indicate
/// onset of subharmonic emission consistent with histotripsy bubble activity.
///
/// # Reference
/// Cramer et al. (2021), *Ultrasound Med. Biol.* 47, 2102.
#[must_use]
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
