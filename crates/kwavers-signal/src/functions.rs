//! Free-standing signal utility and generation functions.

use super::traits::Signal;
use super::window::{get_win, SignalWindowType};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Sample a signal at evenly spaced time points starting at `t0`.
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
///
pub fn sample_signal<S: Signal + ?Sized>(
    signal: &S,
    dt: f64,
    n: usize,
    t0: f64,
) -> KwaversResult<Vec<f64>> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "dt must be finite and > 0, got {dt}"
        )));
    }
    if !t0.is_finite() {
        return Err(KwaversError::InvalidInput(format!(
            "t0 must be finite, got {t0}"
        )));
    }

    Ok((0..n)
        .map(|i| signal.amplitude((i as f64).mul_add(dt, t0)))
        .collect())
}

/// Return the smallest power of two ≥ `n`.
#[must_use]
pub fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Zero-pad `signal` to `target_len`, truncating if longer.
#[must_use]
pub fn pad_zeros(signal: &[f64], target_len: usize) -> Vec<f64> {
    let mut padded = vec![0.0; target_len];
    let copy_len = signal.len().min(target_len);
    padded[..copy_len].copy_from_slice(&signal[..copy_len]);
    padded
}

/// Compute the discrete sample count for a tone burst.
///
/// For positive `sample_rate_hz`, `signal_freq_hz`, and `num_cycles` the burst
/// length is `floor(num_cycles * sample_rate_hz / signal_freq_hz) + 1`.
#[must_use]
pub(super) fn tone_burst_sample_count(
    sample_rate_hz: f64,
    signal_freq_hz: f64,
    num_cycles: f64,
) -> usize {
    (num_cycles * sample_rate_hz / signal_freq_hz)
        .floor()
        .max(0.0) as usize
        + 1
}

/// Tukey (cosine-tapered) window aligned to k-Wave conventions.
#[must_use]
pub(super) fn k_wave_tukey_window(n: usize, alpha: f64) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    if alpha <= 0.0 {
        return vec![1.0; n];
    }
    if alpha >= 1.0 {
        return get_win(SignalWindowType::Hann, n, true);
    }

    let mut window = vec![1.0; n];
    let taper_len = (((n - 1) as f64 * alpha / 2.0) + 1e-8).floor() as usize + 1;
    let taper_len = taper_len.min(n);
    let alpha_n = alpha * n as f64;

    for i in 0..taper_len {
        let idx = i as f64;
        let value = 0.5 * (1.0 + (TWO_PI / alpha_n * (idx - alpha_n / 2.0)).cos());
        window[i] = value;
        window[n - 1 - i] = value;
    }

    window
}

/// Gaussian burst window aligned to k-wave-python's `tone_burst`.
#[must_use]
pub(super) fn k_wave_gaussian_burst_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }

    let step = 6.0 / (n - 1) as f64;
    let taper = k_wave_tukey_window(n, 0.05);

    (0..n)
        .map(|i| {
            let x = (i as f64).mul_add(step, -3.0);
            (-0.5 * x * x).exp() * taper[i]
        })
        .collect()
}

/// Complete specification for a tone-burst signal aligned to k-Wave conventions.
///
/// Groups all parameters required by [`tone_burst_series`] into a single struct,
/// eliminating positional argument confusion and satisfying the `too_many_arguments`
/// constraint at the call site.
#[derive(Debug, Clone, PartialEq)]
pub struct ToneBurstSpec {
    /// Sampling frequency in Hz. Must be finite and > 0.
    pub sample_rate_hz: f64,
    /// Carrier frequency in Hz. Must be finite and > 0.
    pub signal_freq_hz: f64,
    /// Number of complete oscillation cycles. Must be finite and > 0.
    pub num_cycles: f64,
    /// Number of leading zero-valued samples before the burst onset.
    pub signal_offset: usize,
    /// Total output length; defaults to `signal_offset + burst_samples` when `None`.
    pub signal_length: Option<usize>,
    /// Envelope window applied to the burst region.
    pub window: SignalWindowType,
    /// Peak amplitude. Must be finite and ≥ 0.
    pub amplitude: f64,
    /// Initial carrier phase in radians. Must be finite.
    pub phase: f64,
}

impl ToneBurstSpec {
    /// Construct a minimal burst spec with unit amplitude, zero offset, and a Hann window.
    #[must_use]
    pub fn new(sample_rate_hz: f64, signal_freq_hz: f64, num_cycles: f64) -> Self {
        Self {
            sample_rate_hz,
            signal_freq_hz,
            num_cycles,
            signal_offset: 0,
            signal_length: None,
            window: SignalWindowType::Hann,
            amplitude: 1.0,
            phase: 0.0,
        }
    }
}

/// Generate a tone-burst signal with an envelope aligned to k-wave-python.
///
/// ## Theorem: Sample Count
/// For positive `sample_rate_hz`, `signal_freq_hz`, and `num_cycles`, the
/// discrete burst length is `floor(num_cycles * sample_rate_hz / signal_freq_hz) + 1`.
///
/// ## Reference
/// `external/k-wave-python/kwave/utils/signals.py::tone_burst`
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
///
pub fn tone_burst_series(spec: &ToneBurstSpec) -> KwaversResult<Vec<f64>> {
    let ToneBurstSpec {
        sample_rate_hz,
        signal_freq_hz,
        num_cycles,
        signal_offset,
        signal_length,
        window,
        amplitude,
        phase,
    } = spec;
    let (sample_rate_hz, signal_freq_hz, num_cycles, amplitude, phase) = (
        *sample_rate_hz,
        *signal_freq_hz,
        *num_cycles,
        *amplitude,
        *phase,
    );
    let (signal_offset, signal_length) = (*signal_offset, *signal_length);

    if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "sample_rate_hz must be finite and > 0, got {sample_rate_hz}"
        )));
    }
    if !signal_freq_hz.is_finite() || signal_freq_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "signal_freq_hz must be finite and > 0, got {signal_freq_hz}"
        )));
    }
    if !num_cycles.is_finite() || num_cycles <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "num_cycles must be finite and > 0, got {num_cycles}"
        )));
    }
    if !amplitude.is_finite() || amplitude < 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "amplitude must be finite and >= 0, got {amplitude}"
        )));
    }
    if !phase.is_finite() {
        return Err(KwaversError::InvalidInput(format!(
            "phase must be finite, got {phase}"
        )));
    }

    let burst_samples = tone_burst_sample_count(sample_rate_hz, signal_freq_hz, num_cycles);
    if burst_samples == 0 {
        return Err(KwaversError::InvalidInput(
            "tone burst has zero sample count".to_owned(),
        ));
    }

    let required_len = signal_offset + burst_samples;
    let out_len = signal_length.unwrap_or(required_len);
    if out_len < required_len {
        return Err(KwaversError::InvalidInput(format!(
            "signal_length {out_len} is less than required length {required_len}"
        )));
    }

    let dt = 1.0 / sample_rate_hz;
    let win = match window {
        SignalWindowType::Gaussian => k_wave_gaussian_burst_window(burst_samples),
        _ => get_win(*window, burst_samples, true),
    };

    let mut out = vec![0.0; out_len];
    for i in 0..burst_samples {
        let t = i as f64 * dt;
        let carrier = (TWO_PI * signal_freq_hz).mul_add(t, phase).sin();
        out[signal_offset + i] = amplitude * win[i] * carrier;
    }
    Ok(out)
}

/// Generate a continuous-wave sinusoidal signal.
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
///
pub fn create_cw_signal(
    t: &[f64],
    frequency_hz: f64,
    amplitude: f64,
    phase: f64,
) -> KwaversResult<Vec<f64>> {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "frequency_hz must be finite and > 0, got {frequency_hz}"
        )));
    }
    if !amplitude.is_finite() {
        return Err(KwaversError::InvalidInput(format!(
            "amplitude must be finite, got {amplitude}"
        )));
    }
    if !phase.is_finite() {
        return Err(KwaversError::InvalidInput(format!(
            "phase must be finite, got {phase}"
        )));
    }
    if t.iter().any(|&x| !x.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "time vector contains non-finite values".to_owned(),
        ));
    }

    Ok(t.iter()
        .map(|&ti| amplitude * (TWO_PI * frequency_hz).mul_add(ti, phase).sin())
        .collect())
}

/// Generate multiple continuous-wave signals with broadcasting of amplitude and phase.
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn create_cw_signals(
    t: &[f64],
    frequency_hz: f64,
    amplitudes: &[f64],
    phases: &[f64],
) -> KwaversResult<Array2<f64>> {
    if amplitudes.is_empty() || phases.is_empty() {
        return Err(KwaversError::InvalidInput(
            "amplitudes and phases must be non-empty".to_owned(),
        ));
    }

    let n_signals = amplitudes.len().max(phases.len());
    if !(amplitudes.len() == 1 || amplitudes.len() == n_signals) {
        return Err(KwaversError::InvalidInput(format!(
            "amplitudes length must be 1 or {n_signals}, got {}",
            amplitudes.len()
        )));
    }
    if !(phases.len() == 1 || phases.len() == n_signals) {
        return Err(KwaversError::InvalidInput(format!(
            "phases length must be 1 or {n_signals}, got {}",
            phases.len()
        )));
    }

    let n_cols = t.len();
    let mut out = Array2::<f64>::zeros([n_signals, n_cols]);
    for s in 0..n_signals {
        let a = if amplitudes.len() == 1 {
            amplitudes[0]
        } else {
            amplitudes[s]
        };
        let p = if phases.len() == 1 {
            phases[0]
        } else {
            phases[s]
        };
        let row = create_cw_signal(t, frequency_hz, a, p)?;
        for (c, &val) in row.iter().enumerate() {
            out[[s, c]] = val;
        }
    }
    Ok(out)
}

/// Add Gaussian noise to a signal at a target SNR (dB).
/// # Errors
/// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn add_noise(signal: &[f64], snr_db: f64, seed: Option<u64>) -> KwaversResult<Vec<f64>> {
    if signal.is_empty() {
        return Ok(Vec::new());
    }
    if !snr_db.is_finite() {
        if snr_db.is_infinite() && snr_db.is_sign_positive() {
            return Ok(signal.to_vec());
        }
        return Err(KwaversError::InvalidInput(format!(
            "snr_db must be finite or +inf, got {snr_db}"
        )));
    }
    if signal.iter().any(|&x| !x.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "signal contains non-finite values".to_owned(),
        ));
    }

    let signal_power = signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64;
    if signal_power == 0.0 {
        return Err(KwaversError::InvalidInput(
            "cannot add noise to zero-power signal".to_owned(),
        ));
    }

    let snr_linear = 10.0_f64.powf(snr_db / 10.0);
    let noise_power = signal_power / snr_linear;
    if !noise_power.is_finite() || noise_power < 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "computed noise power is invalid: {noise_power}"
        )));
    }

    let sigma = noise_power.sqrt();
    let normal = Normal::new(0.0, sigma).map_err(|e| KwaversError::Other(anyhow::anyhow!(e)))?;
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    Ok(signal
        .iter()
        .map(|&x| x + normal.sample(&mut rng))
        .collect())
}
