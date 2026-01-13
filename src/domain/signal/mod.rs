// signal/mod.rs
//! Signal generation and processing module
//!
//! Comprehensive signal generation library including:
//! - Basic waveforms (sine, square, triangle)
//! - Pulse signals (Gaussian, rectangular, tone burst, Ricker)
//! - Frequency sweeps (linear, logarithmic, hyperbolic)
//! - Modulation techniques (AM, FM, PM, QAM, PWM)
//! - Windowing functions

use self::window::get_win;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

pub mod amplitude;
pub mod analytic;
pub mod filter;
pub mod frequency;
pub mod frequency_sweep;
pub mod modulation;
pub mod phase;
pub mod pulse;
pub mod special;
pub mod waveform;
pub mod window;

pub use filter::Filter;
pub use special::{NullSignal, TimeVaryingSignal};

// Core Signal trait moved to domain
// Core Signal trait
pub mod traits;
pub use traits::Signal;

// Clone impl is in traits.rs

// Re-export commonly used signal types
pub use waveform::{SineWave, SquareWave, TriangleWave};
pub use window::{window_value, WindowType};

// Re-export pulse signals
pub use pulse::{
    GaussianPulse, PulseShape, PulseTrain, RectangularPulse, RickerWavelet, ToneBurst,
};

// Re-export frequency sweeps
pub use frequency_sweep::{
    ExponentialSweep, FrequencySweep, HyperbolicSweep, LinearChirp, LogarithmicSweep, SteppedSweep,
    SweepConfig, SweepDirection, SweepType,
};

// Re-export modulation types
pub use modulation::{
    AmplitudeModulation, FrequencyModulation, PhaseModulation, PulseWidthModulation,
    QuadratureAmplitudeModulation,
};

pub fn sample_signal(signal: &dyn Signal, dt: f64, n: usize, t0: f64) -> KwaversResult<Vec<f64>> {
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
        .map(|i| signal.amplitude(t0 + i as f64 * dt))
        .collect())
}

#[must_use]
pub fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

#[must_use]
pub fn pad_zeros(signal: &[f64], target_len: usize) -> Vec<f64> {
    let mut padded = vec![0.0; target_len];
    let copy_len = signal.len().min(target_len);
    padded[..copy_len].copy_from_slice(&signal[..copy_len]);
    padded
}

pub fn tone_burst_series(
    sample_rate_hz: f64,
    signal_freq_hz: f64,
    num_cycles: f64,
    signal_offset: usize,
    signal_length: Option<usize>,
    window: WindowType,
    amplitude: f64,
    phase: f64,
) -> KwaversResult<Vec<f64>> {
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

    let burst_samples = ((num_cycles * sample_rate_hz) / signal_freq_hz).round();
    if !burst_samples.is_finite() || burst_samples <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "tone burst has non-positive sample count ({burst_samples})"
        )));
    }
    let burst_samples = burst_samples as usize;

    let required_len = signal_offset + burst_samples;
    let out_len = signal_length.unwrap_or(required_len);
    if out_len < required_len {
        return Err(KwaversError::InvalidInput(format!(
            "signal_length {out_len} is less than required length {required_len}"
        )));
    }

    let dt = 1.0 / sample_rate_hz;
    let win = get_win(window, burst_samples, true);

    let mut out = vec![0.0; out_len];
    for i in 0..burst_samples {
        let t = i as f64 * dt;
        let carrier = (2.0 * PI * signal_freq_hz * t + phase).sin();
        out[signal_offset + i] = amplitude * win[i] * carrier;
    }
    Ok(out)
}

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
            "time vector contains non-finite values".to_string(),
        ));
    }

    Ok(t.iter()
        .map(|&ti| amplitude * (2.0 * PI * frequency_hz * ti + phase).sin())
        .collect())
}

pub fn create_cw_signals(
    t: &[f64],
    frequency_hz: f64,
    amplitudes: &[f64],
    phases: &[f64],
) -> KwaversResult<Array2<f64>> {
    if amplitudes.is_empty() || phases.is_empty() {
        return Err(KwaversError::InvalidInput(
            "amplitudes and phases must be non-empty".to_string(),
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

    let mut out = Array2::<f64>::zeros((n_signals, t.len()));
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
        out.row_mut(s).assign(&ndarray::Array1::from_vec(row));
    }
    Ok(out)
}

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
            "signal contains non-finite values".to_string(),
        ));
    }

    let signal_power = signal.iter().map(|&x| x * x).sum::<f64>() / signal.len() as f64;
    if signal_power == 0.0 {
        return Err(KwaversError::InvalidInput(
            "cannot add noise to zero-power signal".to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn tone_burst_series_respects_offset_and_length() {
        let y = tone_burst_series(
            1_000.0,
            100.0,
            2.0,
            10,
            Some(40),
            WindowType::Hann,
            1.0,
            0.0,
        )
        .unwrap();
        assert_eq!(y.len(), 40);
        assert!(y[..10].iter().all(|&v| v == 0.0));
        assert!(y[10..].iter().any(|&v| v != 0.0));
    }

    #[test]
    fn add_noise_achieves_target_snr_reasonably() {
        let sample_rate_hz = 10_000.0;
        let f0 = 100.0;
        let n = 65_536;
        let dt = 1.0 / sample_rate_hz;
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * (i as f64 * dt)).sin())
            .collect();

        let snr_db_target = 20.0;
        let noisy = add_noise(&clean, snr_db_target, Some(123)).unwrap();
        let noise: Vec<f64> = noisy
            .iter()
            .zip(clean.iter())
            .map(|(&y, &x)| y - x)
            .collect();

        let signal_power = clean.iter().map(|&x| x * x).sum::<f64>() / n as f64;
        let noise_power = noise.iter().map(|&x| x * x).sum::<f64>() / n as f64;
        let snr_db_measured = 10.0 * (signal_power / noise_power).log10();

        assert!((snr_db_measured - snr_db_target).abs() < 0.5);
    }

    #[test]
    fn create_cw_signals_broadcasts_phase() {
        let t = [0.0, 0.25, 0.5, 0.75];
        let out = create_cw_signals(&t, 1.0, &[1.0, 2.0], &[0.0]).unwrap();
        assert_eq!(out.dim(), (2, 4));
        assert!((out[[0, 1]] - 1.0).abs() < 1e-12);
        assert!((out[[1, 1]] - 2.0).abs() < 1e-12);
    }

    proptest! {
        #[test]
        fn next_pow2_is_power_of_two_and_ge_n(n in 1usize..(1<<20)) {
            let p = next_pow2(n);
            prop_assert!(p >= n);
            prop_assert!(p.is_power_of_two());
        }

        #[test]
        fn pad_zeros_preserves_prefix(n in 0usize..2048, m in 0usize..2048) {
            let src: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let dst = pad_zeros(&src, m);
            prop_assert_eq!(dst.len(), m);
            let k = n.min(m);
            prop_assert_eq!(&dst[..k], &src[..k]);
        }
    }
}
