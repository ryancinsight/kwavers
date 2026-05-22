//! Pulsed-Wave Doppler — Spectral Velocity Waveform Extraction
//!
//! ## Mathematical Foundation
//!
//! ### IQ Demodulation and Range Gating
//!
//! In pulsed-wave (PW) Doppler, the transducer emits short pulses at pulse
//! repetition frequency `f_prf`.  After range gating, the received signal at
//! sample-volume depth `z` consists of a complex I/Q ensemble:
//!
//! ```text
//! ŝ[n] = A[n] · exp(j·φ[n])   n = 0, 1, …, N−1
//! ```
//!
//! where `A[n]` is the echo amplitude and `φ[n]` is the phase due to scatterer
//! motion.  The Doppler frequency shift relates to blood velocity by:
//!
//! ```text
//! f_d = (2 · f₀ · v · cos θ) / c
//! v   = f_d · c / (2 · f₀ · cos θ)
//! ```
//!
//! ### Spectral Estimation
//!
//! The velocity spectrum at the sample volume is obtained by:
//!
//! 1. Apply Hann window to the ensemble `ŝ[n]·w[n]`
//! 2. Zero-pad to `fft_size` if the ensemble is shorter
//! 3. Take the FFT: `Ŝ[k] = FFT(ŝ[n]·w[n])`
//! 4. The one-sided magnitude spectrum `|Ŝ[k]|` represents the power at each
//!    Doppler frequency (and hence each velocity component)
//!
//! Velocity range: `±v_max` where `v_max = f_prf · c / (4 · f₀)`
//! (Nyquist limit for PW Doppler, alias-free range).
//!
//! ### Wall Filter
//!
//! Before spectral analysis, a high-pass wall filter (mean subtraction or
//! polynomial regression) removes the strong low-frequency clutter from
//! vessel walls and tissue.  This module uses the `WallFilter` from
//! `super::wall_filter`.
//!
//! ## References
//! - Evans DH, McDicken WN (2000). *Doppler Ultrasound: Physics, Instrumentation
//!   and Signal Processing* (2nd ed.). Wiley.  Chapters 3–5.
//! - Jensen JA (1996). *Estimation of Blood Velocities Using Ultrasound*.
//!   Cambridge University Press.  §4.3 (PW Doppler), §6.2 (spectral estimation).
//! - Kasai C et al. (1985). "Real-time two-dimensional blood flow imaging using
//!   an autocorrelation technique." *IEEE Trans Sonics Ultrason* 32(3):458–464.

use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
use crate::core::error::{KwaversError, KwaversResult};
use crate::math::fft::{fft_1d_complex, Complex64};
use ndarray::{Array1, ArrayView1};
use std::f64::consts::PI;

/// Pulsed-wave Doppler configuration
#[derive(Debug, Clone)]
pub struct PWDConfig {
    /// Transducer centre frequency (Hz)
    pub center_frequency: f64,
    /// Pulse repetition frequency (Hz)
    pub prf: f64,
    /// Sample volume depth (m)
    pub sample_volume_depth: f64,
    /// Sample volume gate length (m)
    pub sample_volume_length: f64,
    /// FFT length for spectral estimation (power of 2 recommended)
    pub fft_size: usize,
    /// Speed of sound (m/s)
    pub c_sound: f64,
    /// Beam-to-flow angle (radians) (used for velocity axis scaling)
    pub beam_angle: f64,
}

impl Default for PWDConfig {
    fn default() -> Self {
        Self {
            center_frequency: 5.0e6,     // 5 MHz
            prf: 4e3,                    // 4 kHz PRF
            sample_volume_depth: 0.05,   // 5 cm
            sample_volume_length: 0.005, // 5 mm gate
            fft_size: 128,
            c_sound: SOUND_SPEED_TISSUE, // m/s
            beam_angle: 0.0, // 0° (parallel to flow)
        }
    }
}

/// Spectral Doppler waveform — one-sided magnitude spectrum
///
/// Index k corresponds to Doppler frequency `f_d[k] = k · f_prf / fft_size`,
/// or equivalently velocity `v[k] = f_d[k] · c / (2 · f₀ · cos θ)`.
pub type SpectralWaveform = Array1<f64>;

/// Pulsed-wave Doppler processor
#[derive(Debug, Clone)]
pub struct PulsedWaveDoppler {
    config: PWDConfig,
}

impl PulsedWaveDoppler {
    #[must_use]
    pub fn new(config: PWDConfig) -> Self {
        Self { config }
    }

    /// Extract the Doppler velocity spectrum from a range-gated I/Q ensemble.
    ///
    /// # Algorithm (Evans & McDicken 2000, §5.2)
    ///
    /// 1. Subtract ensemble mean (high-pass wall filter — DC clutter removal)
    /// 2. Apply Hann window to suppress spectral leakage
    /// 3. Zero-pad to `config.fft_size` if ensemble is shorter
    /// 4. FFT → one-sided magnitude spectrum `|Ŝ[k]|`
    ///
    /// # Arguments
    /// * `iq_ensemble` – Complex I/Q samples at the sample volume, one per PW
    ///   pulse (length = ensemble size, typically 64–256 pulses)
    ///
    /// # Returns
    /// One-sided magnitude spectrum of length `fft_size/2 + 1`.  Bin k
    /// corresponds to Doppler frequency `f_d = k · f_prf / fft_size` and
    /// velocity `v = f_d · c / (2 · f₀ · cos θ)`.
    ///
    /// # Errors
    /// Returns `InvalidInput` if `fft_size < 2` or the ensemble is empty.
    pub fn extract_waveform(
        &self,
        iq_ensemble: ArrayView1<Complex64>,
    ) -> KwaversResult<SpectralWaveform> {
        let ensemble_len = iq_ensemble.len();

        if ensemble_len == 0 {
            return Err(KwaversError::InvalidInput(
                "I/Q ensemble must be non-empty".to_owned(),
            ));
        }

        let fft_size = self.config.fft_size;
        if fft_size < 2 {
            return Err(KwaversError::InvalidInput(
                "fft_size must be ≥ 2".to_owned(),
            ));
        }

        // ── Step 1: High-pass wall filter — subtract ensemble mean ────────────
        // Removes DC clutter from slow-moving vessel walls and stationary tissue.
        let mean: Complex64 = iq_ensemble.iter().sum::<Complex64>() / ensemble_len as f64;
        let filtered: Vec<Complex64> = iq_ensemble.iter().map(|&s| s - mean).collect();

        // ── Step 2: Hann window + zero-pad to fft_size ────────────────────────
        // Window length = min(ensemble_len, fft_size); zero-pad remaining.
        let win_len = ensemble_len.min(fft_size);
        let mut windowed = Array1::<Complex64>::zeros(fft_size);

        for n in 0..win_len {
            let w = 0.5 * (1.0 - (2.0 * PI * n as f64 / (win_len - 1).max(1) as f64).cos());
            windowed[n] = filtered[n] * w;
        }

        // ── Step 3: FFT ───────────────────────────────────────────────────────
        let spectrum = fft_1d_complex(&windowed);

        // ── Step 4: One-sided magnitude spectrum ──────────────────────────────
        // Length = fft_size/2 + 1; bin k → f_d = k·f_prf/fft_size.
        let out_len = fft_size / 2 + 1;
        let waveform: SpectralWaveform = Array1::from_shape_fn(out_len, |k| spectrum[k].norm());

        Ok(waveform)
    }

    /// Velocity axis corresponding to `extract_waveform` output (m/s).
    ///
    /// Uses the Doppler equation: `v[k] = k·f_prf/fft_size · c / (2·f₀·cos θ)`.
    /// Maximum alias-free velocity: `v_max = f_prf·c/(4·f₀·cos θ)`.
    #[must_use]
    pub fn velocity_axis(&self) -> Array1<f64> {
        let fft_size = self.config.fft_size;
        let out_len = fft_size / 2 + 1;
        let cos_theta = self.config.beam_angle.cos().max(1e-6); // avoid division by zero
        let df_to_v = self.config.c_sound / (2.0 * self.config.center_frequency * cos_theta);
        let df = self.config.prf / fft_size as f64;

        Array1::from_shape_fn(out_len, |k| k as f64 * df * df_to_v)
    }

    /// Maximum alias-free velocity (m/s) (Nyquist limit for PW Doppler).
    ///
    /// ```text
    /// v_max = f_prf · c / (4 · f₀ · cos θ)
    /// ```
    ///
    /// Above this velocity, Doppler aliasing occurs (Evans & McDicken 2000, §3.5).
    #[must_use]
    pub fn max_velocity(&self) -> f64 {
        let cos_theta = self.config.beam_angle.cos().max(1e-6);
        self.config.prf * self.config.c_sound / (4.0 * self.config.center_frequency * cos_theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// **Test: zero ensemble produces zero-magnitude waveform**
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_extract_waveform_zero_signal() {
        let config = PWDConfig::default();
        let pwd = PulsedWaveDoppler::new(config);
        let ensemble: Array1<Complex64> = Array1::zeros(64);

        let waveform = pwd.extract_waveform(ensemble.view()).unwrap();
        assert!(
            waveform.iter().all(|&v| v == 0.0),
            "Waveform of zero ensemble should be all-zero"
        );
    }

    /// **Test: single-frequency complex exponential peaks at correct Doppler bin**
    ///
    /// Signal: `s[n] = exp(j 2π f_d n / f_prf)` with `f_d = k₀ · f_prf / fft_size`.
    /// Expected peak: bin `k₀`.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_extract_waveform_single_tone_peak() {
        let config = PWDConfig {
            fft_size: 128,
            prf: 4_000.0,
            ..Default::default()
        };
        let pwd = PulsedWaveDoppler::new(config.clone());

        // Tone at bin k=10 of the 128-point FFT with PRF=4kHz
        // f_d = 10 * 4000 / 128 = 312.5 Hz
        let k0 = 10usize;
        let f_d = k0 as f64 * config.prf / config.fft_size as f64;
        let n_samples = 128usize;

        let ensemble: Array1<Complex64> = Array1::from_shape_fn(n_samples, |n| {
            let phase = 2.0 * PI * f_d * n as f64 / config.prf;
            Complex64::new(phase.cos(), phase.sin())
        });

        let waveform = pwd.extract_waveform(ensemble.view()).unwrap();

        let peak_bin = waveform
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(
            peak_bin, k0,
            "Doppler waveform peak at bin {peak_bin}, expected bin {k0} (f_d={f_d:.1} Hz)"
        );
    }

    /// **Test: waveform output has correct length**
    ///
    /// Expected length = `fft_size/2 + 1` (one-sided spectrum).
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_extract_waveform_output_length() {
        let config = PWDConfig {
            fft_size: 128,
            ..Default::default()
        };
        let pwd = PulsedWaveDoppler::new(config.clone());
        let ensemble: Array1<Complex64> = Array1::from_shape_fn(64, |_| Complex64::new(1.0, 0.0));

        let waveform = pwd.extract_waveform(ensemble.view()).unwrap();
        assert_eq!(waveform.len(), config.fft_size / 2 + 1);
    }

    /// **Test: waveform is non-negative (magnitude spectrum)**
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_extract_waveform_non_negative() {
        let config = PWDConfig::default();
        let pwd = PulsedWaveDoppler::new(config);
        let ensemble: Array1<Complex64> = Array1::from_shape_fn(64, |n| {
            Complex64::new((n as f64 * 0.1).sin(), (n as f64 * 0.1).cos())
        });

        let waveform = pwd.extract_waveform(ensemble.view()).unwrap();
        for (k, &v) in waveform.iter().enumerate() {
            assert!(v >= 0.0, "Waveform[{k}] = {v} < 0 (magnitude must be ≥ 0)");
        }
    }

    /// **Test: velocity axis has correct length and maximum velocity (Nyquist)**
    ///
    /// For a 5 MHz probe with f_prf = 4 kHz and c = 1540 m/s at 0° angle:
    /// `v_max = f_prf · c / (4 · f₀) = 4000 × 1540 / (4 × 5e6) = 0.308 m/s`
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_velocity_axis_nyquist_limit() {
        let config = PWDConfig {
            center_frequency: 5e6,
            prf: 4e3,
            c_sound: SOUND_SPEED_TISSUE,
            beam_angle: 0.0,
            fft_size: 128,
            ..Default::default()
        };
        let pwd = PulsedWaveDoppler::new(config.clone());

        let v_axis = pwd.velocity_axis();
        assert_eq!(v_axis.len(), config.fft_size / 2 + 1);

        let v_max = pwd.max_velocity();
        let expected_v_max = 4000.0 * SOUND_SPEED_TISSUE / (4.0 * 5e6);
        let rel_err = (v_max - expected_v_max).abs() / expected_v_max;
        assert!(
            rel_err < 1e-10,
            "v_max = {v_max:.4e} m/s, expected {expected_v_max:.4e} m/s"
        );

        // Last bin of velocity axis should equal v_max
        let last_v = *v_axis.last().unwrap();
        assert!(
            (last_v - v_max).abs() < 1e-10,
            "Last velocity bin {last_v:.4e} should equal v_max {v_max:.4e}"
        );
    }
}
