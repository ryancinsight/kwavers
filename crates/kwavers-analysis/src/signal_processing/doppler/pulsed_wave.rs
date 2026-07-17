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

use apollo::{fft_1d_complex, Complex64};
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array1, ArrayView1};

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
            center_frequency: 5.0 * MHZ_TO_HZ, // 5 MHz
            prf: 4e3,                          // 4 kHz PRF
            sample_volume_depth: 0.05,         // 5 cm
            sample_volume_length: 0.005,       // 5 mm gate
            fft_size: 128,
            c_sound: SOUND_SPEED_TISSUE, // m/s
            beam_angle: 0.0,             // 0° (parallel to flow)
        }
    }
}

/// Spectral Doppler waveform — one-sided magnitude spectrum
///
/// Index k corresponds to Doppler frequency `f_d[k] = k · f_prf / fft_size`,
/// or equivalently velocity `v[k] = f_d[k] · c / (2 · f₀ · cos θ)`.
pub type SpectralWaveform = Array1<f64>;

/// Centered, two-sided pulsed-wave Doppler spectrum.
///
/// `frequency_hz`, `velocity_m_s`, and `power` share the same ascending signed
/// Doppler-bin order. The first bin is the most negative observable frequency;
/// the final bin is the largest positive frequency below Nyquist. Unlike the
/// legacy one-sided magnitude waveform, this type retains reverse-flow energy.
#[derive(Debug, Clone, PartialEq)]
pub struct SignedSpectralWaveform {
    /// Signed Doppler frequency bins in hertz.
    pub frequency_hz: Array1<f64>,
    /// Signed beam-projected velocity bins in metres per second.
    pub velocity_m_s: Array1<f64>,
    /// Squared complex-FFT magnitudes at the matching signed bins.
    pub power: Array1<f64>,
}

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
    /// Returns `InvalidInput` if `fft_size < 2`, the ensemble is empty, or the
    /// ensemble is longer than `fft_size`. The latter is rejected rather than
    /// silently discarding acquired pulses.
    pub fn extract_waveform(
        &self,
        iq_ensemble: ArrayView1<Complex64>,
    ) -> KwaversResult<SpectralWaveform> {
        let spectrum = fft_1d_complex(&self.windowed_iq(iq_ensemble)?);

        // One-sided magnitude spectrum.
        // Length = fft_size/2 + 1; bin k → f_d = k·f_prf/fft_size.
        let out_len = self.config.fft_size / 2 + 1;
        let waveform: SpectralWaveform =
            Array1::from_shape_fn([out_len], |idx| spectrum[idx[0]].norm());

        Ok(waveform)
    }

    /// Extract a centered, signed Doppler-power spectrum from range-gated I/Q.
    ///
    /// The shared preprocessing removes static clutter, applies a Hann window,
    /// and zero-pads before an FFT. The returned bins use the conventional
    /// centered order `[-f_prf/2, ..., 0, ..., +f_prf/2)`, preserving both
    /// approaching and receding flow components.
    ///
    /// # Errors
    ///
    /// Returns [`KwaversError::InvalidInput`] for empty or overlong I/Q, invalid
    /// physical Doppler parameters, or a beam angle perpendicular to flow.
    pub fn signed_spectrum(
        &self,
        iq_ensemble: ArrayView1<Complex64>,
    ) -> KwaversResult<SignedSpectralWaveform> {
        let velocity_per_hz = self.signed_velocity_per_hz()?;
        let fft_size = self.config.fft_size;
        let fft_size_i32 = i32::try_from(fft_size).map_err(|_| {
            KwaversError::InvalidInput("fft_size exceeds signed spectral indexing range".to_owned())
        })?;
        let spectrum = fft_1d_complex(&self.windowed_iq(iq_ensemble)?);
        let midpoint = fft_size_i32 / 2;
        let bin_width_hz = self.config.prf / f64::from(fft_size_i32);
        let frequency_hz = Array1::from_shape_vec(
            [fft_size],
            (-midpoint..(fft_size_i32 - midpoint))
                .map(|signed_bin| f64::from(signed_bin) * bin_width_hz)
                .collect(),
        )
        .map_err(|error| {
            KwaversError::InvalidInput(format!("invalid signed spectrum shape: {error}"))
        })?;
        let velocity_m_s = frequency_hz.mapv(|frequency_hz| frequency_hz * velocity_per_hz);
        let power = Array1::from_shape_fn([fft_size], |index| {
            let source_index = (index[0] + fft_size - fft_size / 2) % fft_size;
            spectrum[source_index].norm_sqr()
        });

        Ok(SignedSpectralWaveform {
            frequency_hz,
            velocity_m_s,
            power,
        })
    }

    /// Velocity axis corresponding to `extract_waveform` output (m/s).
    ///
    /// Uses the Doppler equation: `v[k] = k·f_prf/fft_size · c / (2·f₀·cos θ)`.
    /// Maximum alias-free velocity: `v_max = f_prf·c/(4·f₀·cos θ)`.
    #[must_use]
    pub fn velocity_axis(&self) -> Array1<f64> {
        let fft_size = self.config.fft_size;
        let out_len = fft_size / 2 + 1;
        let cos_theta = self.config.beam_angle.cos().abs().max(f64::EPSILON);
        let df_to_v = self.config.c_sound / (2.0 * self.config.center_frequency * cos_theta);
        let df = self.config.prf / fft_size as f64;

        Array1::from_shape_fn([out_len], |idx| idx[0] as f64 * df * df_to_v)
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
        let cos_theta = self.config.beam_angle.cos().abs().max(f64::EPSILON);
        self.config.prf * self.config.c_sound / (4.0 * self.config.center_frequency * cos_theta)
    }

    fn windowed_iq(&self, iq_ensemble: ArrayView1<Complex64>) -> KwaversResult<Array1<Complex64>> {
        let ensemble_len = iq_ensemble.size();
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
        if ensemble_len > fft_size {
            return Err(KwaversError::InvalidInput(
                "I/Q ensemble length must not exceed fft_size".to_owned(),
            ));
        }

        // Subtracting the ensemble mean removes the zero-frequency tissue term
        // before the spectral window is applied.
        let ensemble_len_f64 = f64::from(u32::try_from(ensemble_len).map_err(|_| {
            KwaversError::InvalidInput("I/Q ensemble exceeds supported window length".to_owned())
        })?);
        let mean: Complex64 = iq_ensemble.iter().sum::<Complex64>() / ensemble_len_f64;
        let window_denominator =
            f64::from(u32::try_from((ensemble_len - 1).max(1)).map_err(|_| {
                KwaversError::InvalidInput(
                    "I/Q ensemble exceeds supported window length".to_owned(),
                )
            })?);
        let mut windowed = Vec::with_capacity(fft_size);
        for n in 0..fft_size {
            if n >= ensemble_len {
                windowed.push(Complex64::new(0.0, 0.0));
                continue;
            }
            let sample_index = f64::from(u32::try_from(n).map_err(|_| {
                KwaversError::InvalidInput(
                    "I/Q ensemble exceeds supported window length".to_owned(),
                )
            })?);
            let window = 0.5 * (1.0 - (TWO_PI * sample_index / window_denominator).cos());
            windowed.push((iq_ensemble[n] - mean) * window);
        }
        Array1::from_shape_vec([fft_size], windowed)
            .map_err(|error| KwaversError::InvalidInput(format!("invalid window shape: {error}")))
    }

    fn signed_velocity_per_hz(&self) -> KwaversResult<f64> {
        for (name, value) in [
            ("center_frequency", self.config.center_frequency),
            ("prf", self.config.prf),
            ("c_sound", self.config.c_sound),
            ("sample_volume_depth", self.config.sample_volume_depth),
            ("sample_volume_length", self.config.sample_volume_length),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(KwaversError::InvalidInput(format!(
                    "{name} must be finite and positive"
                )));
            }
        }
        if !self.config.beam_angle.is_finite() {
            return Err(KwaversError::InvalidInput(
                "beam_angle must be finite".to_owned(),
            ));
        }
        let beam_alignment = self.config.beam_angle.cos();
        if beam_alignment.abs() <= f64::EPSILON {
            return Err(KwaversError::InvalidInput(
                "beam_angle must not be perpendicular to flow".to_owned(),
            ));
        }
        Ok(self.config.c_sound / (2.0 * self.config.center_frequency * beam_alignment))
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
        let ensemble: Array1<Complex64> = Array1::zeros([64]);

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

        let ensemble: Array1<Complex64> = Array1::from_shape_fn([n_samples], |idx| {
            let n = idx[0];
            let phase = 2.0 * PI * f_d * n as f64 / config.prf;
            Complex64::new(phase.cos(), phase.sin())
        });

        let waveform = pwd.extract_waveform(ensemble.view()).unwrap();

        let peak_bin = waveform
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
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
        let ensemble: Array1<Complex64> = Array1::from_shape_fn([64], |_| Complex64::new(1.0, 0.0));

        let waveform = pwd.extract_waveform(ensemble.view()).unwrap();
        assert_eq!(waveform.len(), config.fft_size / 2 + 1);
    }

    #[test]
    fn waveform_rejects_an_ensemble_longer_than_its_fft() {
        let pwd = PulsedWaveDoppler::new(PWDConfig {
            fft_size: 8,
            ..Default::default()
        });
        let ensemble = Array1::<Complex64>::zeros([9]);

        let error = pwd.extract_waveform(ensemble.view()).unwrap_err();
        let KwaversError::InvalidInput(message) = error else {
            panic!("expected an invalid I/Q ensemble error");
        };
        assert_eq!(message, "I/Q ensemble length must not exceed fft_size");
    }

    /// **Test: waveform is non-negative (magnitude spectrum)**
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_extract_waveform_non_negative() {
        let config = PWDConfig::default();
        let pwd = PulsedWaveDoppler::new(config);
        let ensemble: Array1<Complex64> = Array1::from_shape_fn([64], |idx| {
            let n = idx[0];
            Complex64::new((n as f64 * 0.1).sin(), (n as f64 * 0.1).cos())
        });

        let waveform = pwd.extract_waveform(ensemble.view()).unwrap();
        for (k, &v) in waveform.iter().enumerate() {
            assert!(v >= 0.0, "Waveform[{k}] = {v} < 0 (magnitude must be ≥ 0)");
        }
    }

    #[test]
    fn signed_spectrum_retains_reverse_flow_energy_and_velocity_axis() {
        let config = PWDConfig {
            fft_size: 64,
            prf: 4_096.0,
            center_frequency: 5.0 * MHZ_TO_HZ,
            c_sound: SOUND_SPEED_TISSUE,
            beam_angle: 0.0,
            ..Default::default()
        };
        let signed_bin = -7_i32;
        let fft_size = u32::try_from(config.fft_size)
            .expect("invariant: test FFT size fits the documented provider range");
        let doppler_frequency_hz = f64::from(signed_bin) * config.prf / f64::from(fft_size);
        let ensemble = Array1::<Complex64>::from_shape_fn([config.fft_size], |index| {
            let sample =
                f64::from(u32::try_from(index[0]).expect("invariant: test FFT index fits u32"));
            let phase = TWO_PI * doppler_frequency_hz * sample / config.prf;
            Complex64::new(phase.cos(), phase.sin())
        });
        let spectrum = PulsedWaveDoppler::new(config.clone())
            .signed_spectrum(ensemble.view())
            .unwrap();
        let (peak_index, peak_power) = spectrum
            .power
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .unwrap();
        let expected_velocity_m_s =
            doppler_frequency_hz * config.c_sound / (2.0 * config.center_frequency);
        // A radix-2 FFT applies O(N log₂N) floating-point operations per bin.
        // γ₁₀₂₄ bounds this 64-sample spectral-axis comparison conservatively.
        let roundoff = 1024.0 * f64::EPSILON;
        let velocity_bound = expected_velocity_m_s.abs() * roundoff / (1.0 - roundoff);

        assert_eq!(peak_index, 25);
        assert!(peak_power > 0.0);
        assert_eq!(spectrum.frequency_hz[0], -config.prf / 2.0);
        assert_eq!(
            spectrum.frequency_hz[spectrum.frequency_hz.len() - 1],
            config.prf / 2.0 - config.prf / f64::from(fft_size)
        );
        assert!(spectrum.velocity_m_s[peak_index] < 0.0);
        assert!(
            (spectrum.velocity_m_s[peak_index] - expected_velocity_m_s).abs() <= velocity_bound
        );
    }

    #[test]
    fn signed_spectrum_rejects_perpendicular_beam_geometry() {
        let config = PWDConfig {
            beam_angle: std::f64::consts::FRAC_PI_2,
            ..Default::default()
        };
        let ensemble = Array1::<Complex64>::zeros([2]);

        let error = PulsedWaveDoppler::new(config)
            .signed_spectrum(ensemble.view())
            .unwrap_err();
        let KwaversError::InvalidInput(message) = error else {
            panic!("expected a perpendicular beam-angle error");
        };
        assert_eq!(message, "beam_angle must not be perpendicular to flow");
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
            center_frequency: 5.0 * MHZ_TO_HZ,
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
        let expected_v_max = 4000.0 * SOUND_SPEED_TISSUE / (4.0 * 5.0 * MHZ_TO_HZ);
        let rel_err = (v_max - expected_v_max).abs() / expected_v_max;
        assert!(
            rel_err < 1e-10,
            "v_max = {v_max:.4e} m/s, expected {expected_v_max:.4e} m/s"
        );

        // Last bin of velocity axis should equal v_max
        let last_v = v_axis[v_axis.len() - 1];
        assert!(
            (last_v - v_max).abs() < 1e-10,
            "Last velocity bin {last_v:.4e} should equal v_max {v_max:.4e}"
        );
    }
}
