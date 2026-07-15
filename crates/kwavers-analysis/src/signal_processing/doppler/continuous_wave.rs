//! Continuous-Wave (CW) Doppler.
//!
//! Continuous-wave Doppler insonifies with an unmodulated tone and listens on a
//! separate element. Because there is no pulsing, there is **no pulse-repetition
//! frequency and hence no Nyquist velocity limit**: the full Doppler spectrum is
//! recovered without aliasing, at the cost of range resolution (all scatterers
//! along the beam contribute). It is the method of choice for high-velocity jets
//! (aortic stenosis, regurgitation) that alias under pulsed-wave Doppler.
//!
//! # Method
//!
//! The received tone at frequency `f₀ + f_d` is quadrature-demodulated against
//! the transmit carrier `f₀`, producing a complex baseband whose instantaneous
//! frequency is the signed Doppler shift `f_d`. A short moving-average removes
//! the `2f₀` image. The baseband spectrum (two-sided, via FFT + `fftshift`) maps
//! bin frequency to signed velocity through the Doppler equation
//!
//! ```text
//! v = f_d · c / (2 · f₀ · cos θ).
//! ```
//!
//! The sign of `f_d` encodes flow direction (toward/away), preserved by the
//! quadrature demodulation.
//!
//! # References
//! - Evans, D. H., & McDicken, W. N. (2000). *Doppler Ultrasound: Physics,
//!   Instrumentation and Signal Processing* (2nd ed.). Wiley.
//! - Jensen, J. A. (1996). *Estimation of Blood Velocities Using Ultrasound*.
//!   Cambridge University Press.

use apollo::{fft_1d_complex, fftshift, Complex64};
use kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array1;

/// Continuous-wave Doppler configuration.
#[derive(Debug, Clone, Copy)]
pub struct CwDopplerConfig {
    /// Transmit (carrier) frequency `f₀` [Hz].
    pub center_frequency: f64,
    /// Receiver sampling rate `f_s` [Hz].
    pub sampling_rate: f64,
    /// Baseband (post-demodulation) analysis rate `f_bb` [Hz].
    ///
    /// The mixed-down signal is decimated to this rate before spectral analysis.
    /// It sets the velocity resolution (`Δv = c·f_bb/(2 f₀ N_bb)`) and the
    /// unambiguous velocity range (`±c·f_bb/(4 f₀)`). Because `f_bb` is chosen by
    /// the receiver — not tied to a pulse-repetition frequency or depth — CW
    /// Doppler avoids the pulsed-wave Nyquist limit.
    pub baseband_rate: f64,
    /// Speed of sound `c` [m/s].
    pub sound_speed: f64,
    /// Beam-to-flow angle `θ` [rad].
    pub angle: f64,
}

impl CwDopplerConfig {
    /// Construct with explicit parameters; `sound_speed` defaults to soft tissue
    /// and the baseband rate to 100 kHz (covers physiological velocities).
    #[must_use]
    pub fn new(center_frequency: f64, sampling_rate: f64, angle: f64) -> Self {
        Self {
            center_frequency,
            sampling_rate,
            baseband_rate: 100e3,
            sound_speed: SOUND_SPEED_TISSUE,
            angle,
        }
    }

    /// Decimation factor `f_s / f_bb` (≥ 1).
    #[must_use]
    pub fn decimation(&self) -> usize {
        ((self.sampling_rate / self.baseband_rate).round() as usize).max(1)
    }

    /// Convert a signed Doppler frequency [Hz] to a signed velocity [m/s].
    #[must_use]
    pub fn velocity_from_frequency(&self, f_d: f64) -> f64 {
        f_d * self.sound_speed / (2.0 * self.center_frequency * self.angle.cos())
    }
}

/// Two-sided Doppler spectrum: signed velocities and matching spectral power.
#[derive(Debug, Clone)]
pub struct CwSpectrum {
    /// Signed velocity axis [m/s], ascending (negative = away from transducer).
    pub velocity: Array1<f64>,
    /// Power at each velocity bin (|FFT|²).
    pub power: Array1<f64>,
}

impl CwSpectrum {
    /// Power-weighted mean velocity (spectral centroid) [m/s].
    #[must_use]
    pub fn mean_velocity(&self) -> f64 {
        let total: f64 = leto::sum_all(&self.power).unwrap_or(0.0);
        if total <= 0.0 {
            return 0.0;
        }
        self.velocity
            .iter()
            .zip(self.power.iter())
            .map(|(&v, &p)| v * p)
            .sum::<f64>()
            / total
    }

    /// Peak (modal) velocity [m/s] — the bin of maximum power.
    #[must_use]
    pub fn peak_velocity(&self) -> f64 {
        let mut best = 0usize;
        let mut best_p = f64::NEG_INFINITY;
        for (k, &p) in self.power.iter().enumerate() {
            if p > best_p {
                best_p = p;
                best = k;
            }
        }
        self.velocity[best]
    }
}

/// Continuous-wave Doppler processor.
#[derive(Debug, Clone, Copy)]
pub struct ContinuousWaveDoppler {
    config: CwDopplerConfig,
}

impl ContinuousWaveDoppler {
    /// Create a processor.
    #[must_use]
    pub fn new(config: CwDopplerConfig) -> Self {
        Self { config }
    }

    /// Quadrature-demodulate a real received signal to complex baseband and
    /// decimate to the configured baseband rate.
    ///
    /// Mixes with `exp(−j2π f₀ t)` (gain 2), then decimates by averaging
    /// non-overlapping blocks of `D = f_s/f_bb` samples. The block average is
    /// the anti-alias filter (first spectral null at `f_s/D = f_bb`) and also
    /// suppresses the `2f₀` image, leaving the signed Doppler baseband.
    #[must_use]
    pub fn demodulate(&self, rf: &[f64]) -> Array1<Complex64> {
        let fs = self.config.sampling_rate;
        let f0 = self.config.center_frequency;
        let d = self.config.decimation();
        let n = rf.len();
        let n_bb = n / d;
        let mut baseband = Vec::with_capacity(n_bb.max(1));
        for m in 0..n_bb.max(1) {
            let (mut sx, mut sy) = (0.0, 0.0);
            let count = if n_bb == 0 { n } else { d };
            for k in 0..count {
                let i = m * d + k;
                if i >= n {
                    break;
                }
                let phase = TWO_PI * f0 * (i as f64) / fs;
                sx += 2.0 * rf[i] * phase.cos();
                sy += -2.0 * rf[i] * phase.sin();
            }
            let inv = 1.0 / count as f64;
            baseband.push(Complex64::new(sx * inv, sy * inv));
        }
        Array1::from_vec([baseband.len()], baseband).expect("demodulate: vec matches length")
    }

    /// Compute the two-sided Doppler velocity spectrum from a received signal.
    ///
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] when `rf` is empty.
    pub fn spectrum(&self, rf: &[f64]) -> KwaversResult<CwSpectrum> {
        if rf.is_empty() {
            return Err(KwaversError::InvalidInput(
                "CW Doppler received signal is empty".to_owned(),
            ));
        }
        let baseband = self.demodulate(rf);
        let n = baseband.len();
        let spectrum = fft_1d_complex(&baseband);
        // Center zero frequency for a symmetric signed-velocity axis.
        let shifted = fftshift(
            spectrum
                .as_slice()
                .expect("invariant: FFT output is contiguous"),
        );
        let f_bb = self.config.baseband_rate;
        // Bin k (after shift) → baseband frequency (k − n/2)·f_bb/n.
        let half = (n / 2) as f64;
        let velocity = Array1::from_shape_fn([n], |idx| {
            let f_d = (idx[0] as f64 - half) * f_bb / n as f64;
            self.config.velocity_from_frequency(f_d)
        });
        let power = Array1::from_shape_fn([n], |idx| shifted[idx[0]].norm_sqr());
        Ok(CwSpectrum { velocity, power })
    }
}
