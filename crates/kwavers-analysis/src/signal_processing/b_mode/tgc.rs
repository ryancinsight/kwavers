//! Time-Gain Compensation (TGC).
//!
//! Echoes from deeper scatterers are weaker because the round-trip path
//! accumulates more attenuation. TGC restores a depth-uniform brightness by
//! applying a depth-increasing gain that exactly cancels the expected
//! attenuation, so equal reflectors read equally bright regardless of depth.
//!
//! # Model
//!
//! Soft-tissue attenuation follows a power law `α(f) = a₀·f` with `a₀` in
//! dB·cm⁻¹·MHz⁻¹. A reflector at depth `z` (cm) suffers round-trip attenuation
//!
//! ```text
//! A(z) = 2 · a₀ · f · z   [dB],
//! ```
//!
//! so the compensating linear gain is `g(z) = 10^{A(z)/20}`. With axial sample
//! index `i` at sampling rate `f_s` and sound speed `c`, the round-trip depth is
//! `z = c·i/(2 f_s)`.
//!
//! # Reference
//! - Szabo, T. L. (2014). *Diagnostic Ultrasound Imaging: Inside Out* (2nd ed.),
//!   §4.2 (attenuation) and §10.4 (TGC). Academic Press.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array1;

/// Time-gain-compensation parameters.
#[derive(Debug, Clone, Copy)]
pub struct TgcConfig {
    /// Attenuation slope `a₀` [dB·cm⁻¹·MHz⁻¹] (soft tissue ≈ 0.5).
    pub attenuation_db_cm_mhz: f64,
    /// Imaging frequency [MHz].
    pub frequency_mhz: f64,
    /// Sound speed [m/s].
    pub sound_speed: f64,
    /// Axial sampling rate [Hz].
    pub sampling_rate: f64,
}

impl TgcConfig {
    /// Round-trip depth [m] of axial sample `i`.
    #[must_use]
    pub fn depth_m(&self, i: usize) -> f64 {
        self.sound_speed * i as f64 / (2.0 * self.sampling_rate)
    }

    /// Compensating linear gain at axial sample `i`.
    #[must_use]
    pub fn gain(&self, i: usize) -> f64 {
        let z_cm = self.depth_m(i) * 100.0;
        let gain_db = 2.0 * self.attenuation_db_cm_mhz * self.frequency_mhz * z_cm;
        10.0_f64.powf(gain_db / 20.0)
    }

    /// Per-sample linear gain curve for `n` axial samples.
    #[must_use]
    pub fn gain_curve(&self, n: usize) -> Array1<f64> {
        Array1::from_shape_fn(n, |i| self.gain(i))
    }

    /// Apply TGC to an axial RF/envelope line.
    ///
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] when configuration values are
    /// non-positive (which would make the gain ill-defined).
    pub fn apply(&self, line: &Array1<f64>) -> KwaversResult<Array1<f64>> {
        if self.sound_speed <= 0.0 || self.sampling_rate <= 0.0 || self.frequency_mhz <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "TGC requires positive sound speed, sampling rate, and frequency".to_owned(),
            ));
        }
        let curve = self.gain_curve(line.len());
        Ok(line * &curve)
    }
}
