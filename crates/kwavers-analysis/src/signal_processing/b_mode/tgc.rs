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
//! A(z) = 2 · a₀ · f · z   `dB`,
//! ```
//!
//! so the compensating linear gain is `g(z) = 10^{A(z)/20}`. With axial sample
//! index `i` at sampling rate `f_s` and sound speed `c`, the round-trip depth is
//! `z = c·i/(2 f_s)`.
//!
//! # Reference
//! - Szabo, T. L. (2014). *Diagnostic Ultrasound Imaging: Inside Out* (2nd ed.),
//!   §4.2 (attenuation) and §10.4 (TGC). Academic Press.

use aequitas::systems::si::quantities::{Dimensionless, Frequency, Length, Velocity};
use aequitas::Quantity;
use eunomia::RealField;
use kwavers_core::constants::bone_acoustics::AttenuationSlope;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array1;

/// Calibrated axial sample spacing and log-amplitude gain increment.
///
/// Construct the SI attenuation slope by dividing the loss at a reference
/// frequency by that frequency. The slope applies to linear-in-frequency loss.
///
/// ```compile_fail,E0308
/// use aequitas::systems::si::quantities::{Frequency, Length, Velocity};
/// use kwavers_analysis::signal_processing::b_mode::TgcConfig;
/// let frequency = Frequency::<f64>::from_base(5e6);
/// let _ = TgcConfig::new(Length::from_base(1.0), frequency,
///     Velocity::from_base(1540.0), frequency);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TgcConfig<T = f64> {
    depth_step: Length<T>,
    log_gain_step: Dimensionless<T>,
}

// Extract binary scale while preserving the dimension of each operand.
// Positive mantissas lie in [1,2), so products and quotients of them
// stay in range even when the original SI factors span the scalar range.
fn normalized<T: RealField, D>(value: Quantity<T, D>) -> (Quantity<T, D>, i32) {
    let scalar = value.into_base();
    let exponent = scalar.binary_exponent().unwrap_or(0);
    (
        Quantity::from_base(scalar.scale_binary(-exponent)),
        exponent,
    )
}

const GAIN_BLOCK: usize = 64;

impl<T: RealField> TgcConfig<T> {
    /// Build the round-trip model in the supplied scalar precision.
    ///
    /// # Errors
    /// Rejects negative slope, nonpositive speed or frequencies, nonfinite
    /// inputs, and unrepresentable sample spacing or log-gain increment.
    pub fn new(
        slope: AttenuationSlope<T>,
        frequency: Frequency<T>,
        sound_speed: Velocity<T>,
        sampling_rate: Frequency<T>,
    ) -> KwaversResult<Self> {
        if !slope.into_base().is_finite()
            || slope.into_base() < T::ZERO
            || [
                frequency.into_base(),
                sound_speed.into_base(),
                sampling_rate.into_base(),
            ]
            .iter()
            .any(|&v| !v.is_finite() || v <= T::ZERO)
        {
            return Err(KwaversError::InvalidInput(
                "TGC requires a finite nonnegative attenuation slope and finite positive frequencies and speed".into()
            ));
        }
        let (speed, c) = normalized(sound_speed);
        let (rate, r) = normalized(sampling_rate);
        let (slope, a) = normalized(slope);
        let (frequency, f) = normalized(frequency);
        let path = speed / rate;
        let log_gain: Dimensionless<T> = slope * frequency * path;
        let depth_step = Length::from_base(path.into_base().scale_binary(c - r - 1));
        let log_gain_step =
            Dimensionless::from_base(log_gain.into_base().scale_binary(c - r + a + f));
        if depth_step.into_base() == T::ZERO
            || !depth_step.into_base().is_finite()
            || !log_gain_step.into_base().is_finite()
            || (slope.into_base() > T::ZERO && log_gain_step.into_base() == T::ZERO)
        {
            return Err(KwaversError::InvalidInput(
                "TGC sample increment exceeds scalar range".into(),
            ));
        }
        Ok(Self {
            depth_step,
            log_gain_step,
        })
    }

    /// One-way reflector depth at a round-trip sample index.
    ///
    /// # Errors
    /// Returns an error when the depth exceeds the scalar range.
    pub fn depth(&self, index: usize) -> KwaversResult<Length<T>> {
        let depth = self.depth_step * T::from_count(index);
        if !depth.into_base().is_finite() {
            return Err(KwaversError::InvalidInput(
                "TGC depth exceeds scalar range".into(),
            ));
        }
        Ok(depth)
    }

    #[inline]
    fn exponential(&self, index: usize) -> T {
        (self.log_gain_step * T::from_count(index))
            .into_base()
            .exp()
    }

    // Reuse the offsets from a short block of independent exponentials.
    // For 4096 samples, B + 4096/B is minimized at B=64. Each point
    // requires just one product, independent of the length of the line.
    fn gain_value(&self, index: usize) -> T {
        let offset = index % GAIN_BLOCK;
        self.exponential(index - offset) * self.exponential(offset)
    }

    fn gains(&self, len: usize) -> KwaversResult<impl Iterator<Item = T> + '_> {
        self.check_length(len)?;
        let mut offsets = [T::ONE; GAIN_BLOCK];
        for (i, factor) in offsets.iter_mut().take(len).enumerate() {
            *factor = self.exponential(i);
        }
        let mut anchor = T::ONE;
        Ok((0..len).map(move |i| {
            let offset = i % GAIN_BLOCK;
            if offset == 0 {
                anchor = self.exponential(i);
            }
            anchor * offsets[offset]
        }))
    }

    /// Linear pressure-amplitude gain at a sample.
    ///
    /// # Errors
    /// Returns an error when the gain exceeds the scalar range.
    pub fn gain(&self, index: usize) -> KwaversResult<Dimensionless<T>> {
        let gain = self.gain_value(index);
        if !gain.is_finite() {
            return Err(KwaversError::InvalidInput(
                "TGC gain exceeds scalar range".into(),
            ));
        }
        Ok(Dimensionless::from_base(gain))
    }

    fn check_length(&self, len: usize) -> KwaversResult<()> {
        if let Some(last) = len.checked_sub(1) {
            self.gain(last)?;
            // Rounding the block anchor and the offset independently can
            // change which of two adjacent blocks has the larger endpoint.
            // All complete blocks share one offset maximum and increasing
            // anchors. Only the last complete and partial endpoints matter.
            let start = last - last % GAIN_BLOCK;
            if start != 0 && last % GAIN_BLOCK != GAIN_BLOCK - 1 {
                self.gain(start - 1)?;
            }
        }
        Ok(())
    }

    /// Linear gains in axial sample order.
    ///
    /// # Errors
    /// Returns an error when the largest requested gain exceeds the scalar range.
    pub fn gain_curve(&self, len: usize) -> KwaversResult<Array1<T>> {
        Ok(Array1::from(self.gains(len)?.collect::<Vec<_>>()))
    }

    /// Apply the gain to signed RF or envelope samples in one output allocation.
    /// RF multiplication follows the sample type's IEEE arithmetic.
    ///
    /// # Errors
    /// Returns an error when the largest requested gain exceeds the scalar range.
    pub fn apply(&self, line: &Array1<T>) -> KwaversResult<Array1<T>> {
        let gains = self.gains(line.len())?;
        let values = if let Some(samples) = line.as_slice() {
            gains.zip(samples).map(|(g, &s)| s * g).collect::<Vec<_>>()
        } else {
            gains.zip(line.iter()).map(|(g, &s)| s * g).collect()
        };
        Ok(Array1::from(values))
    }

    /// Apply compensation to an existing RF buffer in logical sample order.
    ///
    /// # Errors
    /// Rejects gain overflow or overlapping samples before modifying the line.
    pub fn apply_in_place(&self, line: &mut Array1<T>) -> KwaversResult<()> {
        let gains = self.gains(line.len())?;
        if let Some(samples) = line.as_slice_mut() {
            for (sample, gain) in samples.iter_mut().zip(gains) {
                *sample *= gain;
            }
        } else {
            let samples = line
                .try_iter_mut()
                .map_err(|e| KwaversError::InvalidInput(e.to_string()))?;
            for (sample, gain) in samples.zip(gains) {
                *sample *= gain;
            }
        }
        Ok(())
    }
}
