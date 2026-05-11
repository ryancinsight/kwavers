//! IIR High-Pass Clutter Filter
//!
//! This module implements Infinite Impulse Response (IIR) high-pass filtering for
//! ultrasound Doppler clutter rejection. IIR filters are computationally efficient
//! and provide sharp cutoff characteristics with fewer coefficients than FIR filters.
//!
//! # Algorithm Overview
//!
//! The filter implements a recursive difference equation:
//! ```text
//! y[n] = Σᵢ bᵢx[n-i] - Σⱼ aⱼy[n-j]
//! ```
//! where:
//! - bᵢ are feedforward (numerator) coefficients
//! - aⱼ are feedback (denominator) coefficients
//! - `x[n]` is the input signal
//! - `y[n]` is the output signal
//!
//! # High-Pass Filter Design
//!
//! A simple first-order IIR high-pass filter has transfer function:
//! ```text
//! H(z) = (1 - z⁻¹) / (1 - α·z⁻¹)
//! ```
//! where α controls the cutoff frequency.
//!
//! The cutoff frequency fc relates to α via:
//! ```text
//! α = exp(-2π·fc/fs)
//! ```
//! where fs is the sampling frequency.
//!
//! # References
//!
//! - Oppenheim, A. V., & Schafer, R. W. (2009). *Discrete-Time Signal Processing*. Prentice Hall.
//! - Bjaerum, S., et al. (2002). "Clutter filters adapted to tissue motion in ultrasound color flow imaging"
//! - Brands, P. J., Hoeks, A. P., Reneman, R. S., et al. (1995). "A noninvasive method to estimate wall shear rate"
//!   *Ultrasound in Medicine & Biology*, 21(2), 171-185.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array2;

#[cfg(test)]
mod tests;

/// Configuration for IIR high-pass clutter filter
#[derive(Debug, Clone)]
pub struct IirFilterConfig {
    /// Cutoff frequency as fraction of sampling frequency (0 < fc < 0.5)
    ///
    /// Typical values:
    /// - 0.01-0.05: Very low cutoff (removes only DC and slow drift)
    /// - 0.05-0.10: Low cutoff (typical for blood flow imaging)
    /// - 0.10-0.20: Medium cutoff (aggressive clutter rejection)
    ///
    /// The cutoff frequency should be set based on the maximum tissue
    /// velocity to avoid removing blood flow signal.
    pub cutoff_frequency: f64,

    /// Filter order (1-4)
    ///
    /// Higher orders provide sharper cutoff but:
    /// - Increase computational cost
    /// - May introduce phase distortion
    /// - Can cause numerical instability
    ///
    /// First-order (default) is usually sufficient for clutter filtering.
    pub order: usize,

    /// Use zero-phase filtering (forward-backward)
    ///
    /// When true, applies filter twice (forward then backward) to eliminate
    /// phase distortion. Doubles computational cost but preserves waveform shape.
    pub zero_phase: bool,
}

impl Default for IirFilterConfig {
    fn default() -> Self {
        Self {
            cutoff_frequency: 0.05, // 5% of sampling frequency
            order: 1,               // First-order for efficiency
            zero_phase: false,      // Real-time compatible
        }
    }
}

impl IirFilterConfig {
    /// Create configuration with specific cutoff frequency
    #[must_use]
    pub fn with_cutoff(cutoff_frequency: f64) -> Self {
        Self {
            cutoff_frequency,
            ..Default::default()
        }
    }

    /// Enable zero-phase filtering
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn with_zero_phase(mut self) -> Self {
        self.zero_phase = true;
        self
    }

    /// Set filter order
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Validate configuration parameters
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.cutoff_frequency <= 0.0 || self.cutoff_frequency >= 0.5 {
            return Err(KwaversError::InvalidInput(format!(
                "Cutoff frequency {} must be in range (0, 0.5)",
                self.cutoff_frequency
            )));
        }
        if self.order == 0 || self.order > 4 {
            return Err(KwaversError::InvalidInput(format!(
                "Filter order {} must be in range [1, 4]",
                self.order
            )));
        }
        Ok(())
    }
}

/// IIR high-pass clutter filter
///
/// Removes slow-moving tissue clutter using recursive high-pass filtering.
#[derive(Debug)]
pub struct IirFilter {
    config: IirFilterConfig,
    /// Feedforward coefficients (numerator)
    b_coeffs: Vec<f64>,
    /// Feedback coefficients (denominator)
    a_coeffs: Vec<f64>,
}

impl IirFilter {
    /// Create a new IIR filter with given configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    pub fn new(config: IirFilterConfig) -> KwaversResult<Self> {
        config.validate()?;

        // Design filter coefficients based on order
        let (b_coeffs, a_coeffs) =
            Self::design_highpass_filter(config.cutoff_frequency, config.order)?;

        Ok(Self {
            config,
            b_coeffs,
            a_coeffs,
        })
    }

    /// Design high-pass IIR filter coefficients
    ///
    /// Uses bilinear transform from analog prototype
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn design_highpass_filter(cutoff: f64, order: usize) -> KwaversResult<(Vec<f64>, Vec<f64>)> {
        // For first-order high-pass filter:
        // H(z) = (1 - z^-1) / (1 - α·z^-1)
        // where α = exp(-2π·fc)

        let alpha = (-2.0 * std::f64::consts::PI * cutoff).exp();

        let mut b = vec![1.0, -1.0]; // Numerator: [1, -1]
        let mut a = vec![1.0, -alpha]; // Denominator: [1, -α]

        // For higher orders, cascade multiple first-order sections
        if order > 1 {
            // Simplified: just scale cutoff for each stage
            // Production version would use proper Butterworth/Chebyshev design
            for _ in 1..order {
                b = Self::convolve(&b, &[1.0, -1.0]);
                a = Self::convolve(&a, &[1.0, -alpha]);
            }
        }

        // Normalize so a[0] = 1
        if !a.is_empty() && a[0] != 1.0 {
            let a0 = a[0];
            b.iter_mut().for_each(|x| *x /= a0);
            a.iter_mut().for_each(|x| *x /= a0);
        }

        Ok((b, a))
    }

    /// Convolve two sequences (for cascading filters)
    fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = a.len() + b.len() - 1;
        let mut result = vec![0.0; n];

        for i in 0..a.len() {
            for j in 0..b.len() {
                result[i + j] += a[i] * b[j];
            }
        }

        result
    }

    /// Apply IIR high-pass filter to slow-time data
    ///
    /// # Arguments
    ///
    /// * `slow_time_data` - Input data with shape (n_pixels, n_frames)
    ///
    /// # Returns
    ///
    /// Filtered data with same shape, with low-frequency components removed
    ///
    /// # Algorithm
    ///
    /// For each pixel:
    /// 1. Apply forward IIR filter: `y[n] = Σbᵢx[n-i] - Σaⱼy[n-j]`
    /// 2. If zero_phase enabled, reverse and filter again
    /// 3. Return filtered signal
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn filter(&self, slow_time_data: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let (n_pixels, n_frames) = slow_time_data.dim();

        if n_frames < self.b_coeffs.len() {
            return Err(KwaversError::InvalidInput(format!(
                "Number of frames ({}) must be >= filter length ({})",
                n_frames,
                self.b_coeffs.len()
            )));
        }

        let mut filtered_data = Array2::<f64>::zeros((n_pixels, n_frames));

        for pixel_idx in 0..n_pixels {
            // Extract temporal signal
            let signal = slow_time_data.row(pixel_idx);

            // Apply forward filter
            let mut filtered = self.apply_iir_filter(&signal.to_vec());

            // Apply backward filter if zero-phase requested
            if self.config.zero_phase {
                filtered.reverse();
                filtered = self.apply_iir_filter(&filtered);
                filtered.reverse();
            }

            // Store result
            for (t, &val) in filtered.iter().enumerate() {
                filtered_data[[pixel_idx, t]] = val;
            }
        }

        Ok(filtered_data)
    }

    /// Apply IIR filter to a single signal
    ///
    /// Implements the difference equation:
    /// y[n] = Σᵢ bᵢx[n-i] - Σⱼ₌₁ aⱼy[n-j]
    fn apply_iir_filter(&self, signal: &[f64]) -> Vec<f64> {
        let n = signal.len();
        let mut output = vec![0.0; n];

        let b = &self.b_coeffs;
        let a = &self.a_coeffs;

        for n_idx in 0..n {
            let mut y = 0.0;

            // Feedforward (FIR part): Σ bᵢx[n-i]
            for (i, &bi) in b.iter().enumerate() {
                if n_idx >= i {
                    y += bi * signal[n_idx - i];
                }
            }

            // Feedback (IIR part): -Σ aⱼy[n-j] (skip a[0] which is always 1)
            for (j, &aj) in a.iter().enumerate().skip(1) {
                if n_idx >= j {
                    y -= aj * output[n_idx - j];
                }
            }

            output[n_idx] = y;
        }

        output
    }
}
