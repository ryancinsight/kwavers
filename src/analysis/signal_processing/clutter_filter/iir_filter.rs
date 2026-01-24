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
//! - x[n] is the input signal
//! - y[n] is the output signal
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
    #[must_use]
    pub fn with_zero_phase(mut self) -> Self {
        self.zero_phase = true;
        self
    }

    /// Set filter order
    #[must_use]
    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Validate configuration parameters
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
    /// 1. Apply forward IIR filter: y[n] = Σbᵢx[n-i] - Σaⱼy[n-j]
    /// 2. If zero_phase enabled, reverse and filter again
    /// 3. Return filtered signal
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = IirFilterConfig::with_cutoff(0.1);
        assert!(config.validate().is_ok());

        let bad_cutoff_low = IirFilterConfig {
            cutoff_frequency: 0.0,
            ..Default::default()
        };
        assert!(bad_cutoff_low.validate().is_err());

        let bad_cutoff_high = IirFilterConfig {
            cutoff_frequency: 0.6,
            ..Default::default()
        };
        assert!(bad_cutoff_high.validate().is_err());

        let bad_order = IirFilterConfig {
            cutoff_frequency: 0.1,
            order: 0,
            zero_phase: false,
        };
        assert!(bad_order.validate().is_err());
    }

    #[test]
    fn test_iir_filter_creation() {
        let config = IirFilterConfig::with_cutoff(0.05);
        let filter = IirFilter::new(config);
        assert!(filter.is_ok());
    }

    #[test]
    fn test_filter_removes_dc_component() {
        // Use higher cutoff for more aggressive DC removal
        let config = IirFilterConfig::with_cutoff(0.05).with_zero_phase();
        let filter = IirFilter::new(config).unwrap();

        // Create signal with DC offset + oscillation
        let n_pixels = 5;
        let n_frames = 100;
        let mut data = Array2::<f64>::zeros((n_pixels, n_frames));

        for i in 0..n_pixels {
            for t in 0..n_frames {
                let dc = 10.0; // DC component (clutter)
                let ac = 2.0 * (2.0 * std::f64::consts::PI * (t as f64) / 20.0).sin();
                data[[i, t]] = dc + ac;
            }
        }

        let filtered = filter.filter(&data).unwrap();

        // Check that DC component is significantly reduced
        let original_mean = data.mean().unwrap();
        let filtered_mean = filtered.mean().unwrap();

        // Filtered mean should be much smaller than original
        assert!(filtered_mean.abs() < 0.3 * original_mean.abs());
    }

    #[test]
    fn test_filter_preserves_high_frequency() {
        let config = IirFilterConfig::with_cutoff(0.05);
        let filter = IirFilter::new(config).unwrap();

        // Create high-frequency oscillation
        let n_pixels = 3;
        let n_frames = 100;
        let mut data = Array2::<f64>::zeros((n_pixels, n_frames));

        for i in 0..n_pixels {
            for t in 0..n_frames {
                // High-frequency signal (blood flow)
                data[[i, t]] = 3.0 * (2.0 * std::f64::consts::PI * (t as f64) / 5.0).sin();
            }
        }

        let filtered = filter.filter(&data).unwrap();

        // Check that high-frequency content is largely preserved
        let filtered_std = filtered.std(0.0);
        let original_std = data.std(0.0);

        assert!(filtered_std > 0.8 * original_std); // Should retain most amplitude
    }

    #[test]
    fn test_zero_phase_filtering() {
        let config = IirFilterConfig::with_cutoff(0.1).with_zero_phase();
        let filter = IirFilter::new(config).unwrap();

        let n_frames = 50;
        let signal =
            Array2::from_shape_fn((1, n_frames), |(_, t)| 5.0 + 2.0 * ((t as f64) * 0.4).sin());

        let filtered = filter.filter(&signal);
        assert!(filtered.is_ok());

        // Zero-phase filtering should preserve signal shape better
        // (this is a weak test - full test would compare phase spectrum)
        let result = filtered.unwrap();
        assert_eq!(result.dim(), signal.dim());
    }

    #[test]
    fn test_higher_order_filter() {
        let config = IirFilterConfig::with_cutoff(0.05).with_order(2);
        let filter = IirFilter::new(config).unwrap();

        let data = Array2::from_shape_fn((5, 100), |(_, t)| 10.0 + (t as f64) * 0.1);

        let filtered = filter.filter(&data);
        assert!(filtered.is_ok());
    }
}
