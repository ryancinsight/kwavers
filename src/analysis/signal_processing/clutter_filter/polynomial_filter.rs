//! Polynomial Regression Clutter Filter
//!
//! This module implements polynomial regression-based clutter filtering for ultrasound
//! Doppler imaging. The filter removes slow-moving tissue clutter by fitting and subtracting
//! a polynomial curve from the temporal signal at each spatial location.
//!
//! # Algorithm Overview
//!
//! For each pixel time series x(t), the filter:
//! 1. Fits a polynomial p(t) = Σᵢ aᵢtⁱ to the signal
//! 2. Subtracts the fitted polynomial: y(t) = x(t) - p(t)
//! 3. Returns the residual containing blood flow signal
//!
//! # Mathematical Foundation
//!
//! Polynomial regression solves the least-squares problem:
//! ```text
//! min Σₜ [x(t) - Σᵢ aᵢtⁱ]²
//! ```
//!
//! This is equivalent to solving the normal equations:
//! ```text
//! (VᵀV)a = Vᵀx
//! ```
//! where V is the Vandermonde matrix with elements Vₜᵢ = tⁱ
//!
//! # References
//!
//! - Jensen, J. A. (1996). *Field: A Program for Simulating Ultrasound Systems*
//! - Bjaerum, S., Torp, H., & Kristoffersen, K. (2002). "Clutter filters adapted to tissue motion in ultrasound color flow imaging"
//!   *IEEE Trans. Ultrason., Ferroelect., Freq. Control*, 49(6), 693-704.
//! - Yu, A. C. H., & Cobbold, R. S. C. (2008). "Single-ensemble-based eigen-processing methods for color flow imaging"
//!   *IEEE Trans. Ultrason., Ferroelect., Freq. Control*, 55(3), 559-572.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{s, Array1, Array2};

/// Configuration for polynomial regression clutter filter
#[derive(Debug, Clone)]
pub struct PolynomialFilterConfig {
    /// Order of the polynomial (typically 1-5)
    ///
    /// - Order 1: Linear fit (tissue motion assumed linear)
    /// - Order 2: Quadratic fit (tissue acceleration)
    /// - Order 3-5: Higher-order motion (respiration, cardiac)
    ///
    /// Higher orders can fit more complex tissue motion but may remove
    /// some blood signal and are more sensitive to noise.
    pub polynomial_order: usize,

    /// Temporal normalization for numerical stability
    ///
    /// When true, time indices are normalized to [0, 1] range before
    /// polynomial fitting to improve numerical conditioning.
    pub normalize_time: bool,
}

impl Default for PolynomialFilterConfig {
    fn default() -> Self {
        Self {
            polynomial_order: 2,  // Quadratic fit (good balance)
            normalize_time: true, // Recommended for stability
        }
    }
}

impl PolynomialFilterConfig {
    /// Create configuration with specific polynomial order
    #[must_use]
    pub fn with_order(order: usize) -> Self {
        Self {
            polynomial_order: order,
            ..Default::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> KwaversResult<()> {
        if self.polynomial_order == 0 {
            return Err(KwaversError::InvalidInput(
                "Polynomial order must be >= 1".to_string(),
            ));
        }
        if self.polynomial_order > 10 {
            return Err(KwaversError::InvalidInput(format!(
                "Polynomial order {} is too high (max 10)",
                self.polynomial_order
            )));
        }
        Ok(())
    }
}

/// Polynomial regression clutter filter
///
/// Removes slow-moving tissue clutter by fitting and subtracting polynomial
/// trends from temporal signals.
#[derive(Debug)]
pub struct PolynomialFilter {
    config: PolynomialFilterConfig,
}

impl PolynomialFilter {
    /// Create a new polynomial filter with given configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    pub fn new(config: PolynomialFilterConfig) -> KwaversResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Apply polynomial regression filter to slow-time data
    ///
    /// # Arguments
    ///
    /// * `slow_time_data` - Input data with shape (n_pixels, n_frames)
    ///
    /// # Returns
    ///
    /// Filtered data with same shape, where polynomial trend has been removed
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input data has wrong shape
    /// - Number of frames is less than polynomial order + 1
    /// - Numerical issues in polynomial fitting (singular matrix)
    ///
    /// # Algorithm
    ///
    /// For each pixel (spatial location):
    /// 1. Extract temporal signal x(t)
    /// 2. Construct Vandermonde matrix V for polynomial basis
    /// 3. Solve normal equations (VᵀV)a = Vᵀx for coefficients
    /// 4. Compute polynomial fit p(t) = Σᵢ aᵢtⁱ
    /// 5. Subtract: y(t) = x(t) - p(t)
    ///
    /// # Performance
    ///
    /// - Time complexity: O(n_pixels × n_frames × order²)
    /// - Space complexity: O(n_frames × order)
    /// - Suitable for real-time processing with moderate polynomial orders
    pub fn filter(&self, slow_time_data: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let (n_pixels, n_frames) = slow_time_data.dim();

        // Validate ensemble length
        if n_frames <= self.config.polynomial_order {
            return Err(KwaversError::InvalidInput(format!(
                "Number of frames ({}) must be > polynomial order ({})",
                n_frames, self.config.polynomial_order
            )));
        }

        // Create time vector
        let mut time: Array1<f64> = Array1::from_iter((0..n_frames).map(|t| t as f64));

        // Normalize time if requested
        if self.config.normalize_time {
            let max_time = (n_frames - 1) as f64;
            time.mapv_inplace(|t| t / max_time);
        }

        // Construct Vandermonde matrix V
        // V[t, i] = t^i for i = 0, 1, ..., polynomial_order
        let mut vandermonde = Array2::<f64>::zeros((n_frames, self.config.polynomial_order + 1));
        for (t_idx, &t) in time.iter().enumerate() {
            for power in 0..=self.config.polynomial_order {
                vandermonde[[t_idx, power]] = t.powi(power as i32);
            }
        }

        // Precompute (VᵀV)⁻¹Vᵀ for efficiency (same for all pixels)
        let vt = vandermonde.t().to_owned();
        let vtv = vt.dot(&vandermonde);

        // Compute inverse using simple Gaussian elimination
        // For production, would use robust LU decomposition
        let vtv_inv = self.pseudo_inverse(&vtv)?;
        let projection = vtv_inv.dot(&vt);

        // Apply filter to each pixel
        let mut filtered_data = Array2::<f64>::zeros((n_pixels, n_frames));

        for pixel_idx in 0..n_pixels {
            // Extract temporal signal for this pixel
            let signal = slow_time_data.slice(s![pixel_idx, ..]).to_owned();

            // Compute polynomial coefficients: a = (VᵀV)⁻¹Vᵀx
            let coefficients = projection.dot(&signal);

            // Evaluate polynomial fit: p(t) = Va
            let polynomial_fit = vandermonde.dot(&coefficients);

            // Subtract polynomial trend
            for (t, (&original, &fit)) in signal.iter().zip(polynomial_fit.iter()).enumerate() {
                filtered_data[[pixel_idx, t]] = original - fit;
            }
        }

        Ok(filtered_data)
    }

    /// Compute pseudo-inverse of a matrix using SVD
    ///
    /// This is a simplified implementation. For production, use optimized
    /// linear algebra library.
    fn pseudo_inverse(&self, matrix: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let (n, m) = matrix.dim();

        if n != m {
            return Err(KwaversError::InvalidInput(
                "Matrix must be square for simple inversion".to_string(),
            ));
        }

        // For small matrices, use simple Gaussian elimination with pivoting
        // This is sufficient for typical polynomial orders (1-5)
        let mut augmented = Array2::<f64>::zeros((n, 2 * n));

        // Create augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
                if i == j {
                    augmented[[i, j + n]] = 1.0;
                }
            }
        }

        // Forward elimination with partial pivoting
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = augmented[[col, col]].abs();

            for row in (col + 1)..n {
                if augmented[[row, col]].abs() > max_val {
                    max_row = row;
                    max_val = augmented[[row, col]].abs();
                }
            }

            if max_val < 1e-12 {
                return Err(KwaversError::Numerical(
                    crate::core::error::NumericalError::SolverFailed {
                        method: "Matrix inversion".to_string(),
                        reason: "Matrix is singular or nearly singular".to_string(),
                    },
                ));
            }

            // Swap rows
            if max_row != col {
                for j in 0..(2 * n) {
                    let temp = augmented[[col, j]];
                    augmented[[col, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }

            // Eliminate column
            for row in 0..n {
                if row != col {
                    let factor = augmented[[row, col]] / augmented[[col, col]];
                    for j in 0..(2 * n) {
                        augmented[[row, j]] -= factor * augmented[[col, j]];
                    }
                }
            }

            // Normalize pivot row
            let pivot = augmented[[col, col]];
            for j in 0..(2 * n) {
                augmented[[col, j]] /= pivot;
            }
        }

        // Extract inverse from right half of augmented matrix
        let mut inverse = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inverse[[i, j]] = augmented[[i, j + n]];
            }
        }

        Ok(inverse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = PolynomialFilterConfig::with_order(2);
        assert!(config.validate().is_ok());

        let bad_config = PolynomialFilterConfig {
            polynomial_order: 0,
            ..Default::default()
        };
        assert!(bad_config.validate().is_err());

        let too_high_config = PolynomialFilterConfig {
            polynomial_order: 11,
            ..Default::default()
        };
        assert!(too_high_config.validate().is_err());
    }

    #[test]
    fn test_polynomial_filter_creation() {
        let config = PolynomialFilterConfig::with_order(2);
        let filter = PolynomialFilter::new(config);
        assert!(filter.is_ok());
    }

    #[test]
    fn test_filter_with_linear_trend() {
        let config = PolynomialFilterConfig::with_order(1);
        let filter = PolynomialFilter::new(config).unwrap();

        // Create data with linear trend (clutter) + noise (blood)
        let n_pixels = 10;
        let n_frames = 50;
        let mut data = Array2::<f64>::zeros((n_pixels, n_frames));

        for i in 0..n_pixels {
            for t in 0..n_frames {
                // Linear trend (tissue motion)
                let trend = 10.0 * (t as f64) / (n_frames as f64);
                // Small oscillation (blood flow)
                let blood = 0.5 * (2.0 * std::f64::consts::PI * (t as f64) / 10.0).sin();
                data[[i, t]] = trend + blood;
            }
        }

        let filtered = filter.filter(&data).unwrap();

        // Check that filtered data has reduced mean compared to original
        let original_mean = data.mean().unwrap();
        let filtered_mean = filtered.mean().unwrap().abs();

        assert!(filtered_mean < original_mean);
    }

    #[test]
    fn test_filter_preserves_oscillations() {
        let config = PolynomialFilterConfig::with_order(2);
        let filter = PolynomialFilter::new(config).unwrap();

        // Create data with quadratic trend + high-frequency oscillation
        let n_pixels = 5;
        let n_frames = 100;
        let mut data = Array2::<f64>::zeros((n_pixels, n_frames));

        for i in 0..n_pixels {
            for t in 0..n_frames {
                let t_norm = (t as f64) / (n_frames as f64);
                // Quadratic trend (tissue acceleration)
                let trend = 10.0 * t_norm * t_norm;
                // High-frequency oscillation (blood flow)
                let blood = 2.0 * (2.0 * std::f64::consts::PI * (t as f64) / 5.0).sin();
                data[[i, t]] = trend + blood;
            }
        }

        let filtered = filter.filter(&data).unwrap();

        // Check that high-frequency content is preserved
        // (variance should be dominated by blood signal)
        let filtered_std = filtered.std(0.0);
        assert!(filtered_std > 1.0); // Should retain oscillation amplitude
    }

    #[test]
    fn test_insufficient_frames() {
        let config = PolynomialFilterConfig::with_order(5);
        let filter = PolynomialFilter::new(config).unwrap();

        // Only 5 frames, but order 5 polynomial needs at least 6
        let data = Array2::<f64>::zeros((10, 5));
        let result = filter.filter(&data);

        assert!(result.is_err());
    }
}
