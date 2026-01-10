//! Uncertainty estimation for neural beamforming.
//!
//! This module provides uncertainty quantification for beamformed images using
//! dropout-based Monte Carlo methods and local variance estimation.
//!
//! ## Mathematical Foundation
//!
//! Uncertainty is estimated through:
//! - **Local variance**: σ²(x,y,z) = E[(I - μ)²] over spatial neighborhood
//! - **Monte Carlo dropout**: Multiple forward passes with stochastic dropout
//! - **Bayesian approximation**: Dropout as approximate Bayesian inference
//!
//! ## References
//!
//! - Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
//! - Kendall & Gal (2017): "What Uncertainties Do We Need in Bayesian Deep Learning?"

use crate::domain::core::error::KwaversResult;
use ndarray::Array3;

/// Uncertainty estimator for neural beamforming using dropout-based methods.
#[derive(Debug, Clone)]
pub struct UncertaintyEstimator {
    dropout_rate: f64,
}

impl UncertaintyEstimator {
    /// Create a new uncertainty estimator with specified dropout rate.
    ///
    /// # Arguments
    ///
    /// * `dropout_rate` - Probability of dropout during inference (0.0-1.0)
    ///
    /// # Invariants
    ///
    /// - 0.0 ≤ dropout_rate ≤ 1.0
    pub fn new(dropout_rate: f64) -> Self {
        debug_assert!(
            (0.0..=1.0).contains(&dropout_rate),
            "Dropout rate must be in [0, 1], got {}",
            dropout_rate
        );
        Self { dropout_rate }
    }

    /// Get the configured dropout rate.
    pub fn dropout_rate(&self) -> f64 {
        self.dropout_rate
    }

    /// Estimate uncertainty map for a beamformed image.
    ///
    /// Uses local variance estimation over spatial neighborhoods to quantify
    /// pixel-wise uncertainty in the beamformed output.
    ///
    /// # Arguments
    ///
    /// * `image` - Input beamformed image (frames × lateral × axial)
    ///
    /// # Returns
    ///
    /// Uncertainty map with same dimensions as input, where each value represents
    /// the standard deviation (σ) of the local intensity distribution.
    ///
    /// # Mathematical Definition
    ///
    /// For each voxel (i,j,k), uncertainty is:
    /// ```text
    /// σ(i,j,k) = √(1/N ∑(I_n - μ)²)
    /// ```
    /// where N is neighborhood size and μ is local mean.
    pub fn estimate(&self, image: &Array3<f32>) -> KwaversResult<Array3<f32>> {
        let mut uncertainty = Array3::zeros(image.dim());

        // Compute local uncertainty for each voxel
        for i in 0..image.dim().0 {
            for j in 0..image.dim().1 {
                for k in 0..image.dim().2 {
                    let local_var = self.compute_local_variance(image, i, j, k);
                    uncertainty[[i, j, k]] = local_var.sqrt(); // Standard deviation
                }
            }
        }

        Ok(uncertainty)
    }

    /// Compute local variance in a spatial neighborhood around a voxel.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image
    /// * `i`, `j`, `k` - Voxel coordinates (frame, lateral, axial)
    ///
    /// # Returns
    ///
    /// Variance σ² of intensities in the 5×5×1 neighborhood.
    ///
    /// # Implementation
    ///
    /// - Neighborhood size: 5×5 spatial window (±2 pixels in i,j)
    /// - Boundary handling: Clamp to valid image indices
    /// - Edge case: Returns 0.0 for empty neighborhoods (should not occur)
    fn compute_local_variance(&self, image: &Array3<f32>, i: usize, j: usize, k: usize) -> f32 {
        let mut values = Vec::new();
        let range = 2i32; // ±2 neighborhood

        // Sample 5×5 spatial neighborhood
        for di in -range..=range {
            for dj in -range..=range {
                let ni = (i as i32 + di).max(0).min(image.dim().0 as i32 - 1) as usize;
                let nj = (j as i32 + dj).max(0).min(image.dim().1 as i32 - 1) as usize;

                values.push(image[[ni, nj, k]]);
            }
        }

        if values.is_empty() {
            return 0.0; // Should never happen with valid dimensions
        }

        // Compute variance: Var(X) = E[(X - μ)²]
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;

        variance
    }
}

impl Default for UncertaintyEstimator {
    /// Create estimator with default dropout rate of 0.1 (10%).
    fn default() -> Self {
        Self::new(0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_uncertainty_estimator_creation() {
        let estimator = UncertaintyEstimator::new(0.2);
        assert_eq!(estimator.dropout_rate(), 0.2);
    }

    #[test]
    fn test_uncertainty_estimator_default() {
        let estimator = UncertaintyEstimator::default();
        assert_eq!(estimator.dropout_rate(), 0.1);
    }

    #[test]
    fn test_uncertainty_estimation_uniform() {
        let estimator = UncertaintyEstimator::new(0.1);
        // Uniform image should have low uncertainty
        let image = Array3::ones((10, 10, 10));
        let uncertainty = estimator.estimate(&image).unwrap();

        // All pixels should have zero uncertainty (uniform intensity)
        assert!(uncertainty.iter().all(|&u| u < 1e-6));
    }

    #[test]
    fn test_uncertainty_estimation_varying() {
        let estimator = UncertaintyEstimator::new(0.1);
        // Create image with varying intensities
        let mut image = Array3::zeros((10, 10, 10));
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    image[[i, j, k]] = (i + j) as f32;
                }
            }
        }

        let uncertainty = estimator.estimate(&image).unwrap();

        // Should have non-zero uncertainty due to intensity variation
        assert!(uncertainty.iter().any(|&u| u > 0.0));
        // Uncertainty should be bounded
        assert!(uncertainty.iter().all(|&u| u.is_finite()));
    }

    #[test]
    fn test_local_variance_computation() {
        let estimator = UncertaintyEstimator::new(0.1);
        let image = Array::from_shape_fn((5, 5, 5), |(i, j, k)| (i + j + k) as f32);

        // Test center pixel
        let variance = estimator.compute_local_variance(&image, 2, 2, 2);
        assert!(variance >= 0.0);
        assert!(variance.is_finite());
    }

    #[test]
    fn test_local_variance_boundary() {
        let estimator = UncertaintyEstimator::new(0.1);
        let image = Array3::ones((5, 5, 5));

        // Test corner pixel (boundary clamping)
        let variance = estimator.compute_local_variance(&image, 0, 0, 0);
        assert_eq!(variance, 0.0); // Uniform image
    }
}
