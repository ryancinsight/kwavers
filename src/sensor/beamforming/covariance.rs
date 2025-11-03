//! Covariance estimation for beamforming
//!
//! ## Mathematical Foundation
//! **Sample Covariance**: R = (1/N) Σ x_n x_nᴴ
//! **Forward-Backward Averaging**: Reduces correlation matrix bias for finite samples
//! **Spatial Smoothing**: Decorrelates coherent sources using subarray averaging

use ndarray::{Array2, s};
use crate::error::KwaversResult;

/// Covariance matrix estimator with multiple methods
#[derive(Debug, Clone)]
pub struct CovarianceEstimator {
    /// Use forward-backward averaging for improved estimation
    pub forward_backward_averaging: bool,
    /// Number of snapshots for averaging
    pub num_snapshots: usize,
}

impl Default for CovarianceEstimator {
    fn default() -> Self {
        Self {
            forward_backward_averaging: true,
            num_snapshots: 1,
        }
    }
}

impl CovarianceEstimator {
    /// Create new covariance estimator
    #[must_use]
    pub fn new(forward_backward_averaging: bool, num_snapshots: usize) -> Self {
        Self {
            forward_backward_averaging,
            num_snapshots,
        }
    }

    /// Estimate sample covariance matrix: R = (1/N) Σ x_n x_nᴴ
    /// where x_n are the sensor snapshots and N is the number of snapshots
    pub fn estimate(&self, data: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let (num_sensors, num_snapshots) = data.dim();

        if num_snapshots == 0 {
            return Err(crate::error::KwaversError::InvalidInput(
                "No snapshots available for covariance estimation".to_string()
            ));
        }

        // Initialize covariance matrix
        let mut covariance = Array2::zeros((num_sensors, num_sensors));

        // Compute sample covariance: R = (1/N) Σ x_n x_nᴴ
        for snapshot in 0..num_snapshots {
            let x = data.column(snapshot);
            // Outer product: x * xᴴ (conjugate transpose, but real-valued so just transpose)
            for i in 0..num_sensors {
                for j in 0..num_sensors {
                    covariance[[i, j]] += x[i] * x[j];
                }
            }
        }

        // Normalize by number of snapshots
        covariance.mapv_inplace(|x| x / num_snapshots as f64);

        // Apply forward-backward averaging if enabled
        if self.forward_backward_averaging {
            covariance = self.apply_forward_backward_averaging(&covariance);
        }

        Ok(covariance)
    }

    /// Apply forward-backward averaging to reduce estimation bias
    /// R_fb = 0.5 * (R + J R* J) where J is the exchange matrix
    #[must_use]
    pub fn apply_forward_backward_averaging(&self, covariance: &Array2<f64>) -> Array2<f64> {
        let n = covariance.nrows();
        let mut fb_covariance = Array2::zeros((n, n));

        // Forward part: R
        // Backward part: J R* J where J is the exchange matrix (flips rows and columns)
        for i in 0..n {
            for j in 0..n {
                let forward = covariance[[i, j]];
                let backward = covariance[[n - 1 - i, n - 1 - j]]; // J R* J for real matrices
                fb_covariance[[i, j]] = 0.5 * (forward + backward);
            }
        }

        fb_covariance
    }

    /// Estimate covariance matrix with spatial smoothing for coherent sources
    pub fn estimate_with_spatial_smoothing(
        &self,
        data: &Array2<f64>,
        subarray_size: usize,
    ) -> KwaversResult<Array2<f64>> {
        let base_covariance = self.estimate(data)?;
        let spatial_smoothing = SpatialSmoothing::new(subarray_size);
        spatial_smoothing.apply(&base_covariance)
    }
}

/// Spatial smoothing for coherent source decorrelation
#[derive(Debug, Clone)]
pub struct SpatialSmoothing {
    /// Size of subarrays for smoothing
    pub subarray_size: usize,
}

impl SpatialSmoothing {
    /// Create new spatial smoothing processor
    #[must_use]
    pub fn new(subarray_size: usize) -> Self {
        Self { subarray_size }
    }

    /// Apply spatial smoothing to covariance matrix
    /// Creates multiple subarray covariance matrices and averages them
    pub fn apply(&self, covariance: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let n = covariance.nrows();

        if self.subarray_size >= n {
            return Ok(covariance.clone());
        }

        let num_subarrays = n - self.subarray_size + 1;
        let mut smoothed = Array2::zeros((self.subarray_size, self.subarray_size));

        // Average covariance matrices from all possible subarrays
        for start_idx in 0..num_subarrays {
            let end_idx = start_idx + self.subarray_size;

            // Extract subarray covariance matrix
            let sub_cov = covariance.slice(s![start_idx..end_idx, start_idx..end_idx]);

            // Add to smoothed matrix
            smoothed += &sub_cov;
        }

        // Normalize by number of subarrays
        smoothed.mapv_inplace(|x| x / num_subarrays as f64);

        Ok(smoothed)
    }
}
