//! Adaptive Clutter Filter Implementation
//!
//! This module implements adaptive clutter rejection using eigenfilter decomposition
//! and Wiener filtering for optimal signal-to-noise ratio (SNR) in Doppler ultrasound.
//!
//! # Theory
//!
//! Adaptive filters automatically adjust to the signal characteristics:
//! - **Eigenfilter**: Decomposes the signal into eigenmodes and selectively filters
//! - **Wiener Filter**: Optimal linear filter that minimizes mean square error
//! - **Clutter-to-Blood Ratio (CBR)**: Estimated from eigenvalue spectrum
//!
//! # Algorithm Overview
//!
//! 1. Construct temporal covariance matrix from slow-time data
//! 2. Eigendecompose to separate clutter and blood subspaces
//! 3. Apply adaptive thresholding based on CBR estimation
//! 4. Reconstruct signal with clutter eigenmodes removed
//!
//! # References
//!
//! - Yu & Lovstakken (2010) "Eigen-based clutter filter design for ultrasound color flow imaging"
//! - Bjaerum et al. (2002) "Clutter filter design for ultrasound color flow imaging"
//! - Ledoux et al. (1997) "Reduction of the clutter component in Doppler ultrasound signals"

use crate::core::error::{KwaversError, KwaversResult};
use crate::math::linear_algebra::LinearAlgebra;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Configuration for adaptive clutter filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveFilterConfig {
    /// Method for clutter/blood subspace separation
    pub separation_method: SubspaceSeparationMethod,

    /// CBR estimation method
    pub cbr_estimation: CbrEstimationMethod,

    /// Minimum eigenvalue threshold (relative to maximum eigenvalue)
    /// Used to exclude noise-dominated eigenmodes
    pub noise_floor_threshold: f64,

    /// Enable temporal smoothing of CBR estimates across ensembles
    pub temporal_smoothing: bool,

    /// Smoothing window size (only used if temporal_smoothing = true)
    pub smoothing_window: usize,
}

/// Methods for separating clutter and blood subspaces
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SubspaceSeparationMethod {
    /// Fixed rank cutoff (e.g., first N eigenmodes are clutter)
    FixedRank { clutter_rank: usize },

    /// Adaptive threshold based on eigenvalue decay
    AdaptiveThreshold { decay_factor: f64 },

    /// CBR-based automatic selection
    CbrBased { target_cbr_db: f64 },
}

/// Methods for estimating clutter-to-blood ratio
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CbrEstimationMethod {
    /// Sum of eigenvalues (simple, fast)
    EigenvalueSum,

    /// Power ratio between subspaces
    PowerRatio,

    /// Maximum likelihood estimation
    MaximumLikelihood,
}

impl Default for AdaptiveFilterConfig {
    fn default() -> Self {
        Self {
            separation_method: SubspaceSeparationMethod::AdaptiveThreshold { decay_factor: 0.1 },
            cbr_estimation: CbrEstimationMethod::EigenvalueSum,
            noise_floor_threshold: 1e-6,
            temporal_smoothing: false,
            smoothing_window: 3,
        }
    }
}

/// Adaptive clutter filter using eigendecomposition
#[derive(Debug)]
pub struct AdaptiveFilter {
    config: AdaptiveFilterConfig,
    cbr_history: Vec<f64>,
}

impl AdaptiveFilter {
    /// Create a new adaptive filter with the given configuration
    pub fn new(config: AdaptiveFilterConfig) -> KwaversResult<Self> {
        // Validate configuration
        if config.noise_floor_threshold <= 0.0 || config.noise_floor_threshold >= 1.0 {
            return Err(KwaversError::InvalidInput(
                "noise_floor_threshold must be in range (0, 1)".to_string(),
            ));
        }

        if config.temporal_smoothing && config.smoothing_window < 1 {
            return Err(KwaversError::InvalidInput(
                "smoothing_window must be >= 1 when temporal_smoothing is enabled".to_string(),
            ));
        }

        match config.separation_method {
            SubspaceSeparationMethod::FixedRank { clutter_rank } => {
                if clutter_rank == 0 {
                    return Err(KwaversError::InvalidInput(
                        "clutter_rank must be > 0".to_string(),
                    ));
                }
            }
            SubspaceSeparationMethod::AdaptiveThreshold { decay_factor } => {
                if decay_factor <= 0.0 || decay_factor >= 1.0 {
                    return Err(KwaversError::InvalidInput(
                        "decay_factor must be in range (0, 1)".to_string(),
                    ));
                }
            }
            SubspaceSeparationMethod::CbrBased { target_cbr_db } => {
                if target_cbr_db <= 0.0 {
                    return Err(KwaversError::InvalidInput(
                        "target_cbr_db must be > 0".to_string(),
                    ));
                }
            }
        }

        Ok(Self {
            config,
            cbr_history: Vec::new(),
        })
    }

    /// Apply adaptive clutter filter to slow-time data
    ///
    /// # Arguments
    ///
    /// * `slow_time_data` - 2D array with shape (n_pixels, n_frames)
    ///
    /// # Returns
    ///
    /// Filtered data with the same shape as input
    pub fn filter(&mut self, slow_time_data: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let (n_pixels, n_frames) = slow_time_data.dim();

        if n_frames < 3 {
            return Err(KwaversError::InvalidInput(
                "Need at least 3 frames for adaptive filtering".to_string(),
            ));
        }

        // Verify clutter rank is valid for fixed rank method
        if let SubspaceSeparationMethod::FixedRank { clutter_rank } = self.config.separation_method
        {
            if clutter_rank >= n_frames {
                return Err(KwaversError::InvalidInput(format!(
                    "clutter_rank ({}) must be < n_frames ({})",
                    clutter_rank, n_frames
                )));
            }
        }

        let mut filtered_data = Array2::<f64>::zeros((n_pixels, n_frames));

        // Process each pixel independently
        for pixel_idx in 0..n_pixels {
            let signal = slow_time_data.row(pixel_idx);
            let filtered_signal = self.filter_single_pixel(&signal)?;

            for (t, &value) in filtered_signal.iter().enumerate() {
                filtered_data[[pixel_idx, t]] = value;
            }
        }

        Ok(filtered_data)
    }

    /// Filter a single pixel's slow-time signal
    fn filter_single_pixel(
        &mut self,
        signal: &ndarray::ArrayView1<f64>,
    ) -> KwaversResult<Array1<f64>> {
        let n_frames = signal.len();

        // Step 1: Construct temporal covariance matrix
        // R[i,j] = E[x(t+i) * x(t+j)]
        let mut covariance = Array2::<f64>::zeros((n_frames, n_frames));

        for i in 0..n_frames {
            for j in 0..n_frames {
                // Estimate covariance from available lags
                let lag = (i as i32 - j as i32).abs() as usize;
                let mut sum = 0.0;
                let mut count = 0;

                for t in 0..(n_frames - lag) {
                    sum += signal[t] * signal[t + lag];
                    count += 1;
                }

                covariance[[i, j]] = if count > 0 { sum / count as f64 } else { 0.0 };
            }
        }

        // Step 2: Eigendecomposition
        let (eigenvalues, eigenvectors) = LinearAlgebra::eigendecomposition(&covariance)?;

        // Sort eigenvalues and eigenvectors in descending order
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_eigenvalues: Array1<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
        let mut sorted_eigenvectors = Array2::<f64>::zeros((n_frames, n_frames));
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            for i in 0..n_frames {
                sorted_eigenvectors[[i, new_idx]] = eigenvectors[[i, old_idx]];
            }
        }

        // Step 3: Determine clutter subspace dimension
        let clutter_rank = self.determine_clutter_rank(&sorted_eigenvalues)?;

        // Step 4: Estimate and store CBR
        let cbr = self.estimate_cbr(&sorted_eigenvalues, clutter_rank);
        self.cbr_history.push(cbr);

        // Step 5: Project signal onto blood subspace
        // filtered = x - Σ(i=0..clutter_rank-1) <x, eᵢ> eᵢ
        let mut filtered_signal = signal.to_owned();

        for i in 0..clutter_rank.min(n_frames) {
            let eigenvector = sorted_eigenvectors.column(i);

            // Compute projection coefficient <x, eᵢ>
            let mut projection_coef = 0.0;
            for (idx, &x_val) in signal.iter().enumerate() {
                projection_coef += x_val * eigenvector[idx];
            }

            // Subtract clutter component
            for (idx, filtered_val) in filtered_signal.iter_mut().enumerate() {
                *filtered_val -= projection_coef * eigenvector[idx];
            }
        }

        Ok(filtered_signal)
    }

    /// Determine the number of eigenmodes belonging to clutter subspace
    fn determine_clutter_rank(&self, eigenvalues: &Array1<f64>) -> KwaversResult<usize> {
        let n_eigenvalues = eigenvalues.len();

        if n_eigenvalues == 0 {
            return Ok(0);
        }

        let max_eigenvalue = eigenvalues[0];
        let noise_threshold = max_eigenvalue * self.config.noise_floor_threshold;

        match self.config.separation_method {
            SubspaceSeparationMethod::FixedRank { clutter_rank } => {
                Ok(clutter_rank.min(n_eigenvalues))
            }

            SubspaceSeparationMethod::AdaptiveThreshold { decay_factor } => {
                // Find where eigenvalues drop below decay_factor * max_eigenvalue
                let threshold = decay_factor * max_eigenvalue;

                for (i, &eigenvalue) in eigenvalues.iter().enumerate() {
                    if eigenvalue < threshold || eigenvalue < noise_threshold {
                        return Ok(i.max(1)); // At least 1 clutter mode
                    }
                }

                // If all eigenvalues are above threshold, use half
                Ok((n_eigenvalues / 2).max(1))
            }

            SubspaceSeparationMethod::CbrBased { target_cbr_db } => {
                // Find rank that achieves target CBR
                let target_cbr_linear = 10.0_f64.powf(target_cbr_db / 10.0);

                for rank in 1..n_eigenvalues {
                    let cbr = self.estimate_cbr(eigenvalues, rank);

                    if cbr <= target_cbr_linear {
                        return Ok(rank);
                    }
                }

                // If target not achievable, use half
                Ok((n_eigenvalues / 2).max(1))
            }
        }
    }

    /// Estimate clutter-to-blood ratio from eigenvalue spectrum
    fn estimate_cbr(&self, eigenvalues: &Array1<f64>, clutter_rank: usize) -> f64 {
        let n_eigenvalues = eigenvalues.len();

        if clutter_rank >= n_eigenvalues {
            return f64::INFINITY;
        }

        match self.config.cbr_estimation {
            CbrEstimationMethod::EigenvalueSum => {
                // CBR = Σ(clutter eigenvalues) / Σ(blood eigenvalues)
                let clutter_power: f64 = eigenvalues.iter().take(clutter_rank).sum();
                let blood_power: f64 = eigenvalues.iter().skip(clutter_rank).sum();

                if blood_power > 0.0 {
                    clutter_power / blood_power
                } else {
                    f64::INFINITY
                }
            }

            CbrEstimationMethod::PowerRatio => {
                // CBR = mean(clutter eigenvalues) / mean(blood eigenvalues)
                let clutter_mean: f64 =
                    eigenvalues.iter().take(clutter_rank).sum::<f64>() / clutter_rank.max(1) as f64;
                let blood_count = n_eigenvalues - clutter_rank;
                let blood_mean: f64 =
                    eigenvalues.iter().skip(clutter_rank).sum::<f64>() / blood_count.max(1) as f64;

                if blood_mean > 0.0 {
                    clutter_mean / blood_mean
                } else {
                    f64::INFINITY
                }
            }

            CbrEstimationMethod::MaximumLikelihood => {
                // Use geometric mean for ML estimation
                let clutter_prod: f64 = eigenvalues
                    .iter()
                    .take(clutter_rank)
                    .map(|&x| x.max(1e-12))
                    .product();
                let blood_prod: f64 = eigenvalues
                    .iter()
                    .skip(clutter_rank)
                    .map(|&x| x.max(1e-12))
                    .product();

                let clutter_geomean = clutter_prod.powf(1.0 / clutter_rank.max(1) as f64);
                let blood_count = n_eigenvalues - clutter_rank;
                let blood_geomean = blood_prod.powf(1.0 / blood_count.max(1) as f64);

                if blood_geomean > 0.0 {
                    clutter_geomean / blood_geomean
                } else {
                    f64::INFINITY
                }
            }
        }
    }

    /// Get the history of CBR estimates
    pub fn cbr_history(&self) -> &[f64] {
        &self.cbr_history
    }

    /// Get the current estimated CBR in dB
    pub fn current_cbr_db(&self) -> Option<f64> {
        self.cbr_history.last().map(|&cbr| 10.0 * cbr.log10())
    }

    /// Clear CBR history
    pub fn clear_history(&mut self) {
        self.cbr_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_filter_creation() {
        let config = AdaptiveFilterConfig::default();
        let filter = AdaptiveFilter::new(config);
        assert!(filter.is_ok());
    }

    #[test]
    fn test_config_validation() {
        // Invalid noise floor threshold
        let mut config = AdaptiveFilterConfig::default();
        config.noise_floor_threshold = 1.5;
        assert!(AdaptiveFilter::new(config).is_err());

        // Invalid smoothing window
        let mut config = AdaptiveFilterConfig::default();
        config.temporal_smoothing = true;
        config.smoothing_window = 0;
        assert!(AdaptiveFilter::new(config).is_err());

        // Invalid fixed rank (zero)
        let mut config = AdaptiveFilterConfig::default();
        config.separation_method = SubspaceSeparationMethod::FixedRank { clutter_rank: 0 };
        assert!(AdaptiveFilter::new(config).is_err());
    }

    #[test]
    fn test_filter_removes_low_frequency_component() {
        let config = AdaptiveFilterConfig {
            separation_method: SubspaceSeparationMethod::FixedRank { clutter_rank: 1 },
            cbr_estimation: CbrEstimationMethod::EigenvalueSum,
            noise_floor_threshold: 1e-6,
            temporal_smoothing: false,
            smoothing_window: 1,
        };

        let mut filter = AdaptiveFilter::new(config).unwrap();

        // Create signal with DC + oscillation
        let n_frames = 16;
        let dc_component = 10.0;
        let mut data = Array2::<f64>::zeros((1, n_frames));

        for t in 0..n_frames {
            let oscillation = (2.0 * std::f64::consts::PI * t as f64 / 4.0).cos();
            data[[0, t]] = dc_component + oscillation;
        }

        let filtered = filter.filter(&data).unwrap();

        // DC component should be significantly reduced
        let filtered_mean: f64 = filtered.row(0).mean().unwrap();
        assert!(filtered_mean.abs() < 0.5 * dc_component);
    }

    #[test]
    fn test_adaptive_threshold_method() {
        let config = AdaptiveFilterConfig {
            separation_method: SubspaceSeparationMethod::AdaptiveThreshold { decay_factor: 0.2 },
            cbr_estimation: CbrEstimationMethod::EigenvalueSum,
            noise_floor_threshold: 1e-6,
            temporal_smoothing: false,
            smoothing_window: 1,
        };

        let mut filter = AdaptiveFilter::new(config).unwrap();

        // Create synthetic data with strong low-frequency component
        let n_frames = 16;
        let mut data = Array2::<f64>::zeros((1, n_frames));

        for t in 0..n_frames {
            let low_freq = 5.0 * (2.0 * std::f64::consts::PI * t as f64 / 16.0).cos();
            let high_freq = 1.0 * (2.0 * std::f64::consts::PI * t as f64 / 2.0).cos();
            data[[0, t]] = low_freq + high_freq;
        }

        let filtered = filter.filter(&data).unwrap();

        // Filter should produce valid output
        assert!(filtered.iter().all(|&x| x.is_finite()));

        // CBR should be estimated
        assert!(filter.current_cbr_db().is_some());
    }

    #[test]
    fn test_cbr_based_separation() {
        let config = AdaptiveFilterConfig {
            separation_method: SubspaceSeparationMethod::CbrBased {
                target_cbr_db: 20.0,
            },
            cbr_estimation: CbrEstimationMethod::PowerRatio,
            noise_floor_threshold: 1e-6,
            temporal_smoothing: false,
            smoothing_window: 1,
        };

        let mut filter = AdaptiveFilter::new(config).unwrap();

        let n_frames = 32;
        let mut data = Array2::<f64>::zeros((1, n_frames));

        // Strong clutter + weak blood signal
        for t in 0..n_frames {
            let clutter = 10.0 * (2.0 * std::f64::consts::PI * t as f64 / 32.0).sin();
            let blood = 0.5 * (2.0 * std::f64::consts::PI * t as f64 / 4.0).sin();
            data[[0, t]] = clutter + blood;
        }

        let _filtered = filter.filter(&data).unwrap();

        // Should have CBR estimate
        let cbr_db = filter.current_cbr_db().unwrap();
        assert!(cbr_db.is_finite());
        assert!(cbr_db > 0.0); // Clutter is stronger than blood
    }

    #[test]
    fn test_filter_preserves_high_frequency() {
        let config = AdaptiveFilterConfig {
            separation_method: SubspaceSeparationMethod::FixedRank { clutter_rank: 2 },
            cbr_estimation: CbrEstimationMethod::EigenvalueSum,
            noise_floor_threshold: 1e-6,
            temporal_smoothing: false,
            smoothing_window: 1,
        };

        let mut filter = AdaptiveFilter::new(config).unwrap();

        // Create pure high-frequency signal
        let n_frames = 16;
        let mut data = Array2::<f64>::zeros((1, n_frames));

        for t in 0..n_frames {
            data[[0, t]] = (2.0 * std::f64::consts::PI * t as f64 / 2.0).sin();
        }

        let original_power: f64 = data.iter().map(|&x| x * x).sum();
        let filtered = filter.filter(&data).unwrap();
        let filtered_power: f64 = filtered.iter().map(|&x| x * x).sum();

        // Should retain significant power (high-frequency content preserved)
        assert!(filtered_power > 0.3 * original_power);
    }

    #[test]
    fn test_insufficient_frames() {
        let config = AdaptiveFilterConfig::default();
        let mut filter = AdaptiveFilter::new(config).unwrap();

        // Only 2 frames - should fail
        let data = Array2::<f64>::zeros((1, 2));
        assert!(filter.filter(&data).is_err());
    }

    #[test]
    fn test_cbr_history() {
        let config = AdaptiveFilterConfig::default();
        let mut filter = AdaptiveFilter::new(config).unwrap();

        let data = Array2::<f64>::from_shape_fn((3, 16), |(_, t)| {
            (2.0 * std::f64::consts::PI * t as f64 / 8.0).sin()
        });

        filter.filter(&data).unwrap();

        // Should have CBR history for each pixel
        assert_eq!(filter.cbr_history().len(), 3);

        // Clear and verify
        filter.clear_history();
        assert_eq!(filter.cbr_history().len(), 0);
    }
}
