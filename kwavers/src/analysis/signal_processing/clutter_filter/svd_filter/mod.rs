//! Spatiotemporal SVD Clutter Filter for Functional Ultrasound
//!
//! Implements the spatiotemporal Singular Value Decomposition (SVD) clutter filtering
//! technique for separating blood flow signals from tissue motion in ultrasound imaging.
//!
//! # Mathematical Foundation
//!
//! Given a slow-time signal matrix S[space × time]:
//! 1. Perform SVD: S = UΣV^T
//! 2. Tissue clutter occupies low-rank subspace (first K singular values)
//! 3. Blood flow occupies remaining subspace (singular values K+1 to N)
//! 4. Reconstruct blood-only signal: S_blood = U[:,K+1:] Σ[K+1:,K+1:] V[:,K+1:]^T
//!
//! # References
//!
//! - Demené, C., et al. (2015). "Spatiotemporal clutter filtering of ultrafast
//!   ultrasound data highly increases Doppler and fUltrasound sensitivity."
//!   *Scientific Reports*, 5, 11203. DOI: 10.1038/srep11203
//!
//! - Baranger, J., et al. (2018). "Adaptive spatiotemporal SVD clutter filtering."
//!   *IEEE Trans. Medical Imaging*, 37(7), 1574-1586. DOI: 10.1109/TMI.2018.2789499

use crate::core::error::{KwaversError, KwaversResult};
use crate::math::linear_algebra::LinearAlgebra;
use ndarray::{Array1, Array2, Axis};

#[cfg(test)]
mod tests;

/// Configuration for SVD clutter filter
#[derive(Debug, Clone)]
pub struct SvdClutterFilterConfig {
    /// Number of clutter components to remove (tissue rank)
    /// Typical: 2-5 for most tissue motion scenarios
    pub clutter_rank: usize,

    /// Minimum ensemble length (number of temporal frames)
    /// Must be >= 2 * clutter_rank for proper rank separation
    /// Typical: 100-200 frames
    pub min_ensemble_length: usize,

    /// Use automatic rank selection instead of fixed rank
    /// When true, uses energy threshold or knee detection
    pub auto_rank_selection: bool,

    /// Energy threshold for automatic rank selection (0.0 to 1.0)
    /// Keep singular values until this fraction of total energy is removed
    /// Typical: 0.999 (99.9% of clutter energy)
    pub energy_threshold: f64,
}

impl Default for SvdClutterFilterConfig {
    fn default() -> Self {
        Self {
            clutter_rank: 3,
            min_ensemble_length: 100,
            auto_rank_selection: false,
            energy_threshold: 0.999,
        }
    }
}

impl SvdClutterFilterConfig {
    /// Create configuration with manual rank selection
    #[must_use]
    pub fn with_fixed_rank(clutter_rank: usize) -> Self {
        Self {
            clutter_rank,
            auto_rank_selection: false,
            ..Default::default()
        }
    }

    /// Create configuration with automatic rank selection
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn with_auto_rank(energy_threshold: f64) -> Self {
        Self {
            auto_rank_selection: true,
            energy_threshold,
            ..Default::default()
        }
    }

    /// Validate configuration
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.clutter_rank == 0 {
            return Err(KwaversError::InvalidInput(
                "Clutter rank must be at least 1".to_owned(),
            ));
        }

        if self.min_ensemble_length < 2 * self.clutter_rank {
            return Err(KwaversError::InvalidInput(format!(
                "Ensemble length ({}) must be at least 2 × clutter_rank ({})",
                self.min_ensemble_length,
                2 * self.clutter_rank
            )));
        }

        if self.energy_threshold <= 0.0 || self.energy_threshold >= 1.0 {
            return Err(KwaversError::InvalidInput(
                "Energy threshold must be in range (0.0, 1.0)".to_owned(),
            ));
        }

        Ok(())
    }
}

/// Spatiotemporal SVD clutter filter
///
/// Separates tissue clutter from blood flow signals using singular value
/// decomposition of the slow-time data matrix.
#[derive(Debug)]
pub struct SignalSvdClutterFilter {
    config: SvdClutterFilterConfig,
}

impl SignalSvdClutterFilter {
    /// Create new SVD clutter filter with configuration
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: SvdClutterFilterConfig) -> KwaversResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Apply clutter filter to slow-time data matrix
    ///
    /// # Arguments
    ///
    /// * `slow_time_data` - Matrix of shape (n_pixels, n_frames)
    ///   Each row is the temporal signal at a single spatial pixel
    ///   Each column is a single time frame across all pixels
    ///
    /// # Returns
    ///
    /// Filtered data matrix with tissue clutter removed, same shape as input
    ///
    /// # Algorithm
    ///
    /// 1. Center the data (subtract temporal mean from each pixel)
    /// 2. Compute SVD: S = UΣV^T
    /// 3. Determine clutter rank K (fixed or automatic)
    /// 4. Zero out first K singular values: Σ[0:K, 0:K] = 0
    /// 5. Reconstruct: S_filtered = UΣ_filtered V^T
    /// 6. Add back temporal means
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn filter(&self, slow_time_data: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let (n_pixels, n_frames) = slow_time_data.dim();

        // Validate input dimensions
        if n_frames < self.config.min_ensemble_length {
            return Err(KwaversError::InvalidInput(format!(
                "Ensemble length ({}) is below minimum ({})",
                n_frames, self.config.min_ensemble_length
            )));
        }

        // Step 1: Center the data (remove temporal mean from each pixel)
        let temporal_means = slow_time_data.mean_axis(Axis(1)).unwrap();
        let mut centered_data = slow_time_data.clone();
        for (i, mean) in temporal_means.iter().enumerate() {
            for j in 0..n_frames {
                centered_data[[i, j]] -= mean;
            }
        }

        // Step 2: Compute SVD: S = UΣV^T
        // Note: LinearAlgebra::svd returns (U, Σ, V), not (U, Σ, V^T)
        let (u, mut sigma, v) = LinearAlgebra::svd(&centered_data)?;

        // Step 3: Determine clutter rank
        let clutter_rank = if self.config.auto_rank_selection {
            self.estimate_clutter_rank(&sigma)?
        } else {
            self.config.clutter_rank.min(sigma.len())
        };

        // Step 4: Zero out first K singular values (clutter subspace)
        for i in 0..clutter_rank {
            sigma[i] = 0.0;
        }

        // Step 5: Reconstruct filtered signal: S_filtered = U * Σ * V^T
        // SVD returns U (n_pixels × min(n_pixels, n_frames))
        //             Σ (min(n_pixels, n_frames))
        //             V (n_frames × min(n_pixels, n_frames))

        let rank = sigma.len();

        // First compute U * Σ (multiply each column of U by corresponding singular value)
        let mut u_sigma = Array2::<f64>::zeros((n_pixels, rank));
        for j in 0..rank {
            for i in 0..n_pixels {
                u_sigma[[i, j]] = u[[i, j]] * sigma[j];
            }
        }

        // Then compute (U * Σ) * V^T
        let filtered_data = u_sigma.dot(&v.t());

        // Step 6: Add back temporal means
        let mut final_data = filtered_data;
        for (i, mean) in temporal_means.iter().enumerate() {
            for j in 0..n_frames {
                final_data[[i, j]] += mean;
            }
        }

        Ok(final_data)
    }

    /// Estimate clutter rank using energy threshold
    ///
    /// Finds the minimum rank K such that the first K singular values
    /// capture at least energy_threshold fraction of total signal energy.
    ///
    /// This corresponds to removing the dominant tissue motion components.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn estimate_clutter_rank(&self, singular_values: &Array1<f64>) -> KwaversResult<usize> {
        // Compute total energy (sum of squared singular values)
        let total_energy: f64 = singular_values.iter().map(|&s| s * s).sum();

        if total_energy == 0.0 {
            return Ok(0); // No signal, no clutter
        }

        // Find minimum K such that first K components have >= threshold energy
        let mut cumulative_energy = 0.0;
        for (k, &sigma) in singular_values.iter().enumerate() {
            cumulative_energy += sigma * sigma;
            let fraction = cumulative_energy / total_energy;

            if fraction >= self.config.energy_threshold {
                // Found the knee - these components are clutter
                return Ok((k + 1).min(singular_values.len()));
            }
        }

        // If we get here, threshold was too high - use all but last component
        Ok(singular_values.len().saturating_sub(1))
    }

    /// Compute Power Doppler intensity from filtered slow-time data
    ///
    /// Power Doppler is the temporal variance of the filtered signal,
    /// indicating blood flow presence and intensity.
    ///
    /// # Arguments
    ///
    /// * `filtered_data` - Output from `filter()` method (n_pixels × n_frames)
    ///
    /// # Returns
    ///
    /// Power Doppler image (n_pixels,) where each value is the temporal
    /// variance of the filtered signal at that pixel
    #[must_use]
    pub fn compute_power_doppler(&self, filtered_data: &Array2<f64>) -> Array1<f64> {
        let (n_pixels, n_frames) = filtered_data.dim();
        let mut power_doppler = Array1::zeros(n_pixels);

        for i in 0..n_pixels {
            // Extract temporal signal for this pixel
            let temporal_signal = filtered_data.row(i);

            // Compute mean
            let mean = temporal_signal.sum() / (n_frames as f64);

            // Compute variance
            let variance: f64 = temporal_signal
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<f64>()
                / (n_frames as f64);

            power_doppler[i] = variance;
        }

        power_doppler
    }

    /// Get current configuration
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn config(&self) -> &SvdClutterFilterConfig {
        &self.config
    }

    /// Estimate signal-to-clutter ratio (SCR) improvement
    ///
    /// Compares clutter energy before and after filtering.
    ///
    /// # Returns
    ///
    /// SCR improvement in dB
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn estimate_scr_improvement(
        &self,
        original: &Array2<f64>,
        filtered: &Array2<f64>,
    ) -> KwaversResult<f64> {
        if original.dim() != filtered.dim() {
            return Err(KwaversError::InvalidInput(
                "Original and filtered data must have same dimensions".to_owned(),
            ));
        }

        // Compute clutter (difference between original and filtered)
        let clutter = original - filtered;

        // Compute energy of original clutter-dominated signal
        let original_energy: f64 = original.iter().map(|&x| x * x).sum();

        // Compute energy of remaining clutter after filtering
        let clutter_energy: f64 = clutter.iter().map(|&x| x * x).sum();

        if clutter_energy == 0.0 {
            return Ok(f64::INFINITY); // Perfect filtering
        }

        // SCR improvement in dB
        let scr_db = 10.0 * (original_energy / clutter_energy).log10();

        Ok(scr_db)
    }
}
