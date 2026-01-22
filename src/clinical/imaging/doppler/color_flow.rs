//! Color Flow Imaging
//!
//! Generates 2D velocity maps for real-time flow visualization.

use crate::clinical::imaging::doppler::{
    AutocorrelationConfig, AutocorrelationEstimator, DopplerResult, WallFilter, WallFilterConfig,
};
use crate::core::error::KwaversResult;
use ndarray::{Array2, ArrayView3};
use num_complex::Complex64;

/// Color flow imaging configuration
#[derive(Debug, Clone)]
pub struct ColorFlowConfig {
    pub autocorrelation: AutocorrelationConfig,
    pub wall_filter: WallFilterConfig,
    pub spatial_averaging: Option<(usize, usize)>, // (axial, lateral) kernel size
}

impl Default for ColorFlowConfig {
    fn default() -> Self {
        Self {
            autocorrelation: AutocorrelationConfig::default(),
            wall_filter: WallFilterConfig::default(),
            spatial_averaging: Some((3, 3)),
        }
    }
}

/// 2D velocity map result
pub type VelocityMap = Array2<f64>;

/// Color flow imaging processor
#[derive(Debug)]
pub struct ColorFlowImaging {
    config: ColorFlowConfig,
    estimator: AutocorrelationEstimator,
    wall_filter: WallFilter,
}

impl ColorFlowImaging {
    pub fn new(config: ColorFlowConfig) -> Self {
        let estimator = AutocorrelationEstimator::new(config.autocorrelation.clone());
        let wall_filter = WallFilter::new(config.wall_filter.clone());

        Self {
            config,
            estimator,
            wall_filter,
        }
    }

    /// Process I/Q data to generate color flow image
    pub fn process(&self, iq_data: &ArrayView3<Complex64>) -> KwaversResult<DopplerResult> {
        // Apply wall filter to remove clutter
        let filtered = self.wall_filter.apply(iq_data)?;

        // Estimate velocities
        let (mut velocity, variance) = self.estimator.estimate(&filtered.view())?;

        // Apply spatial averaging if configured
        if let Some((axial_size, lateral_size)) = self.config.spatial_averaging {
            velocity = self.spatial_average(&velocity, axial_size, lateral_size);
        }

        Ok(DopplerResult {
            velocity,
            variance,
            power: None,
            center_frequency: self.config.autocorrelation.center_frequency,
            prf: self.config.autocorrelation.prf,
        })
    }

    fn spatial_average(&self, data: &Array2<f64>, axial: usize, lateral: usize) -> Array2<f64> {
        let (n_depths, n_beams) = data.dim();
        let mut averaged = data.clone();

        let axial_half = axial / 2;
        let lateral_half = lateral / 2;

        for i in axial_half..(n_depths - axial_half) {
            for j in lateral_half..(n_beams - lateral_half) {
                let mut sum = 0.0;
                let mut count = 0;

                for di in 0..axial {
                    for dj in 0..lateral {
                        let ii = i + di - axial_half;
                        let jj = j + dj - lateral_half;
                        sum += data[[ii, jj]];
                        count += 1;
                    }
                }

                averaged[[i, j]] = sum / (count as f64);
            }
        }

        averaged
    }
}
