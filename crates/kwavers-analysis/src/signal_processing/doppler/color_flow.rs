//! Color Flow Imaging
//!
//! Generates 2D velocity maps for real-time flow visualization.

use crate::signal_processing::doppler::{
    AutocorrelationConfig, AutocorrelationEstimator, DopplerResult, WallFilter, WallFilterConfig,
};
use eunomia::Complex64;
use kwavers_core::error::KwaversResult;
use leto::{Array2, ArrayView3};

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
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
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
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
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

    /// Exposed for testing — the method is otherwise private.
    #[cfg(test)]
    pub(crate) fn spatial_average_pub(
        &self,
        data: &Array2<f64>,
        axial: usize,
        lateral: usize,
    ) -> Array2<f64> {
        self.spatial_average(data, axial, lateral)
    }

    fn spatial_average(&self, data: &Array2<f64>, axial: usize, lateral: usize) -> Array2<f64> {
        let [n_depths, n_beams] = data.shape();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal_processing::doppler::{
        AutocorrelationConfig, WallFilterConfig, WallFilterType,
    };
    use eunomia::Complex64;
    use leto::{Array2, Array3};

    // Helper: config with HighPass wall filter and no spatial averaging.
    // Uses default AutocorrelationConfig (ensemble_size=10, f₀=5 MHz, PRF=4 kHz).
    fn cfg_no_avg() -> ColorFlowConfig {
        ColorFlowConfig {
            autocorrelation: AutocorrelationConfig::default(),
            wall_filter: WallFilterConfig {
                filter_type: WallFilterType::HighPass,
                prf: 4e3,
            },
            spatial_averaging: None,
        }
    }

    // ─── process: zero input ──────────────────────────────────────────────────

    /// Zero IQ data → zero velocity at every pixel.
    ///
    /// Pipeline:
    ///   1. HighPass(zeros) = zeros (mean = 0)
    ///   2. R₁ = 0 → phase = atan2(0, 0) = 0
    ///   3. velocity = 0
    #[test]
    fn color_flow_zero_input_produces_zero_velocity() {
        let cfi = ColorFlowImaging::new(cfg_no_avg());
        let iq = Array3::<Complex64>::zeros((10, 8, 4));
        let result = cfi.process(&iq.view()).unwrap();
        assert_eq!(result.velocity.shape(), [8, 4]);
        for &val in result.velocity.iter() {
            assert_eq!(val, 0.0, "zero IQ must yield zero velocity, got {val}");
        }
    }

    // ─── process: output struct fields ────────────────────────────────────────

    /// Result fields match configured autocorrelation parameters;
    /// `power` is always `None` (not computed by this estimator).
    #[test]
    fn color_flow_result_fields_match_config() {
        let cfg = cfg_no_avg();
        let f0 = cfg.autocorrelation.center_frequency; // 5e6
        let prf = cfg.autocorrelation.prf; // 4e3
        let cfi = ColorFlowImaging::new(cfg);
        let iq = Array3::<Complex64>::zeros((10, 4, 4));
        let result = cfi.process(&iq.view()).unwrap();
        assert_eq!(
            result.center_frequency, f0,
            "center_frequency mismatch: expected {f0}, got {}",
            result.center_frequency
        );
        assert_eq!(
            result.prf, prf,
            "prf mismatch: expected {prf}, got {}",
            result.prf
        );
        assert!(result.power.is_none(), "power should be None");
    }

    // ─── process: output shape ────────────────────────────────────────────────

    /// Velocity and variance maps must have shape (n_depths, n_beams).
    #[test]
    fn color_flow_velocity_map_shape_matches_input() {
        let cfi = ColorFlowImaging::new(cfg_no_avg());
        let iq = Array3::<Complex64>::zeros((10, 16, 8));
        let result = cfi.process(&iq.view()).unwrap();
        assert_eq!(result.velocity.shape(), [16, 8], "velocity shape wrong");
        assert_eq!(result.variance.shape(), [16, 8], "variance shape wrong");
    }

    // ─── process: spatial averaging ───────────────────────────────────────────

    /// Box averaging on a uniform (all-zero) velocity map preserves zeros.
    ///
    /// For a constant map C, a box filter of any integer kernel size returns C
    /// at every interior cell. Boundary cells are untouched (copied from the
    /// pre-average array, which is also C).  With C = 0 the entire map stays 0.
    #[test]
    fn color_flow_spatial_averaging_preserves_uniform_zero_velocity() {
        let cfg = ColorFlowConfig {
            autocorrelation: AutocorrelationConfig::default(),
            wall_filter: WallFilterConfig {
                filter_type: WallFilterType::HighPass,
                prf: 4e3,
            },
            spatial_averaging: Some((3, 3)),
        };
        let cfi = ColorFlowImaging::new(cfg);
        let iq = Array3::<Complex64>::zeros((10, 8, 8));
        let result = cfi.process(&iq.view()).unwrap();
        for &val in result.velocity.iter() {
            assert_eq!(
                val, 0.0,
                "uniform zero map must remain zero after averaging, got {val}"
            );
        }
    }

    // ─── spatial_average: box filter arithmetic ───────────────────────────────

    /// Box filter of kernel (3,3) on a constant 5×5 map returns that constant at
    /// every interior cell (i∈[1..4), j∈[1..4)).
    ///
    /// Sum = 9 × C, count = 9, averaged = C. Boundary cells are left equal to C
    /// because the impl clones the input before averaging.
    #[test]
    fn color_flow_spatial_average_box_filter_on_constant_map() {
        let cfg = ColorFlowConfig {
            autocorrelation: AutocorrelationConfig::default(),
            wall_filter: WallFilterConfig {
                filter_type: WallFilterType::HighPass,
                prf: 4e3,
            },
            spatial_averaging: Some((3, 3)),
        };
        let cfi = ColorFlowImaging::new(cfg);
        let c = 7.5_f64;
        let data = Array2::from_elem((5, 5), c);
        let averaged = cfi.spatial_average_pub(&data, 3, 3);
        // All interior cells (i∈1..4, j∈1..4) must equal c.
        for i in 1..4 {
            for j in 1..4 {
                assert!(
                    (averaged[[i, j]] - c).abs() < 1e-12,
                    "interior cell ({i},{j}): expected {c}, got {}",
                    averaged[[i, j]]
                );
            }
        }
    }

    // ─── process: error path ─────────────────────────────────────────────────

    /// Providing fewer ensemble pulses than the configured ensemble_size returns Err.
    #[test]
    fn color_flow_wrong_ensemble_size_returns_error() {
        let cfi = ColorFlowImaging::new(cfg_no_avg()); // expects 10
        let iq = Array3::<Complex64>::zeros((5, 8, 4)); // only 5
        assert!(
            cfi.process(&iq.view()).is_err(),
            "mismatched ensemble size must return Err"
        );
    }
}
