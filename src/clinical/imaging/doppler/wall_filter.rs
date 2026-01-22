//! Wall Filter for Clutter Rejection
//!
//! Removes slow-moving clutter from vessel walls and tissue while preserving
//! blood flow signals. Essential for clean Doppler velocity estimation.

use crate::core::error::KwaversResult;
use ndarray::{Array3, ArrayView3};
use num_complex::Complex64;

/// Wall filter types
#[derive(Debug, Clone, Copy)]
pub enum FilterType {
    /// Simple high-pass filter (remove DC component)
    HighPass,
    /// Polynomial regression filter (Hoeks et al.)
    Polynomial { order: usize },
    /// IIR filter (infinite impulse response)
    IIR { cutoff_frequency: f64 },
}

/// Wall filter configuration
#[derive(Debug, Clone)]
pub struct WallFilterConfig {
    pub filter_type: FilterType,
    pub prf: f64,
}

impl Default for WallFilterConfig {
    fn default() -> Self {
        Self {
            filter_type: FilterType::Polynomial { order: 2 },
            prf: 4e3,
        }
    }
}

/// Wall filter for clutter rejection
#[derive(Debug, Clone)]
pub struct WallFilter {
    config: WallFilterConfig,
}

impl WallFilter {
    pub fn new(config: WallFilterConfig) -> Self {
        Self { config }
    }

    /// Apply wall filter to I/Q data
    ///
    /// Removes slow-moving clutter (tissue, vessel walls) while preserving
    /// blood flow signals.
    pub fn apply(&self, iq_data: &ArrayView3<Complex64>) -> KwaversResult<Array3<Complex64>> {
        let (ensemble_size, n_depths, n_beams) = iq_data.dim();
        let mut filtered = Array3::<Complex64>::zeros((ensemble_size, n_depths, n_beams));

        match self.config.filter_type {
            FilterType::HighPass => {
                // Simple DC removal: subtract mean from each ensemble
                for depth in 0..n_depths {
                    for beam in 0..n_beams {
                        let mean = (0..ensemble_size)
                            .map(|n| iq_data[[n, depth, beam]])
                            .sum::<Complex64>()
                            / (ensemble_size as f64);

                        for n in 0..ensemble_size {
                            filtered[[n, depth, beam]] = iq_data[[n, depth, beam]] - mean;
                        }
                    }
                }
            }
            FilterType::Polynomial { order: _ } => {
                // Polynomial regression filter (to be implemented)
                // For now, use simple high-pass
                for depth in 0..n_depths {
                    for beam in 0..n_beams {
                        let mean = (0..ensemble_size)
                            .map(|n| iq_data[[n, depth, beam]])
                            .sum::<Complex64>()
                            / (ensemble_size as f64);

                        for n in 0..ensemble_size {
                            filtered[[n, depth, beam]] = iq_data[[n, depth, beam]] - mean;
                        }
                    }
                }
            }
            FilterType::IIR { .. } => {
                // IIR filter (to be implemented)
                // For now, use simple high-pass
                for depth in 0..n_depths {
                    for beam in 0..n_beams {
                        let mean = (0..ensemble_size)
                            .map(|n| iq_data[[n, depth, beam]])
                            .sum::<Complex64>()
                            / (ensemble_size as f64);

                        for n in 0..ensemble_size {
                            filtered[[n, depth, beam]] = iq_data[[n, depth, beam]] - mean;
                        }
                    }
                }
            }
        }

        Ok(filtered)
    }
}
