//! Short-Lag Spatial Coherence (SLSC) Beamforming
//!
//! # Overview
//!
//! Short-Lag Spatial Coherence (SLSC) beamforming is an advanced imaging technique
//! that improves image quality by leveraging the spatial coherence of backscattered
//! ultrasound signals. Unlike conventional delay-and-sum beamforming which only uses
//! amplitude information, SLSC exploits the phase coherence between signals received
//! at different array elements.
//!
//! ## Key Advantages
//!
//! - **Improved Clutter Rejection**: Suppresses incoherent noise and reverberation clutter
//! - **Better Contrast**: Enhances tissue boundaries and cyst visualization
//! - **Robust to Phase Aberration**: Less sensitive to sound speed variations
//! - **No Additional Hardware**: Uses same data as conventional beamforming
//!
//! # Mathematical Foundation
//!
//! ## Spatial Coherence
//!
//! The spatial coherence between signals received at elements i and j is defined as:
//!
//! ```text
//! C(d) = |Σ_{k=1}^{N-d} s_k · s_{k+d}^*| / √[Σ|s_k|² · Σ|s_{k+d}|²]
//! ```
//!
//! where:
//! - `d` = element lag (distance between elements)
//! - `s_k` = signal at element k after delay compensation
//! - `N` = total number of elements
//! - `*` = complex conjugate
//!
//! # References
//!
//! - Lediju, M. A., et al. (2011). "Short-lag spatial coherence of backscattered echoes."
//!   *IEEE Trans. Ultrason., Ferroelect., Freq. Control*, 58(7).
//!
//! - Jakovljevic, M., et al. (2013). "In vivo application of short-lag spatial coherence
//!   imaging in human liver." *Ultrasonic Imaging*, 35(3).

pub use adaptive::AdaptiveSlsc;
pub use batch::process_slsc_batch;
pub use beamformer::SlscBeamformer;
pub use multi::MultiLagSlsc;

mod adaptive;
mod batch;
mod beamformer;
mod multi;
#[cfg(test)]
mod tests;

use kwavers_core::error::{KwaversError, KwaversResult};

/// Configuration for SLSC beamforming
#[derive(Debug, Clone)]
pub struct SlscConfig {
    /// Maximum lag to use (M). Typically 10-20% of array elements.
    pub max_lag: usize,
    /// Weighting function for different lags
    pub weighting: LagWeighting,
    /// Whether to use normalized coherence
    pub normalize: bool,
}

impl Default for SlscConfig {
    fn default() -> Self {
        Self {
            max_lag: 10,
            weighting: LagWeighting::Uniform,
            normalize: true,
        }
    }
}

impl SlscConfig {
    /// Create a new config with specified max lag
    #[must_use]
    pub fn with_max_lag(max_lag: usize) -> Self {
        Self {
            max_lag,
            ..Default::default()
        }
    }

    /// Create a new config with triangular weighting
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn with_triangular_weighting() -> Self {
        Self {
            weighting: LagWeighting::Triangular,
            ..Default::default()
        }
    }

    /// Validate the configuration
    ///
    /// # Errors
    /// Returns error if max_lag is 0
    pub fn validate(&self) -> KwaversResult<()> {
        if self.max_lag == 0 {
            return Err(KwaversError::Config(
                kwavers_core::error::ConfigError::InvalidValue {
                    parameter: "max_lag".to_owned(),
                    value: "0".to_owned(),
                    constraint: "max_lag must be greater than 0".to_owned(),
                },
            ));
        }
        Ok(())
    }
}

/// Weighting function for lag contributions
#[derive(Debug, Clone, PartialEq)]
pub enum LagWeighting {
    /// Uniform weighting (all lags equal)
    Uniform,
    /// Triangular weighting (linear decrease with lag)
    Triangular,
    /// Hamming window weighting
    Hamming,
    /// Custom weighting with user-defined weights
    Custom { weights: Box<[f64; 64]>, len: usize },
}

impl LagWeighting {
    /// Get the weight for a specific lag
    #[must_use]
    pub fn weight(&self, lag: usize, max_lag: usize) -> f64 {
        match self {
            Self::Uniform => 1.0,
            Self::Triangular => {
                if lag >= max_lag {
                    0.0
                } else {
                    1.0 - (lag as f64 / max_lag as f64)
                }
            }
            Self::Hamming => {
                if lag >= max_lag {
                    0.0
                } else {
                    let alpha = 0.54;
                    let beta = 0.46;
                    let pi = std::f64::consts::PI;
                    alpha - beta * (2.0 * pi * lag as f64 / max_lag as f64).cos()
                }
            }
            Self::Custom { weights, len } => {
                if lag < *len {
                    weights[lag]
                } else {
                    0.0
                }
            }
        }
    }
}
