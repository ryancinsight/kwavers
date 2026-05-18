//! MUSIC (Multiple Signal Classification) Algorithm
//!
//! Implements super-resolution direction-of-arrival (DoA) estimation using subspace methods.
//!
//! # Theory
//!
//! MUSIC exploits the eigenstructure of the spatial covariance matrix R to achieve
//! super-resolution source localization beyond the Rayleigh limit.
//!
//! Given M sensors and K sources (K < M):
//! - Signal subspace: Span of K steering vectors
//! - Noise subspace: Orthogonal complement (M-K dimensional)
//!
//! The MUSIC pseudospectrum is defined as:
//!
//! P_MUSIC(θ) = 1 / (a(θ)^H E_n E_n^H a(θ))
//!
//! where:
//! - a(θ) is the steering vector for location θ
//! - E_n is the noise subspace eigenvector matrix
//! - ^H denotes conjugate transpose
//!
//! Source locations correspond to peaks (nulls in denominator) where a(θ) ⊥ E_n.
//!
//! # References
//!
//! - Schmidt, R. O. (1986). "Multiple emitter location and signal parameter estimation"
//!   IEEE Trans. Antennas Propag., 34(3), 276-280.
//! - Stoica, P., & Nehorai, A. (1989). "MUSIC, maximum likelihood, and Cramér–Rao bound"
//!   IEEE Trans. Acoust., Speech, Signal Process., 37(5), 720-741.
//! - Van Trees, H. L. (2002). "Optimum Array Processing" - Part IV of Detection, Estimation,
//!   and Modulation Theory. Wiley-Interscience.
//! - Wax, M., & Kailath, T. (1985). "Detection of signals by information theoretic criteria"
//!   IEEE Trans. Acoust., Speech, Signal Process., 33(2), 387-392.

use super::config::AcousticLocalizationConfig;
use super::model_order::ModelOrderCriterion;
use crate::analysis::signal_processing::localization::SourceLocation;

mod localization_impl;
mod processor;
mod spectrum;
#[cfg(test)]
mod tests;

pub use processor::MUSICProcessor;

/// MUSIC configuration
#[derive(Debug, Clone)]
pub struct MUSICConfig {
    /// Base localization config
    pub config: AcousticLocalizationConfig,

    /// Number of sources to detect (None = automatic via AIC/MDL)
    pub num_sources: Option<usize>,

    /// Model order selection criterion (used if num_sources is None)
    pub model_order_criterion: ModelOrderCriterion,

    /// Search grid resolution (number of points per dimension)
    pub grid_resolution: usize,

    /// Search region bounds [xmin, xmax, ymin, ymax, zmin, zmax] in meters
    pub search_bounds: [f64; 6],

    /// Minimum separation between detected sources (m)
    pub min_source_separation: f64,

    /// Number of temporal snapshots for covariance estimation
    pub num_snapshots: usize,

    /// Diagonal loading factor for covariance regularization
    ///
    /// Added to diagonal: R_reg = R + δI where δ = loading_factor × trace(R)/M
    /// Prevents ill-conditioning. Typical value: 1e-6 to 1e-3.
    pub diagonal_loading: f64,

    /// Center frequency for steering vector calculation (Hz)
    pub center_frequency: f64,
}

impl MUSICConfig {
    /// Create new MUSIC configuration
    #[must_use]
    pub fn new(config: AcousticLocalizationConfig, num_sources: Option<usize>) -> Self {
        let center_frequency = config.sampling_frequency / 4.0;
        Self {
            config,
            num_sources,
            model_order_criterion: ModelOrderCriterion::MDL,
            grid_resolution: 50,
            search_bounds: [-0.1, 0.1, -0.1, 0.1, -0.1, 0.1],
            min_source_separation: 0.01,
            num_snapshots: 100,
            diagonal_loading: 1e-6,
            center_frequency,
        }
    }

    /// Set model order selection criterion
    #[must_use]
    pub fn with_criterion(mut self, criterion: ModelOrderCriterion) -> Self {
        self.model_order_criterion = criterion;
        self
    }

    /// Set grid resolution
    #[must_use]
    pub fn with_grid_resolution(mut self, resolution: usize) -> Self {
        self.grid_resolution = resolution;
        self
    }

    /// Set search region bounds
    #[must_use]
    pub fn with_search_bounds(mut self, bounds: [f64; 6]) -> Self {
        self.search_bounds = bounds;
        self
    }

    /// Set minimum source separation
    #[must_use]
    pub fn with_min_separation(mut self, separation: f64) -> Self {
        self.min_source_separation = separation;
        self
    }

    /// Set number of snapshots
    #[must_use]
    pub fn with_num_snapshots(mut self, snapshots: usize) -> Self {
        self.num_snapshots = snapshots;
        self
    }

    /// Set diagonal loading factor
    #[must_use]
    pub fn with_diagonal_loading(mut self, loading: f64) -> Self {
        self.diagonal_loading = loading;
        self
    }

    /// Set center frequency
    #[must_use]
    pub fn with_center_frequency(mut self, frequency: f64) -> Self {
        self.center_frequency = frequency;
        self
    }
}

impl Default for MUSICConfig {
    fn default() -> Self {
        Self::new(AcousticLocalizationConfig::default(), Some(1))
    }
}

/// MUSIC result with multiple sources
#[derive(Debug, Clone)]
pub struct MUSICResult {
    /// Detected source locations
    pub sources: Vec<SourceLocation>,

    /// MUSIC pseudospectrum (flattened grid)
    pub pseudospectrum: Vec<f64>,

    /// Grid dimensions [nx, ny, nz]
    pub grid_dims: [usize; 3],

    /// Search bounds used [xmin, xmax, ymin, ymax, zmin, zmax]
    pub search_bounds: [f64; 6],

    /// Number of sources detected
    pub num_sources: usize,

    /// Noise subspace dimension
    pub noise_subspace_dim: usize,
}
