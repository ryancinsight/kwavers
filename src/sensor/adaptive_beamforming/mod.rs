//! # Adaptive Beamforming Algorithms
//!
//! This module provides a comprehensive suite of adaptive beamforming algorithms for
//! array signal processing, organized into focused submodules for maintainability and clarity.
//!
//! ## Overview
//!
//! Adaptive beamforming optimizes array antenna patterns in real-time based on received
//! signal statistics. Unlike conventional beamforming (delay-and-sum), adaptive techniques
//! minimize interference while maintaining gain in the desired direction.
//!
//! ## Available Algorithms
//!
//! ### Conventional Beamforming
//! - [`DelayAndSum`]: Simple phase-aligned summation (conventional beamforming)
//!
//! ### Adaptive Algorithms
//! - [`MinimumVariance`]: MVDR/Capon beamformer - minimizes output power with unit gain constraint
//! - [`RobustCapon`]: Robust version of MVDR with uncertainty bounds
//!
//! ### High-Resolution Algorithms
//! - [`MUSIC`]: Multiple Signal Classification using eigenstructure
//! - [`EigenspaceMV`]: Eigenspace-based minimum variance beamformer
//!
//! ### Subspace Tracking
//! - [`SubspaceTracker`]: PAST algorithm for online subspace estimation
//! - [`OrthonormalSubspaceTracker`]: OPAST algorithm with orthonormal constraints
//!
//! ### Utilities
//! - [`CovarianceTaper`]: Spatial smoothing for improved covariance estimation
//! - [`ArrayGeometry`]: Array element positioning and configuration
//! - [`SteeringVector`]: Steering vector computation utilities
//!
//! ## Basic Usage
//!
//! ```rust,no_run
//! use kwavers::sensor::adaptive_beamforming::{MinimumVariance, ArrayGeometry};
//!
//! // Create a linear array with 8 elements spaced λ/2 apart
//! let geometry = ArrayGeometry::linear(8, 0.5);
//!
//! // Create MVDR beamformer for adaptive beamforming
//! let beamformer = MinimumVariance::default();
//!
//! // In practice, you would compute weights from covariance and steering:
//! // let weights = beamformer.compute_weights(&covariance, &steering);
//! // let output = weights.dot(&received_signals); // Apply to array data
//! ```
//!
//! ## Advanced Usage with Subspace Methods
//!
//! ```rust,no_run
//! use kwavers::sensor::adaptive_beamforming::{
//!     MUSIC, EigenspaceMV, estimate_num_sources, SourceEstimationCriterion
//! };
//! use ndarray::Array2;
//! use num_complex::Complex64;
//!
//! // Covariance matrix from signal processing
//! let covariance: Array2<Complex64> = Array2::eye(8);
//!
//! // Estimate number of sources using information-theoretic criteria
//! let num_snapshots = 100; // Snapshots used to compute covariance
//! let num_sources = estimate_num_sources(
//!     &covariance, num_snapshots, SourceEstimationCriterion::AIC
//! );
//!
//! // Create high-resolution beamformers
//! let music = MUSIC::new(num_sources);
//! let esmv = EigenspaceMV::new(num_sources);
//!
//! // In practice, compute spectra/beam patterns:
//! // let spectrum = music.pseudospectrum(&cov, &steering);
//! // let weights = esmv.compute_weights(&cov, &steering);
//! ```
//!
//! ## Performance Considerations
//!
//! - **MVDR**: O(N³) complexity due to matrix inversion, suitable for small arrays (< 32 elements)
//! - **MUSIC**: O(N³) for eigendecomposition, excellent resolution for multiple sources
//! - **Subspace Tracking**: O(N²) per update, suitable for real-time applications
//!
//! Use [`CovarianceTaper`] for improved performance with coherent sources.
//!
//! ## References
//!
//! - Van Trees, H. L. (2002). *Optimum Array Processing*. Wiley.
//! - Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis".
//!   *Proceedings of the IEEE*, 57(8), 1408-1418.
//! - Schmidt, R. (1986). "Multiple emitter location and signal parameter estimation".
//!   *IEEE Transactions on Antennas and Propagation*, 34(3), 276-280.

pub mod adaptive;
pub mod algorithms;
#[cfg(feature = "legacy_algorithms")]
#[allow(dead_code)] // Temporary: contains code being refactored
pub(crate) mod algorithms_old;
pub mod array_geometry;
pub mod beamformer;
pub mod conventional;
pub mod matrix_utils;
pub mod opast;
pub mod past;
pub mod source_estimation;
pub mod steering;
pub mod subspace;
pub mod tapering;
pub mod weights;

// Re-export main types - single implementation
pub use adaptive::{MinimumVariance, RobustCapon};
pub use algorithms::{BeamformingAlgorithm, DelayAndSum, MinimumVariance as MvLegacy};
pub use array_geometry::{ArrayGeometry, ElementPosition};
pub use beamformer::AdaptiveBeamformer;
pub use conventional::{BeamformingAlgorithm as ConventionalAlgorithm, DelayAndSum as DsLegacy};
pub use matrix_utils::{eigen_hermitian, invert_matrix};
pub use opast::OrthonormalSubspaceTracker;
pub use past::SubspaceTracker;
pub use source_estimation::{estimate_num_sources, SourceEstimationCriterion};
pub use steering::{SteeringMatrix, SteeringVector};
pub use subspace::{EigenspaceMV, MUSIC};
pub use tapering::CovarianceTaper;
pub use weights::{WeightCalculator, WeightingScheme};

// Legacy algorithms (deprecated) - available with --features legacy_algorithms
#[cfg(feature = "legacy_algorithms")]
pub use algorithms_old::{
    CovarianceTaper as LegacyCovarianceTaper, DelayAndSum as LegacyDelayAndSum,
    EigenspaceMV as LegacyEigenspaceMV, MinimumVariance as LegacyMinimumVariance,
    OrthonormalSubspaceTracker as LegacyOrthonormalSubspaceTracker,
    RobustCapon as LegacyRobustCapon, SubspaceTracker as LegacySubspaceTracker,
    MUSIC as LegacyMUSIC,
};
