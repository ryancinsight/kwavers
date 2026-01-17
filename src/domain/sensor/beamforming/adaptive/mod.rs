//! # Adaptive Beamforming Algorithms (DEPRECATED)
//!
//! **⚠️ DEPRECATION NOTICE**: This module has been moved to `analysis::signal_processing::beamforming::adaptive`.
//!
//! ## Migration Guide
//!
//! Update your imports as follows:
//!
//! ```rust,ignore
//! // OLD (deprecated):
//! use kwavers::domain::sensor::beamforming::adaptive::{MinimumVariance, MUSIC, EigenspaceMV};
//!
//! // NEW:
//! use kwavers::analysis::signal_processing::beamforming::adaptive::{MinimumVariance, MUSIC, EigenspaceMV};
//! ```
//!
//! ## Rationale
//!
//! Beamforming algorithms are **signal processing / analysis** operations, not domain primitives.
//! Moving them to the analysis layer enforces proper architectural layering:
//!
//! - **Domain layer** (`domain::sensor`): Sensor geometry, array configuration, data acquisition
//! - **Analysis layer** (`analysis::signal_processing`): Beamforming, DOA estimation, filtering
//!
//! This module now provides backward-compatible re-exports for one minor version cycle.
//!
//! ## Timeline
//!
//! - **Current (v2.14)**: Deprecated re-exports available (with warnings)
//! - **Next minor (v2.15)**: This module will be removed
//!
//! ## References
//! - Van Trees, H. L. (2002). *Optimum Array Processing*. Wiley.
//! - Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis".
//!   *Proceedings of the IEEE*, 57(8), 1408-1418.
//! - Schmidt, R. (1986). "Multiple emitter location and signal parameter estimation".
//!   *IEEE Transactions on Antennas and Propagation*, 34(3), 276-280.

#[cfg(feature = "legacy_algorithms")]
#[path = "adaptive.rs"]
pub mod legacy;

pub mod array_geometry;
pub mod beamformer;
pub mod conventional;

// SSOT: forbid re-exporting duplicated numerics from this module.
// Keep `matrix_utils` available only for legacy-compat shims, not for public SSOT use.
#[cfg(feature = "legacy_algorithms")]
pub mod matrix_utils;

#[cfg(feature = "legacy_algorithms")]
pub mod source_estimation;

pub mod steering;

// Subspace methods depend on complex Hermitian eigendecomposition.
// In strict SSOT mode, these are only compiled if explicitly enabled as legacy.
#[cfg(feature = "legacy_algorithms")]
pub mod subspace;

pub mod tapering;
pub mod weights;

#[cfg(feature = "legacy_algorithms")]
pub use legacy::{MinimumVariance as LegacyMinimumVariance, RobustCapon as LegacyRobustCapon};

pub use array_geometry::{ArrayGeometry, ElementPosition};
pub use beamformer::AdaptiveBeamformer;

#[cfg(feature = "legacy_algorithms")]
pub use source_estimation::SourceEstimationCriterion;

#[cfg(feature = "legacy_algorithms")]
pub use subspace::{EigenspaceMV, MUSIC};

pub use steering::{SteeringMatrix, SteeringVector};
pub use tapering::CovarianceTaper;
pub use weights::{WeightCalculator, WeightingScheme};
