//! # Adaptive Beamforming Algorithms
//!
//! This module provides adaptive beamforming algorithms for array signal processing.
//!
//! # SSOT enforcement (strict)
//! This crate enforces a **Single Source of Truth** (SSOT) for numerical linear algebra and
//! prohibits error masking. Concretely:
//! - Adaptive algorithms must not implement ad-hoc matrix inversion / eigensolvers locally.
//! - Adaptive algorithms must not silently fall back to dummy weights (e.g. `steering.clone()`)
//!   or dummy pseudospectrum values (e.g. `0.0`) on numerical failure.
//!
//! ## Architectural boundary
//! - **Numerics (solves/inversion/eigendecomposition)** belong to `crate::utils::linear_algebra`.
//! - **Beamforming math orchestration** belongs to `crate::sensor::beamforming` SSOT modules.
//! - This module is a thin composition/orchestration layer that re-exports SSOT-correct APIs.
//!
//! ## Strict mode note
//! Some high-resolution subspace methods (e.g. MUSIC/ESMV) require **complex Hermitian
//! eigendecomposition**. If SSOT does not yet provide a complex eigensolver, those methods are
//! deliberately **disabled** rather than shipped with local numerics or silent fallbacks.
//!
//! # References
//! - Van Trees, H. L. (2002). *Optimum Array Processing*. Wiley.
//! - Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis".
//!   *Proceedings of the IEEE*, 57(8), 1408-1418.
//! - Schmidt, R. (1986). "Multiple emitter location and signal parameter estimation".
//!   *IEEE Transactions on Antennas and Propagation*, 34(3), 276-280.

#[cfg(feature = "legacy_algorithms")]
pub mod adaptive;

pub mod algorithms;

#[cfg(feature = "legacy_algorithms")]
pub(crate) mod algorithms_old;

pub mod array_geometry;
pub mod beamformer;
pub mod conventional;

// SSOT: forbid re-exporting duplicated numerics from this module.
// Keep `matrix_utils` available only for legacy-compat shims, not for public SSOT use.
#[cfg(feature = "legacy_algorithms")]
pub mod matrix_utils;

#[cfg(feature = "legacy_algorithms")]
pub mod opast;

#[cfg(feature = "legacy_algorithms")]
pub mod past;

#[cfg(feature = "legacy_algorithms")]
pub mod source_estimation;

pub mod steering;

// Subspace methods depend on complex Hermitian eigendecomposition.
// In strict SSOT mode, these are only compiled if explicitly enabled as legacy.
#[cfg(feature = "legacy_algorithms")]
pub mod subspace;

pub mod tapering;
pub mod weights;

// Re-export SSOT-correct main algorithm types.
pub use algorithms::{BeamformingAlgorithm, DelayAndSum, MinimumVariance};

#[cfg(feature = "legacy_algorithms")]
pub use adaptive::{MinimumVariance as LegacyMinimumVariance, RobustCapon as LegacyRobustCapon};

pub use array_geometry::{ArrayGeometry, ElementPosition};
pub use beamformer::AdaptiveBeamformer;
pub use conventional::{BeamformingAlgorithm as ConventionalAlgorithm, DelayAndSum as DsLegacy};

#[cfg(feature = "legacy_algorithms")]
pub use opast::OrthonormalSubspaceTracker;

#[cfg(feature = "legacy_algorithms")]
pub use past::SubspaceTracker;

#[cfg(feature = "legacy_algorithms")]
pub use source_estimation::SourceEstimationCriterion;

pub use steering::{SteeringMatrix, SteeringVector};
pub use tapering::CovarianceTaper;
pub use weights::{WeightCalculator, WeightingScheme};

// Legacy algorithms and any legacy numerics are gated explicitly.
#[cfg(feature = "legacy_algorithms")]
pub use algorithms_old::{
    CovarianceTaper as LegacyCovarianceTaper, DelayAndSum as LegacyDelayAndSum,
    EigenspaceMV as LegacyEigenspaceMV, MinimumVariance as LegacyMinimumVariance2,
    OrthonormalSubspaceTracker as LegacyOrthonormalSubspaceTracker,
    RobustCapon as LegacyRobustCapon2, SubspaceTracker as LegacySubspaceTracker,
    MUSIC as LegacyMUSIC,
};
