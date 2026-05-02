//! # Covariance Matrix Estimation
//!
//! This module provides utilities for computing sample covariance matrices from
//! sensor array data. Covariance matrices are central to adaptive beamforming
//! algorithms (MVDR, MUSIC, etc.) and capture spatial correlation patterns.
//!
//! # Architectural Intent (SSOT + Analysis Layer)
//!
//! ## Design Principles
//!
//! 1. **Single Source of Truth**: All covariance estimation logic lives here
//! 2. **Explicit Failure**: No silent fallbacks, error masking, or dummy outputs
//! 3. **Mathematical Rigor**: Enforce positive semi-definite Hermitian structure
//! 4. **Layer Separation**: Operates on processed data, not domain primitives
//!
//! ## SSOT Enforcement (Strict)
//!
//! This module is the **only** place for covariance estimation:
//!
//! - ❌ **NO local covariance computation** in beamforming algorithms
//! - ❌ **NO silent fallbacks** to identity matrices on failure
//! - ❌ **NO error masking** via dummy outputs
//! - ❌ **NO bypassing** validation checks
//!
//! ## Layer Dependencies
//!
//! ```text
//! analysis::signal_processing::beamforming::covariance (Layer 7)
//!   ↓ imports from
//! math::linear_algebra (Layer 1) - matrix operations, eigendecomposition
//! core::error (Layer 0) - error types
//! ```
//!
//! # Mathematical Foundation
//!
//! ## Sample Covariance Matrix
//!
//! For N sensors and M snapshots (time samples or frequency bins), the sample
//! covariance matrix is:
//!
//! ```text
//! R = (1/M) ∑ₘ₌₁ᴹ x[m] x[m]^H
//! ```
//!
//! where:
//! - `x[m]` (N×1) = snapshot m (sensor data vector)
//! - `H` = Hermitian transpose (conjugate transpose)
//! - `R` (N×N) = Hermitian positive semi-definite matrix
//!
//! ## Properties
//!
//! The covariance matrix **R** satisfies:
//!
//! 1. **Hermitian**: R = R^H (symmetric for real data)
//! 2. **Positive Semi-Definite**: x^H R x ≥ 0 for all x
//! 3. **Rank**: rank(R) ≤ min(N, M)
//! 4. **Trace**: tr(R) = ∑ᵢ σᵢ² (sum of signal powers)
//!
//! ## Diagonal Loading
//!
//! To improve numerical stability and robustness, diagonal loading adds a small
//! positive value to the diagonal:
//!
//! ```text
//! R_loaded = R + ε·I
//! ```
//!
//! where ε > 0 is the loading factor (typically 1e-6 to 1e-2).
//!
//! **Effect**: Regularizes singular/ill-conditioned matrices, provides robustness
//! to model mismatch.
//!
//! # Estimation Methods
//!
//! | Method | Snapshots Required | Computational Cost | Rank | Use Case |
//! |--------|-------------------|-------------------|------|----------|
//! | Sample | M ≥ 2N | O(N²·M) | min(N,M) | General |
//! | Forward-Backward | M ≥ N | O(N²·M) | min(N,2M) | Linear arrays |
//! | Spatial Smoothing | M ≥ L | O(N²·M·L) | Enhanced | Coherent signals |
//!
//! # Literature References
//!
//! - Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis."
//!   *Proceedings of the IEEE*, 57(8), 1408-1418.
//! - Shan, T. J., et al. (1985). "On spatial smoothing for direction-of-arrival
//!   estimation of coherent signals." *IEEE Trans. Acoust., Speech, Signal Process.*
//! - Pillai, S. U., & Kwon, B. H. (1989). "Forward/backward spatial smoothing
//!   techniques for coherent signal identification." *IEEE Trans. Acoust., Speech,
//!   Signal Process.*, 37(1), 8-15.

mod estimation;
mod sensor_covariance;
#[cfg(test)]
mod tests;
mod validation;

pub use estimation::{estimate_forward_backward_covariance, estimate_sample_covariance};
pub use sensor_covariance::{
    CovarianceEstimator, CovariancePostProcess, SpatialSmoothing, SpatialSmoothingComplex,
};
pub use validation::{is_hermitian, trace, validate_covariance_matrix};
