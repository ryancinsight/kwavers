//! Adaptive Beamforming Algorithms
//!
//! # Overview
//!
//! Adaptive beamforming algorithms adjust their weights based on the received data
//! to optimize performance criteria such as signal-to-interference-plus-noise ratio (SINR).
//! Unlike fixed beamformers (e.g., delay-and-sum), adaptive beamformers automatically
//! form nulls in interference directions and maximize desired signal reception.
//!
//! # Mathematical Foundation
//!
//! ## General Framework
//!
//! Given:
//! - **Covariance matrix R** (N×N Hermitian): Captures spatial correlation of sensor array outputs
//! - **Steering vector a** (N×1 complex): Expected signal response for look direction
//!
//! Adaptive beamformers compute optimal weight vector **w** (N×1 complex) according to
//! different optimization criteria.
//!
//! ## Common Optimization Problems
//!
//! ### 1. Minimum Variance Distortionless Response (MVDR)
//!
//! ```text
//! minimize   w^H R w
//! subject to w^H a = 1
//! ```
//!
//! Solution: `w = R^{-1} a / (a^H R^{-1} a)`
//!
//! ### 2. Maximum SINR
//!
//! ```text
//! maximize   w^H R_s w / w^H R_n w
//! ```
//!
//! where R_s is signal covariance and R_n is noise+interference covariance.
//!
//! ### 3. Subspace Methods (MUSIC, ESPRIT)
//!
//! Exploit eigenstructure of covariance matrix to separate signal and noise subspaces.
//!
//! # Architectural Intent (SSOT + Analysis Layer)
//!
//! This module is the **single source of truth** for adaptive beamforming in kwavers.
//! It is placed in the **analysis layer** (Layer 7) because:
//!
//! 1. **Layering**: Beamforming is signal processing (analysis), not sensor geometry (domain)
//! 2. **Dependencies**: Analysis can import domain (geometry) and math (linear algebra)
//! 3. **Reusability**: Works with data from simulations, sensors, and clinical workflows
//!
//! ## SSOT Enforcement (Strict)
//!
//! This module enforces **Single Source of Truth** for numerical operations:
//!
//! - ❌ **NO local matrix inversion** - use `math::linear_algebra::LinearAlgebra`
//! - ❌ **NO silent fallbacks** - return `Err(...)` on numerical failure
//! - ❌ **NO error masking** - all failures are explicit
//! - ❌ **NO dummy weights** - never return steering vector as disguised fallback
//! - ❌ **NO dummy pseudospectrum** - never return 0.0 on failure
//!
//! ## Layer Dependencies
//!
//! ```text
//! analysis::signal_processing::beamforming::adaptive (Layer 7)
//!   ↓ imports from
//! math::linear_algebra (Layer 1) - linear solvers, eigendecomposition
//! domain::sensor (Layer 2) - sensor array geometry
//! core::error (Layer 0) - error types
//! ```
//!
//! # Algorithm Categories
//!
//! ## Data-Adaptive Methods
//!
//! - **MVDR/Capon**: Minimum variance with unit gain constraint
//! - **Robust Capon**: MVDR with uncertainty modeling
//! - **Loaded MVDR**: Diagonal loading for numerical stability
//!
//! ## Subspace Methods
//!
//! - **MUSIC**: Multiple Signal Classification
//! - **ESMV**: Eigenspace Minimum Variance
//! - **ESPRIT**: Estimation of Signal Parameters via Rotational Invariance Techniques
//!
//! ## Tracking Methods
//!
//! - **PAST**: Projection Approximation Subspace Tracking
//! - **OPAST**: Orthonormal PAST
//! - **RLS**: Recursive Least Squares
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use kwavers::analysis::signal_processing::beamforming::adaptive::{
//!     MinimumVariance, AdaptiveBeamformer
//! };
//! use ndarray::{Array1, Array2};
//! use num_complex::Complex64;
//!
//! // Create 8-element linear array covariance matrix
//! let n = 8;
//! let covariance = estimate_covariance_matrix(&sensor_data);
//!
//! // Create steering vector for look direction (e.g., broadside)
//! let steering = compute_steering_vector(n, 0.0_f64.to_radians());
//!
//! // Create MVDR beamformer
//! let mvdr = MinimumVariance::with_diagonal_loading(1e-4);
//!
//! // Compute optimal weights
//! let weights = mvdr.compute_weights(&covariance, &steering)?;
//!
//! // Apply to data
//! let output: Array1<Complex64> = apply_weights(&sensor_data, &weights);
//! ```
//!
//! # Performance Considerations
//!
//! | Algorithm | Complexity | SNR Requirement | Robustness | Use Case |
//! |-----------|------------|-----------------|------------|----------|
//! | DAS | O(N) | Any | High | Baseline |
//! | MVDR | O(N³) | Moderate-High | Medium | Interference rejection |
//! | Robust Capon | O(N³) | Moderate | High | Model uncertainty |
//! | MUSIC | O(N³) | High | Low | High-resolution DOA |
//!
//! # Literature References
//!
//! ## Foundational Papers
//!
//! - Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis."
//!   *Proceedings of the IEEE*, 57(8), 1408-1418.
//!   DOI: 10.1109/PROC.1969.7278
//!
//! - Schmidt, R. O. (1986). "Multiple emitter location and signal parameter estimation."
//!   *IEEE Transactions on Antennas and Propagation*, 34(3), 276-280.
//!   DOI: 10.1109/TAP.1986.1143830
//!
//! - Van Trees, H. L. (2002). *Optimum Array Processing: Part IV of Detection,
//!   Estimation, and Modulation Theory*. Wiley-Interscience.
//!
//! ## Advanced Topics
//!
//! - Li, J., Stoica, P., & Wang, Z. (2003). "On robust Capon beamforming and diagonal loading."
//!   *IEEE Transactions on Signal Processing*, 51(7), 1702-1715.
//!   DOI: 10.1109/TSP.2003.812831
//!
//! - Vorobyov, S. A., et al. (2003). "Robust adaptive beamforming using worst-case
//!   performance optimization: A solution to the signal mismatch problem."
//!   *IEEE Transactions on Signal Processing*, 51(2), 313-324.
//!   DOI: 10.1109/TSP.2002.806865
//!
//! # Migration Note
//!
//! This module was migrated from `domain::sensor::beamforming::adaptive` to
//! `analysis::signal_processing::beamforming::adaptive` as part of the
//! architectural purification effort (ADR 003). The API remains unchanged.
//!
//! ## Migration Timeline
//!
//! - **Phase 2 (Week 3)**: Structure creation ✅
//! - **Phase 3 (Week 3-4)**: Algorithm migration (MVDR, MUSIC, ESMV) 🟡 IN PROGRESS
//! - **Phase 4 (Week 4-5)**: Deprecation and cleanup
//!
//! ## Backward Compatibility
//!
//! The old location `domain::sensor::beamforming::adaptive` will continue to exist
//! with deprecation warnings for one minor version cycle.
//!
//! # Planned Extensions
//!
//! Future adaptive beamforming algorithms (not yet implemented):
//! - `robust_capon`: Robust Capon beamformer with uncertainty quantification
//! - `lcmv`: Linearly Constrained Minimum Variance beamformer with multiple constraints
//! - `gsc`: Generalized Sidelobe Canceller for adaptive interference suppression

use crate::core::error::KwaversResult;
use ndarray::{Array1, Array2};
use num_complex::Complex64;

// Algorithm implementations
pub mod music;
pub mod mvdr;
pub mod subspace;

// Re-export main types
pub use music::MUSIC;
pub use mvdr::MinimumVariance;
pub use subspace::EigenspaceMV;

/// Adaptive beamforming algorithm trait.
///
/// This trait defines the interface for adaptive beamforming algorithms that compute
/// optimal weight vectors from covariance matrices and steering vectors.
///
/// # SSOT Error Semantics (Strict)
///
/// Implementations **MUST**:
/// - Return `Err(...)` on any numerical failure (singular matrices, invalid inputs, etc.)
/// - **NEVER** silently fall back to dummy values (e.g., steering vector, zeros)
/// - **NEVER** mask errors or return incorrect results
///
/// # Mathematical Contract
///
/// Given:
/// - Covariance matrix **R** (N×N Hermitian positive semi-definite)
/// - Steering vector **a** (N×1 complex)
///
/// Compute weight vector **w** (N×1 complex) according to algorithm-specific criteria.
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::adaptive::{
///     AdaptiveBeamformer, MinimumVariance
/// };
///
/// let mvdr = MinimumVariance::default();
/// let weights = mvdr.compute_weights(&covariance, &steering)?;
/// ```
pub trait AdaptiveBeamformer {
    /// Compute beamforming weights for given covariance matrix and steering vector.
    ///
    /// # Parameters
    ///
    /// - `covariance`: Sample covariance matrix **R** (N×N Hermitian)
    /// - `steering`: Steering vector **a** (N×1 complex)
    ///
    /// # Returns
    ///
    /// Optimal weight vector **w** (N×1 complex) according to algorithm's optimization criterion.
    ///
    /// # Errors
    ///
    /// Returns `Err(...)` if:
    /// - Input dimensions are inconsistent (covariance not square, steering length mismatch)
    /// - Covariance matrix is singular or ill-conditioned
    /// - Any numerical operation fails (solve, eigendecomposition, etc.)
    /// - Algorithm-specific constraints are violated
    ///
    /// # Mathematical Guarantees
    ///
    /// Implementations must ensure:
    /// - Output weights satisfy algorithm-specific constraints (e.g., unit gain for MVDR)
    /// - No silent fallbacks or error masking
    /// - Numerically stable computation (e.g., via diagonal loading)
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: Typically O(N²) to O(N³) depending on algorithm
    /// - **Space Complexity**: O(N²) for covariance matrix
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>>;
}

// Implement trait for MinimumVariance
impl AdaptiveBeamformer for MinimumVariance {
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>> {
        // Delegate to inherent method
        self.compute_weights(covariance, steering)
    }
}

// Implement trait for EigenspaceMV
impl AdaptiveBeamformer for EigenspaceMV {
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> KwaversResult<Array1<Complex64>> {
        // Delegate to inherent method
        self.compute_weights(covariance, steering)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::signal_processing::beamforming::test_utilities;
    use approx::assert_relative_eq;

    #[test]
    fn test_adaptive_beamformer_trait() {
        let n = 4;
        let cov = test_utilities::create_diagonal_dominant_covariance(n, 0.05);
        let steering = test_utilities::create_steering_vector(n, 0.0);

        let beamformer: Box<dyn AdaptiveBeamformer> = Box::new(MinimumVariance::default());
        let weights = beamformer
            .compute_weights(&cov, &steering)
            .expect("trait method should work");

        assert_eq!(weights.len(), n);
        for &w in &weights {
            assert!(w.is_finite());
        }
    }

    #[test]
    fn test_mvdr_via_trait() {
        let n = 8;
        let cov = test_utilities::create_diagonal_dominant_covariance(n, 0.05);
        let steering = test_utilities::create_steering_vector(n, 0.0);

        let mvdr = MinimumVariance::with_diagonal_loading(1e-4);
        let weights = mvdr
            .compute_weights(&cov, &steering)
            .expect("should compute");

        // Verify unit gain constraint
        let gain: Complex64 = weights
            .iter()
            .zip(steering.iter())
            .map(|(w, a)| w.conj() * a)
            .sum();

        assert_relative_eq!(gain.re, 1.0, epsilon = 1e-6);
        assert_relative_eq!(gain.im, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_module_exports() {
        let mvdr = MinimumVariance::with_diagonal_loading(1e-4);
        assert_eq!(mvdr.diagonal_loading, 1e-4);

        let music = MUSIC::new(2);
        assert_eq!(music.num_sources, 2);

        let esmv = EigenspaceMV::with_diagonal_loading(2, 1e-4);
        assert_eq!(esmv.num_sources, 2);
        assert_eq!(esmv.diagonal_loading, 1e-4);
    }

    #[test]
    fn test_subspace_via_trait() {
        let n = 8;
        let cov = test_utilities::create_test_covariance(n, 0.2, 0.1);
        let steering = test_utilities::create_steering_vector(n, 0.0);

        // Test ESMV via trait
        let beamformer: Box<dyn AdaptiveBeamformer> =
            Box::new(EigenspaceMV::with_diagonal_loading(2, 1e-4));
        let weights = beamformer
            .compute_weights(&cov, &steering)
            .expect("ESMV via trait should work");

        // Verify unit gain
        let gain: Complex64 = weights
            .iter()
            .zip(steering.iter())
            .map(|(w, a)| w.conj() * a)
            .sum();

        assert_relative_eq!(gain.re, 1.0, epsilon = 1e-6);
        assert_relative_eq!(gain.im, 0.0, epsilon = 1e-6);
    }
}
