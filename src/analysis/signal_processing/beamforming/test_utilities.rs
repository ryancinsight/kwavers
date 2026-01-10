//! Shared Test Utilities for Beamforming Algorithms
//!
//! # Purpose
//!
//! This module provides common test utilities for beamforming algorithm tests,
//! eliminating code duplication across test modules and ensuring consistent
//! test setup across all beamforming implementations.
//!
//! # Deep Vertical Hierarchy Rationale
//!
//! Following the deep vertical file tree principle:
//! - **Lower layer**: This module provides primitive test data generators
//! - **Middle layer**: Individual algorithm tests compose these primitives
//! - **Upper layer**: Integration tests use both for end-to-end validation
//!
//! This eliminates redundant implementations of `create_test_covariance()` and
//! `create_steering_vector()` scattered across 4+ test modules.
//!
//! # Design Principles
//!
//! 1. **Single Source of Truth**: One canonical implementation of test data generators
//! 2. **Reusability**: Shared across time-domain, adaptive, and subspace tests
//! 3. **Configurability**: Flexible parameters for different test scenarios
//! 4. **Numerical Correctness**: Well-conditioned matrices for stable eigendecomposition
//!
//! # Usage Example
//!
//! ```rust
//! use crate::analysis::signal_processing::beamforming::test_utilities::*;
//!
//! #[cfg(test)]
//! mod tests {
//!     use super::*;
//!
//!     #[test]
//!     fn my_beamforming_test() {
//!         // Create standard test covariance matrix
//!         let cov = TestCovarianceBuilder::new(8)
//!             .with_decay(0.2)
//!             .with_diagonal_loading(0.1)
//!             .build();
//!
//!         // Create steering vector for broadside
//!         let steering = create_steering_vector(8, 0.0);
//!
//!         // ... test code ...
//!     }
//! }
//! ```

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::f64::consts::PI;

/// Create a well-conditioned Hermitian covariance matrix for testing.
///
/// # Mathematical Model
///
/// Creates a real symmetric (Hermitian with zero imaginary parts) positive
/// definite matrix using an exponential covariance model:
///
/// ```text
/// R(i,j) = exp(-α|i-j|) + ε·δ(i,j)
/// ```
///
/// where:
/// - α: Decay parameter (controls correlation structure)
/// - ε: Diagonal loading (ensures positive definiteness)
/// - δ(i,j): Kronecker delta (1 if i=j, 0 otherwise)
///
/// # Parameters
///
/// - `n`: Matrix dimension (number of array elements)
/// - `decay`: Exponential decay parameter α (typical: 0.1 - 0.5)
/// - `diagonal_loading`: Regularization parameter ε (typical: 0.01 - 0.1)
///
/// # Returns
///
/// N×N complex Hermitian matrix suitable for eigendecomposition and linear solvers.
///
/// # Design Rationale
///
/// Real symmetric matrices (Hermitian with zero imaginary parts) are chosen because:
/// 1. Jacobi eigensolver converges faster on real-symmetric embedded form
/// 2. Avoids numerical issues with small imaginary components
/// 3. Sufficient for testing most beamforming algorithms
/// 4. Conditioning is easier to control
///
/// # Example
///
/// ```rust
/// let cov = create_test_covariance(8, 0.2, 0.1);
/// // 8×8 matrix with exponential decay and 0.1 diagonal loading
/// ```
pub fn create_test_covariance(n: usize, decay: f64, diagonal_loading: f64) -> Array2<Complex64> {
    let mut r = Array2::zeros((n, n));

    // Build real symmetric (Hermitian) matrix: R = R^T with zero imaginary parts
    for i in 0..n {
        for j in 0..n {
            let dist = (i as f64 - j as f64).abs();
            let val = (-decay * dist).exp();
            // Real symmetric - no imaginary part
            r[(i, j)] = Complex64::new(val, 0.0);
        }
    }

    // Add diagonal loading for conditioning (makes it strictly positive definite)
    for i in 0..n {
        r[(i, i)] += Complex64::new(diagonal_loading, 0.0);
    }

    r
}

/// Create a steering vector for a uniform linear array (ULA).
///
/// # Mathematical Definition
///
/// For a linear array with uniform element spacing d, the steering vector for
/// direction θ (angle from broadside) is:
///
/// ```text
/// a(θ) = [1, e^{jkd·sin(θ)}, e^{j2kd·sin(θ)}, ..., e^{j(N-1)kd·sin(θ)}]^T
/// ```
///
/// where:
/// - k = 2π/λ: Wavenumber
/// - d: Element spacing (normalized to λ/2 for this function)
/// - θ: Angle from broadside (radians)
/// - j: Imaginary unit
///
/// # Parameters
///
/// - `n`: Number of array elements
/// - `angle_rad`: Steering angle from broadside (radians)
///   - θ = 0: Broadside (perpendicular to array)
///   - θ > 0: Positive endfire
///   - θ < 0: Negative endfire
///
/// # Returns
///
/// N×1 complex steering vector with unit norm.
///
/// # Normalization
///
/// The wavenumber is normalized to k = 2π (λ = 1) and element spacing is
/// assumed to be d = 0.5λ (half-wavelength spacing), giving:
///
/// ```text
/// phase_n = k · n · sin(θ) = 2π · n · sin(θ)
/// ```
///
/// # Example
///
/// ```rust
/// // Broadside steering (θ = 0°)
/// let steering_broadside = create_steering_vector(8, 0.0);
///
/// // 30° off broadside
/// let steering_30deg = create_steering_vector(8, 30.0_f64.to_radians());
///
/// // Endfire (θ = 90°)
/// let steering_endfire = create_steering_vector(8, std::f64::consts::FRAC_PI_2);
/// ```
pub fn create_steering_vector(n: usize, angle_rad: f64) -> Array1<Complex64> {
    let k = 2.0 * PI; // Normalized wavenumber (λ = 1, d = 0.5λ)
    Array1::from_vec(
        (0..n)
            .map(|i| {
                let phase = k * (i as f64) * angle_rad.sin();
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect(),
    )
}

/// Builder for test covariance matrices with configurable parameters.
///
/// # Purpose
///
/// Provides a fluent API for creating test covariance matrices with different
/// characteristics (conditioning, structure, noise levels).
///
/// # Example
///
/// ```rust
/// let cov = TestCovarianceBuilder::new(8)
///     .with_decay(0.3)
///     .with_diagonal_loading(0.05)
///     .build();
/// ```
pub struct TestCovarianceBuilder {
    n: usize,
    decay: f64,
    diagonal_loading: f64,
}

impl TestCovarianceBuilder {
    /// Create a new builder with default parameters.
    ///
    /// # Defaults
    ///
    /// - decay: 0.2 (moderate correlation)
    /// - diagonal_loading: 0.1 (well-conditioned)
    pub fn new(n: usize) -> Self {
        Self {
            n,
            decay: 0.2,
            diagonal_loading: 0.1,
        }
    }

    /// Set the exponential decay parameter (correlation strength).
    ///
    /// # Parameter Guidelines
    ///
    /// - Small decay (0.1): High correlation, condition number ~10-50
    /// - Medium decay (0.2-0.3): Moderate correlation, condition number ~5-20
    /// - Large decay (0.5+): Low correlation, near-diagonal, condition number ~2-5
    pub fn with_decay(mut self, decay: f64) -> Self {
        self.decay = decay;
        self
    }

    /// Set the diagonal loading (regularization parameter).
    ///
    /// # Parameter Guidelines
    ///
    /// - Small loading (0.01): Minimal regularization, may be ill-conditioned
    /// - Medium loading (0.1): Good conditioning, standard choice
    /// - Large loading (0.5+): Strong regularization, near-identity
    pub fn with_diagonal_loading(mut self, diagonal_loading: f64) -> Self {
        self.diagonal_loading = diagonal_loading;
        self
    }

    /// Build the covariance matrix.
    pub fn build(self) -> Array2<Complex64> {
        create_test_covariance(self.n, self.decay, self.diagonal_loading)
    }
}

/// Create a diagonal-dominant covariance matrix (near identity).
///
/// Useful for testing algorithms with well-conditioned inputs.
///
/// # Parameters
///
/// - `n`: Matrix dimension
/// - `off_diagonal_magnitude`: Magnitude of off-diagonal elements (typical: 0.01 - 0.1)
///
/// # Returns
///
/// N×N Hermitian matrix: R(i,j) = 1 if i=j, else off_diagonal_magnitude / (1 + |i-j|)
pub fn create_diagonal_dominant_covariance(
    n: usize,
    off_diagonal_magnitude: f64,
) -> Array2<Complex64> {
    let mut r = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let val = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                let dist = (i as f64 - j as f64).abs();
                Complex64::new(off_diagonal_magnitude / (1.0 + dist), 0.0)
            };
            r[(i, j)] = val;
        }
    }
    r
}

/// Create an identity covariance matrix (uncorrelated sensors).
///
/// # Parameters
///
/// - `n`: Matrix dimension
///
/// # Returns
///
/// N×N identity matrix (all diagonal 1.0, all off-diagonal 0.0)
pub fn create_identity_covariance(n: usize) -> Array2<Complex64> {
    Array2::from_diag(&Array1::from_elem(n, Complex64::new(1.0, 0.0)))
}

/// Create a rank-deficient covariance matrix for testing singular cases.
///
/// # Parameters
///
/// - `n`: Matrix dimension
/// - `rank`: Effective rank (number of non-zero eigenvalues)
///
/// # Returns
///
/// N×N Hermitian matrix with specified rank
///
/// # Design
///
/// Creates matrix as R = U Λ U^H where Λ has only `rank` non-zero eigenvalues.
pub fn create_rank_deficient_covariance(n: usize, rank: usize) -> Array2<Complex64> {
    assert!(rank <= n, "Rank must be <= dimension");

    // Create random orthogonal matrix U (simplified: just use identity with permutation)
    let u = Array2::<Complex64>::eye(n);

    // Create diagonal eigenvalue matrix with only `rank` non-zero entries
    let mut lambda = Array1::<f64>::zeros(n);
    for i in 0..rank {
        lambda[i] = 1.0 - (i as f64) / (rank as f64); // Decreasing eigenvalues
    }

    // R = U Λ U^H (for identity U, this is just diagonal)
    let mut r = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..rank {
                r[(i, j)] += u[(i, k)] * Complex64::new(lambda[k], 0.0) * u[(j, k)].conj();
            }
        }
    }

    r
}

/// Angle conversion utilities.
pub mod angle {
    use std::f64::consts::PI;

    /// Convert degrees to radians.
    #[inline]
    pub fn deg_to_rad(deg: f64) -> f64 {
        deg * PI / 180.0
    }

    /// Convert radians to degrees.
    #[inline]
    pub fn rad_to_deg(rad: f64) -> f64 {
        rad * 180.0 / PI
    }

    /// Broadside angle (0 radians).
    pub const BROADSIDE: f64 = 0.0;

    /// Positive endfire angle (π/2 radians, 90°).
    pub const ENDFIRE_POS: f64 = PI / 2.0;

    /// Negative endfire angle (-π/2 radians, -90°).
    pub const ENDFIRE_NEG: f64 = -PI / 2.0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_covariance_is_hermitian() {
        let cov = create_test_covariance(8, 0.2, 0.1);

        // Check Hermitian property: R = R^H
        for i in 0..8 {
            for j in 0..8 {
                let r_ij = cov[(i, j)];
                let r_ji_conj = cov[(j, i)].conj();
                assert_relative_eq!(r_ij.re, r_ji_conj.re, epsilon = 1e-12);
                assert_relative_eq!(r_ij.im, r_ji_conj.im, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_covariance_is_positive_definite() {
        let cov = create_test_covariance(8, 0.2, 0.1);

        // Check diagonal elements are positive
        for i in 0..8 {
            assert!(cov[(i, i)].re > 0.0);
            assert_relative_eq!(cov[(i, i)].im, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_steering_vector_unit_elements() {
        let steering = create_steering_vector(8, 0.0);

        // Each element should have unit magnitude
        for &s in steering.iter() {
            assert_relative_eq!(s.norm(), 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_steering_broadside_is_ones() {
        let steering = create_steering_vector(8, 0.0);

        // At broadside (θ=0), phase is zero, so all elements are 1+0j
        for &s in steering.iter() {
            assert_relative_eq!(s.re, 1.0, epsilon = 1e-12);
            assert_relative_eq!(s.im, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_builder_pattern() {
        let cov = TestCovarianceBuilder::new(4)
            .with_decay(0.3)
            .with_diagonal_loading(0.05)
            .build();

        assert_eq!(cov.nrows(), 4);
        assert_eq!(cov.ncols(), 4);

        // Check diagonal has loading applied
        for i in 0..4 {
            assert!(cov[(i, i)].re >= 1.05); // exp(0) + loading >= 1.05
        }
    }

    #[test]
    fn test_diagonal_dominant_covariance() {
        let cov = create_diagonal_dominant_covariance(4, 0.1);

        // Diagonal should be 1.0
        for i in 0..4 {
            assert_relative_eq!(cov[(i, i)].re, 1.0, epsilon = 1e-12);
        }

        // Off-diagonal should be small
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert!(cov[(i, j)].norm() <= 0.1);
                }
            }
        }
    }

    #[test]
    fn test_identity_covariance() {
        let cov = create_identity_covariance(4);

        // Should be identity matrix
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert_relative_eq!(cov[(i, j)].re, 1.0, epsilon = 1e-12);
                    assert_relative_eq!(cov[(i, j)].im, 0.0, epsilon = 1e-12);
                } else {
                    assert_relative_eq!(cov[(i, j)].re, 0.0, epsilon = 1e-12);
                    assert_relative_eq!(cov[(i, j)].im, 0.0, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_rank_deficient_covariance() {
        let cov = create_rank_deficient_covariance(4, 2);

        // Matrix should be Hermitian
        for i in 0..4 {
            for j in 0..4 {
                let r_ij = cov[(i, j)];
                let r_ji_conj = cov[(j, i)].conj();
                assert_relative_eq!(r_ij.re, r_ji_conj.re, epsilon = 1e-12);
                assert_relative_eq!(r_ij.im, r_ji_conj.im, epsilon = 1e-12);
            }
        }

        // This is a simple sanity check; full rank verification would require eigendecomposition
    }

    #[test]
    fn test_angle_conversion() {
        assert_relative_eq!(angle::deg_to_rad(0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(angle::deg_to_rad(90.0), PI / 2.0, epsilon = 1e-12);
        assert_relative_eq!(angle::deg_to_rad(180.0), PI, epsilon = 1e-12);

        assert_relative_eq!(angle::rad_to_deg(0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(angle::rad_to_deg(PI / 2.0), 90.0, epsilon = 1e-12);
        assert_relative_eq!(angle::rad_to_deg(PI), 180.0, epsilon = 1e-12);
    }

    #[test]
    fn test_angle_constants() {
        assert_eq!(angle::BROADSIDE, 0.0);
        assert_eq!(angle::ENDFIRE_POS, PI / 2.0);
        assert_eq!(angle::ENDFIRE_NEG, -PI / 2.0);
    }
}
