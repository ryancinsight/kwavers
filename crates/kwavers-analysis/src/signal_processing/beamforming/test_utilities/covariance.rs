//! Test covariance matrix generators.

use eunomia::Complex64;
use leto::{
    Array1,
    Array2,
};

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
pub fn create_test_covariance(n: usize, decay: f64, diagonal_loading: f64) -> Array2<Complex64> {
    let mut r = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let dist = (i as f64 - j as f64).abs();
            let val = (-decay * dist).exp();
            r[(i, j)] = Complex64::new(val, 0.0);
        }
    }

    for i in 0..n {
        r[(i, i)] += Complex64::new(diagonal_loading, 0.0);
    }

    r
}

/// Builder for test covariance matrices with configurable parameters.
///
/// # Example
///
/// ```rust,ignore
/// let cov = TestCovarianceBuilder::new(8)
///     .with_decay(0.3)
///     .with_diagonal_loading(0.05)
///     .build();
/// ```
#[derive(Debug)]
pub struct TestCovarianceBuilder {
    n: usize,
    decay: f64,
    diagonal_loading: f64,
}

impl TestCovarianceBuilder {
    /// Create a new builder with default parameters (decay=0.2, loading=0.1).
    pub fn new(n: usize) -> Self {
        Self {
            n,
            decay: 0.2,
            diagonal_loading: 0.1,
        }
    }

    /// Set the exponential decay parameter (correlation strength).
    pub fn with_decay(mut self, decay: f64) -> Self {
        self.decay = decay;
        self
    }

    /// Set the diagonal loading (regularization parameter).
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
/// R(i,j) = 1 if i=j, else off_diagonal_magnitude / (1 + |i-j|)
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
pub fn create_identity_covariance(n: usize) -> Array2<Complex64> {
    Array2::from_diag(&Array1::from_elem(n, Complex64::new(1.0, 0.0)))
}

/// Create a rank-deficient covariance matrix for testing singular cases.
///
/// R = U Λ U^H where Λ has only `rank` non-zero eigenvalues.
/// # Panics
/// - Panics if assertion fails: `Rank must be <= dimension`.
///
pub fn create_rank_deficient_covariance(n: usize, rank: usize) -> Array2<Complex64> {
    assert!(rank <= n, "Rank must be <= dimension");

    let u = Array2::<Complex64>::eye(n);

    let mut lambda = Array1::<f64>::zeros(n);
    for i in 0..rank {
        lambda[i] = 1.0 - (i as f64) / (rank as f64);
    }

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
