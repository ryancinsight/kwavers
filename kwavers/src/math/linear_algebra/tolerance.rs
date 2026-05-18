/// Default tolerance for convergence checks
pub const DEFAULT: f64 = 1e-12;
/// Tolerance for matrix rank determination
pub const RANK: f64 = 1e-10;
/// Maximum iterations for iterative methods
pub const MAX_ITERATIONS: usize = 1000;

/// Tolerance used for detecting near-zero complex pivots during LU factorization.
///
/// Intentionally aligned with `RANK` to preserve existing conditioning policy.
pub const COMPLEX_PIVOT: f64 = RANK;

/// Convergence tolerance for the SSOT complex Hermitian eigensolver (Jacobi on real-embedded form).
///
/// Bounds the maximum absolute off-diagonal entry of the embedded real symmetric matrix
/// before declaring convergence.
pub const HERMITIAN_EIG_TOL: f64 = 1e-12;

/// Maximum sweeps (major iterations) for the SSOT complex Hermitian eigensolver (Jacobi).
pub const HERMITIAN_EIG_MAX_SWEEPS: usize = 2048;

/// Convergence tolerance for tridiagonal QR eigensolver (off-diagonal magnitude threshold).
pub const SYMM_TRIDIAG_QR_TOL: f64 = 1e-12;

/// Maximum iterations for implicit QR on symmetric tridiagonal matrices.
pub const SYMM_TRIDIAG_QR_MAX_ITERS: usize = 256;

/// Below this dimension (2n for the embedded real symmetric problem), Jacobi is fine and
/// often faster due to lower constant factors and simpler code paths.
pub const HERMITIAN_EIG_JACOBI_CUTOFF_DIM: usize = 64;
