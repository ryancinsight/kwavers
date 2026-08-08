//! Linear algebra operations.
//!
//! Submodules:
//! - `numeric_ops`: generic float trait (`NumericOps`)
//! - `tolerance`: numerical tolerance constants
//! - `sparse`: sparse matrix operations (CSR, COO, iterative solvers)
//!
//! Note: Basic linear algebra operations (solve, inv, LU, QR, Cholesky, symmetric eigen,
//! norms, window functions, optimization) are provided by `leto-ops` and should be used
//! directly instead of the deprecated modules.

pub mod numeric_ops;
pub mod sparse;
pub mod tolerance;

pub use numeric_ops::NumericOps;

// Re-export from leto-ops for backward compatibility
pub use leto_ops::application::linalg::{
    // Complex linear algebra
    complex_inv,
    complex_solve,
    // Eigendecomposition
    eigenvalues,
    hermitian_eigen_jacobi,
    hermitian_eigen_qr,
    l2_normalize,
    l2_normalize_into,
    // Norms
    norm,
    norm_l1,
    norm_l2,
    norm_max,
    symmetric_eigen_jacobi,
    // Iterative solvers
    LsqrConfig,
    LsqrResult,
    LsqrSolver,
};

#[cfg(test)]
mod tests;
