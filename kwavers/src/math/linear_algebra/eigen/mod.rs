//! Eigenvalue Decomposition Operations
//!
//! Provides eigenvalue and eigenvector computation for real symmetric and
//! complex Hermitian matrices using a two-tier Jacobi strategy.
//!
//! ## Algorithm Selection
//!
//! **Tier 1: Jacobi (n < 64)** — direct iteration, O(n³) per sweep.
//! **Tier 2: Householder + QR (n ≥ 64)** — O(n³) tridiagonalization + O(n²) per eigenvalue.
//!
//! ## References
//!
//! - Golub & Van Loan (2013) — Matrix Computations (4th ed.), Chapters 8–9
//! - Parlett (1998) — The Symmetric Eigenvalue Problem

mod decomposition;
#[cfg(test)]
mod tests;

pub use decomposition::EigenDecomposition;
