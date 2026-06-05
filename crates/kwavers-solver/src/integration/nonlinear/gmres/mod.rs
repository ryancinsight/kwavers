//! GMRES (Generalized Minimal Residual) Krylov subspace solver.
//!
//! Implements restarted GMRES(m) with Modified Gram-Schmidt orthogonalization
//! for large sparse linear systems A·x = b.
//!
//! # References
//!
//! - Saad & Schultz (1986): SIAM JSC 7(3), 856–869. DOI: 10.1137/0907058

pub mod config;
pub mod solver;
#[cfg(test)]
mod tests;
pub mod types;

pub use config::GMRESConfig;
pub use solver::GMRESSolver;
pub use types::GmresConvergenceInfo;
