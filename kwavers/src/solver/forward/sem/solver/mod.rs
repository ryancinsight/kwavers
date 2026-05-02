//! Spectral Element Method Solver Implementation
//!
//! Integrates high-order GLL basis functions, hexahedral element assembly,
//! Newmark time integration, and boundary condition management.
//!
//! ## Reference
//!
//! - Komatitsch D, Tromp J (1999). Geophys J Int 139, 806–822.

mod config;
mod physics;
mod sem_solver;
#[cfg(test)]
mod tests;

pub use config::SemConfig;
pub use sem_solver::SemSolver;
