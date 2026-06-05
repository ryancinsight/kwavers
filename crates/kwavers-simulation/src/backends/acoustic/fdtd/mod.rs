//! FDTD backend adapter for the simulation layer.
//!
//! Adapts `FdtdSolver` (low-level numerics) to `AcousticSolverBackend` (simulation
//! orchestration trait), following the Adapter Pattern.

pub mod backend;
pub mod impl_trait;
#[cfg(test)]
mod tests;

pub use backend::FdtdBackend;
