//! FEM Helmholtz frequency-domain backend.
//!
//! This backend adapts `FemHelmholtzSolver` to the simulation layer without
//! pretending that a frequency-domain solve is a time-stepper.

pub mod backend;
pub mod impl_trait;
#[cfg(test)]
mod tests;

pub use backend::FemHelmholtzBackend;
