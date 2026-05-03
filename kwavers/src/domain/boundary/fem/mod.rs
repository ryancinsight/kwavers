//! FEM Boundary Conditions for Variational Methods
//!
//! Handles boundary conditions for finite element and other variational methods.

pub mod manager;
#[cfg(test)]
mod tests;
pub mod types;

pub use manager::FemBoundaryManager;
pub use types::FemBoundaryCondition;
