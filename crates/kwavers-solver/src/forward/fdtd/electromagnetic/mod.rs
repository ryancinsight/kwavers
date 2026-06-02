//! Electromagnetic FDTD solver
//!
//! This module adapts the existing FDTD solver to solve Maxwell's equations
//! for electromagnetic wave propagation using the Yee staggered grid scheme.

mod maxwell;
mod solver;
#[cfg(test)]
mod tests;
mod types;

pub use types::ElectromagneticFdtdSolver;
