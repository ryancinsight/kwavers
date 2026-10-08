//! Axisymmetric (cylindrical) KZK solver.
//!
//! Provides [CylindricalKZKConfig] and [CylindricalKZKSolver] for
//! propagating finite-amplitude acoustic beams in azimuthally symmetric
//! geometry using Crank-Nicolson radial diffraction.

mod operators;
mod solver;

#[cfg(test)]
mod tests;

pub use solver::{CylindricalKZKConfig, CylindricalKZKSolver};
