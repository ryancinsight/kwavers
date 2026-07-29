//! Analytical test solutions for physics validation
//!
//! This module provides exact analytical solutions for various wave propagation
//! scenarios to validate numerical solvers.

pub mod dispersion;
pub mod patterns;
pub mod plane_wave;
pub mod propagation;
pub mod physics_validation;

// Re-export main test utilities
pub use physics_validation::PhysicsTestUtils;

