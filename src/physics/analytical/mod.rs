//! Analytical test solutions for physics validation
//!
//! This module provides exact analytical solutions for various wave propagation
//! scenarios to validate numerical solvers.

pub mod dispersion;
pub mod plane_wave;
pub mod utils;

// Re-export main test utilities
pub use utils::PhysicsTestUtils;
