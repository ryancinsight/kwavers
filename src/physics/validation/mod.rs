//! Physics validation tests with known analytical solutions
//!
//! This module contains tests that validate our numerical implementations
//! against known analytical solutions from physics literature.

pub mod analytical_solutions;
pub mod conservation_laws;
pub mod kuznetsov_implicit_test;
pub mod material_properties;
pub mod nonlinear_acoustics;
pub mod numerical_methods;
pub mod wave_equations;

// Re-export test utilities
pub use material_properties::*;
