// physics/optics/polarization/mod.rs
//! Optical polarization physics implementation
//!
//! This module implements proper optical polarization using Jones calculus,
//! providing mathematically correct descriptions of polarization states and transformations.
//!
//! # Jones Calculus
//!
//! Jones vectors describe the polarization state of light:
//! ```text
//! |E_x|   |E_0x|
//! |E_y| = |E_0y|
//! ```
//!
//! Jones matrices describe optical elements that transform polarization:
//! ```text
//! |E_x'|   |m11 m12| |E_x|
//! |E_y'| = |m21 m22| |E_y|
//! ```
//!
//! # References
//!
//! - Jones, R. C. (1941). "A new calculus for the treatment of optical systems"
//! - Born, M., & Wolf, E. (1999). Principles of Optics

pub mod jones_matrix;
pub mod jones_model;
pub mod jones_vector;
pub mod linear;

#[cfg(test)]
mod tests;

use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::{Array3, Array4};
use num_complex::Complex64;
use std::fmt::Debug;

// Re-exports for backward compatibility
pub use jones_matrix::JonesMatrix;
pub use jones_model::JonesPolarizationModel;
pub use jones_vector::JonesVector;
pub use linear::LinearPolarization;

pub trait PolarizationModel: Debug + Send + Sync {
    /// Apply polarization transformation to light field
    fn apply_polarization(
        &mut self,
        fluence: &mut Array3<f64>,
        polarization_state: &mut Array4<Complex64>,
        grid: &Grid,
        medium: &dyn Medium,
    );
}
