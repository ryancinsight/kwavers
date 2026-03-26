//! Image registration and resampling for multi-modal fusion.
//!
//! This module provides functionality for aligning images from different modalities
//! into a common coordinate system, including transformation application, resampling,
//! and interpolation methods.

pub mod coordinates;
pub mod resampling;
pub mod transforms;
pub mod validation;

pub use coordinates::generate_coordinate_arrays;
pub use resampling::resample_to_target_grid;
pub use validation::validate_registration_compatibility;

#[cfg(test)]
mod tests;
