//! Image registration and resampling for multi-modal fusion.
//!
//! This module provides functionality for aligning images from different modalities
//! into a common coordinate system, including transformation application, resampling,
//! and interpolation methods.

pub mod adapter;
pub mod coordinates;
pub mod resampling;
pub mod transforms;
pub mod validation;

/// Identity 4×4 homogeneous transform, flattened row-major.
///
/// Single source of truth for the seed passed to ritk's intensity-based
/// registration (`rigid_registration_mutual_info` / `affine_registration_mutual_info`),
/// which now accept a raw `&[f64; 16]` rather than a dedicated transform type.
pub const IDENTITY_HOMOGENEOUS: [f64; 16] = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

pub use adapter::{
    FusionBenchmarkCase, FusionRegistrationResult, FusionValidationCase, RitkRegistrationEngine,
};
pub use coordinates::generate_coordinate_arrays;
pub use resampling::resample_to_target_grid;
pub use validation::validate_registration_compatibility;

#[cfg(test)]
mod tests;
