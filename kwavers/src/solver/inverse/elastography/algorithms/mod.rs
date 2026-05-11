//! Shared Algorithms for Elastography Inversion
//!
//! Common utility algorithms used across multiple inversion methods, including
//! spatial smoothing, boundary filling, and feature detection.
//!
//! Partitioned by domain:
//! - `smoothing`  — box, bilateral, and directional 3-D filters.
//! - `boundary`   — nearest-interior boundary extrapolation.
//! - `detection`  — acoustic push-location peak finder.

mod boundary;
mod detection;
mod smoothing;
#[cfg(test)]
mod tests;

pub use boundary::fill_boundaries;
pub use detection::find_push_locations;
pub use smoothing::{directional_smoothing, spatial_smoothing, volumetric_smoothing};
