//! Physics validation module
//!
//! Provides analytical solutions and validation tools for verifying
//! numerical implementations against known physics

pub mod gaussian_beam;

pub use gaussian_beam::{measure_beam_radius, GaussianBeamParameters};
