//! Physics validation module
//!
//! Provides analytical solutions and validation tools for verifying
//! numerical implementations against known physics
//! TODO_AUDIT: P1 - Experimental Validation - Implement experimental validation against Brenner, Yasui, and Putterman sonoluminescence datasets, adding benchmark tests against real-world measurements

pub mod gaussian_beam;

pub use gaussian_beam::{measure_beam_radius, GaussianBeamParameters};
