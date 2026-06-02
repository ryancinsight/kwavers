//! Optical domain primitives.
//!
//! Reference optical-material data shared across the stack (e.g. chromophore
//! extinction spectra), parallel to the acoustic [`crate::medium`] primitives.

pub mod chromophores;

pub use chromophores::{ExtinctionSpectrum, HemoglobinDatabase};
