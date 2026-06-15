//! Optical domain primitives.
//!
//! Reference optical-material data shared across the stack (e.g. chromophore
//! extinction spectra), parallel to the acoustic `kwavers-medium` primitives.

pub mod chromophores;
pub mod optical_transport;

pub use chromophores::{ExtinctionSpectrum, HemoglobinDatabase};
