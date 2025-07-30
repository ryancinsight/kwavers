// src/physics/optics/mod.rs
pub mod diffusion;
pub mod polarization;
pub mod thermal;
pub mod sonoluminescence;

// Re-export commonly used types
pub use diffusion::*;
pub use polarization::*;
pub use sonoluminescence::{SonoluminescenceEmission, EmissionParameters};