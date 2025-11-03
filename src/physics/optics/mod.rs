// src/physics/optics/mod.rs
pub mod diffusion;
pub mod polarization;
pub mod scattering;
pub mod sonoluminescence;

// Re-export commonly used types
pub use diffusion::*;
pub use polarization::*;
pub use scattering::{MieCalculator, MieParameters, MieResult, RayleighScattering};
pub use sonoluminescence::{EmissionParameters, SonoluminescenceEmission};
