// src/physics/optics/mod.rs
pub mod diffusion;
pub mod polarization;
pub mod scattering;
pub mod sonoluminescence;

// Re-export commonly used types
pub use diffusion::{LightDiffusion, OpticalProperties};
pub use polarization::{
    JonesMatrix, JonesPolarizationModel, JonesVector, LinearPolarization, PolarizationModel,
};
pub use scattering::{MieCalculator, MieParameters, MieResult, RayleighScattering};
pub use sonoluminescence::{EmissionParameters, SonoluminescenceEmission};
