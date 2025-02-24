// physics/optics/mod.rs
pub mod diffusion;
pub mod polarization;
pub mod thermal;

pub use diffusion::LightDiffusion;
pub use polarization::PolarizationModel;
pub use thermal::OpticalThermalModel;