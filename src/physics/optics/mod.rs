// optics/mod.rs
pub mod diffusion;
pub mod photoacoustic;
pub mod polarization;
pub mod thermal;

pub use photoacoustic::PhotoacousticSource;
pub use polarization::PolarizationModel;
pub use thermal::OpticalThermalModel;