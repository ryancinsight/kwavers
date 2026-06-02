//! Fundamental physics traits defining model interfaces.
//!
//! This module decouples concrete implementations from their abstract contracts,
//! ensuring that cross-cutting concerns adhere to the Dependency Inversion Principle.

pub mod cavitation;
pub mod chemical;
pub mod heterogeneity;
pub mod optical;
pub mod scattering;
pub mod streaming;
pub mod thermal;
pub mod wave;

pub use cavitation::CavitationModelBehavior;
pub use chemical::ChemicalModelTrait;
pub use heterogeneity::HeterogeneityModelTrait;
pub use optical::LightDiffusionModelTrait;
pub use scattering::AcousticScatteringModelTrait;
pub use streaming::StreamingModelTrait;
pub use thermal::ThermalModelTrait;
pub use wave::AcousticWaveModel;
