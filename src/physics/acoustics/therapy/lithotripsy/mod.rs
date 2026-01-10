//! Lithotripsy physics submodules.
//!
//! This module provides the physics components for extracorporeal shock wave
//! lithotripsy (ESWL) simulation, including shock wave generation, stone
//! fracture mechanics, cavitation dynamics, and bioeffects assessment.

pub mod bioeffects;
pub mod cavitation_cloud;
pub mod shock_wave;
pub mod stone_fracture;

// Re-export main types
pub use bioeffects::{BioeffectsModel, BioeffectsParameters, SafetyAssessment};
pub use cavitation_cloud::{CavitationCloudDynamics, CloudParameters};
pub use shock_wave::{ShockWaveGenerator, ShockWaveParameters, ShockWavePropagation};
pub use stone_fracture::{StoneFractureModel, StoneMaterial};
