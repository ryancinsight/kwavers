// src/physics/effects/mod.rs
//! Unified physics effects module
//! 
//! This module organizes all physics effects into a clear hierarchy,
//! following the improved architecture design.

pub mod wave;
pub mod particle;
pub mod thermal;
pub mod optical;
pub mod chemical;
pub mod mechanical;

// Re-export commonly used effects
pub use wave::{AcousticWaveEffect, ElasticWaveEffect};
pub use particle::{BubbleDynamicsEffect, CavitationEffect, StreamingEffect};
pub use thermal::{HeatDiffusionEffect, ThermalShockEffect};
pub use optical::{SonoluminescenceEffect, PhotoacousticEffect, LightDiffusionEffect};
pub use chemical::{SonochemistryEffect, RadicalProductionEffect};
pub use mechanical::{MaterialDamageEffect, ErosionEffect};