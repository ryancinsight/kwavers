// src/physics/mod.rs
pub mod chemistry;
pub mod composable;
pub mod heterogeneity;
pub mod mechanics;
pub mod optics;
pub mod scattering;
pub mod thermodynamics;
pub mod traits;

// Re-export commonly used types
pub use composable::{PhysicsComponent, PhysicsContext, PhysicsPipeline, AcousticWaveComponent, ThermalDiffusionComponent};
pub use traits::*;