// src/physics/mod.rs
pub mod bubble_dynamics;
pub mod chemistry;
pub mod composable;
pub mod heterogeneity;
pub mod mechanics;
pub mod optics;
pub mod plugin; // NEW: Plugin architecture for extensible physics
pub mod scattering;
pub mod state;
pub mod thermodynamics;
pub mod traits;

#[cfg(test)]
pub mod analytical_tests;

// Re-export commonly used types
pub use bubble_dynamics::{BubbleField, BubbleState, BubbleParameters};
pub use composable::{PhysicsComponent, PhysicsContext, PhysicsPipeline, AcousticWaveComponent, ThermalDiffusionComponent, KuznetsovWaveComponent};
pub use state::{PhysicsState, FieldAccessor, field_indices};
pub use traits::*;
pub use plugin::{PhysicsPlugin, PluginManager, PluginMetadata, PluginContext}; // NEW: Plugin exports
pub use optics::sonoluminescence::{SonoluminescenceEmission, EmissionParameters};