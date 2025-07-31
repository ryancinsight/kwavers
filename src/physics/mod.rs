// src/physics/mod.rs
pub mod bubble_dynamics;
pub mod chemistry;
pub mod composable;
pub mod core; // Core infrastructure for improved architecture
pub mod effects; // Unified physics effects
pub mod heterogeneity;
pub mod mechanics;
pub mod optics;
pub mod pipeline; // Pipeline system for effect orchestration
pub mod plugin; // Plugin architecture for extensible physics
pub mod scattering;
pub mod state;
pub mod thermodynamics;
pub mod traits;

#[cfg(test)]
pub mod analytical_tests;

// Re-export commonly used types
pub use bubble_dynamics::{BubbleField, BubbleState, BubbleParameters};
pub use composable::{PhysicsComponent, PhysicsContext, ThermalDiffusionComponent, KuznetsovWaveComponent};
pub use state::{PhysicsState, FieldAccessor, field_indices};
pub use traits::*;
pub use plugin::{PhysicsPlugin, PluginManager, PluginMetadata, PluginContext};
pub use optics::sonoluminescence::{SonoluminescenceEmission, EmissionParameters};

// Re-export core types from the new architecture
pub use core::{
    PhysicsEffect, EffectCategory, EffectId, EffectContext, EffectState,
    Entity, EntityId, EntityManager, Component,
    PhysicsEvent, EventBus, EventHandler,
    PhysicsSystem, SystemScheduler, SystemContext, EffectSystem,
};

// Re-export pipeline types
pub use pipeline::{PhysicsPipeline, PipelineBuilder, PipelineConfig};