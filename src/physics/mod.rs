// src/physics/mod.rs
pub mod bubble_dynamics;
pub mod cavitation_control;
pub mod chemistry;
pub mod constants;  // Physical constants (SSOT)
pub mod phase_modulation;
// composable module removed - use plugin system instead
pub mod field_indices; // Unified field indices (SSOT)
pub mod field_mapping; // NEW: Unified field mapping system
pub mod heterogeneity;
pub mod mechanics;
// migration module removed - composable has been fully removed
pub mod imaging; // Unified imaging physics module
pub mod optics;
pub mod plugin; // Plugin architecture for extensible physics
pub mod sonoluminescence_detector;
pub mod state;
pub mod therapy; // Unified therapy physics module
pub mod thermal; // Unified thermal physics module
pub mod traits;
pub mod wave_propagation; // NEW: Wave propagation with reflection and refraction

pub mod analytical;
pub mod conservation;

#[cfg(test)]
pub mod validation;

// Re-export commonly used types
pub use bubble_dynamics::{BubbleField, BubbleParameters, BubbleState};
// Removed composable exports - use plugin system instead
pub use field_mapping::{
    FieldAccessor as UnifiedFieldAccessor, FieldAccessorMut, UnifiedFieldType,
};
pub use optics::sonoluminescence::{EmissionParameters, SonoluminescenceEmission};
pub use plugin::{PhysicsPlugin, PluginContext, PluginManager, PluginMetadata}; // NEW: Plugin exports
pub use state::PhysicsState;
pub use traits::*;
pub use wave_propagation::{
    AttenuationCalculator, Polarization, PropagationCoefficients, WaveMode,
    WavePropagationCalculator,
};

// Re-export commonly used constants
pub use constants::*;

/// Functional programming utilities for physics calculations
///
/// This module provides a comprehensive set of functional programming tools
/// organized into focused submodules for maintainability and performance.
pub mod functional;